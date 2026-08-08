"""OpenVLA 对抗纹理优化使用的 nvdiffrast 可微 renderer。

该 module 负责 mesh/UV 数据加载、顶点颜色扰动、可微光栅化、简化光照模型、
光照校准和 UV texture bake。它不负责 MuJoCo 相机矩阵、背景采集、图像合成
或攻击损失，因此导入 module 时不会创建环境或加载 VLA 模型。

``DifferentiableRenderer`` 的构造会创建 CUDA rasterizer context；只有实例化
需要可用 GPU，导入类定义和阅读类型信息不需要 GPU。

主要数据流为：

``OBJ + 原始 PNG → 几何/UV/原始纹理 → Surface Delta
→ MVP + nvdiffrast → 对抗前景与 mask → MuJoCo 背景合成 → OpenVLA``。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    overload,
)

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from numpy.typing import NDArray
from PIL import Image

from .configuration import TextureParameterizationKind
from .spectral_geometry import (
    build_render_to_geometry_map,
    load_obj_geometry,
    load_spectral_basis,
    validate_basis_geometry,
)
from .texture_parameterization import (
    GeometryVertexTextureParameterization,
    SpectralTextureParameterization,
    SurfaceStepStats,
    SurfaceTextureParameterization,
    surface_normalized_step_,
)


Tensor: TypeAlias = torch.Tensor
ImageResolution: TypeAlias = tuple[int, int]
Device: TypeAlias = str | torch.device
PathLike: TypeAlias = str | Path
FloatingArray: TypeAlias = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class AdversarialTextureLoadResult:
    """加载外部对抗纹理后的参数化方式与扰动统计。"""

    source_kind: Literal["parameter", "baked_texture"]
    max_absolute_delta: float
    nonzero_percentage: float


@dataclass
class SurfaceDeltaGradientCapture:
    """一次受控 forward 中实际进入 renderer 的 Surface Delta tensors。

    Gate 2E 在 backward 后读取这些非叶子 tensor 的 ``.grad``。普通训练默认
    不启用捕获，因此不会额外保留中间梯度或延长 autograd graph 生命周期。
    """

    tensors: list[Tensor]

    def summed_gradient(self) -> Tensor:
        """按 renderer 调用累加同 shape 梯度并返回 detach tensor。"""

        if not self.tensors:
            raise RuntimeError("Surface Delta 捕获没有记录任何 renderer forward")
        gradients: list[Tensor] = []
        for tensor in self.tensors:
            if tensor.grad is None:
                raise RuntimeError("捕获的 Surface Delta 没有 backward 梯度")
            gradients.append(tensor.grad.detach())
        reference_shape = gradients[0].shape
        if any(gradient.shape != reference_shape for gradient in gradients):
            raise RuntimeError("多次 renderer 调用的 Surface Delta shape 不一致")
        return torch.stack(gradients, dim=0).sum(dim=0)


@dataclass(frozen=True)
class RendererEvidence:
    """同一次 nvdiffrast rasterization 产生的完整实例证据。

    RGB 为 float32 NHWC ``[1,H,W,3]``，visibility mask 为 float32 NHWC
    ``[1,H,W,1]``，raw raster 为 float32 ``[1,H,W,4]``。raw raster 用于
    triangle/barycentric correspondence；不得从 RGB 或 mask 反推 face。
    """

    adversarial_rgb: Tensor
    clean_rgb: Tensor
    visibility_mask: Tensor
    raster: Tensor


def resolve_position_offset(
    pos_offset: Optional[Sequence[float]],
) -> tuple[float, float, float]:
    """校验并返回 renderer 的 model-space XYZ 平移。

    未提供校准值时返回精确零偏移。历史实现把
    ``[0.02, 0.01, 0.025]`` 无条件用于所有物体，真实轮廓诊断表明它会给
    Spatial bowl 和 Object 物体引入十余像素的系统性错位。特殊资产如果确实
    需要平移，仍可通过 ``pos_offset`` 显式传入。
    """
    if pos_offset is None:
        return (0.0, 0.0, 0.0)

    offset_array: FloatingArray = np.asarray(
        pos_offset,
        dtype=np.float32,
    )
    if offset_array.shape != (3,) or not np.isfinite(offset_array).all():
        raise ValueError("pos_offset 必须包含三个有限的 XYZ 浮点数")
    return (
        float(offset_array[0]),
        float(offset_array[1]),
        float(offset_array[2]),
    )


class DifferentiableRenderer(nn.Module):
    """在保留原始 UV 的前提下施加可学习曲面颜色增量。

    ``legacy_vertex`` 保留原实现，优化 seam-split 渲染顶点的无界
    ``adv_noise [V, 3]``。``geometry_vertex`` 与 ``spectral`` 共享新的
    **Surface Delta** 路径：先在原始 UV texture 上采样 clean color，再把
    ``render_delta [V, 3]`` 插值到像素并相加。谱方法的唯一可学习参数为
    ``coefficients [K, 3]``。
    """

    def __init__(
        self,
        mesh_path: PathLike,
        orig_texture_path: Optional[PathLike] = None,
        device: Device = "cuda",
        scale_xyz: Optional[Sequence[float]] = None,
        pos_offset: Optional[Sequence[float]] = None,
        epsilon: float = 128.0 / 255.0,
        texture_parameterization: TextureParameterizationKind = (
            "legacy_vertex"
        ),
        spectral_basis_path: Optional[PathLike] = None,
        spectral_basis_count: int = 128,
    ) -> None:
        """加载物体资产并创建 CUDA rasterizer context。

        Args:
            mesh_path: OBJ 等 trimesh 支持的 mesh 文件路径。加载失败时沿用
                原实现，打印 warning 并使用 ``0.1 × 0.1 × 0.001`` 的薄盒。
            orig_texture_path: 原始 RGB texture 路径；不存在时使用常量灰色纹理。
            device: renderer tensor 所在 device。当前 nvdiffrast 实现要求 CUDA。
            scale_xyz: mesh 三轴缩放，长度应为 3；缺省为 ``[1, 1, 1]``。
            pos_offset: 可选的 model-space 顶点平移，长度必须为 3。缺省为
                精确零偏移；只有经过轮廓对齐测量的特殊资产才应显式传值。
            epsilon: 实际 **Surface Delta** 的最大逐通道颜色扰动幅度。
            texture_parameterization: ``legacy_vertex`` 保留旧行为；
                ``geometry_vertex`` 为公平的高维 UV 保真基线；
                ``spectral`` 优化低维谱系数。
            spectral_basis_path: ``spectral`` 模式必需的 NPZ 谱基产物。
            spectral_basis_count: 从产物中取前 K 个非恒定低频模态。
        """
        super().__init__()
        self.device: Device = device
        self.epsilon: float = epsilon
        self.texture_parameterization_kind: TextureParameterizationKind = (
            texture_parameterization
        )
        self.tex_h: int = 256
        self.tex_w: int = 256

        resolved_offset: tuple[float, float, float] = resolve_position_offset(
            pos_offset
        )
        self.pos_offset: Tensor
        self.register_buffer(
            "pos_offset",
            torch.tensor(
                resolved_offset,
                dtype=torch.float32,
                device=device,
            ),
        )  # [3]

        resolved_scale: Sequence[float] = (
            scale_xyz if scale_xyz is not None else [1.0, 1.0, 1.0]
        )
        scale_array: FloatingArray = np.asarray(resolved_scale, dtype=np.float64)

        mesh_load_succeeded: bool = True
        try:
            mesh: trimesh.Trimesh = trimesh.load(mesh_path, force="mesh")
        except Exception:
            mesh_load_succeeded = False
            print(f"[WARNING] Failed to load mesh from {mesh_path}, using dummy box.")
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.001])

        # vertices: [num_vertices, 3]；faces: [num_faces, 3]。
        render_vertices_unscaled: FloatingArray = np.asarray(
            mesh.vertices,
            dtype=np.float64,
        )
        render_faces: NDArray[np.int64] = np.asarray(
            mesh.faces,
            dtype=np.int64,
        )
        vertices: FloatingArray = (
            render_vertices_unscaled * scale_array[None, :]
        )
        self.num_vertices: int = len(vertices)
        self.pos: Tensor
        self.faces: Tensor
        self.register_buffer(
            "pos", torch.from_numpy(vertices.astype(np.float32)).to(device)
        )
        self.register_buffer(
            "faces", torch.from_numpy(render_faces.astype(np.int32)).to(device)
        )

        # uv: [num_uv_vertices, 2]；uv_idx: [num_faces, 3]。
        self.uv: Tensor
        self.uv_idx: Tensor
        has_uv: bool = (
            hasattr(mesh.visual, "uv")
            and mesh.visual.uv is not None
            and len(mesh.visual.uv) > 0
        )
        if has_uv:
            uv_array: FloatingArray = np.asarray(mesh.visual.uv).astype(np.float32)
            self.register_buffer("uv", torch.from_numpy(uv_array).to(device))
            has_face_uv: bool = (
                hasattr(mesh.visual, "face_uv")
                and mesh.visual.face_uv is not None
                and len(mesh.visual.face_uv) == len(mesh.faces)
            )
            if has_face_uv:
                face_uv_array: NDArray[np.int32] = np.asarray(
                    mesh.visual.face_uv
                ).astype(np.int32)
                self.register_buffer(
                    "uv_idx", torch.from_numpy(face_uv_array).to(device)
                )
            else:
                # 没有独立 UV face index 时，沿用几何 face index。
                self.register_buffer("uv_idx", self.faces)
        else:
            # 没有 UV 时，用顶点 XY 的包围盒归一化结果生成平面 UV。
            vertex_tensor: Tensor = torch.from_numpy(vertices).to(device)
            generated_uv: Tensor = vertex_tensor[:, :2]
            if generated_uv.numel() > 0:
                uv_min: Tensor = generated_uv.min(0, keepdim=True)[0]
                uv_max: Tensor = generated_uv.max(0, keepdim=True)[0]
                generated_uv = (generated_uv - uv_min) / (uv_max - uv_min + 1e-8)
            else:
                generated_uv = torch.zeros((len(vertices), 2), device=device)
            self.register_buffer("uv", generated_uv.float())
            self.register_buffer("uv_idx", self.faces)

        # vn: [num_vertices, 3]，每行是单位 vertex normal。
        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            mesh.fix_normals()
        # np.array 创建独立副本，避免下面的 fallback 修正意外修改 trimesh 对象。
        vertex_normals: FloatingArray = np.array(
            getattr(mesh, "vertex_normals", np.zeros_like(vertices))
        )
        if vertex_normals.sum() == 0:
            vertex_normals[:, 2] = 1.0
        vertex_normals = vertex_normals / (
            np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8
        )
        self.vn: Tensor
        self.register_buffer(
            "vn", torch.from_numpy(vertex_normals.astype(np.float32)).to(device)
        )

        # 这是构造阶段唯一必须使用 CUDA 的操作。module import 不会执行到这里。
        self.glctx: dr.RasterizeCudaContext = dr.RasterizeCudaContext()

        # orig_texture 使用 NHWC：[1, texture_height, texture_width, 3]。
        self.orig_texture: Tensor
        if orig_texture_path is not None and os.path.exists(orig_texture_path):
            image_raw: Image.Image = Image.open(orig_texture_path).convert("RGB")
            self.tex_w, self.tex_h = image_raw.size
            flipped_image: Image.Image = image_raw.transpose(Image.FLIP_TOP_BOTTOM)
            texture_tensor: Tensor = (
                torch.from_numpy(np.array(flipped_image)).float() / 255.0
            )
            self.register_buffer(
                "orig_texture",
                texture_tensor.unsqueeze(0).to(device).contiguous(),
            )
        else:
            fallback_texture: Tensor = (
                torch.tensor([0.45, 0.45, 0.45], device=device)
                .view(1, 1, 1, 3)
                .expand(1, self.tex_h, self.tex_w, 3)
                .contiguous()
            )
            self.register_buffer("orig_texture", fallback_texture)

        # orig_vertex_colors: [num_vertices, 3]。
        original_vertex_colors: Tensor = self._sample_uv_texture_at_vertices()
        self.orig_vertex_colors: Tensor
        self.register_buffer("orig_vertex_colors", original_vertex_colors)
        # legacy 参数始终保留，以兼容历史 .pt 产物和默认实验命令。新的
        # Geometry/Spectral 路径只通过 get_texture_param 暴露自己的参数。
        self.adv_noise: nn.Parameter = nn.Parameter(
            torch.zeros(self.num_vertices, 3, dtype=torch.float32, device=device)
        )
        self.surface_parameterization: Optional[
            GeometryVertexTextureParameterization
            | SpectralTextureParameterization
        ] = None
        # spectral_eigenvalues: float [K]。仅 spectral adapter 设置；作为
        # buffer 随 module device 迁移，并与 basis 列严格同序。
        self.spectral_eigenvalues: Optional[Tensor]
        self.register_buffer("spectral_eigenvalues", None)
        if texture_parameterization != "legacy_vertex":
            if not mesh_load_succeeded:
                raise ValueError(
                    "曲面参数化要求 mesh 成功加载，不能使用 dummy fallback"
                )
            geometry_vertices, geometry_faces = load_obj_geometry(mesh_path)
            render_to_geometry: NDArray[np.int64] = (
                build_render_to_geometry_map(
                    geometry_vertices,
                    geometry_faces,
                    render_vertices_unscaled,
                    render_faces,
                )
            )
            if texture_parameterization == "geometry_vertex":
                self.surface_parameterization = (
                    GeometryVertexTextureParameterization(
                        render_to_geometry,
                        num_geometry_vertices=len(geometry_vertices),
                        epsilon=epsilon,
                        device=device,
                    )
                )
            elif texture_parameterization == "spectral":
                if spectral_basis_path is None:
                    raise ValueError(
                        "spectral 参数化必须提供 spectral_basis_path"
                    )
                basis_data = load_spectral_basis(
                    spectral_basis_path,
                    max_basis=spectral_basis_count,
                    include_constant=False,
                )
                self.spectral_eigenvalues = torch.as_tensor(
                    basis_data.eigenvalues,
                    dtype=torch.float32,
                    device=device,
                )
                validate_basis_geometry(
                    basis_data,
                    geometry_vertices,
                    geometry_faces,
                )
                self.surface_parameterization = (
                    SpectralTextureParameterization(
                        basis_data.basis,
                        render_to_geometry,
                        epsilon=epsilon,
                        device=device,
                    )
                )
            else:
                raise ValueError(
                    f"未知纹理参数化: {texture_parameterization}"
                )
        self.light_dir: Tensor = F.normalize(
            torch.tensor([0.2, 0.2, 1.0], device=device), dim=0
        )  # [3]

        # 三个校准 buffer 都是逐 RGB channel 参数，shape 为 [3]。
        self.calib_scale: Tensor
        self.calib_bias: Tensor
        self.calib_gamma: Tensor
        self.register_buffer("calib_scale", torch.ones(3, device=device))
        self.register_buffer("calib_bias", torch.zeros(3, device=device))
        self.register_buffer("calib_gamma", torch.ones(3, device=device))

        self.ambient_strength: float = 0.42
        self.diffuse_strength: float = 0.48
        self.specular_strength: float = 0.05
        self.specular_shininess: float = 24.0
        self.shadow_strength: float = 0.15
        self.shadow_gamma: float = 1.8
        self.min_light: float = 0.16
        self._surface_delta_gradient_capture: Optional[
            SurfaceDeltaGradientCapture
        ] = None

    @contextmanager
    def capture_surface_delta_gradients(
        self,
    ) -> Iterator[SurfaceDeltaGradientCapture]:
        """仅在 ``with`` 范围捕获实际渲染路径的 Surface Delta 梯度。"""

        if self._surface_delta_gradient_capture is not None:
            raise RuntimeError("Surface Delta 梯度捕获不能嵌套")
        capture = SurfaceDeltaGradientCapture(tensors=[])
        self._surface_delta_gradient_capture = capture
        try:
            yield capture
        finally:
            self._surface_delta_gradient_capture = None

    def _sample_uv_texture_at_vertices(self) -> Tensor:
        """从原始 UV texture 采样每个几何顶点的 RGB。

        Returns:
            浮点 tensor，形状为 ``[num_vertices, 3]``，数值位于 ``[0, 1]``。
        """
        with torch.no_grad():
            if len(self.uv) == self.num_vertices:
                uv_vertices: Tensor = self.uv
            else:
                # 一个几何顶点可能由多个 UV 顶点引用；这里对其 UV 坐标求平均。
                uv_sum: Tensor = torch.zeros(
                    self.num_vertices, 2, device=self.device
                )  # [num_vertices, 2]
                uv_count: Tensor = torch.zeros(
                    self.num_vertices, 1, device=self.device
                )  # [num_vertices, 1]
                face_vertex_indices: Tensor = self.faces.long()  # [num_faces, 3]
                face_uv_indices: Tensor = self.uv_idx.long()  # [num_faces, 3]
                face_ones: Tensor = torch.ones(
                    len(face_vertex_indices), 1, device=self.device
                )
                for local_corner in range(3):
                    vertex_indices: Tensor = face_vertex_indices[:, local_corner]
                    uv_indices: Tensor = face_uv_indices[:, local_corner]
                    uv_sum.scatter_add_(
                        0,
                        vertex_indices.unsqueeze(1).expand(-1, 2),
                        self.uv[uv_indices],
                    )
                    uv_count.scatter_add_(0, vertex_indices.unsqueeze(1), face_ones)
                uv_vertices = uv_sum / uv_count.clamp_min(1)

            # nvdiffrast texture query 使用 [batch, query_height, query_width, 2]。
            uv_query: Tensor = uv_vertices.unsqueeze(0).unsqueeze(0)
            sampled_colors: Tensor = dr.texture(
                self.orig_texture.contiguous(),
                uv_query.contiguous(),
                filter_mode="linear",
            )
            return sampled_colors.squeeze(0).squeeze(0).contiguous()

    def get_texture_param(self) -> nn.Parameter:
        """返回当前 adapter 的唯一可学习纹理参数。

        legacy/Geometry Vertex 返回 ``[V, 3]`` 或 ``[N, 3]``；Spectral 返回
        ``[K, 3]``。
        """
        if self.surface_parameterization is not None:
            return self.surface_parameterization.coefficients
        return self.adv_noise

    def reset_texture(self) -> None:
        """把当前纹理参数原地清零。"""
        if self.surface_parameterization is not None:
            self.surface_parameterization.reset_parameters()
            return
        with torch.no_grad():
            self.adv_noise.data.fill_(0.0)

    def get_texture_parameterization_name(self) -> str:
        """返回用于日志和产物命名的稳定 adapter 名称。"""
        return self.texture_parameterization_kind

    def get_render_to_geometry_mapping(self) -> Tensor:
        """返回严格的 ``renderer vertex -> OBJ geometry vertex`` seam 映射。

        legacy 参数化没有独立保存原始 OBJ 几何拓扑，不能作为 Fixed Vertex
        Support correspondence 的来源，因此显式拒绝而不猜测恒等映射。
        """

        if self.surface_parameterization is None:
            raise RuntimeError(
                "legacy renderer 不提供严格 render_to_geometry 映射"
            )
        return self.surface_parameterization.render_to_geometry

    def get_spectral_basis_and_eigenvalues(
        self,
    ) -> tuple[Tensor, Tensor]:
        """返回审计使用的 ``basis [N,K]`` 与 ``eigenvalues [K]``。

        Geometry/Legacy 没有谱模态语义，显式报错，禁止梯度审计把逐顶点参数
        静默解释为谱系数。
        """
        parameterization = self.surface_parameterization
        eigenvalues: Optional[Tensor] = self.spectral_eigenvalues
        if (
            not isinstance(
                parameterization,
                SpectralTextureParameterization,
            )
            or eigenvalues is None
        ):
            raise RuntimeError(
                "当前 renderer 不是 spectral 参数化，无法读取谱基"
            )
        if eigenvalues.shape != (parameterization.num_basis,):
            raise RuntimeError(
                "谱基与特征值数量不一致："
                f"{parameterization.num_basis} != "
                f"{tuple(eigenvalues.shape)}"
            )
        return parameterization.basis, eigenvalues

    def get_surface_delta(self) -> Tensor:
        """返回与 renderer 顶点对齐的 float ``[V, 3]`` Surface Delta。"""
        if self.surface_parameterization is not None:
            surface_delta: Tensor = (
                self.surface_parameterization.render_delta()
            )
        else:
            surface_delta = torch.tanh(self.adv_noise) * self.epsilon
        capture: Optional[SurfaceDeltaGradientCapture] = (
            self._surface_delta_gradient_capture
        )
        if (
            capture is not None
            and torch.is_grad_enabled()
            and surface_delta.requires_grad
        ):
            surface_delta.retain_grad()
            capture.tensors.append(surface_delta)
        return surface_delta

    def step_surface_parameterization_(
        self,
        gradient: Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        """对 Geometry/Spectral adapter 执行统一的曲面归一化更新。"""
        parameterization: Optional[SurfaceTextureParameterization] = (
            self.surface_parameterization
        )
        if parameterization is None:
            raise RuntimeError(
                "legacy_vertex 不支持 surface-normalized 更新"
            )
        return surface_normalized_step_(
            parameterization,
            gradient,
            surface_step,
        )

    def load_adversarial_texture(
        self,
        texture_path: PathLike,
    ) -> AdversarialTextureLoadResult:
        """从参数 ``.pt`` 或 bake 后的 RGB 图像恢复 ``adv_noise``。

        ``.pt`` 文件应直接包含 float tensor，shape ``[num_vertices, 3]``。
        其他后缀沿用现有行为，按 RGB 图像处理：先恢复 renderer 内部使用的
        竖直翻转 UV 方向，再通过当前 mesh UV 把 texture 采样到顶点。最终
        delta 截断到 ``(-epsilon, epsilon)``，通过 ``atanh`` 转回无界参数。

        该方法集中维护 ``orig_texture``、``orig_vertex_colors`` 和私有 UV
        sampler 的不变量；调用方不需要临时修改 renderer buffer。
        """
        resolved_path: Path = Path(texture_path)
        source_kind: Literal["parameter", "baked_texture"]
        delta: Tensor

        texture_parameter: nn.Parameter = self.get_texture_param()
        if str(resolved_path).endswith(".pt"):
            source_kind = "parameter"
            loaded_noise: Tensor = torch.load(
                resolved_path,
                map_location=texture_parameter.device,
            )
            if loaded_noise.shape != texture_parameter.shape:
                raise ValueError(
                    f"纹理参数 shape 不匹配: {tuple(loaded_noise.shape)} != "
                    f"{tuple(texture_parameter.shape)}"
                )
            with torch.no_grad():
                texture_parameter.copy_(loaded_noise)
                if self.surface_parameterization is not None:
                    delta = self.surface_parameterization.geometry_delta()
                else:
                    delta = torch.tanh(loaded_noise) * self.epsilon
        else:
            if self.surface_parameterization is not None:
                raise ValueError(
                    "Geometry/Spectral 参数化不能从 bake PNG 无损恢复参数；"
                    "迁移评估应把 PNG 直接激活为 MuJoCo Active Texture"
                )
            source_kind = "baked_texture"
            with Image.open(resolved_path) as baked_image:
                # baked_pixels: float32 HWC [texture_height, texture_width, 3]。
                baked_pixels: FloatingArray = (
                    np.array(baked_image).astype(np.float32) / 255.0
                )
            # baked_texture: float32 NHWC
            # [1, texture_height, texture_width, 3]。
            baked_texture: Tensor = (
                torch.from_numpy(baked_pixels)
                .unsqueeze(0)
                .to(texture_parameter.device)
            )
            stored_texture: Tensor = torch.flip(
                baked_texture,
                dims=[1],
            ).contiguous()

            original_texture: Tensor = self.orig_texture
            try:
                self.orig_texture = stored_texture
                loaded_vertex_colors: Tensor = (
                    self._sample_uv_texture_at_vertices()
                )
            finally:
                self.orig_texture = original_texture

            delta = (
                loaded_vertex_colors - self.orig_vertex_colors
            ).clamp(
                -self.epsilon + 1e-6,
                self.epsilon - 1e-6,
            )
            loaded_noise = torch.atanh(delta / self.epsilon)
            with torch.no_grad():
                texture_parameter.copy_(loaded_noise)

        max_absolute_delta: float = float(delta.abs().max().item())
        nonzero_percentage: float = float(
            (delta.abs() > 1e-3).float().mean().item() * 100.0
        )
        return AdversarialTextureLoadResult(
            source_kind=source_kind,
            max_absolute_delta=max_absolute_delta,
            nonzero_percentage=nonzero_percentage,
        )

    def calibrate_lighting(
        self,
        mvp: Tensor,
        mujoco_clean_rgb: Tensor,
        ema: float = 0.0,
        model_rot: Optional[Tensor] = None,
    ) -> None:
        """让简化 renderer 的干净颜色拟合 MuJoCo 相机中的目标物体颜色。

        Args:
            mvp: model-view-projection 矩阵，形状 ``[4, 4]``。
            mujoco_clean_rgb: NCHW 相机 RGB，形状 ``[1, 3, H, W]``。
            ema: 多帧校准的历史参数权重；0 表示直接使用当前帧拟合结果。
            model_rot: 世界旋转矩阵，形状 ``[3, 3]``，用于旋转 vertex normal。
        """
        with torch.no_grad():
            height: int = int(mujoco_clean_rgb.shape[-2])
            width: int = int(mujoco_clean_rgb.shape[-1])
            clean_lit: Tensor
            visibility_mask: Tensor
            _, clean_lit, visibility_mask = self.render(
                mvp,
                resolution=(height, width),
                return_clean=True,
                model_rot=model_rot,
            )

            # visible_pixels: [H, W]；predicted/target RGB: [H, W, 3]。
            visible_pixels: Tensor = visibility_mask.squeeze(0).squeeze(-1) > 0.5
            predicted_rgb: Tensor = clean_lit.squeeze(0)
            target_rgb: Tensor = mujoco_clean_rgb.squeeze(0).permute(1, 2, 0)

            if visible_pixels.sum() < 50:
                print(
                    "[WARN] 光照校准: 目标物体在画面里的像素太少，"
                    "跳过校准，沿用默认光照参数。"
                )
                return

            fitted_scales: list[Tensor] = []
            fitted_biases: list[Tensor] = []
            fitted_gammas: list[Tensor] = []
            for channel_index in range(3):
                predicted_channel: Tensor = predicted_rgb[..., channel_index][
                    visible_pixels
                ].clamp(1e-4, 1.0 - 1e-4)
                target_channel: Tensor = target_rgb[..., channel_index][visible_pixels]
                gamma_candidates: Tensor = torch.tensor(
                    [1.00, 1.15, 1.30, 1.45, 1.60],
                    device=predicted_channel.device,
                    dtype=predicted_channel.dtype,
                )
                best_error: Optional[Tensor] = None
                best_scale: Tensor = torch.tensor(
                    1.0, device=predicted_channel.device, dtype=predicted_channel.dtype
                )
                best_bias: Tensor = torch.tensor(
                    0.0, device=predicted_channel.device, dtype=predicted_channel.dtype
                )
                best_gamma: Tensor = torch.tensor(
                    1.0, device=predicted_channel.device, dtype=predicted_channel.dtype
                )

                for gamma in gamma_candidates:
                    gamma_corrected: Tensor = predicted_channel.pow(gamma)
                    predicted_mean: Tensor = gamma_corrected.mean()
                    target_mean: Tensor = target_channel.mean()
                    denominator: Tensor = (
                        (gamma_corrected - predicted_mean) ** 2
                    ).sum().clamp_min(1e-6)
                    scale: Tensor = (
                        (
                            (gamma_corrected - predicted_mean)
                            * (target_channel - target_mean)
                        ).sum()
                        / denominator
                    ).clamp(0.85, 1.35)
                    bias: Tensor = (
                        target_mean - scale * predicted_mean
                    ).clamp(-0.12, 0.0)
                    fitted_channel: Tensor = torch.clamp(
                        scale * gamma_corrected + bias, 0.0, 1.0
                    )
                    error: Tensor = F.mse_loss(fitted_channel, target_channel)
                    if best_error is None or error < best_error:
                        best_error = error
                        best_scale = scale
                        best_bias = bias
                        best_gamma = gamma

                fitted_scales.append(best_scale)
                fitted_biases.append(best_bias)
                fitted_gammas.append(best_gamma)

            new_scale: Tensor = torch.stack(fitted_scales).to(
                self.calib_scale.device
            )
            new_bias: Tensor = torch.stack(fitted_biases).to(self.calib_bias.device)
            new_gamma: Tensor = torch.stack(fitted_gammas).to(
                self.calib_gamma.device
            )

            if ema > 0.0:
                momentum: float = float(max(0.0, min(0.999, ema)))
                self.calib_scale = (
                    momentum * self.calib_scale + (1.0 - momentum) * new_scale
                )
                self.calib_bias = (
                    momentum * self.calib_bias + (1.0 - momentum) * new_bias
                )
                self.calib_gamma = (
                    momentum * self.calib_gamma + (1.0 - momentum) * new_gamma
                )
            else:
                self.calib_scale = new_scale
                self.calib_bias = new_bias
                self.calib_gamma = new_gamma
            self.calib_bias = self.calib_bias.clamp(-0.12, 0.0)

    @overload
    def render(
        self,
        mvp: Tensor,
        resolution: ImageResolution = (256, 256),
        return_clean: Literal[False] = False,
        model_rot: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def render(
        self,
        mvp: Tensor,
        resolution: ImageResolution,
        return_clean: Literal[True],
        model_rot: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def render(
        self,
        mvp: Tensor,
        resolution: ImageResolution = (256, 256),
        return_clean: bool = False,
        model_rot: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """保留历史 tuple interface，并委托给显式 evidence renderer。

        Args:
            mvp: model-view-projection 矩阵，形状 ``[4, 4]``。
            resolution: 输出 ``(height, width)``。
            return_clean: 是否同时返回未添加对抗噪声的干净渲染。
            model_rot: 世界旋转矩阵，形状 ``[3, 3]``。

        Returns:
            ``return_clean=False`` 时返回 ``(adv_rgb, mask)``；否则返回
            ``(adv_rgb, clean_rgb, mask)``。RGB 形状均为 ``[1, H, W, 3]``，
            mask 形状为 ``[1, H, W, 1]``。
        """
        evidence: RendererEvidence = self.render_evidence(
            mvp,
            resolution=resolution,
            model_rot=model_rot,
        )
        if return_clean:
            return (
                evidence.adversarial_rgb,
                evidence.clean_rgb,
                evidence.visibility_mask,
            )
        return evidence.adversarial_rgb, evidence.visibility_mask

    def render_evidence(
        self,
        mvp: Tensor,
        resolution: ImageResolution = (256, 256),
        model_rot: Optional[Tensor] = None,
    ) -> RendererEvidence:
        """把 mesh 渲染为带 raw raster correspondence 的完整实例证据。

        ``adversarial_rgb``、``clean_rgb``、``visibility_mask`` 与 ``raster``
        全部来自同一次 rasterization，防止 coverage/compositor 混用跨姿态或
        跨调用缓存。普通调用方继续使用 :meth:`render` 即可。
        """

        # pos/position_homogeneous: [num_vertices, 3/4]。
        # position: [num_vertices, 3]，把对齐偏移应用到 renderer 顶点。
        position: Tensor = self.pos + self.pos_offset
        position_homogeneous: Tensor = torch.cat(
            [position, torch.ones_like(position[..., :1])], dim=-1
        )
        # position_clip: [num_vertices, 4]，顶点经过 MVP 后的裁剪空间坐标。
        position_clip: Tensor = torch.matmul(position_homogeneous, mvp.t())

        # raster: [1, H, W, 4]，最后一维包含插值坐标和 triangle id。
        raster: Tensor
        raster, _ = dr.rasterize(
            self.glctx,
            position_clip.unsqueeze(0),
            self.faces,
            resolution=resolution,
        )   # 光栅化得到 [1, H, W, 4]

        # clean/adversarial_color: float NHWC [1, H, W, 3]。
        clean_color: Tensor
        adversarial_color: Tensor
        if self.surface_parameterization is None:
            # legacy 路径：先把 UV texture 采样成顶点颜色，再对顶点颜色插值。
            # 这会在 UV bake 时产生已知的重采样模糊，仅用于历史行为对照。
            raw_adversarial_vertex_color: Tensor = (
                self.orig_vertex_colors + self.get_surface_delta()
            )
            adversarial_vertex_color: Tensor = (
                raw_adversarial_vertex_color
                + (
                    raw_adversarial_vertex_color.clamp(0, 1)
                    - raw_adversarial_vertex_color
                ).detach()
            )
            clean_color, _ = dr.interpolate(
                self.orig_vertex_colors.unsqueeze(0).contiguous(),
                raster,
                self.faces,
            )
            adversarial_color, _ = dr.interpolate(
                adversarial_vertex_color.unsqueeze(0).contiguous(),
                raster,
                self.faces,
            )
        else:
            # UV 保真路径：clean RGB 始终直接来自原始 atlas。pixel_uv 使用
            # ``uv_idx``，而 Surface Delta 使用几何 ``faces``；两者的 triangle
            # 顺序相同，所以可以在像素域精确相加。
            pixel_uv: Tensor
            pixel_uv, _ = dr.interpolate(
                self.uv.unsqueeze(0).contiguous(),
                raster,
                self.uv_idx.int(),
            )
            clean_color = dr.texture(
                self.orig_texture.contiguous(),
                pixel_uv.contiguous(),
                filter_mode="linear",
            )
            interpolated_delta: Tensor
            interpolated_delta, _ = dr.interpolate(
                self.get_surface_delta().unsqueeze(0).contiguous(),
                raster,
                self.faces,
            )
            raw_adversarial_color: Tensor = (
                clean_color + interpolated_delta
            )
            # straight-through clamp：forward 保证合法 RGB，backward 保留
            # unclamped 梯度，避免边界像素永久失去优化信号。
            adversarial_color = raw_adversarial_color + (
                raw_adversarial_color.clamp(0, 1)
                - raw_adversarial_color
            ).detach()

        world_vertex_normals: Tensor = (
            (model_rot @ self.vn.T).T if model_rot is not None else self.vn
        )  # [num_vertices, 3]
        interpolated_normals: Tensor
        interpolated_normals, _ = dr.interpolate(
            world_vertex_normals.unsqueeze(0).contiguous(), raster, self.faces
        )
        normal: Tensor = F.normalize(interpolated_normals, dim=-1, eps=1e-6)

        # Blinn-Phong 风格简化光照，所有方向 tensor shape 为 [1, 1, 1, 3]。
        light_direction: Tensor = self.light_dir.view(1, 1, 1, 3)
        view_direction: Tensor = F.normalize(
            torch.tensor([0.0, 0.0, 1.0], device=normal.device), dim=0
        ).view(1, 1, 1, 3)
        halfway_direction: Tensor = F.normalize(
            light_direction + view_direction, dim=-1, eps=1e-6
        )
        normal_dot_light: Tensor = torch.clamp(
            (normal * light_direction).sum(dim=-1, keepdim=True), 0.0, 1.0
        )
        normal_dot_halfway: Tensor = torch.clamp(
            (normal * halfway_direction).sum(dim=-1, keepdim=True), 0.0, 1.0
        )
        specular: Tensor = normal_dot_halfway.pow(self.specular_shininess)
        light: Tensor = (
            self.ambient_strength
            + self.diffuse_strength * normal_dot_light
            + self.specular_strength * specular
        )
        shadow_mask: Tensor = (1.0 - normal_dot_light).clamp(0.0, 1.0).pow(
            self.shadow_gamma
        )
        light = light * (1.0 - self.shadow_strength * shadow_mask)
        light = torch.clamp(light, self.min_light, 1.35)

        clean_shaded: Tensor = torch.clamp(clean_color * light, 0.0, 1.0)
        adversarial_shaded: Tensor = torch.clamp(
            adversarial_color * light, 0.0, 1.0
        )

        calibration_scale: Tensor = self.calib_scale.view(1, 1, 1, 3)
        calibration_bias: Tensor = self.calib_bias.view(1, 1, 1, 3)
        calibration_gamma: Tensor = self.calib_gamma.view(1, 1, 1, 3)
        clean_base: Tensor = torch.clamp(clean_shaded, 1e-6, 1.0).pow(
            calibration_gamma
        )
        clean_lit: Tensor = torch.clamp(
            clean_base * calibration_scale + calibration_bias, 0.0, 1.0
        )
        adversarial_base: Tensor = torch.clamp(
            adversarial_shaded, 1e-6, 1.0
        ).pow(calibration_gamma)
        adversarial_lit: Tensor = torch.clamp(
            adversarial_base * calibration_scale + calibration_bias, 0.0, 1.0
        )

        # visibility_mask: float32 [1, H, W, 1]，三角形覆盖区域为 1。
        visibility_mask: Tensor = (
            (raster[..., 3] > 0).float().unsqueeze(-1)
        )
        return RendererEvidence(
            adversarial_rgb=adversarial_lit,
            clean_rgb=clean_lit,
            visibility_mask=visibility_mask,
            raster=raster,
        )

    def get_baked_adv_texture(self) -> Tensor:
        """把当前 Surface Delta bake 到 UV atlas。3D纹理变成 uv png

        Returns:
            NHWC texture tensor，形状 ``[1, tex_h, tex_w, 3]``，数值位于
            ``[0, 1]``。输出沿高度翻转，以恢复写入图像文件所需的 UV 方向。
        """
        with torch.no_grad():
            height: int = self.tex_h
            width: int = self.tex_w
            num_uv_vertices: int = int(self.uv.shape[0])

            # uv_clip: [num_uv_vertices, 4]，把 [0, 1] UV 映射到 [-1, 1] clip space。
            uv_clip: Tensor = torch.zeros(
                num_uv_vertices, 4, device=self.device
            )
            uv_clip[:, 0] = 2.0 * self.uv[:, 0] - 1.0
            uv_clip[:, 1] = 2.0 * self.uv[:, 1] - 1.0
            uv_clip[:, 3] = 1.0

            uv_raster: Tensor
            uv_raster, _ = dr.rasterize(
                self.glctx,
                uv_clip.unsqueeze(0),
                self.uv_idx.int(),
                resolution=[height, width],
            )
            uv_visibility_mask: Tensor = (
                (uv_raster[..., 3] > 0).unsqueeze(-1).float()
            )
            baked_texture: Tensor
            if self.surface_parameterization is None:
                noise_parameter: Tensor = self.get_surface_delta()
                adversarial_vertex_color: Tensor = (
                    self.orig_vertex_colors + noise_parameter
                ).clamp(0, 1)  # [num_render_vertices, 3]

                # uv_to_position: [num_uv_vertices]，把 UV 顶点映射到渲染顶点。
                uv_to_position: Tensor = torch.zeros(
                    num_uv_vertices,
                    dtype=torch.long,
                    device=self.device,
                )
                face_vertex_indices: Tensor = self.faces.long()
                face_uv_indices: Tensor = self.uv_idx.long()
                for corner_index in range(3):
                    uv_to_position[face_uv_indices[:, corner_index]] = (
                        face_vertex_indices[:, corner_index]
                    )
                uv_vertex_color: Tensor = adversarial_vertex_color[
                    uv_to_position
                ]
                baked_colors: Tensor
                baked_colors, _ = dr.interpolate(
                    uv_vertex_color.unsqueeze(0).contiguous(),
                    uv_raster,
                    self.uv_idx.int(),
                )
                baked_texture = (
                    uv_visibility_mask * baked_colors
                    + (1.0 - uv_visibility_mask) * self.orig_texture
                )
            else:
                # UV 保真 bake 不再执行 UV→顶点→UV 重采样。只将 Surface Delta
                # 插值到 atlas，再与原始 texel 相加；零参数时结果逐元素等于
                # orig_texture。
                baked_delta: Tensor
                baked_delta, _ = dr.interpolate(
                    self.get_surface_delta().unsqueeze(0).contiguous(),
                    uv_raster,
                    self.faces,
                )
                raw_baked_texture: Tensor = (
                    self.orig_texture + uv_visibility_mask * baked_delta
                )
                baked_texture = raw_baked_texture.clamp(0, 1)
            return torch.flip(baked_texture, dims=[1])

    def bake_vertex_colors_to_texture(
        self, resolution: ImageResolution = (256, 256)
    ) -> Tensor:
        """兼容旧调用名称；当前 bake 始终使用原 texture 的实际分辨率。"""
        del resolution
        return self.get_baked_adv_texture()
