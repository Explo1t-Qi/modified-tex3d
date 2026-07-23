"""OpenVLA 对抗纹理优化使用的 nvdiffrast 可微 renderer。

该 module 负责 mesh/UV 数据加载、顶点颜色扰动、可微光栅化、简化光照模型、
光照校准和 UV texture bake。它不负责 MuJoCo 相机矩阵、背景采集、图像合成
或攻击损失，因此导入 module 时不会创建环境或加载 VLA 模型。

``DifferentiableRenderer`` 的构造会创建 CUDA rasterizer context；只有实例化
需要可用 GPU，导入类定义和阅读类型信息不需要 GPU。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, TypeAlias, overload

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from numpy.typing import NDArray
from PIL import Image


Tensor: TypeAlias = torch.Tensor
ImageResolution: TypeAlias = tuple[int, int]
Device: TypeAlias = str | torch.device
PathLike: TypeAlias = str | Path
FloatingArray: TypeAlias = NDArray[np.floating[Any]]


class DifferentiableRenderer(nn.Module):
    """对 mesh 顶点颜色施加可学习扰动并渲染到相机画面。

    可学习参数只有 ``adv_noise``，形状为 ``[num_vertices, 3]``。mesh 几何、
    UV、法线、原始纹理和光照校准参数都注册为 module buffer，随 renderer
    一起迁移 device，但不会被 optimizer 当作可学习参数。
    """

    def __init__(
        self,
        mesh_path: PathLike,
        orig_texture_path: Optional[PathLike] = None,
        device: Device = "cuda",
        scale_xyz: Optional[Sequence[float]] = None,
        pos_offset: Optional[Sequence[float]] = None,
        epsilon: float = 128.0 / 255.0,
    ) -> None:
        """加载物体资产并创建 CUDA rasterizer context。

        Args:
            mesh_path: OBJ 等 trimesh 支持的 mesh 文件路径。加载失败时沿用
                原实现，打印 warning 并使用 ``0.1 × 0.1 × 0.001`` 的薄盒。
            orig_texture_path: 原始 RGB texture 路径；不存在时使用常量灰色纹理。
            device: renderer tensor 所在 device。当前 nvdiffrast 实现要求 CUDA。
            scale_xyz: mesh 三轴缩放，长度应为 3；缺省为 ``[1, 1, 1]``。
            pos_offset: model-space 顶点平移，长度应为 3；保持现有缺省值
                ``[0.02, 0.01, 0.025]``。
            epsilon: ``tanh(adv_noise)`` 的最大逐通道颜色扰动幅度。
        """
        super().__init__()
        self.device: Device = device
        self.epsilon: float = epsilon
        self.tex_h: int = 256
        self.tex_w: int = 256

        # 使用 ``or`` 而不是只判断 None，以保持空 sequence 也回退缺省值的旧行为。
        resolved_offset: Sequence[float] = pos_offset or [0.02, 0.01, 0.025]
        self.pos_offset: Tensor = torch.tensor(
            resolved_offset, dtype=torch.float32, device=device
        )  # [3]

        resolved_scale: Sequence[float] = (
            scale_xyz if scale_xyz is not None else [1.0, 1.0, 1.0]
        )
        scale_array: FloatingArray = np.asarray(resolved_scale, dtype=np.float64)

        try:
            mesh: trimesh.Trimesh = trimesh.load(mesh_path, force="mesh")
        except Exception:
            print(f"[WARNING] Failed to load mesh from {mesh_path}, using dummy box.")
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.001])

        # vertices: [num_vertices, 3]；faces: [num_faces, 3]。
        vertices: FloatingArray = np.asarray(mesh.vertices) * scale_array[None, :]
        self.num_vertices: int = len(vertices)
        self.pos: Tensor
        self.faces: Tensor
        self.register_buffer(
            "pos", torch.from_numpy(vertices.astype(np.float32)).to(device)
        )
        self.register_buffer(
            "faces", torch.from_numpy(np.asarray(mesh.faces).astype(np.int32)).to(device)
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

        self.adv_noise: nn.Parameter = nn.Parameter(
            torch.zeros(self.num_vertices, 3, dtype=torch.float32, device=device)
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
        """返回 optimizer 更新的 ``[num_vertices, 3]`` 顶点颜色噪声。"""
        return self.adv_noise

    def reset_texture(self) -> None:
        """把可学习顶点颜色噪声原地清零。"""
        with torch.no_grad():
            self.adv_noise.data.fill_(0.0)

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
        """把当前 mesh 渲染为带简化光照的 NHWC RGB。

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
        # pos/position_homogeneous: [num_vertices, 3/4]。
        position: Tensor = self.pos + self.pos_offset
        position_homogeneous: Tensor = torch.cat(
            [position, torch.ones_like(position[..., :1])], dim=-1
        )
        position_clip: Tensor = torch.matmul(position_homogeneous, mvp.t())

        # raster: [1, H, W, 4]，最后一维包含插值坐标和 triangle id。
        raster: Tensor
        raster, _ = dr.rasterize(
            self.glctx,
            position_clip.unsqueeze(0),
            self.faces,
            resolution=resolution,
        )

        # straight-through clamp：forward 限制颜色，backward 保留 unclamped 梯度。
        noise_parameter: Tensor = torch.tanh(self.adv_noise) * self.epsilon
        raw_adversarial_vertex_color: Tensor = (
            self.orig_vertex_colors + noise_parameter
        )
        adversarial_vertex_color: Tensor = raw_adversarial_vertex_color + (
            raw_adversarial_vertex_color.clamp(0, 1) - raw_adversarial_vertex_color
        ).detach()

        # clean/adversarial_color: [1, H, W, 3]。
        clean_color: Tensor
        adversarial_color: Tensor
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

        visibility_mask: Tensor = (raster[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            return adversarial_lit, clean_lit, visibility_mask
        return adversarial_lit, visibility_mask

    def get_baked_adv_texture(self) -> Tensor:
        """把对抗顶点颜色完整 bake 到 UV atlas。

        Returns:
            NHWC texture tensor，形状 ``[1, tex_h, tex_w, 3]``，数值位于
            ``[0, 1]``。输出沿高度翻转，以恢复写入图像文件所需的 UV 方向。
        """
        with torch.no_grad():
            noise_parameter: Tensor = torch.tanh(self.adv_noise) * self.epsilon
            adversarial_vertex_color: Tensor = (
                self.orig_vertex_colors + noise_parameter
            ).clamp(0, 1)  # [num_vertices, 3]

            height: int = self.tex_h
            width: int = self.tex_w
            num_uv_vertices: int = int(self.uv.shape[0])

            # uv_to_position: [num_uv_vertices]，把 UV 顶点映射到几何顶点。
            uv_to_position: Tensor = torch.zeros(
                num_uv_vertices, dtype=torch.long, device=self.device
            )
            face_vertex_indices: Tensor = self.faces.long()
            face_uv_indices: Tensor = self.uv_idx.long()
            for corner_index in range(3):
                uv_to_position[face_uv_indices[:, corner_index]] = (
                    face_vertex_indices[:, corner_index]
                )
            uv_vertex_color: Tensor = adversarial_vertex_color[uv_to_position]

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
            baked_colors: Tensor
            baked_colors, _ = dr.interpolate(
                uv_vertex_color.unsqueeze(0).contiguous(),
                uv_raster,
                self.uv_idx.int(),
            )

            uv_visibility_mask: Tensor = (
                (uv_raster[..., 3] > 0).unsqueeze(-1).float()
            )
            baked_texture: Tensor = (
                uv_visibility_mask * baked_colors
                + (1.0 - uv_visibility_mask) * self.orig_texture
            )
            return torch.flip(baked_texture, dims=[1])

    def bake_vertex_colors_to_texture(
        self, resolution: ImageResolution = (256, 256)
    ) -> Tensor:
        """兼容旧调用名称；当前 bake 始终使用原 texture 的实际分辨率。"""
        del resolution
        return self.get_baked_adv_texture()
