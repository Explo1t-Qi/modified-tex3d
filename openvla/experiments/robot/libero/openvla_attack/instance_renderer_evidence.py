"""共享 Active Texture 多实例的同次 renderer evidence 编排。

本模块逐实例调用显式 ``render_evidence()``，把 adv/clean RGB、valid mask 与
raw raster 按同一实例顺序堆叠，并绑定 renderer faces 及严格 seam 映射。它不
导入 nvdiffrast 或具体 renderer 类，因此 CPU 测试可以验证跨模块契约。

一个实例的四项输出必须来自同一次 rasterization。模块还要求公开 mask 与 raw
triangle ID 解码出的 valid mask 逐值相等，防止 compositor、alignment 与
Support correspondence 使用不同的前景定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, TypeAlias

import torch

from .compositing import ImageResolution, TextureRenderInstance
from .renderer_correspondence import (
    RasterSupportCorrespondence,
    decode_raster_support_correspondence,
)


Tensor: TypeAlias = torch.Tensor


class RendererEvidenceLike(Protocol):
    """具体 renderer evidence dataclass 的结构 interface。"""

    adversarial_rgb: Tensor
    clean_rgb: Tensor
    visibility_mask: Tensor
    raster: Tensor


class InstanceEvidenceRenderer(Protocol):
    """多实例 evidence 编排依赖的最小 renderer interface。"""

    faces: Tensor

    def get_render_to_geometry_mapping(self) -> Tensor:
        """返回 integer ``[num_render_vertices]`` 严格 seam 映射。"""
        ...

    def render_evidence(
        self,
        mvp: Tensor,
        resolution: ImageResolution,
        model_rot: Optional[Tensor] = None,
    ) -> RendererEvidenceLike:
        """返回同次 adv/clean/mask/raster。"""
        ...


@dataclass(frozen=True)
class SharedTextureRendererEvidence:
    """按实例顺序堆叠的 renderer 证据。

    RGB 为 float32 NCHW ``[K,3,H,W]``，visibility mask 为 float32 NCHW
    ``[K,1,H,W]``，raster 为 float32 NHWC ``[K,H,W,4]``。
    """

    adversarial_rgb: Tensor
    clean_rgb: Tensor
    visibility_mask: Tensor
    raster: Tensor
    renderer_faces: Tensor
    render_to_geometry: Tensor

    @property
    def num_instances(self) -> int:
        return int(self.raster.shape[0])

    def decode_support(
        self,
        geometry_support: Tensor,
    ) -> RasterSupportCorrespondence:
        """为同一 Fixed Vertex Support 解码所有实例的 ``w_{k,S}``。"""

        correspondence = decode_raster_support_correspondence(
            self.raster,
            self.renderer_faces,
            self.render_to_geometry,
            geometry_support,
        )
        if not torch.equal(
            correspondence.valid_mask,
            self.visibility_mask.to(dtype=torch.bool),
        ):
            raise RuntimeError(
                "renderer visibility mask 与 raster triangle-ID valid mask 不一致"
            )
        return correspondence


def _validate_single_evidence(
    evidence: RendererEvidenceLike,
    *,
    resolution: ImageResolution,
    instance_index: int,
) -> None:
    height, width = resolution
    expected_shapes: tuple[tuple[str, Tensor, tuple[int, ...]], ...] = (
        ("adversarial_rgb", evidence.adversarial_rgb, (1, height, width, 3)),
        ("clean_rgb", evidence.clean_rgb, (1, height, width, 3)),
        ("visibility_mask", evidence.visibility_mask, (1, height, width, 1)),
        ("raster", evidence.raster, (1, height, width, 4)),
    )
    reference_device: torch.device = evidence.raster.device
    for name, value, expected_shape in expected_shapes:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"instance {instance_index} {name} 必须是 tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"instance {instance_index} {name} 必须是 float32")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"instance {instance_index} {name} shape 必须为 "
                f"{expected_shape}，收到 {tuple(value.shape)}"
            )
        if value.device != reference_device:
            raise ValueError(f"instance {instance_index} evidence device 不一致")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"instance {instance_index} {name} 包含 NaN/Inf")
    if evidence.clean_rgb.requires_grad:
        raise ValueError(
            f"instance {instance_index} clean renderer evidence 未断开梯度"
        )
    raster_valid: Tensor = evidence.raster[..., 3] > 0
    if not torch.equal(
        evidence.visibility_mask[..., 0],
        raster_valid.to(dtype=torch.float32),
    ):
        raise ValueError(
            f"instance {instance_index} renderer mask 与 raster valid 不一致"
        )


def render_shared_texture_instances(
    renderer: InstanceEvidenceRenderer,
    instances: Sequence[TextureRenderInstance],
    *,
    resolution: ImageResolution,
) -> SharedTextureRendererEvidence:
    """逐实例采集同次 renderer evidence，并按输入实例顺序堆叠。"""

    height, width = resolution
    if height <= 0 or width <= 0:
        raise ValueError("renderer evidence resolution 必须为正数")
    if not instances:
        raise ValueError("共享纹理 renderer evidence 不得没有实例")

    renderer_faces: Tensor = renderer.faces
    render_to_geometry: Tensor = renderer.get_render_to_geometry_mapping()
    collected: list[RendererEvidenceLike] = []
    for instance_index, instance in enumerate(instances):
        evidence = renderer.render_evidence(
            instance["mvp"],
            resolution=resolution,
            model_rot=instance["model_rot"],
        )
        _validate_single_evidence(
            evidence,
            resolution=resolution,
            instance_index=instance_index,
        )
        collected.append(evidence)

    raster: Tensor = torch.cat(
        [evidence.raster for evidence in collected],
        dim=0,
    )
    if renderer_faces.device != raster.device or (
        render_to_geometry.device != raster.device
    ):
        raise ValueError("renderer topology mapping 与 evidence device 不一致")
    visibility_mask: Tensor = torch.cat(
        [
            evidence.visibility_mask.permute(0, 3, 1, 2)
            for evidence in collected
        ],
        dim=0,
    )
    result = SharedTextureRendererEvidence(
        adversarial_rgb=torch.cat(
            [
                evidence.adversarial_rgb.permute(0, 3, 1, 2)
                for evidence in collected
            ],
            dim=0,
        ),
        clean_rgb=torch.cat(
            [
                evidence.clean_rgb.permute(0, 3, 1, 2)
                for evidence in collected
            ],
            dim=0,
        ),
        visibility_mask=visibility_mask,
        raster=raster,
        renderer_faces=renderer_faces,
        render_to_geometry=render_to_geometry,
    )
    raster_valid = result.raster[..., 3] > 0
    if not torch.equal(
        result.visibility_mask[:, 0],
        raster_valid.to(dtype=torch.float32),
    ):
        raise RuntimeError("堆叠后的 renderer mask/raster 实例顺序不一致")
    return result
