"""OpenVLA 共享 SigLIP patch feature 的强类型访问边界。

OpenVLA 的视觉主干把 DINOv2 与 SigLIP 封装在同一个
``PrismaticVisionBackbone`` 中。攻击代码若直接依赖 ``featurizer`` /
``fused_featurizer`` 的固定位置，会在模型配置改变后静默读取错误分支。本模块
根据 checkpoint 的 ``timm_model_ids`` 找到唯一 SigLIP 分支，并集中校验输出
shape。

本模块只负责视觉 feature 提取，不定义攻击 loss，也不接触 renderer、LIBERO
环境或谱参数化。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, cast

import torch


class VisionFeaturizer(Protocol):
    """Timm ViT feature extractor 的最小调用接口。"""

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """返回 patch features，shape ``[batch_size, patches, dim]``。"""
        ...


class FusedVisionBackbone(Protocol):
    """Prismatic 双视觉主干中用于定位分支的属性集合。"""

    featurizer: VisionFeaturizer
    fused_featurizer: VisionFeaturizer


class SigLIPFeatureModel(Protocol):
    """共享 SigLIP feature objective 所需的 OpenVLA 模型接口。"""

    config: Any
    vision_backbone: FusedVisionBackbone


def _get_timm_model_ids(model: SigLIPFeatureModel) -> tuple[str, ...]:
    """读取并校验 checkpoint 中视觉分支的稳定标识。"""
    raw_model_ids: Any = getattr(model.config, "timm_model_ids", None)
    if not isinstance(raw_model_ids, Sequence) or isinstance(
        raw_model_ids,
        (str, bytes),
    ):
        raise RuntimeError(
            "OpenVLA config 缺少可解析的 timm_model_ids，无法定位 SigLIP"
        )
    model_ids: tuple[str, ...] = tuple(str(item) for item in raw_model_ids)
    if not model_ids:
        raise RuntimeError("OpenVLA config 的 timm_model_ids 不能为空")
    return model_ids


def get_siglip_featurizer(
    model: SigLIPFeatureModel,
) -> VisionFeaturizer:
    """从 OpenVLA checkpoint 配置定位唯一 SigLIP featurizer。

    当前 OpenVLA 的顺序为 ``[DINOv2, SigLIP]``，所以 SigLIP 对应
    ``fused_featurizer``。这里仍根据模型 ID 动态判断，避免把该顺序写成隐式
    假设。
    """
    model_ids: tuple[str, ...] = _get_timm_model_ids(model)
    siglip_indices: list[int] = [
        index
        for index, model_id in enumerate(model_ids)
        if "siglip" in model_id.lower()
    ]
    if len(siglip_indices) != 1:
        raise RuntimeError(
            "预期 timm_model_ids 中恰好有一个 SigLIP 分支，实际为 "
            f"{list(model_ids)}"
        )

    siglip_index: int = siglip_indices[0]
    vision_backbone: FusedVisionBackbone = model.vision_backbone
    if siglip_index == 0:
        featurizer: Any = getattr(vision_backbone, "featurizer", None)
    elif siglip_index == 1:
        featurizer = getattr(
            vision_backbone,
            "fused_featurizer",
            None,
        )
    else:
        raise RuntimeError(
            "当前 Prismatic interface 最多支持两个视觉分支，"
            f"SigLIP 索引却为 {siglip_index}"
        )
    if featurizer is None or not callable(featurizer):
        raise RuntimeError(
            f"OpenVLA 视觉主干没有可调用的 SigLIP 分支（索引 {siglip_index}）"
        )
    return cast(VisionFeaturizer, featurizer)


def extract_siglip_patch_features(
    model: SigLIPFeatureModel,
    normalized_siglip_pixels: torch.Tensor,
) -> torch.Tensor:
    """提取保持输入梯度的 SigLIP patch features。

    Args:
        model: 带 Prismatic 双视觉主干的 OpenVLA。
        normalized_siglip_pixels: SigLIP 归一化后的浮点 NCHW tensor，
            shape ``[batch_size, 3, height, width]``。

    Returns:
        浮点 patch features，shape ``[batch_size, patches, feature_dim]``。
        返回值不 detach；对抗分支必须保留到输入图像和纹理参数的梯度。
    """
    if normalized_siglip_pixels.ndim != 4:
        raise ValueError(
            "SigLIP 输入必须是 NCHW 四维 tensor，实际 shape="
            f"{tuple(normalized_siglip_pixels.shape)}"
        )
    if normalized_siglip_pixels.shape[1] != 3:
        raise ValueError(
            "SigLIP 输入通道数必须为 3，实际 shape="
            f"{tuple(normalized_siglip_pixels.shape)}"
        )

    featurizer: VisionFeaturizer = get_siglip_featurizer(model)
    patch_features: torch.Tensor = featurizer(
        normalized_siglip_pixels
    )
    if patch_features.ndim != 3:
        raise RuntimeError(
            "SigLIP featurizer 应返回 [batch, patches, dim]，实际 shape="
            f"{tuple(patch_features.shape)}"
        )
    if patch_features.shape[0] != normalized_siglip_pixels.shape[0]:
        raise RuntimeError(
            "SigLIP feature batch 与输入不一致："
            f"{patch_features.shape[0]} != "
            f"{normalized_siglip_pixels.shape[0]}"
        )
    return patch_features
