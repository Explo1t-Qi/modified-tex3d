"""OpenVLA-OFT clean/adversarial 成对响应的纯计算与视觉特征边界。

本模块不创建 LIBERO 环境、不加载 checkpoint，也不修改纹理文件。它只完成两
类可独立验证的工作：

1. 根据模型配置定位 OFT 的 SigLIP 分支，并从 processor 的主视角输出中提取
   patch features；
2. 对同一物理状态下的 clean/adv 图像、特征和动作计算统一距离指标。

把这些语义从运行脚本中分离后，诊断指标可以在无 GPU 环境中做回归测试。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence, cast

import numpy as np
import torch


class VisionFeaturizer(Protocol):
    """Timm 视觉分支的最小调用接口。"""

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """输入 ``[B, 3, H, W]``，返回 ``[B, P, D]``。"""
        ...


class FusedVisionBackbone(Protocol):
    """Prismatic 双视觉主干用于动态选择分支的属性。"""

    featurizer: VisionFeaturizer
    fused_featurizer: VisionFeaturizer


class OFTFeatureModel(Protocol):
    """OFT SigLIP 诊断所需的最小模型接口。"""

    config: Any
    vision_backbone: FusedVisionBackbone


@dataclass(frozen=True)
class DistanceMetrics:
    """两个同 shape 数组之间的尺度与方向差异。"""

    mean_absolute: float
    mean_squared: float
    max_absolute: float
    relative_l2: float
    cosine_distance: float


@dataclass(frozen=True)
class ActionDiagnostics:
    """动作 chunk 的整体变化及夹爪符号变化。"""

    all_steps: DistanceMetrics
    first_step: DistanceMetrics
    gripper_sign_flip_count: int
    gripper_step_count: int
    first_gripper_sign_flip: bool


def _timm_model_ids(model: OFTFeatureModel) -> tuple[str, ...]:
    """读取视觉分支 ID，拒绝依赖属性名猜测分支语义。"""
    raw_ids: Any = getattr(model.config, "timm_model_ids", None)
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise RuntimeError("OFT config 缺少可解析的 timm_model_ids")
    model_ids: tuple[str, ...] = tuple(str(value) for value in raw_ids)
    if not model_ids:
        raise RuntimeError("OFT config 的 timm_model_ids 不能为空")
    return model_ids


def extract_primary_siglip_features(
    model: OFTFeatureModel,
    primary_pixel_values: torch.Tensor,
) -> torch.Tensor:
    """从 OFT processor 的主视角输出提取 SigLIP patch features。

    Args:
        model: 带 Prismatic 双视觉主干的 OFT 模型。
        primary_pixel_values: processor 已归一化的主视角 tensor，dtype 与
            模型一致，shape ``[batch_size, 3*num_backbones, H, W]``。

    Returns:
        SigLIP patch features，shape ``[batch_size, patches, feature_dim]``。

    processor 按 ``timm_model_ids`` 的顺序拼接每个分支的三个通道；因此这里
    同时按配置选择通道切片和 featurizer。当前 checkpoint 是
    ``[DINOv2, SigLIP]``，但代码不把该顺序写死。
    """
    if primary_pixel_values.ndim != 4:
        raise ValueError(
            "主视角 pixel_values 必须是 NCHW，实际 shape="
            f"{tuple(primary_pixel_values.shape)}"
        )
    model_ids: tuple[str, ...] = _timm_model_ids(model)
    expected_channels: int = 3 * len(model_ids)
    if primary_pixel_values.shape[1] != expected_channels:
        raise ValueError(
            "主视角通道数与视觉分支数不一致："
            f"{primary_pixel_values.shape[1]} != {expected_channels}"
        )

    siglip_indices: list[int] = [
        index
        for index, model_id in enumerate(model_ids)
        if "siglip" in model_id.lower()
    ]
    if len(siglip_indices) != 1:
        raise RuntimeError(
            "预期 timm_model_ids 中恰好一个 SigLIP，实际为 "
            f"{list(model_ids)}"
        )
    siglip_index: int = siglip_indices[0]
    if siglip_index == 0:
        raw_featurizer: Any = getattr(model.vision_backbone, "featurizer", None)
    elif siglip_index == 1:
        raw_featurizer = getattr(
            model.vision_backbone,
            "fused_featurizer",
            None,
        )
    else:
        raise RuntimeError("当前 Prismatic interface 最多支持两个视觉分支")
    if raw_featurizer is None or not callable(raw_featurizer):
        raise RuntimeError(f"找不到 SigLIP featurizer（分支 {siglip_index}）")
    featurizer: VisionFeaturizer = cast(VisionFeaturizer, raw_featurizer)

    channel_start: int = 3 * siglip_index
    siglip_pixels: torch.Tensor = primary_pixel_values[
        :, channel_start : channel_start + 3
    ]
    patch_features: torch.Tensor = featurizer(siglip_pixels)
    if patch_features.ndim != 3:
        raise RuntimeError(
            "SigLIP feature 必须是 [B, P, D]，实际 shape="
            f"{tuple(patch_features.shape)}"
        )
    return patch_features


def compute_distance(
    clean_values: np.ndarray,
    adversarial_values: np.ndarray,
) -> DistanceMetrics:
    """计算两个同 shape、非空数组的距离，内部使用 float64 累积。"""
    clean: np.ndarray = np.asarray(clean_values, dtype=np.float64)
    adversarial: np.ndarray = np.asarray(
        adversarial_values,
        dtype=np.float64,
    )
    if clean.shape != adversarial.shape:
        raise ValueError(f"成对数组 shape 不同：{clean.shape} != {adversarial.shape}")
    if clean.size == 0:
        raise ValueError("成对数组不能为空")

    difference: np.ndarray = adversarial - clean
    clean_flat: np.ndarray = clean.reshape(-1)
    adversarial_flat: np.ndarray = adversarial.reshape(-1)
    clean_norm: float = float(np.linalg.norm(clean_flat))
    adversarial_norm: float = float(np.linalg.norm(adversarial_flat))
    difference_norm: float = float(np.linalg.norm(difference.reshape(-1)))
    denominator: float = clean_norm * adversarial_norm
    if denominator <= np.finfo(np.float64).eps:
        cosine_distance: float = 0.0 if difference_norm == 0.0 else 1.0
    else:
        cosine_similarity: float = float(
            np.dot(clean_flat, adversarial_flat) / denominator
        )
        cosine_distance = float(1.0 - np.clip(cosine_similarity, -1.0, 1.0))

    return DistanceMetrics(
        mean_absolute=float(np.mean(np.abs(difference))),
        mean_squared=float(np.mean(np.square(difference))),
        max_absolute=float(np.max(np.abs(difference))),
        relative_l2=float(difference_norm / max(clean_norm, 1e-12)),
        cosine_distance=cosine_distance,
    )


def compute_action_diagnostics(
    clean_actions: np.ndarray,
    adversarial_actions: np.ndarray,
) -> ActionDiagnostics:
    """比较动作 chunk；输入 shape 为 ``[chunk_steps, action_dim]``。"""
    clean: np.ndarray = np.asarray(clean_actions, dtype=np.float64)
    adversarial: np.ndarray = np.asarray(adversarial_actions, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[1] < 1:
        raise ValueError(f"动作必须是 [T, A]，实际 shape={clean.shape}")
    if clean.shape != adversarial.shape:
        raise ValueError(f"动作 shape 不同：{clean.shape} != {adversarial.shape}")

    clean_gripper_positive: np.ndarray = clean[:, -1] >= 0.0
    adversarial_gripper_positive: np.ndarray = adversarial[:, -1] >= 0.0
    gripper_flips: np.ndarray = np.not_equal(
        clean_gripper_positive,
        adversarial_gripper_positive,
    )
    return ActionDiagnostics(
        all_steps=compute_distance(clean, adversarial),
        first_step=compute_distance(clean[0], adversarial[0]),
        gripper_sign_flip_count=int(np.count_nonzero(gripper_flips)),
        gripper_step_count=int(clean.shape[0]),
        first_gripper_sign_flip=bool(gripper_flips[0]),
    )


def metrics_to_dict(metrics: DistanceMetrics | ActionDiagnostics) -> dict[str, Any]:
    """将冻结 dataclass 转为可直接写入 JSON 的字典。"""
    return asdict(metrics)
