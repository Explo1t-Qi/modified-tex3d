"""多目标攻击梯度的动态范数保护。

本模块只处理已经在同一参数空间中对齐的 Action/Feature 梯度，不依赖
OpenVLA forward、LIBERO、renderer 或具体纹理参数化。调用方负责先应用 loss
weight 和 frame weight，并在 batch 内累积；这里负责限制 Feature 的相对范数
并返回可持久化诊断量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GradientNormProtectionResult:
    """一次加权 Action/Feature 梯度合并的结果与诊断量。

    三个梯度 tensor 均定义在同一参数空间且 shape 相同；当前谱实验中为
    float ``[num_basis,3]``。输入已经包含 alpha、frame weight 和 batch 累积，
    因此统计口径与训练实际更新一致。
    """

    combined_gradient: torch.Tensor
    weighted_action_norm: float
    weighted_feature_norm: float
    feature_to_action_norm_ratio: float
    feature_scale: float
    action_feature_cosine: float


def combine_gradients_with_feature_norm_protection(
    weighted_action_gradient: torch.Tensor,
    weighted_feature_gradient: torch.Tensor,
    *,
    feature_to_action_ratio_limit: float,
) -> GradientNormProtectionResult:
    """限制加权 Feature 梯度相对 Action 梯度的 L2 范数。

    只在 Feature 超过 ``ratio_limit * ||g_action||`` 时向下缩放；Feature
    较小时保持原样，绝不主动放大。该方法不投影梯度方向，因此与只处理负
    内积的 PCGrad 有不同语义。
    """
    if weighted_action_gradient.shape != weighted_feature_gradient.shape:
        raise ValueError("Action/Feature 梯度 shape 必须一致")
    if (
        weighted_action_gradient.dtype != weighted_feature_gradient.dtype
        or weighted_action_gradient.device
        != weighted_feature_gradient.device
    ):
        raise ValueError("Action/Feature 梯度 dtype 与 device 必须一致")
    if not weighted_action_gradient.is_floating_point():
        raise ValueError("Action/Feature 梯度必须是浮点 tensor")
    if not torch.isfinite(weighted_action_gradient).all():
        raise ValueError("加权 Action 梯度包含 NaN/Inf")
    if not torch.isfinite(weighted_feature_gradient).all():
        raise ValueError("加权 Feature 梯度包含 NaN/Inf")
    if (
        not np.isfinite(feature_to_action_ratio_limit)
        or feature_to_action_ratio_limit <= 0.0
    ):
        raise ValueError("Feature/Action 梯度范数上限必须为有限正数")

    action_norm_tensor: torch.Tensor = torch.linalg.vector_norm(
        weighted_action_gradient
    )
    feature_norm_tensor: torch.Tensor = torch.linalg.vector_norm(
        weighted_feature_gradient
    )
    action_norm: float = float(action_norm_tensor.item())
    feature_norm: float = float(feature_norm_tensor.item())
    numeric_epsilon: float = torch.finfo(
        weighted_action_gradient.dtype
    ).eps

    if feature_norm <= numeric_epsilon:
        feature_scale: float = 1.0
    elif action_norm <= numeric_epsilon:
        # 没有 Action 信号时，任何非零 Feature 都会违反范数上限。首版选择
        # 安全地跳过该次 Feature 更新，并由日志暴露这一退化情况。
        feature_scale = 0.0
    else:
        feature_scale = min(
            1.0,
            feature_to_action_ratio_limit * action_norm / feature_norm,
        )

    stable_action_norm: float = max(action_norm, numeric_epsilon)
    feature_to_action_ratio: float = feature_norm / stable_action_norm
    if action_norm <= numeric_epsilon or feature_norm <= numeric_epsilon:
        action_feature_cosine: float = 0.0
    else:
        flattened_action: torch.Tensor = (
            weighted_action_gradient.reshape(-1)
        )
        flattened_feature: torch.Tensor = (
            weighted_feature_gradient.reshape(-1)
        )
        action_feature_cosine = float(
            torch.dot(flattened_action, flattened_feature).item()
            / (action_norm * feature_norm)
        )

    combined_gradient: torch.Tensor = (
        weighted_action_gradient
        + feature_scale * weighted_feature_gradient
    ).detach()
    if not torch.isfinite(combined_gradient).all():
        raise ValueError("范数保护后的合并梯度包含 NaN/Inf")
    return GradientNormProtectionResult(
        combined_gradient=combined_gradient,
        weighted_action_norm=action_norm,
        weighted_feature_norm=feature_norm,
        feature_to_action_norm_ratio=feature_to_action_ratio,
        feature_scale=feature_scale,
        action_feature_cosine=action_feature_cosine,
    )
