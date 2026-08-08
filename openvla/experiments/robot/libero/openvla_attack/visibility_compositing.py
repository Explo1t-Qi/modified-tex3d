"""Visibility-Masked Renderer Delta Composition 的纯 tensor 实现。

新候选不再用 nvdiffrast 前景整体替换 MuJoCo 外观。每个共享纹理实例只提供
局部可微颜色响应 ``F_adv - F_clean``，MuJoCo 同一静止 state 的 front-most
instance segmentation 决定真实可见像素：

``I_adv = clamp(I_mujoco + sum_k alpha_k m_k (F_adv_k-F_clean_k), 0, 1)``。

输入全部使用 512 Policy Source 坐标系的 NCHW float32。模块显式要求 clean
renderer 分支不连接 trainable graph，并检查实例 alpha 的互斥性；它不负责
采集 state、渲染实例或执行 Deployment Effective View Transform。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch


COMPOSITION_INVARIANT_ATOL: Final[float] = 1e-6


@dataclass(frozen=True)
class VisibilityMaskedComposition:
    """合成 RGB、未截断响应与可复查的饱和统计。

    ``per_instance_delta`` 为 float32 ``[K,3,H,W]``；``total_delta`` 与两个
    RGB tensor 为 ``[1,3,H,W]``。``saturated_pixel_mask`` 为 bool
    ``[1,1,H,W]``，表示至少一个 channel 在 clamp 前越界。
    """

    composited_rgb: torch.Tensor
    unclamped_rgb: torch.Tensor
    per_instance_delta: torch.Tensor
    total_delta: torch.Tensor
    saturated_pixel_mask: torch.Tensor
    saturated_pixel_fraction: float
    saturated_channel_fraction: float


def _validate_float_tensor(
    value: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    invariant_atol: float,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} 必须使用 float32")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} shape 必须为 {expected_shape}，收到 {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} 包含 NaN/Inf")
    if bool((value < -invariant_atol).any()) or bool(
        (value > 1.0 + invariant_atol).any()
    ):
        raise ValueError(f"{name} 超出 [0,1] 容差")


def compose_visibility_masked_renderer_delta(
    mujoco_clean_rgb: torch.Tensor,
    mujoco_instance_alpha: torch.Tensor,
    renderer_adversarial_rgb: torch.Tensor,
    renderer_clean_rgb: torch.Tensor,
    renderer_valid_mask: torch.Tensor,
    *,
    invariant_atol: float = COMPOSITION_INVARIANT_ATOL,
) -> VisibilityMaskedComposition:
    """按 MuJoCo front-most 可见性合成逐实例 renderer 颜色响应。

    Args:
        mujoco_clean_rgb: float32 ``[1,3,H,W]`` 的干净 Policy Source RGB。
        mujoco_instance_alpha: float32 ``[K,1,H,W]`` 的互斥 hard mask。
        renderer_adversarial_rgb: float32 ``[K,3,H,W]``。
        renderer_clean_rgb: float32 ``[K,3,H,W]``，必须与可学习参数断图。
        renderer_valid_mask: float32 ``[K,1,H,W]``，每实例投影有效性。
        invariant_atol: 输入范围和实例 union 的预注册容差。

    Returns:
        保留 adversarial renderer 梯度的合成证据。函数不会依赖实例顺序，且
        不会用 renderer mask 替代 MuJoCo 遮挡关系。
    """

    if not math.isfinite(invariant_atol) or invariant_atol < 0.0:
        raise ValueError("composition invariant_atol 必须为有限非负数")
    if (
        not isinstance(mujoco_instance_alpha, torch.Tensor)
        or mujoco_instance_alpha.ndim != 4
        or mujoco_instance_alpha.shape[0] <= 0
        or mujoco_instance_alpha.shape[1] != 1
        or mujoco_instance_alpha.shape[2] <= 0
        or mujoco_instance_alpha.shape[3] <= 0
    ):
        raise ValueError("mujoco_instance_alpha 必须为非空 [K,1,H,W]")

    num_instances: int = int(mujoco_instance_alpha.shape[0])
    height: int = int(mujoco_instance_alpha.shape[2])
    width: int = int(mujoco_instance_alpha.shape[3])
    _validate_float_tensor(
        mujoco_clean_rgb,
        name="mujoco_clean_rgb",
        expected_shape=(1, 3, height, width),
        invariant_atol=invariant_atol,
    )
    _validate_float_tensor(
        mujoco_instance_alpha,
        name="mujoco_instance_alpha",
        expected_shape=(num_instances, 1, height, width),
        invariant_atol=invariant_atol,
    )
    _validate_float_tensor(
        renderer_adversarial_rgb,
        name="renderer_adversarial_rgb",
        expected_shape=(num_instances, 3, height, width),
        invariant_atol=invariant_atol,
    )
    _validate_float_tensor(
        renderer_clean_rgb,
        name="renderer_clean_rgb",
        expected_shape=(num_instances, 3, height, width),
        invariant_atol=invariant_atol,
    )
    _validate_float_tensor(
        renderer_valid_mask,
        name="renderer_valid_mask",
        expected_shape=(num_instances, 1, height, width),
        invariant_atol=invariant_atol,
    )
    if not (
        mujoco_clean_rgb.device
        == mujoco_instance_alpha.device
        == renderer_adversarial_rgb.device
        == renderer_clean_rgb.device
        == renderer_valid_mask.device
    ):
        raise ValueError("composition 的全部 tensor 必须位于同一 device")
    if renderer_clean_rgb.requires_grad:
        raise ValueError("renderer_clean_rgb 必须与 trainable Surface Delta 断图")

    hard_alpha_values: torch.Tensor = mujoco_instance_alpha.flatten()
    if not bool(
        ((hard_alpha_values == 0.0) | (hard_alpha_values == 1.0)).all()
    ):
        raise ValueError("raw MuJoCo instance alpha 必须是精确 0/1 hard mask")
    if bool(
        (
            mujoco_instance_alpha.sum(dim=0)
            > 1.0 + invariant_atol
        ).any()
    ):
        raise ValueError("MuJoCo instance alpha 不是 front-most 互斥分区")

    renderer_rgb_delta: torch.Tensor = (
        renderer_adversarial_rgb - renderer_clean_rgb
    )
    per_instance_delta: torch.Tensor = (
        mujoco_instance_alpha * renderer_valid_mask * renderer_rgb_delta
    )
    total_delta: torch.Tensor = per_instance_delta.sum(dim=0, keepdim=True)
    unclamped_rgb: torch.Tensor = mujoco_clean_rgb + total_delta
    saturated_channels: torch.Tensor = (unclamped_rgb < 0.0) | (
        unclamped_rgb > 1.0
    )
    saturated_pixel_mask: torch.Tensor = saturated_channels.any(
        dim=1,
        keepdim=True,
    )
    composited_rgb: torch.Tensor = torch.clamp(unclamped_rgb, 0.0, 1.0)
    return VisibilityMaskedComposition(
        composited_rgb=composited_rgb,
        unclamped_rgb=unclamped_rgb,
        per_instance_delta=per_instance_delta,
        total_delta=total_delta,
        saturated_pixel_mask=saturated_pixel_mask,
        saturated_pixel_fraction=float(
            saturated_pixel_mask.float().mean().item()
        ),
        saturated_channel_fraction=float(
            saturated_channels.float().mean().item()
        ),
    )
