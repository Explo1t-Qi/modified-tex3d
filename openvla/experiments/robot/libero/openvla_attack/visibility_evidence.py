"""Visibility observation、renderer alignment 与四类状态判定。

本模块消费已经位于同一 Effective View 坐标系的 MuJoCo/renderer alpha，不负责
采集或空间变换。MuJoCo alpha 的实例维来自单次 front-most segmentation，必须
互斥；renderer alpha 可因缺少场景深度而重叠，因此 union 使用概率并集
``1 - product_k(1-alpha_k)``。软 overlap 固定为逐像素 ``min``。

四类状态严格区分没有观测、观测不足、对齐无效和有效。coverage 不在这里计算，
从而不会把 ``invalid_alignment`` 或 ``not_observable`` 静默解释为 coverage=0。
当前 ``A_obs=1e-3`` 与 recall=0.95 仍是审计候选值；只有 states 0–9 分布审计后
才能由文档正式冻结。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, Optional, TypeAlias

import torch


VisibilityEvidenceStatus: TypeAlias = Literal[
    "not_observable",
    "insufficient_observation",
    "invalid_alignment",
    "valid",
]
A_OBS_MIN_CANDIDATE: Final[float] = 1e-3
RECALL_MIN_CANDIDATE: Final[float] = 0.95
ALIGNMENT_DENOMINATOR_EPSILON: Final[float] = 1e-12
VISIBILITY_INVARIANT_ATOL: Final[float] = 1e-6


@dataclass(frozen=True)
class VisibilityThresholdCandidates:
    """states 0–9 审计前使用、尚未正式冻结的候选门槛。"""

    observation_area_min: float = A_OBS_MIN_CANDIDATE
    recall_min: float = RECALL_MIN_CANDIDATE
    invariant_atol: float = VISIBILITY_INVARIANT_ATOL
    denominator_epsilon: float = ALIGNMENT_DENOMINATOR_EPSILON

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.observation_area_min)
            and 0.0 < self.observation_area_min <= 1.0
        ):
            raise ValueError("A_obs_min candidate 必须位于 (0,1]")
        if not (
            math.isfinite(self.recall_min)
            and 0.0 <= self.recall_min <= 1.0
        ):
            raise ValueError("recall_min candidate 必须位于 [0,1]")
        if not math.isfinite(self.invariant_atol) or self.invariant_atol < 0:
            raise ValueError("visibility invariant_atol 必须为有限非负数")
        if not (
            math.isfinite(self.denominator_epsilon)
            and self.denominator_epsilon > 0.0
        ):
            raise ValueError("alignment denominator epsilon 必须为有限正数")


@dataclass(frozen=True)
class ObservationEvidence:
    """Effective View 中 union/单实例观测分母。"""

    equivalent_visible_pixels: float
    observation_area: float


@dataclass(frozen=True)
class SoftAlignmentMetrics:
    """同一坐标系中两张 soft alpha 的 recall/precision/IoU。"""

    overlap: float
    mujoco_visible_pixels: float
    renderer_visible_pixels: float
    union_visible_pixels: float
    recall_visible: Optional[float]
    precision_visible: Optional[float]
    iou_visible: Optional[float]


@dataclass(frozen=True)
class InstanceVisibilityEvidence:
    """一个共享纹理实例的观测与对齐诊断。"""

    instance_index: int
    observation: ObservationEvidence
    alignment: Optional[SoftAlignmentMetrics]


@dataclass(frozen=True)
class VisibilityEvaluation:
    """一个 ``(state, view)`` 的四类状态与 union/实例证据。"""

    status: VisibilityEvidenceStatus
    observation: ObservationEvidence
    union_alignment: Optional[SoftAlignmentMetrics]
    instances: tuple[InstanceVisibilityEvidence, ...]


def _validate_alpha(
    alpha: torch.Tensor,
    *,
    name: str,
    invariant_atol: float,
) -> None:
    if not isinstance(alpha, torch.Tensor):
        raise TypeError(f"{name} alpha 必须是 torch.Tensor")
    if alpha.dtype != torch.float32:
        raise TypeError(f"{name} alpha 必须使用 float32")
    if (
        alpha.ndim != 4
        or alpha.shape[0] <= 0
        or alpha.shape[1] != 1
        or alpha.shape[2] <= 0
        or alpha.shape[3] <= 0
    ):
        raise ValueError(f"{name} alpha shape 必须为 [K,1,H,W]")
    if not bool(torch.isfinite(alpha).all()):
        raise ValueError(f"{name} alpha 包含 NaN/Inf")
    if bool((alpha < -invariant_atol).any()) or bool(
        (alpha > 1.0 + invariant_atol).any()
    ):
        raise ValueError(f"{name} alpha 超出 [0,1] 容差")


def _observation(alpha: torch.Tensor) -> ObservationEvidence:
    equivalent_visible_pixels = float(alpha.sum().item())
    height = int(alpha.shape[-2])
    width = int(alpha.shape[-1])
    return ObservationEvidence(
        equivalent_visible_pixels=equivalent_visible_pixels,
        observation_area=equivalent_visible_pixels / float(height * width),
    )


def compute_soft_alignment_metrics(
    mujoco_alpha: torch.Tensor,
    renderer_alpha: torch.Tensor,
    *,
    denominator_epsilon: float = ALIGNMENT_DENOMINATOR_EPSILON,
    invariant_atol: float = VISIBILITY_INVARIANT_ATOL,
) -> SoftAlignmentMetrics:
    """计算 soft min-overlap recall、precision 与 IoU。"""

    _validate_alpha(
        mujoco_alpha,
        name="MuJoCo",
        invariant_atol=invariant_atol,
    )
    _validate_alpha(
        renderer_alpha,
        name="renderer",
        invariant_atol=invariant_atol,
    )
    if mujoco_alpha.shape != renderer_alpha.shape:
        raise ValueError("MuJoCo/renderer alpha shape 不一致")
    if mujoco_alpha.device != renderer_alpha.device:
        raise ValueError("MuJoCo/renderer alpha device 不一致")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("alignment denominator epsilon 必须为有限正数")

    overlap = float(torch.minimum(mujoco_alpha, renderer_alpha).sum().item())
    mujoco_visible = float(mujoco_alpha.sum().item())
    renderer_visible = float(renderer_alpha.sum().item())
    union_visible = float(torch.maximum(mujoco_alpha, renderer_alpha).sum().item())
    return SoftAlignmentMetrics(
        overlap=overlap,
        mujoco_visible_pixels=mujoco_visible,
        renderer_visible_pixels=renderer_visible,
        union_visible_pixels=union_visible,
        recall_visible=(
            overlap / (mujoco_visible + denominator_epsilon)
            if mujoco_visible > 0.0
            else None
        ),
        precision_visible=(
            overlap / (renderer_visible + denominator_epsilon)
            if renderer_visible > 0.0
            else None
        ),
        iou_visible=(
            overlap / (union_visible + denominator_epsilon)
            if union_visible > 0.0
            else None
        ),
    )


def evaluate_visibility(
    mujoco_instance_alpha: torch.Tensor,
    renderer_instance_alpha: Optional[torch.Tensor],
    *,
    thresholds: VisibilityThresholdCandidates = (
        VisibilityThresholdCandidates()
    ),
) -> VisibilityEvaluation:
    """按冻结顺序分类，并防止 union 指标掩盖可观测小实例失配。"""

    _validate_alpha(
        mujoco_instance_alpha,
        name="MuJoCo instances",
        invariant_atol=thresholds.invariant_atol,
    )
    if bool(
        (
            mujoco_instance_alpha.sum(dim=0)
            > 1.0 + thresholds.invariant_atol
        ).any()
    ):
        raise ValueError("MuJoCo instance alpha 不是 front-most 互斥分区")

    renderer_available = renderer_instance_alpha is not None
    if renderer_instance_alpha is not None:
        _validate_alpha(
            renderer_instance_alpha,
            name="renderer instances",
            invariant_atol=thresholds.invariant_atol,
        )
        if renderer_instance_alpha.shape != mujoco_instance_alpha.shape:
            raise ValueError("MuJoCo/renderer instance alpha shape 不一致")
        if renderer_instance_alpha.device != mujoco_instance_alpha.device:
            raise ValueError("MuJoCo/renderer instance alpha device 不一致")

    mujoco_union_alpha = mujoco_instance_alpha.sum(dim=0, keepdim=True)
    union_observation = _observation(mujoco_union_alpha)
    instance_evidence: list[InstanceVisibilityEvidence] = []
    per_instance_alignment_failure = False
    for instance_index in range(mujoco_instance_alpha.shape[0]):
        instance_mujoco_alpha = mujoco_instance_alpha[
            instance_index : instance_index + 1
        ]
        instance_observation = _observation(instance_mujoco_alpha)
        instance_alignment: Optional[SoftAlignmentMetrics] = None
        if renderer_instance_alpha is not None:
            instance_alignment = compute_soft_alignment_metrics(
                instance_mujoco_alpha,
                renderer_instance_alpha[
                    instance_index : instance_index + 1
                ],
                denominator_epsilon=thresholds.denominator_epsilon,
                invariant_atol=thresholds.invariant_atol,
            )
        if (
            instance_observation.observation_area
            >= thresholds.observation_area_min
            and (
                instance_alignment is None
                or instance_alignment.recall_visible is None
                or instance_alignment.recall_visible < thresholds.recall_min
            )
        ):
            per_instance_alignment_failure = True
        instance_evidence.append(
            InstanceVisibilityEvidence(
                instance_index=instance_index,
                observation=instance_observation,
                alignment=instance_alignment,
            )
        )

    union_alignment: Optional[SoftAlignmentMetrics] = None
    if renderer_instance_alpha is not None:
        renderer_union_alpha = 1.0 - torch.prod(
            1.0 - renderer_instance_alpha,
            dim=0,
            keepdim=True,
        )
        union_alignment = compute_soft_alignment_metrics(
            mujoco_union_alpha,
            renderer_union_alpha,
            denominator_epsilon=thresholds.denominator_epsilon,
            invariant_atol=thresholds.invariant_atol,
        )

    if union_observation.equivalent_visible_pixels == 0.0:
        status: VisibilityEvidenceStatus = "not_observable"
    elif (
        union_observation.observation_area
        < thresholds.observation_area_min
    ):
        status = "insufficient_observation"
    elif (
        not renderer_available
        or union_alignment is None
        or union_alignment.recall_visible is None
        or union_alignment.recall_visible < thresholds.recall_min
        or per_instance_alignment_failure
    ):
        status = "invalid_alignment"
    else:
        status = "valid"
    return VisibilityEvaluation(
        status=status,
        observation=union_observation,
        union_alignment=union_alignment,
        instances=tuple(instance_evidence),
    )
