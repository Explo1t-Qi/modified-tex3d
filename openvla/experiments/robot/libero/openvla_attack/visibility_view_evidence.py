"""一个 ``(state, view)`` 的 raw visibility 到 Effective View 编排。

MuJoCo front-most instance alpha 可以作为一个互斥 batch 共同变换；各 renderer
实例 mask 可能因缺少场景深度而互相重叠，因此必须逐实例独立执行同一连续
evidence transform，不能误用 MuJoCo union 上界检查。完成变换后，本模块统一
计算 source/pre-crop/effective 观测量与四类 Visibility Evidence Status。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .coverage_evidence import (
    CoverageEvidenceSpecification,
    PremultipliedEvidenceStages,
    transform_premultiplied_evidence,
)
from .visibility_evidence import (
    ObservationEvidence,
    VisibilityEvaluation,
    VisibilityThresholdCandidates,
    evaluate_visibility,
)


@dataclass(frozen=True)
class VisibilityObservationStages:
    """union 与逐实例在三个空间阶段的观测分母。"""

    source: ObservationEvidence
    pre_crop: ObservationEvidence
    effective: ObservationEvidence
    per_instance_source: tuple[ObservationEvidence, ...]
    per_instance_pre_crop: tuple[ObservationEvidence, ...]
    per_instance_effective: tuple[ObservationEvidence, ...]


@dataclass(frozen=True)
class ViewVisibilityEvidence:
    """一个视角的变换 tensor、观测量和最终分类。"""

    mujoco_stages: PremultipliedEvidenceStages
    renderer_effective_alpha: torch.Tensor
    observations: VisibilityObservationStages
    evaluation: VisibilityEvaluation


def _observation(alpha: torch.Tensor) -> ObservationEvidence:
    equivalent_visible_pixels: float = float(alpha.sum().item())
    height: int = int(alpha.shape[-2])
    width: int = int(alpha.shape[-1])
    return ObservationEvidence(
        equivalent_visible_pixels=equivalent_visible_pixels,
        observation_area=equivalent_visible_pixels / float(height * width),
    )


def _union_and_instances(
    alpha: torch.Tensor,
) -> tuple[ObservationEvidence, tuple[ObservationEvidence, ...]]:
    union_alpha: torch.Tensor = alpha.sum(dim=0, keepdim=True)
    return (
        _observation(union_alpha),
        tuple(
            _observation(alpha[index : index + 1])
            for index in range(alpha.shape[0])
        ),
    )


def build_view_visibility_evidence(
    mujoco_instance_alpha: torch.Tensor,
    renderer_instance_mask: torch.Tensor,
    *,
    specification: CoverageEvidenceSpecification = (
        CoverageEvidenceSpecification()
    ),
    thresholds: VisibilityThresholdCandidates = (
        VisibilityThresholdCandidates()
    ),
) -> ViewVisibilityEvidence:
    """共同变换两类 visibility，并按候选门槛完成四状态判定。

    两个输入均为 source-space float32 ``[K,1,H,W]``，实例顺序必须一致。
    renderer mask 只能表达 nvdiffrast 投影，不得替代 MuJoCo 遮挡 alpha。
    """

    if not isinstance(renderer_instance_mask, torch.Tensor):
        raise TypeError("renderer_instance_mask 必须是 torch.Tensor")
    if renderer_instance_mask.shape != mujoco_instance_alpha.shape:
        raise ValueError("MuJoCo alpha 与 renderer mask 实例/空间 shape 不一致")
    if renderer_instance_mask.device != mujoco_instance_alpha.device:
        raise ValueError("MuJoCo alpha 与 renderer mask device 不一致")

    mujoco_stages = transform_premultiplied_evidence(
        mujoco_instance_alpha,
        mujoco_instance_alpha,
        specification=specification,
    )
    renderer_effective_instances: list[torch.Tensor] = []
    for instance_index in range(renderer_instance_mask.shape[0]):
        instance_mask: torch.Tensor = renderer_instance_mask[
            instance_index : instance_index + 1
        ]
        instance_stages = transform_premultiplied_evidence(
            instance_mask,
            instance_mask,
            specification=specification,
        )
        renderer_effective_instances.append(instance_stages.effective.alpha)
    renderer_effective_alpha: torch.Tensor = torch.cat(
        renderer_effective_instances,
        dim=0,
    )

    source_union, source_instances = _union_and_instances(
        mujoco_stages.source.alpha
    )
    pre_crop_union, pre_crop_instances = _union_and_instances(
        mujoco_stages.pre_crop.alpha
    )
    effective_union, effective_instances = _union_and_instances(
        mujoco_stages.effective.alpha
    )
    evaluation = evaluate_visibility(
        mujoco_stages.effective.alpha,
        renderer_effective_alpha,
        thresholds=thresholds,
    )
    return ViewVisibilityEvidence(
        mujoco_stages=mujoco_stages,
        renderer_effective_alpha=renderer_effective_alpha,
        observations=VisibilityObservationStages(
            source=source_union,
            pre_crop=pre_crop_union,
            effective=effective_union,
            per_instance_source=source_instances,
            per_instance_pre_crop=pre_crop_instances,
            per_instance_effective=effective_instances,
        ),
        evaluation=evaluation,
    )
