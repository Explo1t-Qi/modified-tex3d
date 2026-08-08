"""Visibility 四类状态与 soft alignment 的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.visibility_evidence import (  # noqa: E402
    A_OBS_MIN_FROZEN,
    RECALL_MIN_FROZEN,
    VISIBILITY_THRESHOLDS_FROZEN,
    VisibilityThresholdCandidates,
    compute_soft_alignment_metrics,
    evaluate_visibility,
)


def test_visibility_threshold_defaults_are_formally_frozen() -> None:
    thresholds = VisibilityThresholdCandidates()

    assert VISIBILITY_THRESHOLDS_FROZEN
    assert thresholds.observation_area_min == A_OBS_MIN_FROZEN == 1e-3
    assert thresholds.recall_min == RECALL_MIN_FROZEN == 0.95


def _thresholds(
    *,
    observation_area_min: float = 0.1,
    recall_min: float = 0.75,
) -> VisibilityThresholdCandidates:
    return VisibilityThresholdCandidates(
        observation_area_min=observation_area_min,
        recall_min=recall_min,
    )


def test_soft_alignment_uses_min_overlap_and_max_union() -> None:
    mujoco = torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]])
    renderer = torch.tensor([[[[0.5, 1.0], [1.0, 0.0]]]])

    metrics = compute_soft_alignment_metrics(mujoco, renderer)

    assert metrics.overlap == 1.0
    assert metrics.mujoco_visible_pixels == 1.5
    assert metrics.renderer_visible_pixels == 2.5
    assert metrics.union_visible_pixels == 3.0
    assert metrics.recall_visible == pytest.approx(2.0 / 3.0)
    assert metrics.precision_visible == pytest.approx(0.4)
    assert metrics.iou_visible == pytest.approx(1.0 / 3.0)


def test_zero_observation_is_not_observable_without_renderer() -> None:
    evaluation = evaluate_visibility(
        torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        None,
        thresholds=_thresholds(),
    )

    assert evaluation.status == "not_observable"
    assert evaluation.observation.observation_area == 0.0
    assert evaluation.union_alignment is None


def test_small_observation_is_insufficient_before_alignment_check() -> None:
    alpha = torch.zeros((1, 1, 10, 10), dtype=torch.float32)
    alpha[0, 0, 0, 0] = 1.0

    evaluation = evaluate_visibility(
        alpha,
        None,
        thresholds=_thresholds(observation_area_min=0.02),
    )

    assert evaluation.status == "insufficient_observation"
    assert evaluation.observation.equivalent_visible_pixels == 1.0
    assert evaluation.observation.observation_area == 0.01


def test_sufficient_observation_without_renderer_is_invalid_alignment() -> None:
    alpha = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    evaluation = evaluate_visibility(
        alpha,
        None,
        thresholds=_thresholds(),
    )

    assert evaluation.status == "invalid_alignment"


def test_union_recall_failure_is_invalid_alignment() -> None:
    mujoco = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    renderer = torch.zeros_like(mujoco)
    renderer[:, :, :2, :] = 1.0

    evaluation = evaluate_visibility(
        mujoco,
        renderer,
        thresholds=_thresholds(),
    )

    assert evaluation.status == "invalid_alignment"
    assert evaluation.union_alignment is not None
    assert evaluation.union_alignment.recall_visible == pytest.approx(0.5)


def test_observable_small_instance_failure_cannot_hide_behind_union() -> None:
    mujoco = torch.zeros((2, 1, 10, 10), dtype=torch.float32)
    mujoco[0, 0, :8, :] = 1.0
    mujoco[1, 0, 8:, :] = 1.0
    renderer = torch.zeros_like(mujoco)
    renderer[0] = mujoco[0]

    evaluation = evaluate_visibility(
        mujoco,
        renderer,
        thresholds=_thresholds(
            observation_area_min=0.1,
            recall_min=0.75,
        ),
    )

    assert evaluation.union_alignment is not None
    assert evaluation.union_alignment.recall_visible == pytest.approx(0.8)
    assert evaluation.instances[1].observation.observation_area == 0.2
    assert evaluation.instances[1].alignment is not None
    assert evaluation.instances[1].alignment.recall_visible == 0.0
    assert evaluation.status == "invalid_alignment"


def test_all_union_and_instance_recalls_pass_is_valid() -> None:
    mujoco = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    mujoco[0, 0, :2, :] = 1.0
    mujoco[1, 0, 2:, :] = 1.0

    evaluation = evaluate_visibility(
        mujoco,
        mujoco.clone(),
        thresholds=_thresholds(),
    )

    assert evaluation.status == "valid"
    assert all(
        instance.alignment is not None
        and instance.alignment.recall_visible == pytest.approx(1.0)
        for instance in evaluation.instances
    )


def test_mujoco_instance_alpha_must_be_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="不是 front-most 互斥分区"):
        evaluate_visibility(
            torch.full((2, 1, 2, 2), 0.75),
            torch.full((2, 1, 2, 2), 0.5),
            thresholds=_thresholds(),
        )
