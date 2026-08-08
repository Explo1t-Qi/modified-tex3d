"""raw visibility 到 Effective View 编排的 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
sys.path.insert(0, str(OPENVLA_ROOT))

from openvla_attack.coverage_evidence import (  # noqa: E402
    CoverageEvidenceSpecification,
)
from openvla_attack.visibility_evidence import (  # noqa: E402
    VisibilityThresholdCandidates,
)
from openvla_attack.visibility_view_evidence import (  # noqa: E402
    build_view_visibility_evidence,
)
from experiments.robot.openvla_image_transform import (  # noqa: E402
    CenterCropSpecification,
)


def _specification() -> CoverageEvidenceSpecification:
    return CoverageEvidenceSpecification(
        source_resolution=8,
        pre_crop_resolution=4,
        center_crop=CenterCropSpecification(
            input_resolution=4,
            output_resolution=4,
            crop_area=1.0,
        ),
    )


def _thresholds() -> VisibilityThresholdCandidates:
    return VisibilityThresholdCandidates(
        observation_area_min=0.05,
        recall_min=0.75,
    )


def test_view_evidence_tracks_observation_across_all_stages() -> None:
    mujoco = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    mujoco[0, 0, :4, :] = 1.0
    mujoco[1, 0, 4:, :] = 1.0

    result = build_view_visibility_evidence(
        mujoco,
        mujoco.clone(),
        specification=_specification(),
        thresholds=_thresholds(),
    )

    assert result.evaluation.status == "valid"
    assert result.observations.source.observation_area == 1.0
    assert result.observations.pre_crop.observation_area == 1.0
    assert result.observations.effective.observation_area == 1.0
    assert [
        item.observation_area
        for item in result.observations.per_instance_effective
    ] == [0.5, 0.5]


def test_overlapping_renderer_instances_are_transformed_independently() -> None:
    mujoco = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    mujoco[0, 0, :4, :] = 1.0
    mujoco[1, 0, 4:, :] = 1.0
    renderer = torch.ones_like(mujoco)

    result = build_view_visibility_evidence(
        mujoco,
        renderer,
        specification=_specification(),
        thresholds=_thresholds(),
    )

    assert result.evaluation.status == "valid"
    assert result.renderer_effective_alpha.sum().item() == 32.0
    assert result.evaluation.union_alignment is not None
    assert result.evaluation.union_alignment.precision_visible == pytest.approx(
        1.0
    )


def test_missing_renderer_projection_is_invalid_alignment() -> None:
    mujoco = torch.ones((1, 1, 8, 8), dtype=torch.float32)

    result = build_view_visibility_evidence(
        mujoco,
        torch.zeros_like(mujoco),
        specification=_specification(),
        thresholds=_thresholds(),
    )

    assert result.evaluation.status == "invalid_alignment"
    assert result.evaluation.union_alignment is not None
    assert result.evaluation.union_alignment.recall_visible == 0.0
