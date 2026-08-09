"""逐顶点 coverage contribution 伴随重放测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.openvla_image_transform import CenterCropSpecification
from openvla.experiments.robot.libero.openvla_attack.coverage_evidence import (
    CoverageEvidenceSpecification,
    transform_premultiplied_evidence,
)
from openvla.experiments.robot.libero.openvla_attack.support_coverage_contribution import (
    SupportCoverageContributionError,
    build_projection_coverage_evidence,
    effective_sum_source_weights,
)


def _specification() -> CoverageEvidenceSpecification:
    return CoverageEvidenceSpecification(
        source_resolution=4,
        pre_crop_resolution=2,
        center_crop=CenterCropSpecification(
            input_resolution=2,
            output_resolution=2,
            crop_area=1.0,
        ),
    )


def _arrays() -> dict[str, np.ndarray]:
    alpha = np.zeros((2, 1, 4, 4), dtype=np.float32)
    alpha[0, 0, :2, :] = 1.0
    alpha[1, 0, 2:, :] = 1.0
    triangle_indices = np.zeros((2, 4, 4), dtype=np.int64)
    barycentric = np.zeros((2, 4, 4, 3), dtype=np.float32)
    barycentric[..., 0] = 0.25
    barycentric[..., 1] = 0.25
    barycentric[..., 2] = 0.50
    corners = np.empty((2, 4, 4, 3), dtype=np.int64)
    corners[..., 0] = 0
    corners[..., 1] = 1
    corners[..., 2] = 2
    stages = transform_premultiplied_evidence(
        torch.from_numpy(alpha),
        torch.from_numpy(alpha),
        specification=_specification(),
    )
    return {
        "schema_version": np.asarray("openvla-visibility-alignment-v1"),
        "state_id": np.asarray(3, dtype=np.int64),
        "view_name": np.asarray("primary"),
        "mujoco_alpha_source": alpha,
        "mujoco_alpha_pre_crop": stages.pre_crop.alpha.numpy(),
        "mujoco_alpha_effective": stages.effective.alpha.numpy(),
        "triangle_indices": triangle_indices,
        "barycentric": barycentric,
        "geometry_corner_indices": corners,
    }


def test_effective_sum_adjoint_preserves_constant_image() -> None:
    weights = effective_sum_source_weights(_specification())
    assert weights.dtype == np.float64
    assert weights.shape == (4, 4)
    assert float(weights.sum()) == pytest.approx(4.0)
    assert np.all(weights >= 0.0)


def test_vertex_contributions_reproduce_arbitrary_support_transform() -> None:
    arrays = _arrays()
    evidence, diagnostics = build_projection_coverage_evidence(
        arrays,
        status="valid",
        num_vertices=3,
        specification=_specification(),
    )
    support = np.asarray([True, False, True], dtype=np.bool_)
    # 每个像素受 vertex 0 的 0.25 与 vertex 2 的 0.50 控制。
    assert evidence.coverage(support, space="source") == pytest.approx(0.75)
    assert evidence.coverage(support, space="effective") == pytest.approx(0.75)
    assert evidence.per_instance_coverage(support) == pytest.approx((0.75, 0.75))
    assert diagnostics.source_full_support_coverage == pytest.approx(1.0)
    assert diagnostics.effective_full_support_coverage == pytest.approx(1.0)
    assert diagnostics.effective_adjoint_denominator_absolute_error <= 2e-3


def test_union_is_sum_of_instances_not_single_semantic_body() -> None:
    evidence, _ = build_projection_coverage_evidence(
        _arrays(),
        status="valid",
        num_vertices=3,
        specification=_specification(),
    )
    np.testing.assert_allclose(
        evidence.effective_vertex_numerator,
        evidence.effective_instance_vertex_numerator.sum(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    assert evidence.effective_instance_vertex_numerator.shape == (2, 3)


def test_invalid_corner_mapping_fails_before_scatter() -> None:
    arrays = _arrays()
    arrays["geometry_corner_indices"][0, 0, 0, 0] = 9
    with pytest.raises(SupportCoverageContributionError, match="索引越界"):
        build_projection_coverage_evidence(
            arrays,
            status="valid",
            num_vertices=3,
            specification=_specification(),
        )


def test_saved_effective_alpha_must_replay_from_source() -> None:
    arrays = _arrays()
    arrays["mujoco_alpha_effective"] = arrays[
        "mujoco_alpha_effective"
    ].copy()
    arrays["mujoco_alpha_effective"][0, 0, 0, 0] = 0.5
    with pytest.raises(SupportCoverageContributionError, match="不能由 source"):
        build_projection_coverage_evidence(
            arrays,
            status="valid",
            num_vertices=3,
            specification=_specification(),
        )
