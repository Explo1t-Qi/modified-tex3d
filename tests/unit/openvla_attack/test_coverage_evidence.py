"""连续 coverage evidence transform 的纯 CPU 不变量测试。"""

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

from experiments.robot.openvla_image_transform import (  # noqa: E402
    CenterCropSpecification,
)
from openvla_attack.coverage_evidence import (  # noqa: E402
    CoverageEvidenceSpecification,
    summarize_effective_coverage,
    transform_premultiplied_evidence,
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


def test_area_downsampling_preserves_zero_and_full_control() -> None:
    alpha = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    alpha[0, :, :2, :] = 1.0
    alpha[1, :, 2:, :] = 1.0

    zero = transform_premultiplied_evidence(
        alpha,
        torch.zeros_like(alpha),
        specification=_specification(),
    )
    full = transform_premultiplied_evidence(
        alpha,
        alpha.clone(),
        specification=_specification(),
    )

    torch.testing.assert_close(
        zero.effective.alpha_times_control,
        torch.zeros_like(zero.effective.alpha),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        full.effective.alpha_times_control,
        full.effective.alpha,
        rtol=0,
        atol=0,
    )
    expected_first = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        full.effective.alpha[0:1],
        expected_first,
        rtol=0,
        atol=0,
    )


def test_transform_preserves_support_monotonicity() -> None:
    alpha = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    smaller_control = torch.zeros_like(alpha)
    smaller_control[:, :, 1:3, 1:3] = 0.25
    larger_control = smaller_control.clone()
    larger_control[:, :, 0, :] = 0.75

    smaller = transform_premultiplied_evidence(
        alpha,
        alpha * smaller_control,
        specification=_specification(),
    )
    larger = transform_premultiplied_evidence(
        alpha,
        alpha * larger_control,
        specification=_specification(),
    )

    assert bool(
        (
            smaller.effective.alpha_times_control
            <= larger.effective.alpha_times_control + 1e-6
        ).all()
    )


def test_instance_aware_summary_uses_visible_area_weighting() -> None:
    alpha = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    alpha[0, 0, 0, :] = 1.0  # instance 0: 2 pixels, coverage 1
    alpha[1, 0, 1, 0] = 1.0  # instance 1: 1 pixel, coverage 0
    premultiplied = torch.zeros_like(alpha)
    premultiplied[0] = alpha[0]

    stages = transform_premultiplied_evidence(
        alpha,
        premultiplied,
        specification=CoverageEvidenceSpecification(
            source_resolution=2,
            pre_crop_resolution=2,
            center_crop=CenterCropSpecification(
                input_resolution=2,
                output_resolution=2,
                crop_area=1.0,
            ),
        ),
    )
    summary = summarize_effective_coverage(stages.effective)

    assert summary.coverage == pytest.approx(2.0 / 3.0)
    assert summary.numerator == 2.0
    assert summary.equivalent_visible_pixels == 3.0
    assert summary.observation_area == 0.75
    assert summary.per_instance_coverage == pytest.approx((1.0, 0.0))
    assert summary.per_instance_equivalent_visible_pixels == (2.0, 1.0)


@pytest.mark.parametrize(
    ("alpha", "premultiplied", "message"),
    [
        (
            torch.full((1, 1, 2, 2), -0.1),
            torch.zeros((1, 1, 2, 2)),
            "alpha 超出",
        ),
        (
            torch.ones((1, 1, 2, 2)),
            torch.full((1, 1, 2, 2), 1.1),
            "alpha\\*control",
        ),
        (
            torch.full((2, 1, 2, 2), 0.75),
            torch.zeros((2, 1, 2, 2)),
            "不是互斥可见分区",
        ),
        (
            torch.full((1, 1, 2, 2), float("nan")),
            torch.zeros((1, 1, 2, 2)),
            "NaN/Inf",
        ),
    ],
)
def test_transform_rejects_invalid_premultiplied_evidence(
    alpha: torch.Tensor,
    premultiplied: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        transform_premultiplied_evidence(
            alpha.to(torch.float32),
            premultiplied.to(torch.float32),
            specification=CoverageEvidenceSpecification(
                source_resolution=2,
                pre_crop_resolution=2,
                center_crop=CenterCropSpecification(
                    input_resolution=2,
                    output_resolution=2,
                    crop_area=1.0,
                ),
            ),
        )


def test_summary_represents_absent_observation_as_none() -> None:
    alpha = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    stages = transform_premultiplied_evidence(
        alpha,
        torch.zeros_like(alpha),
        specification=CoverageEvidenceSpecification(
            source_resolution=2,
            pre_crop_resolution=2,
            center_crop=CenterCropSpecification(
                input_resolution=2,
                output_resolution=2,
                crop_area=1.0,
            ),
        ),
    )
    summary = summarize_effective_coverage(stages.effective)

    assert summary.coverage is None
    assert summary.per_instance_coverage == (None,)
    assert summary.observation_area == 0.0
