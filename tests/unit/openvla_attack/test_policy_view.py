"""Policy Pre-Crop Canvas 的无 LIBERO 纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
sys.path.insert(0, str(OPENVLA_ROOT))

from openvla_attack.policy_view import (  # noqa: E402
    DeploymentViewSpecification,
    DifferentiablePolicyViewTransform,
    PolicyPreCropSpecification,
    build_exact_deployment_view_stages,
    resize_policy_pre_crop_canvas,
)
from experiments.robot.openvla_image_transform import (  # noqa: E402
    CenterCropSpecification,
)


def test_policy_canvas_uses_explicit_pillow_bicubic() -> None:
    """非平凡输入必须逐值等于显式 Pillow RGB bicubic。"""
    source_rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    specification = PolicyPreCropSpecification(
        source_resolution=4,
        canvas_resolution=2,
    )

    actual = resize_policy_pre_crop_canvas(
        source_rgb,
        specification=specification,
    )
    expected = np.asarray(
        Image.fromarray(source_rgb).resize(
            (2, 2),
            resample=Image.Resampling.BICUBIC,
        )
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.flags.writeable


@pytest.mark.parametrize(
    ("source_rgb", "error_type", "message"),
    [
        (
            np.zeros((4, 4, 3), dtype=np.float32),
            TypeError,
            "必须为 uint8",
        ),
        (
            np.zeros((3, 4, 3), dtype=np.uint8),
            ValueError,
            "shape 与 specification 不一致",
        ),
        (
            np.zeros((4, 4, 4), dtype=np.uint8),
            ValueError,
            "shape 与 specification 不一致",
        ),
    ],
)
def test_policy_canvas_rejects_invalid_source_contract(
    source_rgb: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    specification = PolicyPreCropSpecification(
        source_resolution=4,
        canvas_resolution=2,
    )

    with pytest.raises(error_type, match=message):
        resize_policy_pre_crop_canvas(
            source_rgb,
            specification=specification,
        )


def test_exact_deployment_view_stages_preserve_reference_order() -> None:
    """exact stages 必须严格执行 Pillow resize 后再执行 TF center crop。"""
    source_rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    specification = DeploymentViewSpecification(
        policy_canvas=PolicyPreCropSpecification(
            source_resolution=8,
            canvas_resolution=6,
        ),
        center_crop=CenterCropSpecification(
            input_resolution=6,
            output_resolution=6,
            crop_area=0.9,
        ),
    )

    stages = build_exact_deployment_view_stages(
        source_rgb,
        specification=specification,
    )

    expected_pre_crop = resize_policy_pre_crop_canvas(
        source_rgb,
        specification=specification.policy_canvas,
    )
    np.testing.assert_array_equal(stages.source_rgb, source_rgb)
    np.testing.assert_array_equal(stages.pre_crop_rgb, expected_pre_crop)
    assert stages.effective_view_rgb.shape == (6, 6, 3)
    assert stages.effective_view_rgb.dtype == np.uint8


def test_differentiable_policy_view_uses_exact_forward_and_surrogate_gradient(
) -> None:
    """BPDA forward 必须逐值 exact，backward 必须回到 512-source 空间。"""
    source_uint8 = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    specification = DeploymentViewSpecification(
        policy_canvas=PolicyPreCropSpecification(
            source_resolution=8,
            canvas_resolution=6,
        ),
        center_crop=CenterCropSpecification(
            input_resolution=6,
            output_resolution=6,
            crop_area=0.9,
        ),
    )
    transform = DifferentiablePolicyViewTransform(specification)
    source_nchw = (
        torch.from_numpy(source_uint8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float32)
        .div(255.0)
        .requires_grad_(True)
    )

    exact_stages = build_exact_deployment_view_stages(
        source_uint8,
        specification=specification,
    )
    actual_pre_crop = transform.build_pre_crop_canvas(source_nchw)
    expected_pre_crop_float = (
        torch.from_numpy(exact_stages.pre_crop_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float32)
        .div(255.0)
    )
    torch.testing.assert_close(
        actual_pre_crop.detach(),
        expected_pre_crop_float,
        rtol=0,
        atol=0,
    )

    actual = transform.build_effective_view(source_nchw)
    expected = exact_stages.effective_view_rgb
    expected_float = (
        torch.from_numpy(expected)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float32)
        .div(255.0)
    )
    torch.testing.assert_close(actual.detach(), expected_float, rtol=0, atol=0)
    actual_uint8 = (
        actual.detach()
        .mul(255.0)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_array_equal(actual_uint8, expected)

    upstream = torch.linspace(
        -1.0,
        1.0,
        actual.numel(),
        dtype=torch.float32,
    ).reshape_as(actual)
    torch.sum(actual * upstream).backward()
    assert source_nchw.grad is not None
    assert tuple(source_nchw.grad.shape) == (1, 3, 8, 8)
    assert bool(torch.isfinite(source_nchw.grad).all())
    assert float(source_nchw.grad.abs().max()) > 0.0


def test_deployment_view_rejects_mismatched_stage_resolutions() -> None:
    with pytest.raises(ValueError, match="Canvas.*center-crop"):
        DeploymentViewSpecification(
            policy_canvas=PolicyPreCropSpecification(
                source_resolution=8,
                canvas_resolution=6,
            ),
            center_crop=CenterCropSpecification(
                input_resolution=7,
                output_resolution=7,
                crop_area=0.9,
            ),
        )
