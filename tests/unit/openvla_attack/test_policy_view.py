"""Policy Pre-Crop Canvas 的无 LIBERO 纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.policy_view import (  # noqa: E402
    PolicyPreCropSpecification,
    resize_policy_pre_crop_canvas,
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
