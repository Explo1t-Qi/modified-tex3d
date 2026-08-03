"""OpenVLA processor-equivalent 可微图像预处理的回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            timm_model_ids=(
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            )
        )
    )


def _processor() -> SimpleNamespace:
    image_processor = SimpleNamespace(
        image_resize_strategy="resize-naive",
        input_sizes=((3, 2, 2), (3, 2, 2)),
        tvf_resize_params=(
            {
                "size": (2, 2),
                "interpolation": 3,
                "antialias": True,
                "max_size": None,
            },
            {
                "size": (2, 2),
                "interpolation": 3,
                "antialias": True,
                "max_size": None,
            },
        ),
        tvf_crop_params=(
            {"output_size": (2, 2)},
            {"output_size": (2, 2)},
        ),
        tvf_normalize_params=(
            {
                "mean": (0.0, 0.0, 0.0),
                "std": (1.0, 1.0, 1.0),
                "inplace": False,
            },
            {
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
                "inplace": False,
            },
        ),
    )
    return SimpleNamespace(image_processor=image_processor)


def test_fused_pixels_follow_checkpoint_dino_then_siglip_order() -> None:
    preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model(),
        processor=_processor(),
    )
    image = torch.full(
        (1, 3, 4, 4),
        0.75,
        dtype=torch.float32,
        requires_grad=True,
    )

    fused = preprocessor.build_fused_pixel_values(image)

    assert fused.shape == (1, 6, 2, 2)
    torch.testing.assert_close(
        fused[:, :3],
        torch.full((1, 3, 2, 2), 0.75),
    )
    torch.testing.assert_close(
        fused[:, 3:],
        torch.full((1, 3, 2, 2), 0.5),
    )
    torch.testing.assert_close(
        preprocessor.build_siglip_pixel_values(image),
        fused[:, 3:],
    )
    fused.sum().backward()
    assert image.grad is not None
    assert float(image.grad.abs().sum()) > 0.0


def test_preprocessor_rejects_non_equivalent_resize_configuration() -> None:
    processor = _processor()
    processor.image_processor.tvf_resize_params[0]["interpolation"] = 2

    with pytest.raises(ValueError, match="bicubic"):
        DifferentiableOpenVLAImageProcessor.from_checkpoint(
            model=_model(),
            processor=processor,
        )
