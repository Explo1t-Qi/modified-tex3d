"""OpenVLA processor-equivalent 可微图像预处理的回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TVF
from PIL import Image


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

    # BPDA forward 先量化为部署 uint8；0.75 * 255 = 191.25，最近整数为191。
    exact_rgb_value = 191.0 / 255.0
    exact_rgb = torch.full((1, 3, 2, 2), exact_rgb_value)
    assert fused.shape == (1, 6, 2, 2)
    torch.testing.assert_close(
        fused[:, :3],
        exact_rgb,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fused[:, 3:],
        (exact_rgb - 0.5) / 0.5,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        preprocessor.build_siglip_pixel_values(image),
        fused[:, 3:],
    )
    fused.sum().backward()
    assert image.grad is not None
    assert float(image.grad.abs().sum()) > 0.0


def test_bpda_forward_exactly_matches_uint8_pil_bicubic() -> None:
    """非平凡图像的 forward 必须逐值等于 checkpoint PIL transform。"""
    preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model(),
        processor=_processor(),
    )
    rgb_uint8 = np.asarray(
        [
            [[0, 255, 17], [255, 0, 33], [9, 240, 64], [128, 1, 250]],
            [[250, 8, 0], [4, 127, 255], [200, 40, 80], [3, 251, 140]],
            [[12, 19, 230], [222, 111, 5], [70, 180, 20], [244, 2, 199]],
            [[255, 100, 1], [13, 220, 177], [99, 33, 255], [0, 199, 48]],
        ],
        dtype=np.uint8,
    )
    image = (
        torch.from_numpy(rgb_uint8.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .requires_grad_(True)
    )
    pil_resized = TVF.resize(
        Image.fromarray(rgb_uint8),
        size=(2, 2),
        interpolation=Image.Resampling.BICUBIC,
        antialias=True,
    )
    exact_rgb = TVF.to_tensor(pil_resized).unsqueeze(0)
    expected_fused = torch.cat(
        (exact_rgb, (exact_rgb - 0.5) / 0.5),
        dim=1,
    )

    fused = preprocessor.build_fused_pixel_values(image)

    torch.testing.assert_close(fused, expected_fused, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        preprocessor.build_siglip_pixel_values(image),
        expected_fused[:, 3:],
        rtol=0.0,
        atol=0.0,
    )
    fused.square().mean().backward()
    assert image.grad is not None
    assert bool(torch.isfinite(image.grad).all())
    assert float(image.grad.abs().sum()) > 0.0


def test_preprocessor_rejects_non_equivalent_resize_configuration() -> None:
    processor = _processor()
    processor.image_processor.tvf_resize_params[0]["interpolation"] = 2

    with pytest.raises(ValueError, match="bicubic"):
        DifferentiableOpenVLAImageProcessor.from_checkpoint(
            model=_model(),
            processor=processor,
        )
