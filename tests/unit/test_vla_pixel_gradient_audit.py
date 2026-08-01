"""跨 VLA 像素梯度审计的纯 CPU 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.vla_pixel_gradient_audit import (
    PixelGradientArtifact,
    build_fused_pixel_values,
    compare_pixel_gradient_artifacts,
    differentiable_center_crop,
    visible_perturbation_mask,
)


def _artifact(
    model_name: str,
    scale: float = 1.0,
    *,
    with_wrist: bool = False,
) -> PixelGradientArtifact:
    feature = np.ones((2, 2, 2, 3), dtype=np.float64) * scale
    action = feature * 2.0
    mask = np.ones((2, 2, 2), dtype=np.bool_)
    return PixelGradientArtifact(
        model_name=model_name,
        checkpoint=f"/{model_name}",
        state_ids=np.asarray([10, 11], dtype=np.int64),
        feature_losses=np.asarray([-1.0, -2.0]),
        action_losses=np.asarray([-3.0, -4.0]),
        primary_feature_gradients=feature,
        primary_action_gradients=action,
        perturbation_masks=mask,
        wrist_feature_gradients=feature * 2.0 if with_wrist else None,
        wrist_action_gradients=action * 3.0 if with_wrist else None,
    )


def test_artifact_round_trip_and_aligned_cosines(tmp_path: Path) -> None:
    source = _artifact("source")
    saved_path = source.save(tmp_path / "source.npz")
    loaded = PixelGradientArtifact.load(saved_path)
    target = _artifact("target", scale=3.0, with_wrist=True)

    summary = compare_pixel_gradient_artifacts(loaded, target)

    assert loaded.model_name == "source"
    assert summary["aggregate"]["cross_model_feature_cosine_masked"][
        "mean"
    ] == pytest.approx(1.0)
    assert summary["aggregate"]["cross_model_action_cosine_masked"][
        "mean"
    ] == pytest.approx(1.0)
    assert summary["aggregate"][
        "target_wrist_to_primary_feature_norm_ratio"
    ]["mean"] == pytest.approx(2.0)
    assert summary["aggregate"][
        "target_wrist_to_primary_action_norm_ratio"
    ]["mean"] == pytest.approx(3.0)
    assert summary["aggregate"][
        "source_feature_target_action_cosine_masked"
    ]["mean"] == pytest.approx(1.0)
    assert summary["aggregate"][
        "source_weighted_objective_target_action_cosine_masked"
    ]["mean"] == pytest.approx(1.0)


def test_center_crop_preserves_gradient_and_shape() -> None:
    image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
    image.requires_grad_(True)

    cropped = differentiable_center_crop(image, crop_area=0.9)
    cropped.sum().backward()

    assert cropped.shape == image.shape
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert float(image.grad.abs().sum()) > 0.0


def test_builds_fused_branches_in_caller_order() -> None:
    image = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    pixels = build_fused_pixel_values(
        image,
        means=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        stds=((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)),
    )
    assert tuple(pixels.shape) == (1, 6, 2, 2)
    assert torch.allclose(pixels[:, :3], torch.ones_like(pixels[:, :3]))
    assert torch.allclose(pixels[:, 3:], torch.ones_like(pixels[:, 3:]))


def test_visible_mask_ignores_one_level_png_noise() -> None:
    clean = np.zeros((2, 2, 3), dtype=np.uint8)
    adversarial = clean.copy()
    adversarial[0, 0, 0] = 1
    adversarial[1, 1, 2] = 2

    mask = visible_perturbation_mask(clean, adversarial, threshold=1)

    np.testing.assert_array_equal(
        mask,
        np.asarray([[False, False], [False, True]]),
    )
