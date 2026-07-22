"""OpenVLA 对抗渲染图像合成的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.compositing import (
    ImageResolution,
    SingleViewFrame,
    build_single_view_samples,
    composite_foreground,
    render_and_composite,
)


def test_composite_uses_foreground_mask_and_clamps_rgb() -> None:
    foreground_rgb = torch.tensor([[[[-0.2, 0.5, 1.2], [1.0, 1.0, 1.0]]]])
    foreground_mask = torch.tensor([[[[1.0], [0.0]]]])
    background_rgb = torch.tensor(
        [[[[0.9, 0.25]], [[0.9, 0.25]], [[0.9, 0.25]]]]
    )

    composited_rgb = composite_foreground(
        foreground_rgb, foreground_mask, background_rgb
    )

    expected_rgb = torch.tensor(
        [[[[0.0, 0.25]], [[0.5, 0.25]], [[1.0, 0.25]]]]
    )
    torch.testing.assert_close(composited_rgb, expected_rgb)


def test_render_and_composite_returns_background_when_mvp_is_missing() -> None:
    class RendererThatMustNotBeCalled:
        def render(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            del args, kwargs
            raise AssertionError("mvp=None 时不应调用 renderer")

    background_rgb = torch.rand((1, 3, 2, 2), dtype=torch.float32)

    composited_rgb = render_and_composite(
        RendererThatMustNotBeCalled(), background_rgb, mvp=None
    )

    assert composited_rgb is background_rgb


def test_single_view_builder_prefers_background_without_target() -> None:
    class FakeRenderer:
        def __init__(self) -> None:
            self.calls: list[
                tuple[torch.Tensor, ImageResolution, Optional[torch.Tensor]]
            ] = []

        def render(
            self,
            mvp: torch.Tensor,
            resolution: ImageResolution,
            model_rot: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append((mvp, resolution, model_rot))
            foreground_rgb = torch.full((1, 1, 1, 3), 0.8)
            foreground_mask = torch.zeros((1, 1, 1, 1))
            return foreground_rgb, foreground_mask

    renderer = FakeRenderer()
    mvp = torch.eye(4)
    model_rotation = torch.eye(3)
    regular_background = torch.full((1, 3, 1, 1), 0.1)
    background_without_target = torch.full((1, 3, 1, 1), 0.2)
    frame: SingleViewFrame = {
        "mvp": mvp,
        "model_rot": model_rotation,
        "bg_tensor": regular_background,
        "bg_tensor_no_obj": background_without_target,
    }

    samples = build_single_view_samples(renderer, frame, render_resolution=64)

    assert len(renderer.calls) == 1
    called_mvp, called_resolution, called_rotation = renderer.calls[0]
    assert called_mvp is mvp
    assert called_resolution == (64, 64)
    assert called_rotation is model_rotation
    assert len(samples) == 1
    torch.testing.assert_close(samples[0], background_without_target)
