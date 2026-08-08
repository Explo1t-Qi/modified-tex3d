"""Visibility-Masked Renderer Delta Composition 的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.visibility_compositing import (  # noqa: E402
    compose_visibility_masked_renderer_delta,
)


def _two_instance_inputs() -> tuple[torch.Tensor, ...]:
    mujoco_clean = torch.full((1, 3, 1, 3), 0.5, dtype=torch.float32)
    alpha = torch.tensor(
        [
            [[[[1.0, 0.0, 0.0]]]],
            [[[[0.0, 1.0, 0.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 1, 3)
    renderer_clean = torch.full((2, 3, 1, 3), 0.4, dtype=torch.float32)
    renderer_adversarial = renderer_clean.clone()
    renderer_adversarial[0] += 0.1
    renderer_adversarial[1] -= 0.2
    renderer_mask = torch.ones((2, 1, 1, 3), dtype=torch.float32)
    return (
        mujoco_clean,
        alpha,
        renderer_adversarial,
        renderer_clean,
        renderer_mask,
    )


def test_zero_renderer_delta_is_exact_identity() -> None:
    mujoco_clean, alpha, _, renderer_clean, renderer_mask = (
        _two_instance_inputs()
    )

    result = compose_visibility_masked_renderer_delta(
        mujoco_clean,
        alpha,
        renderer_clean.clone(),
        renderer_clean,
        renderer_mask,
    )

    torch.testing.assert_close(result.composited_rgb, mujoco_clean, rtol=0, atol=0)
    assert not bool(result.total_delta.any())
    assert result.saturated_pixel_fraction == 0.0
    assert result.saturated_channel_fraction == 0.0


def test_frontmost_alpha_and_renderer_mask_jointly_gate_delta() -> None:
    inputs = list(_two_instance_inputs())
    renderer_mask = inputs[-1]
    renderer_mask[1, :, :, 1] = 0.0

    result = compose_visibility_masked_renderer_delta(*inputs)

    expected = torch.full((1, 3, 1, 3), 0.5)
    expected[:, :, :, 0] = 0.6
    torch.testing.assert_close(result.composited_rgb, expected)


def test_multi_instance_sum_is_invariant_to_instance_order() -> None:
    inputs = _two_instance_inputs()
    original = compose_visibility_masked_renderer_delta(*inputs)
    permutation = torch.tensor([1, 0])
    permuted = compose_visibility_masked_renderer_delta(
        inputs[0],
        inputs[1][permutation],
        inputs[2][permutation],
        inputs[3][permutation],
        inputs[4][permutation],
    )

    torch.testing.assert_close(permuted.composited_rgb, original.composited_rgb)
    torch.testing.assert_close(permuted.total_delta, original.total_delta)


def test_only_adversarial_renderer_branch_receives_gradient() -> None:
    inputs = list(_two_instance_inputs())
    adversarial = inputs[2].clone().requires_grad_(True)
    clean = inputs[3].clone()

    result = compose_visibility_masked_renderer_delta(
        inputs[0],
        inputs[1],
        adversarial,
        clean,
        inputs[4],
    )
    result.composited_rgb.sum().backward()

    assert adversarial.grad is not None
    expected_gradient = inputs[1] * inputs[4]
    torch.testing.assert_close(
        adversarial.grad,
        expected_gradient.expand(-1, 3, -1, -1),
    )
    assert clean.grad is None


def test_clamp_reports_pixel_and_channel_saturation() -> None:
    mujoco_clean = torch.tensor([[[[0.95]], [[0.5]], [[0.05]]]])
    alpha = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    renderer_clean = torch.full((1, 3, 1, 1), 0.5)
    renderer_adversarial = torch.tensor([[[[0.7]], [[0.5]], [[0.3]]]])
    renderer_mask = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    result = compose_visibility_masked_renderer_delta(
        mujoco_clean,
        alpha,
        renderer_adversarial,
        renderer_clean,
        renderer_mask,
    )

    torch.testing.assert_close(
        result.unclamped_rgb,
        torch.tensor([[[[1.15]], [[0.5]], [[-0.15]]]]),
    )
    torch.testing.assert_close(
        result.composited_rgb,
        torch.tensor([[[[1.0]], [[0.5]], [[0.0]]]]),
    )
    assert result.saturated_pixel_fraction == 1.0
    assert result.saturated_channel_fraction == pytest.approx(2.0 / 3.0)


def test_overlapping_mujoco_instances_fail_fast() -> None:
    inputs = list(_two_instance_inputs())
    inputs[1][1, 0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="不是 front-most 互斥分区"):
        compose_visibility_masked_renderer_delta(*inputs)


def test_soft_raw_mujoco_alpha_fails_fast() -> None:
    inputs = list(_two_instance_inputs())
    inputs[1][0, 0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="精确 0/1 hard mask"):
        compose_visibility_masked_renderer_delta(*inputs)


def test_trainable_clean_renderer_branch_fails_fast() -> None:
    inputs = list(_two_instance_inputs())
    inputs[3] = inputs[3].requires_grad_(True)

    with pytest.raises(ValueError, match="必须与 trainable Surface Delta 断图"):
        compose_visibility_masked_renderer_delta(*inputs)
