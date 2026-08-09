"""Action-only Spectral Guard可微语义与状态事务测试。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest
import torch
import torch.nn as nn

from openvla.experiments.robot.libero.openvla_attack.spectral_guard import (
    MeanActionGradient,
    SpectralGuardError,
    SpectralNaturalnessRegularizer,
    calibrate_spectral_guard,
)
from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    SurfaceStepStats,
)


_STATE_FINGERPRINTS = {state_id: f"{state_id:x}" * 64 for state_id in range(10)}


class _StatefulCounter:
    def __init__(self) -> None:
        self.value = 4

    def state_dict(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.value = int(state_dict["value"])


class _BrokenCounter(_StatefulCounter):
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.value = -1


class _CompactDelta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros((2, 3)))

    def forward(self) -> torch.Tensor:
        return self.coefficients


def _regularizer(*, rho_nat: float) -> SpectralNaturalnessRegularizer:
    # 零low basis让所有非零delta都落在高频补空间；这里只隔离校准控制语义，
    # M-正交频带本身由test_spectral_naturalness.py验收。
    return SpectralNaturalnessRegularizer(
        np.ones(2, dtype=np.float64),
        np.zeros((2, 129), dtype=np.float64),
        rho_nat=rho_nat,
        energy_epsilon=1e-12,
    )


def _mean_action_gradient(module: _CompactDelta) -> MeanActionGradient:
    loss = torch.mean((module.coefficients - 0.5).square())
    gradient = torch.autograd.grad(loss, module.coefficients)[0]
    return MeanActionGradient(
        loss=float(loss.detach().item()),
        gradient=gradient.detach(),
        num_frames=10,
        state_ids=tuple(range(10)),
        state_fingerprints=tuple(_STATE_FINGERPRINTS.values()),
    )


def _step_stats(module: _CompactDelta, actual_step: float) -> SurfaceStepStats:
    return SurfaceStepStats(
        direction_surface_max=1.0,
        parameter_scale=0.2,
        projection_scale=1.0,
        step_cap_scale=1.0,
        actual_surface_step=actual_step,
        max_abs_delta=float(module.coefficients.detach().abs().amax().item()),
    )


def test_regularizer_stopgrad_denominator_changes_training_gradient() -> None:
    delta = torch.tensor(
        [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]],
        requires_grad=True,
    )
    regularizer = _regularizer(rho_nat=0.1)
    terms = regularizer(delta)
    training_gradient = torch.autograd.grad(
        terms.penalty,
        delta,
        retain_graph=True,
    )[0]
    diagnostic_gradient = torch.autograd.grad(
        (torch.relu(terms.diagnostic_high_ratio - 0.1)).square(),
        delta,
    )[0]

    assert terms.high_ratio.item() == pytest.approx(
        terms.diagnostic_high_ratio.item()
    )
    assert torch.linalg.vector_norm(training_gradient).item() > 0.0
    assert torch.linalg.vector_norm(diagnostic_gradient).item() < 1e-5


def test_calibration_selects_first_five_activation_window_and_restores() -> None:
    module = _CompactDelta()
    module.coefficients.grad = torch.full_like(module.coefficients, 0.25)
    original_gradient = module.coefficients.grad.clone()
    counter = _StatefulCounter()

    def action_only_update(gradient: torch.Tensor) -> SurfaceStepStats:
        before = module.coefficients.detach().clone()
        with torch.no_grad():
            module.coefficients.add_(gradient, alpha=-0.2)
        counter.value += 1
        # 校准内部对所有RNG的消费都必须在结束后恢复。
        np.random.random()
        torch.rand(1)
        actual_step = float(
            (module.coefficients.detach() - before).abs().amax().item()
        )
        return _step_stats(module, actual_step)

    result = calibrate_spectral_guard(
        mutable_module=module,
        texture_parameter=module.coefficients,
        geometry_delta_provider=module,
        mean_action_gradient_provider=lambda: _mean_action_gradient(module),
        action_only_update=action_only_update,
        regularizer=_regularizer(rho_nat=0.1),
        stateful_components={"sampler_and_update": counter},
        expected_state_fingerprints=_STATE_FINGERPRINTS,
        surface_step=0.2,
    )

    assert result.calibration_status == "calibrated_stable_activation"
    assert result.selected_window_start == 1
    assert result.selected_window_end == 5
    assert result.q_median is not None and result.q_median > 0.0
    assert result.lambda_spec == pytest.approx(
        min(1.0, 0.1 / result.q_median)
    )
    assert len(result.iterations) == 6
    assert all(row.num_action_frames == 10 for row in result.iterations)
    assert result.restore_evidence.all_restored
    torch.testing.assert_close(
        module.coefficients,
        torch.zeros_like(module.coefficients),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(module.coefficients.grad, original_gradient)
    assert counter.value == 4


def _simple_update(
    module: _CompactDelta,
    gradient: torch.Tensor,
) -> SurfaceStepStats:
    before = module.coefficients.detach().clone()
    module.coefficients.data.add_(gradient, alpha=-0.1)
    actual_step = float(
        (module.coefficients.detach() - before).abs().amax().item()
    )
    return _step_stats(module, actual_step)


def test_calibration_without_stable_activation_uses_frozen_fallback() -> None:
    module = _CompactDelta()

    result = calibrate_spectral_guard(
        mutable_module=module,
        texture_parameter=module.coefficients,
        geometry_delta_provider=module,
        mean_action_gradient_provider=lambda: _mean_action_gradient(module),
        action_only_update=lambda gradient: _simple_update(module, gradient),
        regularizer=_regularizer(rho_nat=1.0),
        stateful_components={},
        expected_state_fingerprints=_STATE_FINGERPRINTS,
        surface_step=0.2,
        max_iterations=6,
    )

    assert result.calibration_status == "uncalibrated_no_stable_activation"
    assert result.q_median is None
    assert result.lambda_spec == 1.0
    assert result.selected_window_start is None
    assert result.restore_evidence.all_restored


@pytest.mark.parametrize(
    "state_ids",
    (
        tuple(range(9)),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 8),
        tuple(reversed(range(10))),
    ),
)
def test_calibration_rejects_incomplete_duplicate_or_reordered_states(
    state_ids: tuple[int, ...],
) -> None:
    module = _CompactDelta()

    def invalid_sample() -> MeanActionGradient:
        sample = _mean_action_gradient(module)
        return MeanActionGradient(
            loss=sample.loss,
            gradient=sample.gradient,
            num_frames=len(state_ids),
            state_ids=state_ids,
            state_fingerprints=tuple(
                _STATE_FINGERPRINTS[state_id] for state_id in state_ids
            ),
        )

    with pytest.raises(SpectralGuardError, match="10个|0-9"):
        calibrate_spectral_guard(
            mutable_module=module,
            texture_parameter=module.coefficients,
            geometry_delta_provider=module,
            mean_action_gradient_provider=invalid_sample,
            action_only_update=lambda gradient: _step_stats(module, 0.0),
            regularizer=_regularizer(rho_nat=1.0),
            stateful_components={},
            expected_state_fingerprints=_STATE_FINGERPRINTS,
            surface_step=0.2,
            max_iterations=1,
        )


def test_calibration_blocks_result_when_component_restore_is_false() -> None:
    module = _CompactDelta()
    broken = _BrokenCounter()

    with pytest.raises(SpectralGuardError, match="状态恢复失败"):
        calibrate_spectral_guard(
            mutable_module=module,
            texture_parameter=module.coefficients,
            geometry_delta_provider=module,
            mean_action_gradient_provider=lambda: _mean_action_gradient(module),
            action_only_update=lambda gradient: (
                setattr(broken, "value", broken.value + 1)
                or _step_stats(module, 0.0)
            ),
            regularizer=_regularizer(rho_nat=1.0),
            stateful_components={"broken": broken},
            expected_state_fingerprints=_STATE_FINGERPRINTS,
            surface_step=0.2,
            max_iterations=1,
        )
