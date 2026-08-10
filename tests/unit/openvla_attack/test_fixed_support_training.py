"""正式Fixed-Support trainer共享更新核心测试。"""

from __future__ import annotations

import torch
import torch.nn as nn

from openvla.experiments.robot.libero.openvla_attack.fixed_support_training import (
    FixedSupportActionTrainerCore,
    FixedSupportTrainerCore,
    FixedSupportTrainingError,
)
from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    FixedSupportTextureParameterization,
    SurfaceStepStats,
    surface_normalized_step_,
)


class _Renderer:
    epsilon = 0.5

    def __init__(self) -> None:
        self.step_call_count = 0
        self.parameterization = FixedSupportTextureParameterization(
            render_to_geometry=torch.arange(4),
            num_geometry_vertices=4,
            support_vertex_indices=torch.tensor([1, 3]),
            epsilon=self.epsilon,
        )

    def get_texture_parameterization_name(self) -> str:
        return "fixed_support"

    def get_texture_param(self) -> nn.Parameter:
        return self.parameterization.coefficients

    def get_geometry_surface_delta(self) -> torch.Tensor:
        return self.parameterization.geometry_delta()

    def step_surface_parameterization_(
        self,
        gradient: torch.Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        self.step_call_count += 1
        return surface_normalized_step_(
            self.parameterization,
            gradient,
            surface_step,
        )


def test_shared_trainer_core_uses_surface_normalized_projected_step() -> None:
    renderer = _Renderer()
    trainer = FixedSupportActionTrainerCore(renderer, surface_step=0.1)
    gradient = torch.tensor(
        [[2.0, -1.0, 0.5], [-4.0, 0.25, 1.0]],
        dtype=torch.float32,
    )

    stats = trainer.apply_action_gradient(gradient)

    assert trainer.update_count == 1
    assert stats.actual_surface_step <= 0.1 + 1e-7
    assert stats.max_abs_delta <= renderer.epsilon + 1e-7
    assert trainer.texture_parameter.shape == (2, 3)
    assert trainer.geometry_delta().shape == (4, 3)
    torch.testing.assert_close(
        trainer.geometry_delta()[[0, 2]],
        torch.zeros((2, 3)),
        rtol=0.0,
        atol=0.0,
    )


def test_shared_trainer_core_state_round_trip_restores_update_count() -> None:
    trainer = FixedSupportActionTrainerCore(_Renderer(), surface_step=0.1)
    state = trainer.state_dict()
    trainer.apply_action_gradient(torch.ones((2, 3)))

    trainer.load_state_dict(state)

    assert trainer.update_count == 0
    assert trainer.surface_step == 0.1


def test_combined_training_gradient_is_summed_before_exactly_one_update() -> None:
    renderer = _Renderer()
    trainer = FixedSupportTrainerCore(renderer, surface_step=0.1)
    action_gradient = torch.tensor(
        [[2.0, -1.0, 0.5], [-4.0, 0.25, 1.0]],
        dtype=torch.float32,
    )
    spectral_gradient = torch.tensor(
        [[-3.0, 2.0, 1.0], [1.5, -0.5, -2.0]],
        dtype=torch.float32,
    )

    update = trainer.apply_action_spectral_gradients(
        action_gradient,
        spectral_gradient,
        lambda_spec=0.25,
    )

    expected_weighted = spectral_gradient * 0.25
    expected_total = action_gradient + expected_weighted
    torch.testing.assert_close(
        update.weighted_spectral_gradient,
        expected_weighted,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        update.total_gradient,
        expected_total,
        rtol=0.0,
        atol=0.0,
    )
    assert update.combination_residual_linf == 0.0
    expected_action_total_cosine = float(
        torch.nn.functional.cosine_similarity(
            action_gradient.reshape(1, -1),
            expected_total.reshape(1, -1),
        ).item()
    )
    assert update.action_total_cosine is not None
    assert abs(
        update.action_total_cosine - expected_action_total_cosine
    ) < 1e-7
    assert renderer.step_call_count == 1
    assert trainer.update_count == 1
    assert update.surface_step_stats.actual_surface_step <= 0.1 + 1e-7


def test_zero_spectral_gradient_reduces_exactly_to_action_update() -> None:
    renderer = _Renderer()
    trainer = FixedSupportTrainerCore(renderer, surface_step=0.1)
    action_gradient = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, -0.25, 0.125]],
        dtype=torch.float32,
    )

    update = trainer.apply_action_spectral_gradients(
        action_gradient,
        torch.zeros_like(action_gradient),
        lambda_spec=0.0008505366725298619,
    )

    torch.testing.assert_close(
        update.total_gradient,
        action_gradient,
        rtol=0.0,
        atol=0.0,
    )
    assert update.spectral_gradient_l2 == 0.0
    assert update.weighted_spectral_gradient_l2 == 0.0
    assert update.weighted_spectral_action_ratio == 0.0
    assert update.action_spectral_cosine is None
    assert renderer.step_call_count == 1


def test_action_only_control_uses_exact_action_gradient_and_one_shared_step() -> None:
    renderer = _Renderer()
    trainer = FixedSupportTrainerCore(renderer, surface_step=0.1)
    action_gradient = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, -0.25, 0.125]],
        dtype=torch.float32,
    )

    update = trainer.apply_action_only_gradient(action_gradient)

    torch.testing.assert_close(
        update.total_gradient,
        action_gradient,
        rtol=0.0,
        atol=0.0,
    )
    assert update.action_total_cosine == 1.0
    assert update.spectral_gradient_l2 == 0.0
    assert update.weighted_spectral_action_ratio == 0.0
    assert update.combination_residual_linf == 0.0
    assert renderer.step_call_count == 1
    assert trainer.update_count == 1


def test_zero_action_gradient_still_allows_spectral_restoration_update() -> None:
    renderer = _Renderer()
    trainer = FixedSupportTrainerCore(renderer, surface_step=0.1)
    spectral_gradient = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, -0.25, 0.125]],
        dtype=torch.float32,
    )

    update = trainer.apply_action_spectral_gradients(
        torch.zeros_like(spectral_gradient),
        spectral_gradient,
        lambda_spec=0.5,
    )

    torch.testing.assert_close(
        update.total_gradient,
        spectral_gradient * 0.5,
        rtol=0.0,
        atol=0.0,
    )
    assert update.action_spectral_cosine is None
    assert update.weighted_spectral_action_ratio is None
    assert renderer.step_call_count == 1


def test_combined_training_rejects_invalid_lambda_and_gradient_shape() -> None:
    trainer = FixedSupportTrainerCore(_Renderer(), surface_step=0.1)
    gradient = torch.ones((2, 3))

    for invalid_lambda in (-0.1, float("nan"), 1.1):
        try:
            trainer.apply_action_spectral_gradients(
                gradient,
                gradient,
                lambda_spec=invalid_lambda,
            )
        except FixedSupportTrainingError:
            pass
        else:
            raise AssertionError("非法lambda_spec必须失败")

    try:
        trainer.apply_action_spectral_gradients(
            gradient,
            torch.ones((3, 3)),
            lambda_spec=0.1,
        )
    except FixedSupportTrainingError:
        pass
    else:
        raise AssertionError("不匹配的spectral梯度shape必须失败")


def test_legacy_action_core_name_is_compatibility_alias() -> None:
    assert FixedSupportActionTrainerCore is FixedSupportTrainerCore
