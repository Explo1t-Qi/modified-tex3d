"""正式Fixed-Support trainer共享更新核心测试。"""

from __future__ import annotations

import torch
import torch.nn as nn

from openvla.experiments.robot.libero.openvla_attack.fixed_support_training import (
    FixedSupportActionTrainerCore,
)
from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    FixedSupportTextureParameterization,
    SurfaceStepStats,
    surface_normalized_step_,
)


class _Renderer:
    epsilon = 0.5

    def __init__(self) -> None:
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
