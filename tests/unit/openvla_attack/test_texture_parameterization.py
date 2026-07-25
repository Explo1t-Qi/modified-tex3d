"""曲面纹理参数化的预算、梯度和公平更新测试。"""

from __future__ import annotations

import numpy as np
import torch

from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    GeometryVertexTextureParameterization,
    SpectralTextureParameterization,
    surface_normalized_step_,
)


MAPPING = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int64)
BASIS = np.asarray(
    [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [-0.5, 0.5],
    ],
    dtype=np.float32,
)


def test_spectral_render_copies_receive_identical_surface_delta() -> None:
    parameterization = SpectralTextureParameterization(
        BASIS,
        MAPPING,
        epsilon=0.25,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.copy_(
            torch.tensor(
                [[0.1, 0.2, -0.1], [0.05, -0.1, 0.2]]
            )
        )

    render_delta = parameterization.render_delta()

    torch.testing.assert_close(render_delta[0], render_delta[3])
    torch.testing.assert_close(render_delta[2], render_delta[4])


def test_spectral_projection_preserves_subspace_and_budget() -> None:
    parameterization = SpectralTextureParameterization(
        BASIS,
        MAPPING,
        epsilon=0.125,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.fill_(10.0)
        original_coefficients = parameterization.coefficients.clone()

    projection_scale = parameterization.project_coefficients_()

    assert projection_scale < 1.0
    assert parameterization.max_abs_delta() <= 0.125 + 1e-7
    torch.testing.assert_close(
        parameterization.coefficients,
        original_coefficients * projection_scale,
        rtol=1e-6,
        atol=1e-7,
    )


def test_gradient_reaches_spectral_coefficients() -> None:
    parameterization = SpectralTextureParameterization(
        BASIS,
        MAPPING,
        epsilon=0.5,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.fill_(0.02)

    loss = (parameterization.render_delta() ** 2).sum()
    loss.backward()

    gradient = parameterization.coefficients.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_geometry_vertex_shares_uv_seam_and_enforces_budget() -> None:
    parameterization = GeometryVertexTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        epsilon=0.125,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.fill_(10.0)

    parameterization.project_coefficients_()
    render_delta = parameterization.render_delta()

    assert parameterization.max_abs_delta() <= 0.125 + 1e-7
    torch.testing.assert_close(render_delta[0], render_delta[3])
    torch.testing.assert_close(render_delta[2], render_delta[4])


def test_surface_normalized_step_matches_both_parameterizations() -> None:
    geometry = GeometryVertexTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        epsilon=0.5,
        device="cpu",
    )
    spectral = SpectralTextureParameterization(
        BASIS,
        MAPPING,
        epsilon=0.5,
        device="cpu",
    )

    geometry_stats = surface_normalized_step_(
        geometry,
        torch.arange(1, 13, dtype=torch.float32).reshape(4, 3),
        surface_step=0.02,
    )
    spectral_stats = surface_normalized_step_(
        spectral,
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 1.0, 2.0]]),
        surface_step=0.02,
    )

    assert abs(geometry_stats.actual_surface_step - 0.02) < 1e-6
    assert abs(spectral_stats.actual_surface_step - 0.02) < 1e-6
    assert abs(geometry.max_abs_delta() - 0.02) < 1e-6
    assert abs(spectral.max_abs_delta() - 0.02) < 1e-6


def test_surface_step_stays_capped_while_projection_is_active() -> None:
    parameterization = GeometryVertexTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        epsilon=0.1,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.copy_(
            torch.tensor(
                [
                    [0.1, -0.1, 0.05],
                    [0.08, -0.08, 0.0],
                    [0.04, -0.04, 0.02],
                    [0.02, -0.02, 0.01],
                ]
            )
        )
    before_delta = parameterization.geometry_delta().clone()

    stats = surface_normalized_step_(
        parameterization,
        torch.tensor(
            [
                [-4.0, -1.0, 1.0],
                [-3.0, 2.0, -1.0],
                [2.0, -2.0, 1.0],
                [-1.0, 3.0, -2.0],
            ]
        ),
        surface_step=0.02,
    )

    actual_step = float(
        (parameterization.geometry_delta() - before_delta).abs().amax()
    )
    assert actual_step <= 0.02 + 1e-7
    assert abs(actual_step - stats.actual_surface_step) < 1e-7
    assert parameterization.max_abs_delta() <= 0.1 + 1e-7
