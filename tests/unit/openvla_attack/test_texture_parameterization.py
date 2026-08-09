"""曲面纹理参数化的预算、梯度和公平更新测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    FixedSupportTextureParameterization,
    GeometryVertexTextureParameterization,
    SpectralTextureParameterization,
    TextureParameterizationError,
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


def test_fixed_support_uses_compact_parameters_and_scatters_to_geometry() -> None:
    """Support 外严格为零，UV seam 副本仍共享同一几何顶点增量。"""
    parameterization = FixedSupportTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        support_vertex_indices=np.asarray([1, 3], dtype=np.int64),
        epsilon=0.25,
        device="cpu",
    )
    with torch.no_grad():
        parameterization.coefficients.copy_(
            torch.tensor(
                [[0.1, -0.2, 0.05], [-0.1, 0.15, 0.2]],
                dtype=torch.float32,
            )
        )

    geometry_delta = parameterization.geometry_delta()
    render_delta = parameterization.render_delta()

    assert tuple(parameterization.coefficients.shape) == (2, 3)
    assert parameterization.support_vertex_indices.tolist() == [1, 3]
    torch.testing.assert_close(geometry_delta[0], torch.zeros(3))
    torch.testing.assert_close(geometry_delta[2], torch.zeros(3))
    torch.testing.assert_close(
        geometry_delta[1],
        parameterization.coefficients[0],
    )
    torch.testing.assert_close(
        geometry_delta[3],
        parameterization.coefficients[1],
    )
    torch.testing.assert_close(render_delta[0], render_delta[3])
    torch.testing.assert_close(render_delta[2], render_delta[4])


def test_fixed_support_gradient_lives_only_in_compact_delta_s_space() -> None:
    """完整 Surface loss 应反传到 ``[|S|,3]``，而不是伪全顶点参数。"""
    parameterization = FixedSupportTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        support_vertex_indices=torch.tensor([0, 2]),
        epsilon=0.5,
        device="cpu",
    )
    channel_weights = torch.tensor([1.0, 2.0, 3.0])

    loss = (parameterization.geometry_delta() * channel_weights).sum()
    loss.backward()

    assert parameterization.coefficients.grad is not None
    assert tuple(parameterization.coefficients.grad.shape) == (2, 3)
    torch.testing.assert_close(
        parameterization.coefficients.grad,
        channel_weights.expand(2, 3),
    )


@pytest.mark.parametrize(
    "support_indices,match",
    [
        ([], "不得为空"),
        ([1, 1], "重复"),
        ([-1, 2], "越界"),
        ([1, 4], "越界"),
        ([1.5], "整数"),
    ],
)
def test_fixed_support_rejects_invalid_external_support(
    support_indices: list[int],
    match: str,
) -> None:
    """参数化只消费已冻结 Support，必须拒绝不完整或歧义索引。"""
    with pytest.raises(TextureParameterizationError, match=match):
        FixedSupportTextureParameterization(
            MAPPING,
            num_geometry_vertices=4,
            support_vertex_indices=support_indices,
            epsilon=0.25,
            device="cpu",
        )


def test_fixed_support_surface_step_preserves_mask_and_budget() -> None:
    parameterization = FixedSupportTextureParameterization(
        MAPPING,
        num_geometry_vertices=4,
        support_vertex_indices=[1, 3],
        epsilon=0.1,
        device="cpu",
    )
    stats = surface_normalized_step_(
        parameterization,
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 1.0, 2.0]]),
        surface_step=0.02,
    )

    geometry_delta = parameterization.geometry_delta()
    assert abs(stats.actual_surface_step - 0.02) < 1e-6
    assert parameterization.max_abs_delta() <= 0.1 + 1e-7
    torch.testing.assert_close(geometry_delta[0], torch.zeros(3))
    torch.testing.assert_close(geometry_delta[2], torch.zeros(3))


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
