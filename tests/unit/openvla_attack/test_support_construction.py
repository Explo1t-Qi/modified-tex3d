"""Support Construction 冻结规则的 CPU 单元测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openvla.experiments.robot.libero.openvla_attack.support_construction import (
    AKITA_PRIMARY_COVERAGE_MIN,
    ProjectionCoverageEvidence,
    SupportConstructionError,
    build_face_adjacency,
    construct_akita_support_candidates,
    geodesic_distance_from_support,
    grow_equal_mass_regions,
)


def _strip_mesh(num_vertices: int = 9) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[float(index), float(index % 2), 0.0] for index in range(num_vertices)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[index, index + 1, index + 2] for index in range(num_vertices - 2)],
        dtype=np.int64,
    )
    return vertices, faces


def _coverage(
    values: list[float],
    *,
    state_id: int,
    view_name: str = "primary",
    status: str = "valid",
) -> ProjectionCoverageEvidence:
    numerators = np.asarray(values, dtype=np.float64)
    instance_numerators = numerators[None, :].copy()
    return ProjectionCoverageEvidence(
        state_id=state_id,
        view_name=view_name,
        status=status,  # type: ignore[arg-type]
        source_vertex_numerator=numerators.copy(),
        effective_vertex_numerator=numerators.copy(),
        source_denominator=1.0,
        effective_denominator=1.0,
        source_instance_vertex_numerator=instance_numerators.copy(),
        effective_instance_vertex_numerator=instance_numerators.copy(),
        source_instance_denominators=np.ones(1, dtype=np.float64),
        effective_instance_denominators=np.ones(1, dtype=np.float64),
    )


def test_projection_coverage_is_linear_and_preserves_missing_observation() -> None:
    evidence = _coverage([0.1, 0.2, 0.3], state_id=0)
    mask = np.asarray([True, False, True], dtype=np.bool_)
    assert evidence.coverage(mask) == pytest.approx(0.4)

    missing = ProjectionCoverageEvidence(
        state_id=1,
        view_name="primary",
        status="not_observable",
        source_vertex_numerator=np.zeros(3, dtype=np.float64),
        effective_vertex_numerator=np.zeros(3, dtype=np.float64),
        source_denominator=0.0,
        effective_denominator=0.0,
        source_instance_vertex_numerator=np.zeros((1, 3), dtype=np.float64),
        effective_instance_vertex_numerator=np.zeros((1, 3), dtype=np.float64),
        source_instance_denominators=np.zeros(1, dtype=np.float64),
        effective_instance_denominators=np.zeros(1, dtype=np.float64),
    )
    assert missing.coverage(mask) is None


def test_face_adjacency_and_edge_length_geodesic_are_exact() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    adjacency = build_face_adjacency(faces, num_vertices=4)
    assert adjacency == ((1, 2), (0, 2, 3), (0, 1, 3), (1, 2))
    support = np.asarray([True, False, False, False], dtype=np.bool_)
    distances = geodesic_distance_from_support(vertices, adjacency, support)
    assert distances.tolist() == pytest.approx(
        [0.0, 1.0, math.sqrt(2.0), 1.0 + math.sqrt(2.0)]
    )


def test_region_growing_is_best_first_connected_and_one_vertex_overshoot() -> None:
    vertices, faces = _strip_mesh(7)
    adjacency = build_face_adjacency(faces, num_vertices=7)
    mass = np.ones(7, dtype=np.float64)
    density = np.asarray([0.0, 0.3, 1.0, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    regions, support = grow_equal_mass_regions(
        seed_vertex_ids=(0,),
        target_mass_per_region=2.5,
        vertices=vertices,
        adjacency=adjacency,
        vertex_mass=mass,
        smoothed_density=density,
    )
    assert regions[0].vertex_ids == (0, 2, 3)
    assert regions[0].actual_mass == 3.0
    assert regions[0].actual_mass - regions[0].target_mass <= regions[0].last_added_vertex_mass
    assert support.tolist() == [True, False, True, True, False, False, False]


def test_multi_region_growth_round_robins_and_forbids_cross_adjacency() -> None:
    vertices = np.asarray(
        [[float(index), 0.0, 0.0] for index in range(7)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [4, 5, 5], [5, 6, 6]],
        dtype=np.int64,
    )
    # 退化 triangle 会被 adjacency 明确拒绝，避免测试虚假的链式拓扑。
    with pytest.raises(SupportConstructionError, match="退化自环"):
        build_face_adjacency(faces, num_vertices=7)

    vertices, faces = _strip_mesh(10)
    adjacency = build_face_adjacency(faces, num_vertices=10)
    with pytest.raises(SupportConstructionError, match="face-adjacent"):
        grow_equal_mass_regions(
            seed_vertex_ids=(0, 2),
            target_mass_per_region=2.0,
            vertices=vertices,
            adjacency=adjacency,
            vertex_mass=np.ones(10, dtype=np.float64),
            smoothed_density=np.arange(10, dtype=np.float64),
        )


def test_construct_candidates_uses_minimum_r_and_keeps_wrist_read_only() -> None:
    vertices, faces = _strip_mesh(9)
    mass = np.ones(9, dtype=np.float64)
    density = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    primary = (
        _coverage([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], state_id=0),
        _coverage([0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], state_id=1),
    )
    wrist = (
        _coverage(
            [0.0] * 9,
            state_id=0,
            view_name="wrist_source_crop_proxy",
        ),
    )
    result = construct_akita_support_candidates(
        vertices=vertices,
        faces=faces,
        vertex_mass=mass,
        smoothed_density=density,
        primary_evidence=primary,
        wrist_evidence=wrist,
    )
    assert result.selected_candidate_index == 0
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.seed_vertex_ids == (0,)
    assert candidate.primary_gate_pass
    assert candidate.primary_valid_min is not None
    assert candidate.primary_valid_min >= AKITA_PRIMARY_COVERAGE_MIN
    assert candidate.wrist_coverage[0].effective_coverage == 0.0
    assert not result.production_support_constructed


def test_second_seed_uses_worst_state_gain_and_geodesic_separation() -> None:
    vertices, faces = _strip_mesh(12)
    mass = np.full(12, 100.0, dtype=np.float64)
    density = np.asarray(
        [12.0, 10.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        dtype=np.float64,
    )
    # r=1 只选择 seed 0，state 1 成为最差 state；顶点 1 虽密度更高但离已有
    # support 太近，新增 seed 应落到具有正增益且分离的最低 ID 顶点。
    primary = (
        _coverage([0.21] + [0.0] * 11, state_id=0),
        _coverage([0.0, 0.3, 0.0, 0.0, 0.25] + [0.0] * 7, state_id=1),
    )
    result = construct_akita_support_candidates(
        vertices=vertices,
        faces=faces,
        vertex_mass=mass,
        smoothed_density=density,
        primary_evidence=primary,
    )
    assert len(result.candidates) >= 2
    second = result.candidates[1]
    assert second.seed_diagnostics[0].worst_primary_state_id == 1
    assert second.seed_diagnostics[0].marginal_coverage_gain > 0.0
    assert second.seed_diagnostics[0].geodesic_distance + 1e-12 >= result.smoothing_length


def test_primary_invalid_alignment_fails_before_support_construction() -> None:
    vertices, faces = _strip_mesh(6)
    with pytest.raises(SupportConstructionError, match="invalid_alignment"):
        construct_akita_support_candidates(
            vertices=vertices,
            faces=faces,
            vertex_mass=np.ones(6, dtype=np.float64),
            smoothed_density=np.ones(6, dtype=np.float64),
            primary_evidence=(
                _coverage([0.1] * 6, state_id=0, status="invalid_alignment"),
            ),
        )


def test_fixed_constants_are_not_exposed_as_candidate_parameters() -> None:
    assert math.isclose(AKITA_PRIMARY_COVERAGE_MIN, 0.20)
    with pytest.raises(TypeError):
        construct_akita_support_candidates(  # type: ignore[call-arg]
            vertices=np.zeros((3, 3), dtype=np.float64),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            vertex_mass=np.ones(3, dtype=np.float64),
            smoothed_density=np.ones(3, dtype=np.float64),
            primary_evidence=(_coverage([0.1] * 3, state_id=0),),
            primary_coverage_min=0.19,
        )
