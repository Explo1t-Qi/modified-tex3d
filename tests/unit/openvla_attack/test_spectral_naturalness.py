"""Fixed Support 谱自然性与rho_nat校准的CPU测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from openvla.experiments.robot.libero.openvla_attack.production_support import (
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    FrozenProductionSupport,
    ProductionSupportProvenance,
    array_sha256,
    write_production_support_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    mesh_array_sha256,
    save_spectral_basis,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_naturalness import (
    NATURALNESS_CALIBRATION_SCHEMA_VERSION,
    SpectralNaturalnessError,
    build_rho_nat_calibration,
    compute_naturalness_energy,
    evaluate_rho_nat_calibration,
    load_rho_nat_calibration_artifact,
    write_rho_nat_calibration_artifact,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    num_vertices = 130
    vertices = np.column_stack(
        (
            np.arange(num_vertices, dtype=np.float64),
            np.zeros(num_vertices, dtype=np.float64),
            np.zeros(num_vertices, dtype=np.float64),
        )
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mass = np.ones(num_vertices, dtype=np.float64)
    generator = np.random.default_rng(7)
    candidates = np.column_stack(
        (
            np.ones(num_vertices, dtype=np.float64),
            generator.standard_normal((num_vertices, 128)),
        )
    )
    basis, _ = np.linalg.qr(candidates)
    eigenvalues = np.arange(129, dtype=np.float64)
    basis_path = tmp_path / "continuous_k128.npz"
    save_spectral_basis(
        basis_path,
        vertices=vertices,
        faces=faces,
        mass=mass,
        eigenvalues_with_constant=eigenvalues,
        basis_with_constant=basis,
        metadata={"kind": "continuous_geometry_low_frequency"},
    )

    support_mask = np.zeros(num_vertices, dtype=np.bool_)
    support_mask[[3, 17]] = True
    mesh_sha = mesh_array_sha256(vertices, faces)
    support = FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=ProductionSupportProvenance(
            code_commit="1" * 40,
            object_name="akita_black_bowl",
            source_support_manifest_sha256="2" * 64,
            source_candidate_artifact_sha256="3" * 64,
            source_coverage_artifact_sha256="4" * 64,
            seed_score_manifest_sha256="5" * 64,
            seed_score_artifact_sha256="6" * 64,
            visibility_manifest_sha256="7" * 64,
            visibility_metrics_sha256="8" * 64,
            mesh_file_sha256="9" * 64,
            mesh_array_sha256=mesh_sha,
            renderer_faces_sha256="a" * 64,
            render_to_geometry_sha256="b" * 64,
        ),
        selected_candidate_index=0,
        num_geometry_vertices=num_vertices,
        support_mask=support_mask,
        support_vertex_indices=np.asarray([3, 17], dtype=np.int64),
        support_mask_sha256=array_sha256(support_mask),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=1,
        seed_vertex_ids=(3,),
        total_surface_area=float(num_vertices),
        target_area_fraction=2.0 / num_vertices,
        target_mass=2.0,
        actual_mass=2.0,
        actual_area_fraction=2.0 / num_vertices,
        primary_coverage_min_threshold=0.2,
        primary_state_ids=(0,),
        primary_statuses=("valid",),
        primary_source_coverage=(0.3,),
        primary_effective_coverage=(0.3,),
        wrist_state_ids=(),
        wrist_statuses=(),
        wrist_source_coverage=(),
        wrist_effective_coverage=(),
        naturalness_k_nonconstant=128,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    support_path = tmp_path / "production_fixed_support.npz"
    write_production_support_artifact(support_path, support)
    return support_path, basis_path


def test_energy_uses_rgb_joint_mass_projection() -> None:
    delta = np.asarray([[1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.ones(2)
    constant = np.full((2, 1), 1.0 / np.sqrt(2.0))

    energy = compute_naturalness_energy(
        delta,
        mass=mass,
        low_basis=constant,
        epsilon=1e-12,
    )

    assert energy.total == pytest.approx(6.0)
    assert energy.low == pytest.approx(4.0)
    assert energy.high == pytest.approx(2.0)
    assert energy.high_ratio == pytest.approx(2.0 / (6.0 + 1e-12))


def test_rho_nat_round_trip_and_independent_recompute(tmp_path: Path) -> None:
    support_path, basis_path = _write_inputs(tmp_path)
    calibration = build_rho_nat_calibration(
        production_support_path=support_path,
        spectral_basis_path=basis_path,
        code_commit="c" * 40,
        object_name="akita_black_bowl",
    )
    artifact_path = tmp_path / "rho_nat.npz"
    write_rho_nat_calibration_artifact(artifact_path, calibration)
    loaded = load_rho_nat_calibration_artifact(artifact_path)
    decision = evaluate_rho_nat_calibration(
        artifact_path,
        production_support_path=support_path,
        spectral_basis_path=basis_path,
    )

    assert loaded.schema_version == NATURALNESS_CALIBRATION_SCHEMA_VERSION
    assert loaded.num_low_modes_including_constant == 129
    assert loaded.low_band_eigenvalues.shape == (129,)
    assert 0.0 <= loaded.rho_nat <= 1.0
    assert loaded.rho_nat_calibrated
    assert not loaded.lambda_spec_calibrated
    assert not loaded.formal_training_allowed
    assert decision.gate_pass, decision.failures


def test_calibration_rejects_support_basis_mesh_mismatch(
    tmp_path: Path,
) -> None:
    support_path, basis_path = _write_inputs(tmp_path)
    with np.load(basis_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["vertices"] = payload["vertices"].copy()
    payload["vertices"][0, 1] = 1.0
    changed_basis_path = tmp_path / "changed_basis.npz"
    np.savez_compressed(changed_basis_path, **payload)

    with pytest.raises((SpectralNaturalnessError, ValueError), match="mesh"):
        build_rho_nat_calibration(
            production_support_path=support_path,
            spectral_basis_path=changed_basis_path,
            code_commit="c" * 40,
            object_name="akita_black_bowl",
        )


def test_calibration_runner_does_not_import_model_or_training() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "openvla/experiments/robot/libero/openvla_attack"
        / "calibrate_spectral_naturalness.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_rho_nat_calibration" in calls
    assert "get_model" not in calls
    assert "backward" not in calls
    assert "rollout_episode" not in calls
    assert '"formal_training_allowed": False' in source
