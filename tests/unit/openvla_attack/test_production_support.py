"""Production Fixed Support 冻结 artifact 的 CPU 测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from openvla.experiments.robot.libero.openvla_attack.production_support import (
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    FrozenProductionSupport,
    ProductionSupportError,
    ProductionSupportEvidence,
    ProductionSupportProvenance,
    array_sha256,
    evaluate_production_support_artifact,
    load_production_support_artifact,
    write_production_support_artifact,
)


def _provenance() -> ProductionSupportProvenance:
    return ProductionSupportProvenance(
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
        mesh_array_sha256="a" * 64,
        renderer_faces_sha256="b" * 64,
        render_to_geometry_sha256="c" * 64,
    )


def _frozen_support(
    *,
    formal_training_allowed: bool = False,
    support_vertex_indices: np.ndarray | None = None,
) -> FrozenProductionSupport:
    mask = np.asarray([False, True, False, True], dtype=np.bool_)
    indices = (
        np.asarray([1, 3], dtype=np.int64)
        if support_vertex_indices is None
        else support_vertex_indices
    )
    return FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=_provenance(),
        selected_candidate_index=0,
        num_geometry_vertices=4,
        support_mask=mask,
        support_vertex_indices=indices,
        support_mask_sha256=array_sha256(mask),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=1,
        seed_vertex_ids=(1,),
        total_surface_area=10.0,
        target_area_fraction=0.1,
        target_mass=1.0,
        actual_mass=1.01,
        actual_area_fraction=0.101,
        primary_coverage_min_threshold=0.2,
        primary_state_ids=(0, 1),
        primary_statuses=("valid", "valid"),
        primary_source_coverage=(0.3, 0.4),
        primary_effective_coverage=(0.31, 0.41),
        wrist_state_ids=(0, 1),
        wrist_statuses=("valid", "valid"),
        wrist_source_coverage=(0.1, 0.2),
        wrist_effective_coverage=(0.11, 0.21),
        naturalness_k_nonconstant=128,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=formal_training_allowed,
    )


def test_frozen_support_round_trip_preserves_compact_coordinate_order(
    tmp_path: Path,
) -> None:
    support = _frozen_support()
    path = tmp_path / "production_fixed_support.npz"
    artifact_sha = write_production_support_artifact(path, support)
    loaded = load_production_support_artifact(
        path,
        expected_mesh_file_sha256="9" * 64,
        expected_mesh_array_sha256="a" * 64,
        expected_render_to_geometry_sha256="c" * 64,
    )
    np.testing.assert_array_equal(loaded.support_mask, support.support_mask)
    np.testing.assert_array_equal(
        loaded.support_vertex_indices,
        np.asarray([1, 3], dtype=np.int64),
    )
    assert loaded.compact_coordinate_order == "ascending_geometry_vertex_id"
    evidence = ProductionSupportEvidence(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        artifact_relative_path=path.name,
        artifact_sha256=artifact_sha,
        support_mask_sha256=support.support_mask_sha256,
        num_geometry_vertices=4,
        num_support_vertices=2,
        trainable_rgb_scalar_count=6,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    decision = evaluate_production_support_artifact(
        evidence,
        artifact_root=tmp_path,
        expected=support,
    )
    assert decision.gate_pass, decision.failures


def test_freeze_rejects_noncanonical_compact_order(tmp_path: Path) -> None:
    with pytest.raises(ProductionSupportError, match="升序"):
        write_production_support_artifact(
            tmp_path / "invalid.npz",
            _frozen_support(
                support_vertex_indices=np.asarray([3, 1], dtype=np.int64)
            ),
        )


def test_freeze_cannot_claim_formal_training_before_calibration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProductionSupportError, match="不得提前声称"):
        write_production_support_artifact(
            tmp_path / "invalid_training.npz",
            _frozen_support(formal_training_allowed=True),
        )


def test_loader_rejects_renderer_mapping_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "production_fixed_support.npz"
    write_production_support_artifact(path, _frozen_support())
    with pytest.raises(ProductionSupportError, match="render_to_geometry"):
        load_production_support_artifact(
            path,
            expected_render_to_geometry_sha256="d" * 64,
        )


def test_freeze_runner_does_not_import_training_or_reconstruct_support() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "openvla/experiments/robot/libero/openvla_attack/freeze_production_support.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_frozen_support_from_accepted_inputs" in calls
    assert "construct_akita_support_candidates" not in calls
    assert "get_model" not in calls
    assert "backward" not in calls
    assert "formal_training_allowed=False" in source


def test_attack_entry_blocks_fixed_support_before_spectral_calibration() -> None:
    """现有legacy trainer不得提前消费已冻结但未校准的Support。"""

    path = (
        Path(__file__).resolve().parents[3]
        / "openvla/experiments/robot/libero/attack_openvla.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fixed_support_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "texture_parameterization == 'fixed_support'"
    ]

    assert fixed_support_guards
    assert any(
        isinstance(node, ast.Raise)
        for guard in fixed_support_guards
        for node in ast.walk(guard)
    )
    assert "Action-only trainer" in source
