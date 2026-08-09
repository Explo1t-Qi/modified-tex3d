"""Support Construction 双层 artifact 与 repeat comparison 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    SeedScoreArrays,
)
from openvla.experiments.robot.libero.openvla_attack.support_construction import (
    ProjectionCoverageEvidence,
    construct_akita_support_candidates,
)
from openvla.experiments.robot.libero.openvla_attack.support_construction_audit import (
    SUPPORT_AUDIT_SCHEMA_VERSION,
    SupportArtifactEvidence,
    SupportAuditProvenance,
    compare_support_repeat_artifacts,
    evaluate_support_artifacts,
    file_sha256,
    load_coverage_contribution_artifact,
    write_candidate_artifact,
    write_coverage_contribution_artifact,
)


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[float(index), float(index % 2), 0.0] for index in range(9)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[index, index + 1, index + 2] for index in range(7)],
        dtype=np.int64,
    )
    return vertices, faces


def _coverage() -> ProjectionCoverageEvidence:
    numerator = np.asarray([0.25] + [0.0] * 8, dtype=np.float64)
    return ProjectionCoverageEvidence(
        state_id=0,
        view_name="primary",
        status="valid",
        source_vertex_numerator=numerator.copy(),
        effective_vertex_numerator=numerator.copy(),
        source_denominator=1.0,
        effective_denominator=1.0,
        source_instance_vertex_numerator=numerator[None].copy(),
        effective_instance_vertex_numerator=numerator[None].copy(),
        source_instance_denominators=np.ones(1, dtype=np.float64),
        effective_instance_denominators=np.ones(1, dtype=np.float64),
    )


def _score_arrays(density: np.ndarray) -> SeedScoreArrays:
    vertices, faces = _mesh()
    num_vertices = len(vertices)
    zeros = np.zeros(num_vertices, dtype=np.float64)
    return SeedScoreArrays(
        state_ids=np.asarray([0], dtype=np.int64),
        vertices=vertices,
        faces=faces,
        alpha=0.05,
        total_surface_area=9.0,
        tau=0.1,
        state_normalization_scales=np.ones(1, dtype=np.float64),
        state_abs_component_p99=np.ones(1, dtype=np.float64),
        state_max_to_p99_ratio=np.ones(1, dtype=np.float64),
        normalized_gradient_norms=np.zeros((1, num_vertices), dtype=np.float64),
        mean_normalized_gradient=np.zeros((num_vertices, 3), dtype=np.float64),
        mean_sensitivity=zeros.copy(),
        direction_consistency=zeros.copy(),
        raw_score=zeros.copy(),
        vertex_mass=np.ones(num_vertices, dtype=np.float64),
        raw_density=density.copy(),
        smoothed_density=density.copy(),
        ranked_vertex_ids=np.argsort(-density).astype(np.int64),
        local_peak_mask=np.zeros(num_vertices, dtype=np.bool_),
        ranked_local_peak_vertex_ids=np.asarray([], dtype=np.int64),
        linear_system_relative_residual=0.0,
        mass_conservation_relative_error=0.0,
    )


def _provenance() -> SupportAuditProvenance:
    return SupportAuditProvenance(
        code_commit="1" * 40,
        seed_score_artifact_sha256="2" * 64,
        visibility_manifest_sha256="3" * 64,
        visibility_metrics_sha256="4" * 64,
        visibility_npz_relative_paths=("arrays/state_00_primary.npz",),
        visibility_npz_sha256=("5" * 64,),
    )


def _write_bundle(
    root: Path,
    *,
    density: np.ndarray,
) -> tuple[Path, SupportArtifactEvidence, SeedScoreArrays]:
    score = _score_arrays(density)
    coverage = (_coverage(),)
    result = construct_akita_support_candidates(
        vertices=score.vertices,
        faces=score.faces,
        vertex_mass=score.vertex_mass,
        smoothed_density=score.smoothed_density,
        primary_evidence=coverage,
    )
    root.mkdir()
    coverage_path = root / "support_coverage_contributions.npz"
    coverage_sha = write_coverage_contribution_artifact(
        coverage_path,
        rows=coverage,
        provenance=_provenance(),
    )
    candidate_path = root / "support_candidates.npz"
    candidate_sha = write_candidate_artifact(
        candidate_path,
        result=result,
        provenance=_provenance(),
        coverage_artifact_sha256=coverage_sha,
    )
    evidence = SupportArtifactEvidence(
        schema_version=SUPPORT_AUDIT_SCHEMA_VERSION,
        provenance=_provenance(),
        coverage_artifact_relative_path=coverage_path.name,
        coverage_artifact_sha256=coverage_sha,
        candidate_artifact_relative_path=candidate_path.name,
        candidate_artifact_sha256=candidate_sha,
        production_support_constructed=False,
        fixed_support_frozen=False,
    )
    return candidate_path, evidence, score


def test_artifacts_round_trip_and_independently_recompute(tmp_path: Path) -> None:
    candidate_path, evidence, score = _write_bundle(
        tmp_path / "first",
        density=np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
    )
    rows, provenance = load_coverage_contribution_artifact(
        candidate_path.parent / evidence.coverage_artifact_relative_path
    )
    assert rows[0].state_id == 0
    assert provenance == _provenance()
    decision = evaluate_support_artifacts(
        evidence,
        artifact_root=candidate_path.parent,
        score_arrays=score,
    )
    assert decision.gate_pass, decision.failures


def test_candidate_tampering_is_detected_by_hash(tmp_path: Path) -> None:
    candidate_path, evidence, score = _write_bundle(
        tmp_path / "tampered",
        density=np.arange(9, 0, -1, dtype=np.float64),
    )
    with candidate_path.open("ab") as handle:
        handle.write(b"tamper")
    assert file_sha256(candidate_path) != evidence.candidate_artifact_sha256
    decision = evaluate_support_artifacts(
        evidence,
        artifact_root=candidate_path.parent,
        score_arrays=score,
    )
    assert not decision.gate_pass
    assert "candidate artifact SHA-256" in decision.failures[0]


def test_repeat_report_is_read_only_and_has_no_threshold(tmp_path: Path) -> None:
    first_path, _, _ = _write_bundle(
        tmp_path / "repeat_first",
        density=np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
    )
    second_path, _, _ = _write_bundle(
        tmp_path / "repeat_second",
        density=np.asarray([9.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
    )
    report = compare_support_repeat_artifacts(first_path, second_path)
    assert report.compatible
    assert report.selected_region_count_equal
    assert report.selected_seed_vertex_ids_equal
    assert report.selected_support_jaccard is not None
    assert not report.threshold_registered
