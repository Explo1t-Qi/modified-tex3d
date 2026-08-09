"""Seed Score、单位面积density与mass-aware smoothing纯CPU契约测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from openvla.experiments.robot.libero.openvla_attack.action_objective_audit import (
    ACTION_OBJECTIVE_NAME,
    ACTION_OBJECTIVE_SCHEMA_VERSION,
    ActionObjectiveAuditEvidence,
)
from openvla.experiments.robot.libero.openvla_attack.dense_seed_audit import (
    DENSE_SEED_AUDIT_SCHEMA_VERSION,
    DenseSeedCaptureMetadata,
    DenseSeedGradientEvidence,
    array_sha256,
    write_dense_seed_gradient_evidence,
    write_dense_seed_state_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.diagnose_seed_score import (
    SeedScoreAuditConfig,
    run_seed_score_audit,
)
from openvla.experiments.robot.libero.openvla_attack.deployment_backward_audit import (
    GradientEvidence,
)
from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    AKITA_SMOOTHING_ALPHA,
    SCORE_ARTIFACT_KEYS,
    build_seed_score_provenance,
    compare_seed_score_stability,
    compute_seed_score_arrays,
    evaluate_seed_score_artifact,
    evidence_to_json,
    file_sha256,
    load_dense_gradient_stack,
    seed_score_evidence_from_mapping,
    write_seed_score_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    mesh_array_sha256,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    return vertices, faces


def _write_tetrahedron_obj(path: Path) -> Path:
    vertices, faces = _tetrahedron()
    lines = [
        f"v {vertex[0]} {vertex[1]} {vertex[2]}" for vertex in vertices
    ]
    lines.extend(
        "f " + " ".join(str(int(vertex_id) + 1) for vertex_id in face)
        for face in faces
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _gradient_evidence(*shape: int) -> GradientEvidence:
    return GradientEvidence.from_tensor(torch.ones(shape))


def _raw_gradient(state_id: int) -> np.ndarray:
    base = np.asarray(
        [
            [2.0, 0.2, 0.1],
            [0.4, 1.0, 0.2],
            [0.1, 0.3, 0.8],
            [0.5, 0.4, 0.7],
        ],
        dtype=np.float32,
    )
    scale = np.float32(1.0 + state_id / 10.0)
    state_variation = np.float32((state_id % 3) * 0.01)
    return np.ascontiguousarray(base * scale + state_variation)


def _objective(
    state_id: int,
    gradient: np.ndarray,
) -> ActionObjectiveAuditEvidence:
    return ActionObjectiveAuditEvidence(
        schema_version=ACTION_OBJECTIVE_SCHEMA_VERSION,
        code_commit="a" * 40,
        state_id=state_id,
        initial_state_sha256=f"{state_id:x}" * 64,
        objective=ACTION_OBJECTIVE_NAME,
        parameterization="geometry_vertex_dense",
        zero_surface_delta_linf=0.0,
        teacher_forced_clean_prefix=True,
        clean_action_token_ids=[31744, 31745, 31999],
        clean_classes=[0, 1, 255],
        margins=[2.0, 1.0, 0.5],
        hinge_values=[2.0, 1.0, 0.5],
        action_loss=3.5 / 3.0,
        parameter_shape=(4, 3),
        source_rgb_gradient=_gradient_evidence(1, 3, 8, 8),
        pre_crop_gradient=_gradient_evidence(1, 3, 4, 4),
        effective_view_gradient=_gradient_evidence(1, 3, 4, 4),
        render_surface_delta_gradient=_gradient_evidence(6, 3),
        dense_geometry_gradient=GradientEvidence.from_tensor(
            torch.from_numpy(gradient)
        ),
        dense_geometry_gradient_sha256=array_sha256(gradient),
    )


def _metadata(mesh_file_sha256: str) -> DenseSeedCaptureMetadata:
    return DenseSeedCaptureMetadata(
        mesh_sha256=mesh_file_sha256,
        render_to_geometry_sha256="c" * 64,
        policy_source_rgb_sha256="d" * 64,
        effective_view_rgb_sha256="e" * 64,
        mujoco_instance_alpha_sha256="f" * 64,
        renderer_visibility_sha256="1" * 64,
        shared_instance_body_ids=(17, 23),
        shared_instance_body_names=("bowl_main", "bowl_secondary"),
    )


def _dense_bundle(
    root: Path,
    mesh_file_sha256: str,
) -> tuple[Path, list[DenseSeedGradientEvidence]]:
    rows = []
    for state_id in range(10):
        gradient = _raw_gradient(state_id)
        rows.append(
            write_dense_seed_state_artifact(
                output_root=root,
                objective_evidence=_objective(state_id, gradient),
                dense_geometry_gradient=gradient,
                metadata=_metadata(mesh_file_sha256),
            )
        )
    metrics = write_dense_seed_gradient_evidence(
        root / "dense_seed_metrics.jsonl",
        rows,
        artifact_root=root,
    )
    return metrics, rows


def test_score_uses_global_state_linf_and_rgb_direction_consistency() -> None:
    vertices, faces = _tetrahedron()
    gradients = np.asarray(
        [
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    result = compute_seed_score_arrays(
        raw_gradients=gradients,
        state_ids=(0, 1),
        vertices=vertices,
        faces=faces,
    )

    np.testing.assert_allclose(result.state_normalization_scales, [2.0, 4.0])
    np.testing.assert_allclose(result.mean_sensitivity[:3], [1.0, 0.5, 0.5])
    np.testing.assert_allclose(result.direction_consistency[:3], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(result.raw_score[:3], [1.0, 0.0, 0.5])


def test_density_and_implicit_smoothing_preserve_frozen_equation() -> None:
    vertices, faces = _tetrahedron()
    gradients = np.stack([_raw_gradient(i) for i in range(10)])

    result = compute_seed_score_arrays(
        raw_gradients=gradients,
        state_ids=range(10),
        vertices=vertices,
        faces=faces,
    )

    np.testing.assert_allclose(
        result.raw_density,
        result.raw_score / result.vertex_mass,
    )
    assert result.alpha == AKITA_SMOOTHING_ALPHA
    assert result.tau == result.alpha**2 * result.total_surface_area
    assert result.linear_system_relative_residual < 1e-12
    assert result.mass_conservation_relative_error < 1e-12
    assert np.all(np.isfinite(result.smoothed_density))
    assert result.smoothed_density.min() >= -1e-12


def test_score_artifact_round_trip_recomputes_from_dense_payloads(
    tmp_path: Path,
) -> None:
    dense_root = tmp_path / "dense"
    score_root = tmp_path / "score"
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("synthetic mesh identity", encoding="utf-8")
    mesh_file_hash = file_sha256(mesh_path)
    metrics, rows = _dense_bundle(dense_root, mesh_file_hash)
    vertices, faces = _tetrahedron()
    gradients = load_dense_gradient_stack(rows, artifact_root=dense_root)
    arrays = compute_seed_score_arrays(
        raw_gradients=gradients,
        state_ids=range(10),
        vertices=vertices,
        faces=faces,
    )
    provenance = build_seed_score_provenance(
        code_commit="b" * 40,
        dense_metrics_path=metrics,
        dense_rows=rows,
        mesh_file_sha256=mesh_file_hash,
        vertices=vertices,
        faces=faces,
    )
    evidence = write_seed_score_artifact(
        output_root=score_root,
        provenance=provenance,
        arrays=arrays,
    )
    assert seed_score_evidence_from_mapping(evidence_to_json(evidence)) == evidence

    decision = evaluate_seed_score_artifact(
        evidence,
        artifact_root=score_root,
        dense_artifact_root=dense_root,
        dense_rows=rows,
    )

    assert decision.gate_pass
    assert decision.failures == ()
    with np.load(score_root / evidence.artifact_relative_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(SCORE_ARTIFACT_KEYS)
        assert archive["normalized_gradient_norms"].shape == (10, 4)
        assert archive["mean_normalized_gradient"].shape == (4, 3)
        assert not bool(archive["production_support_constructed"].item())
        assert "selected_vertex_ids" not in archive.files
        assert "coverage" not in archive.files


def test_score_artifact_detects_dense_or_score_tampering(tmp_path: Path) -> None:
    dense_root = tmp_path / "dense"
    score_root = tmp_path / "score"
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("synthetic mesh identity", encoding="utf-8")
    mesh_file_hash = file_sha256(mesh_path)
    metrics, rows = _dense_bundle(dense_root, mesh_file_hash)
    vertices, faces = _tetrahedron()
    arrays = compute_seed_score_arrays(
        raw_gradients=load_dense_gradient_stack(rows, artifact_root=dense_root),
        state_ids=range(10),
        vertices=vertices,
        faces=faces,
    )
    provenance = build_seed_score_provenance(
        code_commit="b" * 40,
        dense_metrics_path=metrics,
        dense_rows=rows,
        mesh_file_sha256=mesh_file_hash,
        vertices=vertices,
        faces=faces,
    )
    evidence = write_seed_score_artifact(
        output_root=score_root,
        provenance=provenance,
        arrays=arrays,
    )

    wrong = replace(evidence, artifact_sha256="0" * 64)
    decision = evaluate_seed_score_artifact(
        wrong,
        artifact_root=score_root,
        dense_artifact_root=dense_root,
        dense_rows=rows,
    )
    assert not decision.gate_pass
    assert "Seed Score artifact SHA-256不匹配" in decision.failures

    changed_rows = list(rows)
    changed_rows[0] = replace(changed_rows[0], artifact_sha256="2" * 64)
    dense_decision = evaluate_seed_score_artifact(
        evidence,
        artifact_root=score_root,
        dense_artifact_root=dense_root,
        dense_rows=changed_rows,
    )
    assert not dense_decision.gate_pass
    assert "Dense artifact hashes与score provenance不一致" in dense_decision.failures


def test_repeat_stability_reports_regions_without_registering_threshold() -> None:
    vertices, faces = _tetrahedron()
    first_gradients = np.stack([_raw_gradient(i) for i in range(10)])
    second_gradients = first_gradients.copy()
    second_gradients[:, 2, 1] += np.float32(1e-4)
    first = compute_seed_score_arrays(
        raw_gradients=first_gradients,
        state_ids=range(10),
        vertices=vertices,
        faces=faces,
    )
    second = compute_seed_score_arrays(
        raw_gradients=second_gradients,
        state_ids=range(10),
        vertices=vertices,
        faces=faces,
    )

    report = compare_seed_score_stability(first, second)

    assert report.compatible
    assert report.threshold_registered is False
    assert report.smoothed_density_cosine > 0.999
    assert report.smoothed_density_spearman > 0.99
    assert tuple(fraction for fraction, _ in report.top_fraction_jaccard) == (
        0.001,
        0.005,
        0.01,
        0.05,
    )
    assert mesh_array_sha256(first.vertices, first.faces) == mesh_array_sha256(
        second.vertices,
        second.faces,
    )


def test_cpu_runner_validates_dense_manifest_and_writes_no_support(
    tmp_path: Path,
) -> None:
    dense_root = tmp_path / "dense"
    score_root = tmp_path / "score"
    mesh_path = _write_tetrahedron_obj(tmp_path / "mesh.obj")
    metrics, rows = _dense_bundle(dense_root, file_sha256(mesh_path))
    dense_manifest = {
        "schema_version": DENSE_SEED_AUDIT_SCHEMA_VERSION,
        "code_commit": "a" * 40,
        "summary": {"gate_pass": True},
        "metrics_sha256": file_sha256(metrics),
        "artifact_sha256": {
            row.artifact_relative_path: row.artifact_sha256 for row in rows
        },
        "production_support_constructed": False,
    }
    (dense_root / "dense_seed_manifest.json").write_text(
        json.dumps(dense_manifest),
        encoding="utf-8",
    )

    manifest_path = run_seed_score_audit(
        SeedScoreAuditConfig(
            dense_audit_dir=str(dense_root),
            mesh_path=str(mesh_path),
            output_dir=str(score_root),
            code_commit="b" * 40,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["decision"]["gate_pass"] is True
    assert manifest["production_support_constructed"] is False
    assert manifest["stability_threshold_registered"] is False
    assert manifest["feature_gradient_used"] is False
    assert manifest["oft_used"] is False
