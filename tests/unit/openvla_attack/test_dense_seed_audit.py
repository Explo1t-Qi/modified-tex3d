"""Dense Seed Audit 完整梯度 artifact/evidence 的纯 CPU 测试。"""

from __future__ import annotations

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
    DENSE_SEED_ARTIFACT_KEYS,
    DENSE_SEED_AUDIT_SCHEMA_VERSION,
    DenseSeedCaptureMetadata,
    DenseSeedGradientEvidence,
    array_sha256,
    evaluate_dense_seed_gradient_evidence,
    load_dense_seed_gradient_evidence,
    summarize_dense_seed_gradient_evidence,
    write_dense_seed_gradient_evidence,
    write_dense_seed_state_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.deployment_backward_audit import (
    GradientEvidence,
)


def _gradient(*shape: int) -> GradientEvidence:
    return GradientEvidence.from_tensor(torch.ones(shape))


def _raw_gradient(state_id: int) -> np.ndarray:
    return (
        np.arange(12, dtype=np.float32).reshape(4, 3) + state_id + 1.0
    )


def _objective_evidence(
    state_id: int,
    raw_gradient: np.ndarray,
) -> ActionObjectiveAuditEvidence:
    dense_stats = GradientEvidence.from_tensor(torch.from_numpy(raw_gradient))
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
        margins=[2.0, 0.0, 1.0],
        hinge_values=[2.0, 0.0, 1.0],
        action_loss=1.0,
        parameter_shape=(4, 3),
        source_rgb_gradient=_gradient(1, 3, 8, 8),
        pre_crop_gradient=_gradient(1, 3, 4, 4),
        effective_view_gradient=_gradient(1, 3, 4, 4),
        render_surface_delta_gradient=_gradient(6, 3),
        dense_geometry_gradient=dense_stats,
        dense_geometry_gradient_sha256=array_sha256(raw_gradient),
    )


def _metadata() -> DenseSeedCaptureMetadata:
    return DenseSeedCaptureMetadata(
        mesh_sha256="b" * 64,
        render_to_geometry_sha256="c" * 64,
        policy_source_rgb_sha256="d" * 64,
        effective_view_rgb_sha256="e" * 64,
        mujoco_instance_alpha_sha256="f" * 64,
        renderer_visibility_sha256="1" * 64,
        shared_instance_body_ids=(17, 23),
        shared_instance_body_names=("bowl_main", "bowl_secondary"),
    )


def _write_state(
    root: Path,
    state_id: int,
) -> DenseSeedGradientEvidence:
    gradient = _raw_gradient(state_id)
    return write_dense_seed_state_artifact(
        output_root=root,
        objective_evidence=_objective_evidence(state_id, gradient),
        dense_geometry_gradient=gradient,
        metadata=_metadata(),
    )


def test_dense_seed_artifact_preserves_complete_raw_gradient(tmp_path: Path) -> None:
    evidence = _write_state(tmp_path, 0)

    decision = evaluate_dense_seed_gradient_evidence(
        evidence,
        artifact_root=tmp_path,
    )

    assert decision.gate_pass
    assert decision.failures == ()
    artifact_path = tmp_path / evidence.artifact_relative_path
    with np.load(artifact_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(DENSE_SEED_ARTIFACT_KEYS)
        np.testing.assert_array_equal(
            archive["dense_geometry_gradient"],
            _raw_gradient(0),
        )
        assert archive["dense_geometry_gradient"].dtype == np.float32
        # Artifact contract 中不允许提前出现聚合分数、density 或 Support。
        assert not any(
            forbidden in key
            for key in archive.files
            for forbidden in ("score", "density", "support", "coverage")
        )


def test_dense_seed_evidence_detects_artifact_tampering(tmp_path: Path) -> None:
    evidence = _write_state(tmp_path, 0)
    artifact_path = tmp_path / evidence.artifact_relative_path
    payload = bytearray(artifact_path.read_bytes())
    payload[-1] ^= 1
    artifact_path.write_bytes(payload)

    decision = evaluate_dense_seed_gradient_evidence(
        evidence,
        artifact_root=tmp_path,
    )

    assert not decision.gate_pass
    assert "artifact SHA-256 不匹配" in decision.failures


def test_dense_seed_evidence_rejects_wrong_gradient_binding(
    tmp_path: Path,
) -> None:
    evidence = _write_state(tmp_path, 0)
    wrong = replace(evidence, dense_geometry_gradient_sha256="0" * 64)

    decision = evaluate_dense_seed_gradient_evidence(
        wrong,
        artifact_root=tmp_path,
    )

    assert not decision.gate_pass
    assert "raw G_s SHA-256 与 objective evidence 不一致" in decision.failures


def test_dense_seed_summary_requires_exact_states_and_common_geometry(
    tmp_path: Path,
) -> None:
    rows = [_write_state(tmp_path, state_id) for state_id in range(10)]

    summary = summarize_dense_seed_gradient_evidence(
        rows,
        artifact_root=tmp_path,
    )

    assert summary.gate_pass
    assert summary.state_ids == tuple(range(10))
    assert summary.valid_state_count == 10
    assert summary.num_geometry_vertices == 4
    assert summary.artifact_count == 10
    assert summary.total_gradient_value_count == 120
    assert summary.production_support_constructed is False

    inconsistent = list(rows)
    inconsistent[-1] = replace(
        inconsistent[-1],
        metadata=replace(_metadata(), mesh_sha256="2" * 64),
    )
    failed = summarize_dense_seed_gradient_evidence(
        inconsistent,
        artifact_root=tmp_path,
    )
    assert not failed.gate_pass
    assert "所有 states 必须绑定同一 OBJ mesh" in failed.failures


def test_dense_seed_jsonl_round_trip_and_revalidates_npz(tmp_path: Path) -> None:
    rows = [_write_state(tmp_path, state_id) for state_id in range(10)]
    evidence_path = write_dense_seed_gradient_evidence(
        tmp_path / "dense_seed_metrics.jsonl",
        rows,
        artifact_root=tmp_path,
    )

    loaded = load_dense_seed_gradient_evidence(evidence_path)
    summary = summarize_dense_seed_gradient_evidence(
        loaded,
        artifact_root=tmp_path,
    )

    assert loaded == rows
    assert summary.gate_pass
