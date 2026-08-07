"""Gate 1D deployment forward 逐 state 判定与产物测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.deployment_forward_audit import (  # noqa: E402
    DEPLOYMENT_FORWARD_SCHEMA_VERSION,
    DeploymentForwardEvidence,
    evaluate_deployment_forward_evidence,
    summarize_deployment_forward_rows,
    write_deployment_forward_jsonl,
)


def _equal_evidence(state_id: int) -> DeploymentForwardEvidence:
    source = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pre_crop = source[:2, :2].copy()
    effective = np.flip(pre_crop, axis=0).copy()
    pixels = np.arange(24, dtype=np.float32).reshape(1, 6, 2, 2)
    tokens = np.arange(100, 107, dtype=np.int64)
    action = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    return DeploymentForwardEvidence(
        state_id=state_id,
        source_rgb=source,
        exact_pre_crop_rgb=pre_crop,
        candidate_pre_crop_rgb=pre_crop.copy(),
        rollout_effective_rgb=effective,
        candidate_effective_rgb=effective.copy(),
        rollout_pixel_values=pixels,
        candidate_pixel_values=pixels.copy(),
        rollout_action_token_ids=tokens,
        candidate_action_token_ids=tokens.copy(),
        rollout_action=action,
        candidate_action=action.copy(),
    )


def test_equal_deployment_forward_evidence_passes_all_hard_checks() -> None:
    row = evaluate_deployment_forward_evidence(
        _equal_evidence(3),
        code_commit="abc123",
    )

    assert row["schema_version"] == DEPLOYMENT_FORWARD_SCHEMA_VERSION
    assert row["state_id"] == 3
    assert row["gate_pass"] is True
    assert row["pre_crop_linf"] == 0.0
    assert row["effective_view_linf"] == 0.0
    assert row["processor_pixel_linf"] == 0.0
    assert row["action_token_hamming"] == 0
    assert row["action_linf"] == 0.0
    assert len(row["source_rgb_sha256"]) == 64


def test_single_action_token_difference_fails_gate() -> None:
    evidence = _equal_evidence(4)
    mismatched_tokens = evidence.candidate_action_token_ids.copy()
    mismatched_tokens[2] += 1
    evidence = DeploymentForwardEvidence(
        **{
            **evidence.__dict__,
            "candidate_action_token_ids": mismatched_tokens,
        }
    )

    row = evaluate_deployment_forward_evidence(
        evidence,
        code_commit="abc123",
    )

    assert row["gate_pass"] is False
    assert row["action_token_hamming"] == 1
    assert row["first_action_token_difference"] == 2


def test_summary_rejects_missing_or_duplicate_state_ids() -> None:
    first = evaluate_deployment_forward_evidence(
        _equal_evidence(0),
        code_commit="abc123",
    )
    second = evaluate_deployment_forward_evidence(
        _equal_evidence(1),
        code_commit="abc123",
    )

    summary = summarize_deployment_forward_rows(
        (first, second),
        expected_state_ids=(0, 1),
    )
    assert summary["gate_pass"] is True
    assert summary["passed_state_count"] == 2

    duplicate = summarize_deployment_forward_rows(
        (first, first),
        expected_state_ids=(0, 1),
    )
    assert duplicate["gate_pass"] is False
    assert duplicate["duplicate_state_ids"] == [0]
    assert duplicate["missing_state_ids"] == [1]


def test_jsonl_writer_preserves_state_order_and_returns_hash(
    tmp_path: Path,
) -> None:
    rows = tuple(
        evaluate_deployment_forward_evidence(
            _equal_evidence(state_id),
            code_commit="abc123",
        )
        for state_id in (7, 2)
    )
    output_path = tmp_path / "deployment_forward_metrics.jsonl"

    output_sha256 = write_deployment_forward_jsonl(
        rows,
        output_path=output_path,
    )

    saved_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["state_id"] for row in saved_rows] == [7, 2]
    assert len(output_sha256) == 64
