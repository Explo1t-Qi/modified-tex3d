"""零 Surface Delta Compositor Gate 的纯 CPU 判定测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.compositor_zero_delta_audit import (  # noqa: E402
    COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION,
    ZeroDeltaCompositorEvidence,
    evaluate_zero_delta_compositor_evidence,
    summarize_zero_delta_compositor_rows,
    write_zero_delta_compositor_jsonl,
    write_zero_delta_compositor_manifest,
)
from openvla_attack.deployment_backward_audit import (  # noqa: E402
    GradientEvidence,
)


def _passing_evidence(state_id: int = 0) -> ZeroDeltaCompositorEvidence:
    source = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pre_crop = source[::2, ::2].copy()
    effective = pre_crop.copy()
    pixels = np.arange(24, dtype=np.float32).reshape(1, 6, 2, 2)
    tokens = np.arange(7, dtype=np.int64)
    action = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    return ZeroDeltaCompositorEvidence(
        state_id=state_id,
        clean_source_rgb=source,
        composited_source_rgb=source.copy(),
        clean_pre_crop_rgb=pre_crop,
        composited_pre_crop_rgb=pre_crop.copy(),
        clean_effective_rgb=effective,
        composited_effective_rgb=effective.copy(),
        clean_pixel_values=pixels,
        composited_pixel_values=pixels.copy(),
        clean_action_token_ids=tokens,
        composited_action_token_ids=tokens.copy(),
        clean_action=action,
        composited_action=action.copy(),
        renderer_delta_linf=0.0,
        per_instance_delta_linf=(0.0, 0.0),
        total_delta_linf=0.0,
        saturated_pixel_fraction=0.0,
        saturated_channel_fraction=0.0,
        clean_renderer_requires_grad=False,
        valid_renderer_pixel_count=5,
        instance_visible_pixel_counts=(3, 2),
        surface_parameter_gradient=GradientEvidence.from_tensor(
            torch.ones((8, 3), dtype=torch.float32)
        ),
        transaction_verified=True,
        transaction_before_sha256="f" * 64,
        transaction_after_sha256="f" * 64,
        arrays_npz_sha256="a" * 64,
        artifact_sha256={"arrays/state_00.npz": "a" * 64},
    )


def test_zero_delta_gate_accepts_exact_identity_and_nonzero_adv_gradient() -> None:
    row = evaluate_zero_delta_compositor_evidence(
        _passing_evidence(),
        code_commit="a" * 40,
    )

    assert row["gate_pass"]
    assert row["failures"] == []
    assert row["source_rgb_linf"] == 0.0
    assert row["processor_pixel_linf"] == 0.0
    assert row["action_token_hamming"] == 0


def test_zero_delta_gate_rejects_stage_difference_and_clean_graph() -> None:
    evidence = _passing_evidence()
    changed_effective = evidence.composited_effective_rgb.copy()
    changed_effective[0, 0, 0] += 1
    evidence = replace(
        evidence,
        composited_effective_rgb=changed_effective,
        clean_renderer_requires_grad=True,
    )

    row = evaluate_zero_delta_compositor_evidence(
        evidence,
        code_commit="b" * 40,
    )

    assert not row["gate_pass"]
    assert row["effective_view_linf"] == 1.0
    assert "Effective View 不是逐值恒等" in row["failures"]
    assert "clean renderer 分支仍连接 autograd" in row["failures"]


def test_zero_delta_gate_rejects_zero_adv_gradient_and_bad_transaction() -> None:
    evidence = replace(
        _passing_evidence(),
        surface_parameter_gradient=GradientEvidence.from_tensor(
            torch.zeros((8, 3), dtype=torch.float32)
        ),
        transaction_after_sha256="e" * 64,
    )

    row = evaluate_zero_delta_compositor_evidence(
        evidence,
        code_commit="c" * 40,
    )

    assert not row["gate_pass"]
    assert "adversarial renderer 对 Surface 参数的梯度不是有限非零" in row[
        "failures"
    ]
    assert "static transaction 前后 fingerprint 不一致" in row["failures"]


def test_zero_delta_summary_requires_exact_state_set_and_unique_rows() -> None:
    rows = [
        evaluate_zero_delta_compositor_evidence(
            _passing_evidence(state_id),
            code_commit="d" * 40,
        )
        for state_id in (0, 1)
    ]
    passing = summarize_zero_delta_compositor_rows(
        rows,
        expected_state_ids=(0, 1),
    )
    assert passing["structural_pass"]
    assert passing["gate_pass"]

    duplicate = summarize_zero_delta_compositor_rows(
        [rows[0], rows[0]],
        expected_state_ids=(0, 1),
    )
    assert not duplicate["structural_pass"]
    assert duplicate["duplicate_state_ids"] == [0]
    assert duplicate["missing_state_ids"] == [1]


def test_zero_delta_writers_bind_jsonl_hash_in_manifest(tmp_path: Path) -> None:
    row = evaluate_zero_delta_compositor_evidence(
        _passing_evidence(),
        code_commit="e" * 40,
    )
    summary = summarize_zero_delta_compositor_rows(
        [row],
        expected_state_ids=(0,),
    )
    metrics_path = tmp_path / "compositor_zero_delta_metrics.jsonl"
    metrics_sha256 = write_zero_delta_compositor_jsonl(
        [row],
        output_path=metrics_path,
    )
    manifest_path = tmp_path / "compositor_zero_delta_manifest.json"
    manifest_sha256 = write_zero_delta_compositor_manifest(
        summary=summary,
        metadata={"code_commit": "e" * 40},
        metrics_jsonl_sha256=metrics_sha256,
        output_path=manifest_path,
    )

    assert len(metrics_sha256) == 64
    assert len(manifest_sha256) == 64
    assert metrics_path.read_text(encoding="utf-8").endswith("\n")
    assert f'"metrics_jsonl_sha256": "{metrics_sha256}"' in (
        manifest_path.read_text(encoding="utf-8")
    )
    assert COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION in (
        manifest_path.read_text(encoding="utf-8")
    )
