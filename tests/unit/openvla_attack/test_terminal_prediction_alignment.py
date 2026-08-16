"""Gate 6g--6j 终止诊断 P/I 只读分析的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_prediction_alignment import (  # noqa: E402
    PredictionAlignmentCase,
    analyze_prediction_alignment_cases,
)
from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    array_sha256,
)


def _case(
    *,
    endpoint: str,
    state_id: int,
    arm: str,
    gradient_x: float,
    step_x: float,
    baseline_loss: float,
    arm_loss: float,
    arm_tokens: tuple[int, ...] = (10, 11),
) -> PredictionAlignmentCase:
    endpoint_digit = "1" if endpoint == "action_spectral" else "2"
    arm_digit = "3" if arm == "radial_support" else "4"
    step = np.asarray([[step_x, 0.0, 0.0]], dtype=np.float32)
    return PredictionAlignmentCase(
        endpoint=endpoint,
        state_id=state_id,
        arm=arm,
        state_fingerprint=f"{state_id + 1:064x}",
        gradient_artifact_sha256=(endpoint_digit * 63) + str(state_id),
        baseline_response_artifact_sha256=("5" * 63) + str(state_id),
        arm_response_artifact_sha256=(arm_digit * 63) + str(state_id),
        step_artifact_sha256=arm_digit * 64,
        step_array_sha256=array_sha256(step),
        dense_surface_gradient=np.asarray(
            [[gradient_x, 0.0, 0.0]], dtype=np.float32
        ),
        executed_step=step,
        baseline_action_loss=baseline_loss,
        arm_action_loss=arm_loss,
        baseline_generated_token_ids=np.asarray([10, 11], dtype=np.int64),
        arm_generated_token_ids=np.asarray(arm_tokens, dtype=np.int64),
    )


def _complete_cases() -> list[PredictionAlignmentCase]:
    cases: list[PredictionAlignmentCase] = []
    for endpoint in ("action_spectral", "action_only_control"):
        endpoint_scale = 1.0 if endpoint == "action_spectral" else 2.0
        for state_id, gradient_x in ((0, 1.0), (1, -1.0)):
            cases.append(
                _case(
                    endpoint=endpoint,
                    state_id=state_id,
                    arm="radial_support",
                    gradient_x=gradient_x * endpoint_scale,
                    step_x=-1.0,
                    baseline_loss=5.0,
                    arm_loss=4.0,
                    arm_tokens=(10, 12) if state_id == 0 else (10, 11),
                )
            )
            cases.append(
                _case(
                    endpoint=endpoint,
                    state_id=state_id,
                    arm="matched_box_support",
                    gradient_x=gradient_x * endpoint_scale,
                    step_x=-0.5,
                    baseline_loss=5.0,
                    arm_loss=6.0 if state_id == 0 else 5.0,
                    arm_tokens=(9, 12) if state_id == 0 else (10, 11),
                )
            )
    return cases


def test_alignment_uses_each_state_gradient_and_reports_exact_sign_table() -> None:
    """P必须使用g[e,s]；四组统计不能先跨endpoint或arm混合。"""

    decision = analyze_prediction_alignment_cases(
        _complete_cases(), expected_state_ids=(0, 1)
    )

    assert decision.audit_valid, decision.failures
    assert len(decision.rows) == 8
    spectral_radial_rows = tuple(
        row
        for row in decision.rows
        if row.endpoint == "action_spectral"
        and row.arm == "radial_support"
    )
    assert tuple(row.predicted_decrease for row in spectral_radial_rows) == (
        1.0,
        -1.0,
    )
    assert tuple(row.exact_improvement for row in spectral_radial_rows) == (
        1.0,
        1.0,
    )
    assert spectral_radial_rows[0].generation_token_change_count == 1
    assert spectral_radial_rows[0].first_generation_token_change_index == 1
    assert spectral_radial_rows[1].generation_token_change_count == 0
    assert spectral_radial_rows[1].first_generation_token_change_index is None

    summary = next(
        row
        for row in decision.summaries
        if row.endpoint == "action_spectral"
        and row.arm == "radial_support"
    )
    counts = {
        (cell.prediction_sign, cell.improvement_sign): cell.count
        for cell in summary.sign_contingency
    }
    assert counts[("positive", "positive")] == 1
    assert counts[("negative", "positive")] == 1
    assert sum(counts.values()) == 2
    assert summary.predicted_decrease_mean == 0.0
    assert summary.predicted_decrease_median == 0.0
    assert summary.exact_improvement_mean == 1.0
    assert not hasattr(summary, "bpda_bad")
    assert not hasattr(decision, "common_bottleneck")


def test_alignment_rejects_incomplete_inventory_and_cross_state_drift() -> None:
    """缺行、重复键或同一endpoint/arm的step漂移都必须使分析无效。"""

    incomplete = analyze_prediction_alignment_cases(
        _complete_cases()[:-1], expected_state_ids=(0, 1)
    )
    assert not incomplete.audit_valid
    assert any("inventory" in failure for failure in incomplete.failures)

    drifted = _complete_cases()
    original = drifted[-1]
    drifted[-1] = PredictionAlignmentCase(
        **{
            **original.__dict__,
            "step_array_sha256": "8" * 64,
        }
    )
    invalid = analyze_prediction_alignment_cases(
        drifted, expected_state_ids=(0, 1)
    )
    assert not invalid.audit_valid
    assert any("step" in failure for failure in invalid.failures)
