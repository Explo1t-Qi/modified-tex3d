"""Action-only κ feasibility/margin-drift只读分析测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    array_sha256,
)
from openvla_attack.terminal_margin_calibration import (  # noqa: E402
    MarginDriftCase,
    analyze_margin_drift_cases,
)


def _logits(
    clean_classes: tuple[int, ...], margins: tuple[float, ...]
) -> np.ndarray:
    logits = np.zeros((len(clean_classes), 3), dtype=np.float32)
    for index, (clean_class, margin) in enumerate(
        zip(clean_classes, margins, strict=True)
    ):
        logits[index, clean_class] = np.float32(margin)
    return logits


def _case(
    *,
    state_id: int,
    training_margins: tuple[float, ...],
    deployment_margins: tuple[float, ...],
) -> MarginDriftCase:
    clean_classes = np.asarray([0, 1], dtype=np.int64)
    teacher_ids = np.asarray([7, 8, 9, 10], dtype=np.int64)
    training_rgb = np.full((2, 2, 3), state_id, dtype=np.uint8)
    deployment_rgb = training_rgb.copy()
    deployment_rgb[0, 0, 0] += np.uint8(1)
    return MarginDriftCase(
        state_id=state_id,
        state_fingerprint=f"{state_id + 1:064x}",
        response_artifact_sha256=f"{state_id + 3:064x}",
        clean_classes=clean_classes,
        teacher_input_ids_sha256=array_sha256(teacher_ids),
        training_effective_rgb_sha256=array_sha256(training_rgb),
        deployment_effective_rgb_sha256=array_sha256(deployment_rgb),
        training_teacher_logits=_logits(
            tuple(int(value) for value in clean_classes), training_margins
        ),
        deployment_teacher_logits=_logits(
            tuple(int(value) for value in clean_classes), deployment_margins
        ),
    )


def test_margin_drift_reports_all_tokens_without_recommending_kappa() -> None:
    """crossed、rebound和reversal必须按精确零边界从raw logits复算。"""

    cases = (
        _case(
            state_id=0,
            training_margins=(-0.25, 0.5),
            deployment_margins=(0.25, 0.25),
        ),
        _case(
            state_id=1,
            training_margins=(0.0, -1.0),
            deployment_margins=(0.25, -0.5),
        ),
    )

    decision = analyze_margin_drift_cases(
        cases,
        expected_state_ids=(0, 1),
        expected_action_dim=2,
    )

    assert decision.audit_valid, decision.failures
    assert len(decision.rows) == 4
    assert decision.counts.token_count == 4
    assert decision.counts.uncrossed_token_count == 1
    assert decision.counts.crossed_token_count == 3
    assert decision.counts.positive_drift_token_count == 3
    assert decision.counts.crossed_positive_drift_token_count == 3
    assert decision.counts.deployment_reversal_token_count == 2
    first = decision.rows[0]
    assert first.training_margin == -0.25
    assert first.deployment_margin == 0.25
    assert first.margin_drift == 0.5
    assert first.crossed_in_training
    assert first.positive_deployment_drift
    assert first.deployment_reversal
    assert not hasattr(decision, "recommended_kappa")
    assert not hasattr(decision, "feasibility_pass")

    summaries = {summary.name: summary for summary in decision.distributions}
    assert summaries["training_margin_all"].count == 4
    assert summaries["training_margin_crossed"].count == 3
    assert summaries["positive_drift_crossed"].count == 3
    assert summaries["positive_drift_crossed"].minimum == 0.25
    assert summaries["positive_drift_crossed"].maximum == 0.5


def test_margin_drift_rejects_missing_state_and_duplicate_identity() -> None:
    """正式分析必须保持states完整唯一，不能静默丢token或state。"""

    case = _case(
        state_id=0,
        training_margins=(-0.25, 0.5),
        deployment_margins=(0.25, 0.25),
    )
    missing = analyze_margin_drift_cases(
        (case,),
        expected_state_ids=(0, 1),
        expected_action_dim=2,
    )
    assert not missing.audit_valid
    assert any("inventory" in failure for failure in missing.failures)

    duplicate = analyze_margin_drift_cases(
        (case, case),
        expected_state_ids=(0, 1),
        expected_action_dim=2,
    )
    assert not duplicate.audit_valid
    assert any("重复" in failure for failure in duplicate.failures)
