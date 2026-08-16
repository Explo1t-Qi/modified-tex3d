"""Gate 6j Radial-vs-Matched-Box 纯 CPU 合同行为测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    execute_surface_counterfactual_step,
)
from openvla_attack.terminal_projection_counterfactual import (  # noqa: E402
    TerminalProjectionCounterfactualError,
    build_matched_box_counterfactual,
    build_matched_box_endpoint_evidence,
    evaluate_projection_response_evidence,
    evaluate_matched_box_endpoint_evidence,
    load_matched_box_endpoint_npz,
    load_projection_response_npz,
    write_matched_box_endpoint_npz,
    write_projection_response_npz,
)
from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    build_endpoint_step_evidence,
)
from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    EndpointResponseEvidence,
    compute_action_hinge,
)


def test_matched_box_is_feasible_and_matches_parent_radial_linf() -> None:
    """Box只截断撞墙坐标，再以parent radial实际L∞匹配步长。"""

    endpoint = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gradient = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    parent = execute_surface_counterfactual_step(
        endpoint,
        gradient,
        epsilon=1.0,
        surface_step=0.2,
    )

    result = build_matched_box_counterfactual(parent, epsilon=1.0)

    assert np.allclose(
        result.box_surface_delta,
        np.asarray(
            [
                [-0.93333334, 0.0, 0.0],
                [0.7, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            np.float32,
        ),
    )
    assert 0.0 < result.stats.matched_scale <= 1.0
    assert np.isclose(
        result.stats.matched_actual_linf,
        parent.stats.actual_surface_step,
    )
    assert result.stats.matched_max_abs_delta <= 1.0
    assert result.stats.matched_predicted_decrease > (
        result.stats.radial_predicted_decrease
    )
    assert result.stats.box_l2 > 0.0
    assert result.stats.matched_l2 > 0.0
    assert result.stats.matched_descent_alignment_cosine > (
        result.stats.radial_descent_alignment_cosine
    )


def test_matched_box_endpoint_npz_round_trip_recomputes_parent_contract(
    tmp_path: Path,
) -> None:
    """Child artifact必须由parent step独立复算，不能信任缓存派生量。"""

    endpoint = np.asarray(
        [[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gradient = np.asarray(
        [[-1.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    support_mask = np.ones(3, dtype=np.bool_)
    parent_step = build_endpoint_step_evidence(
        endpoint="action_spectral",
        endpoint_surface_delta=endpoint,
        aggregate_gradient=gradient,
        support_mask=support_mask,
        epsilon=1.0,
        surface_step=0.2,
    )
    evidence = build_matched_box_endpoint_evidence(
        parent_bundle_sha256="a" * 64,
        parent_step_npz_sha256="b" * 64,
        parent_step=parent_step,
    )
    path = tmp_path / "matched-box-step.npz"

    digest = write_matched_box_endpoint_npz(evidence, output_path=path)
    loaded = load_matched_box_endpoint_npz(path)
    decision = evaluate_matched_box_endpoint_evidence(
        loaded,
        parent_bundle_sha256="a" * 64,
        parent_step_npz_sha256="b" * 64,
        parent_step=parent_step,
    )

    assert len(digest) == 64
    assert decision.audit_valid, decision.failures
    assert decision.endpoint == "action_spectral"
    assert loaded.matched_surface_delta_sha256 != (
        loaded.radial_surface_delta_sha256
    )

    invalid = replace(evidence, matched_scale=1.5)
    with pytest.raises(
        TerminalProjectionCounterfactualError,
        match="matched scale",
    ):
        write_matched_box_endpoint_npz(
            invalid,
            output_path=tmp_path / "invalid.npz",
        )


def test_projection_response_npz_accepts_only_frozen_three_arms(
    tmp_path: Path,
) -> None:
    """新response schema复用raw语义，但不污染Gate 6i冻结arm名称。"""

    clean_classes = np.asarray([0, 2], dtype=np.int64)
    teacher_logits = np.zeros((2, 256), dtype=np.float32)
    teacher_logits[np.arange(2), clean_classes] = np.asarray(
        [2.0, 3.0], dtype=np.float32
    )
    objective = compute_action_hinge(teacher_logits, clean_classes)
    generation_logits = np.zeros((2, 256), dtype=np.float32)
    generation_logits[np.arange(2), clean_classes] = 1.0
    evidence = EndpointResponseEvidence(
        endpoint="action_only_control",
        state_id=3,
        arm="matched_box_support",
        state_fingerprint="c" * 64,
        surface_delta_sha256="d" * 64,
        clean_action_token_ids=clean_classes + 31_744,
        clean_classes=clean_classes,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
        generated_token_ids=clean_classes + 31_744,
        generated_classes=clean_classes.copy(),
        generation_logits=generation_logits,
        teacher_logits=teacher_logits,
        decoded_action=np.zeros(2, dtype=np.float32),
        effective_view_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        processor_bf16_bits=np.zeros((1, 6, 224, 224), dtype=np.uint16),
    )
    path = tmp_path / "response.npz"

    digest = write_projection_response_npz(evidence, output_path=path)
    loaded = load_projection_response_npz(path)

    assert len(digest) == 64
    assert loaded.arm == "matched_box_support"
    assert np.array_equal(loaded.teacher_logits, teacher_logits)

    with pytest.raises(
        TerminalProjectionCounterfactualError,
        match="arm",
    ):
        write_projection_response_npz(
            replace(evidence, arm="support"),
            output_path=tmp_path / "invalid-response.npz",
        )


def _response_with_loss(
    *,
    endpoint: str,
    state_id: int,
    arm: str,
    loss: float,
    surface_sha256: str,
) -> EndpointResponseEvidence:
    clean_classes = np.asarray([0], dtype=np.int64)
    teacher = np.zeros((1, 256), dtype=np.float32)
    teacher[0, 0] = np.float32(loss)
    objective = compute_action_hinge(teacher, clean_classes)
    generation = np.zeros((1, 256), dtype=np.float32)
    generation[0, 0] = 1.0
    return EndpointResponseEvidence(
        endpoint=endpoint,
        state_id=state_id,
        arm=arm,
        state_fingerprint=f"{state_id + 1:064x}",
        surface_delta_sha256=surface_sha256,
        clean_action_token_ids=clean_classes + 31_744,
        clean_classes=clean_classes,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
        generated_token_ids=clean_classes + 31_744,
        generated_classes=clean_classes.copy(),
        generation_logits=generation,
        teacher_logits=teacher,
        decoded_action=np.zeros(1, dtype=np.float32),
        effective_view_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        processor_bf16_bits=np.zeros((1, 6, 224, 224), dtype=np.uint16),
    )


def test_response_evaluator_requires_strict_replay_and_keeps_endpoints_separate(
) -> None:
    """D/I_M逐endpoint报告；baseline/radial必须逐raw字段重放parent。"""

    endpoint_surface = np.asarray(
        [[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gradient = np.asarray(
        [[-1.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    support_mask = np.ones(3, dtype=np.bool_)
    matched_steps = {}
    parent_responses = []
    child_responses = []
    losses = {
        "action_spectral": (5.0, 4.0, 3.0),
        "action_only_control": (5.0, 3.0, 4.0),
    }
    for endpoint in losses:
        parent_step = build_endpoint_step_evidence(
            endpoint=endpoint,
            endpoint_surface_delta=endpoint_surface,
            aggregate_gradient=gradient,
            support_mask=support_mask,
            epsilon=1.0,
            surface_step=0.2,
        )
        matched = build_matched_box_endpoint_evidence(
            parent_bundle_sha256="a" * 64,
            parent_step_npz_sha256=(
                "b" * 64 if endpoint == "action_spectral" else "c" * 64
            ),
            parent_step=parent_step,
        )
        matched_steps[endpoint] = matched
        baseline_loss, radial_loss, matched_loss = losses[endpoint]
        for state_id in range(10):
            baseline = _response_with_loss(
                endpoint=endpoint,
                state_id=state_id,
                arm="baseline",
                loss=baseline_loss,
                surface_sha256=parent_step.realized_endpoint_sha256,
            )
            radial = _response_with_loss(
                endpoint=endpoint,
                state_id=state_id,
                arm="support",
                loss=radial_loss,
                surface_sha256=parent_step.support_surface_delta_sha256,
            )
            parent_responses.extend((baseline, radial))
            child_responses.extend(
                (
                    baseline,
                    replace(radial, arm="radial_support"),
                    _response_with_loss(
                        endpoint=endpoint,
                        state_id=state_id,
                        arm="matched_box_support",
                        loss=matched_loss,
                        surface_sha256=matched.matched_surface_delta_sha256,
                    ),
                )
            )

    decision = evaluate_projection_response_evidence(
        child_responses,
        parent_responses=parent_responses,
        matched_steps_by_endpoint=matched_steps,
    )

    assert decision.audit_valid, decision.failures
    assert decision.response_record_count == 60
    spectral, control = decision.endpoint_metrics
    assert spectral.endpoint == "action_spectral"
    assert spectral.radial_minus_matched == 1.0
    assert spectral.baseline_minus_matched == 2.0
    assert control.endpoint == "action_only_control"
    assert control.radial_minus_matched == -1.0
    assert control.baseline_minus_matched == 1.0
    assert not hasattr(decision, "projection_harm")

    smoke = evaluate_projection_response_evidence(
        [row for row in child_responses if row.state_id == 0],
        parent_responses=[
            row for row in parent_responses if row.state_id == 0
        ],
        matched_steps_by_endpoint=matched_steps,
        expected_state_ids=(0,),
    )
    assert smoke.audit_valid, smoke.failures
    assert smoke.response_record_count == 6
    assert tuple(metric.endpoint for metric in smoke.endpoint_metrics) == (
        "action_spectral",
        "action_only_control",
    )

    drifted = list(child_responses)
    radial_index = next(
        index
        for index, row in enumerate(drifted)
        if row.arm == "radial_support"
    )
    logits = drifted[radial_index].teacher_logits.copy()
    logits[0, 0] += np.float32(1.0)
    objective = compute_action_hinge(
        logits,
        drifted[radial_index].clean_classes,
    )
    drifted[radial_index] = replace(
        drifted[radial_index],
        teacher_logits=logits,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
    )
    invalid = evaluate_projection_response_evidence(
        drifted,
        parent_responses=parent_responses,
        matched_steps_by_endpoint=matched_steps,
    )
    assert not invalid.audit_valid
    assert any("parent replay" in failure for failure in invalid.failures)
