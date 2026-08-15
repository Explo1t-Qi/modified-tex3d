"""Gate 6i Terminal Endpoint Action-Gradient/Response 纯 CPU 行为测试。"""

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

EFFECTIVE_RGB = np.zeros((224, 224, 3), dtype=np.uint8)
PROCESSOR_BF16_BITS = np.zeros((1, 6, 224, 224), dtype=np.uint16)

from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    EndpointGradientEvidence,
    EndpointResponseEvidence,
    aggregate_state_gradients,
    compute_action_hinge,
    compute_gradient_retention,
    evaluate_endpoint_evidence,
    evaluate_endpoint_inventory,
    execute_surface_counterfactual_step,
    load_endpoint_gradient_npz,
    load_endpoint_response_npz,
    validate_response_teacher_binding,
    write_endpoint_gradient_npz,
    write_endpoint_response_npz,
)


def test_response_teacher_binding_normalizes_single_batch_dimension() -> None:
    """response保存1D序列，runner输入保留单batch 2D；二者应按合同比较。"""

    response_ids = np.asarray([11, 12, 31_744, 31_745], dtype=np.int64)
    clean_teacher_ids = response_ids[None, :].copy()

    validate_response_teacher_binding(response_ids, clean_teacher_ids)

    changed = clean_teacher_ids.copy()
    changed[0, -1] += 1
    with pytest.raises(ValueError, match="teacher IDs漂移"):
        validate_response_teacher_binding(response_ids, changed)

    with pytest.raises(ValueError, match="shape/dtype"):
        validate_response_teacher_binding(response_ids[None, :], clean_teacher_ids)


def test_retention_and_float32_aggregate_match_frozen_definition() -> None:
    gradients = np.asarray(
        [
            [[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    support_mask = np.asarray([True, False])

    mean_gradient = aggregate_state_gradients(gradients)

    assert mean_gradient.dtype == np.float32
    assert np.array_equal(
        mean_gradient,
        np.asarray([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], np.float32),
    )
    assert compute_gradient_retention(gradients[0], support_mask) == 5.0 / 14.0
    # aggregate retention必须在先平均梯度后计算，不是逐state比例平均。
    assert compute_gradient_retention(mean_gradient, support_mask) == 0.5


def test_support_mask_is_applied_before_each_arm_normalization() -> None:
    endpoint = np.zeros((2, 3), dtype=np.float32)
    gradient = np.asarray(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.asarray([False, True])

    dense = execute_surface_counterfactual_step(
        endpoint,
        gradient,
        epsilon=1.0,
        surface_step=0.2,
    )
    support = execute_surface_counterfactual_step(
        endpoint,
        gradient,
        support_mask=mask,
        epsilon=1.0,
        surface_step=0.2,
    )

    assert dense.stats.direction_surface_max == 2.0
    assert np.isclose(dense.stats.parameter_scale, 0.1)
    assert support.stats.direction_surface_max == 1.0
    assert np.isclose(support.stats.parameter_scale, 0.2)
    assert np.isclose(dense.stats.actual_surface_step, 0.2)
    assert np.isclose(support.stats.actual_surface_step, 0.2)
    assert np.count_nonzero(support.executed_step[~mask]) == 0


def test_global_projection_precedes_post_projection_step_cap() -> None:
    endpoint = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], np.float32)
    # descent direction=-gradient=[+1,+1]；candidate=[1.2,-0.8]。先全局投影
    # 得[1,-2/3]，第二坐标实际变化1/3>0.2，因此必须后置cap。
    gradient = np.asarray([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], np.float32)

    result = execute_surface_counterfactual_step(
        endpoint,
        gradient,
        epsilon=1.0,
        surface_step=0.2,
    )

    assert np.isclose(result.stats.projection_scale, 1.0 / 1.2)
    assert np.isclose(result.stats.step_cap_scale, 0.6)
    assert np.allclose(
        result.realized_endpoint + result.executed_step,
        np.asarray([[1.0, 0.0, 0.0], [-0.8, 0.0, 0.0]], np.float32),
        atol=1e-6,
    )
    assert np.isclose(result.stats.actual_surface_step, 0.2)
    assert not np.array_equal(result.projected_step, result.executed_step)


def _response(endpoint: str, state_id: int, arm: str) -> EndpointResponseEvidence:
    clean_classes = np.asarray([0, 2], dtype=np.int64)
    teacher_logits = np.zeros((2, 256), dtype=np.float32)
    teacher_logits[0, :4] = np.asarray([3.0, 1.0, 0.0, -1.0])
    teacher_logits[1, :4] = np.asarray([0.0, 2.5, 4.0, 1.0])
    objective = compute_action_hinge(teacher_logits, clean_classes)
    generated_classes = np.asarray([0, 2], dtype=np.int64)
    generation_logits = np.zeros((2, 256), dtype=np.float32)
    generation_logits[np.arange(2), generated_classes] = 1.0
    return EndpointResponseEvidence(
        endpoint=endpoint,
        state_id=state_id,
        arm=arm,
        state_fingerprint="a" * 64,
        surface_delta_sha256="b" * 64,
        clean_action_token_ids=clean_classes + 31_744,
        clean_classes=clean_classes,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
        generated_token_ids=generated_classes + 31_744,
        generated_classes=generated_classes,
        generation_logits=generation_logits,
        teacher_logits=teacher_logits,
        decoded_action=np.asarray([0.0, 1.0], dtype=np.float32),
        effective_view_rgb=EFFECTIVE_RGB,
        processor_bf16_bits=PROCESSOR_BF16_BITS,
    )


def test_action_hinge_and_response_npz_are_independently_recomputed(
    tmp_path: Path,
) -> None:
    evidence = _response("action_spectral", 0, "baseline")
    path = tmp_path / "response.npz"

    digest = write_endpoint_response_npz(evidence, output_path=path)
    loaded = load_endpoint_response_npz(path)

    assert len(digest) == 64
    assert loaded.action_loss == np.float32(1.75)
    assert np.array_equal(loaded.margins, np.asarray([2.0, 1.5], np.float32))
    assert np.array_equal(loaded.hinge_values, loaded.margins)

    invalid_margins = evidence.margins.copy()
    invalid_margins[0] += 1.0
    with np.testing.assert_raises_regex(ValueError, "margin"):
        write_endpoint_response_npz(
            replace(evidence, margins=invalid_margins),
            output_path=tmp_path / "invalid-response.npz",
        )


def test_gradient_npz_round_trip_rejects_nonfinite_or_zero_payload(
    tmp_path: Path,
) -> None:
    evidence = EndpointGradientEvidence(
        endpoint="action_only_control",
        state_id=9,
        state_fingerprint="c" * 64,
        realized_endpoint_sha256="d" * 64,
        dense_surface_gradient=np.ones((3, 3), dtype=np.float32),
    )
    path = tmp_path / "gradient.npz"

    digest = write_endpoint_gradient_npz(evidence, output_path=path)
    loaded = load_endpoint_gradient_npz(path)

    assert len(digest) == 64
    assert loaded.endpoint == evidence.endpoint
    assert np.array_equal(
        loaded.dense_surface_gradient,
        evidence.dense_surface_gradient,
    )
    invalid = evidence.dense_surface_gradient.copy()
    invalid[0, 0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "finite nonzero"):
        write_endpoint_gradient_npz(
            replace(evidence, dense_surface_gradient=invalid),
            output_path=tmp_path / "invalid-gradient.npz",
        )


def test_inventory_requires_exact_20_gradient_and_60_response_keys() -> None:
    endpoints = ("action_spectral", "action_only_control")
    gradient_keys = [(endpoint, state) for endpoint in endpoints for state in range(10)]
    response_keys = [
        (endpoint, state, arm)
        for endpoint in endpoints
        for state in range(10)
        for arm in ("baseline", "dense", "support")
    ]

    valid = evaluate_endpoint_inventory(gradient_keys, response_keys)
    duplicate = evaluate_endpoint_inventory(
        gradient_keys,
        response_keys[:-1] + [response_keys[0]],
    )

    assert valid.audit_valid
    assert valid.gradient_case_count == 20
    assert valid.response_record_count == 60
    assert not duplicate.audit_valid
    assert any("60个唯一response" in failure for failure in duplicate.failures)


def _response_with_loss(
    endpoint: str,
    state_id: int,
    arm: str,
    loss: float,
) -> EndpointResponseEvidence:
    clean_classes = np.asarray([0], dtype=np.int64)
    teacher_logits = np.zeros((1, 256), dtype=np.float32)
    teacher_logits[0, 0] = np.float32(loss)
    objective = compute_action_hinge(teacher_logits, clean_classes)
    generation_logits = np.zeros((1, 256), dtype=np.float32)
    generation_logits[0, 0] = np.float32(1.0)
    endpoint_index = (
        ("action_spectral", "action_only_control").index(endpoint)
    )
    arm_index = ("baseline", "dense", "support").index(arm)
    return EndpointResponseEvidence(
        endpoint=endpoint,
        state_id=state_id,
        arm=arm,
        state_fingerprint=f"{state_id + 1:064x}",
        surface_delta_sha256=f"{10 + 3 * endpoint_index + arm_index:064x}",
        clean_action_token_ids=clean_classes + 31_744,
        clean_classes=clean_classes,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
        generated_token_ids=clean_classes + 31_744,
        generated_classes=clean_classes.copy(),
        generation_logits=generation_logits,
        teacher_logits=teacher_logits,
        decoded_action=np.asarray([0.0], dtype=np.float32),
        effective_view_rgb=EFFECTIVE_RGB,
        processor_bf16_bits=PROCESSOR_BF16_BITS,
    )


def test_complete_evidence_recomputes_retention_advantage_and_cosine() -> None:
    endpoints = ("action_spectral", "action_only_control")
    gradient = np.asarray(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    zero_gradient = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gradient_cases = [
        EndpointGradientEvidence(
            endpoint=endpoint,
            state_id=state_id,
            state_fingerprint=f"{state_id + 1:064x}",
            realized_endpoint_sha256=("d" if endpoint_index == 0 else "e")
            * 64,
            dense_surface_gradient=gradient.copy(),
        )
        for endpoint_index, endpoint in enumerate(endpoints)
        for state_id in range(10)
    ]
    response_records = [
        _response_with_loss(
            endpoint,
            state_id,
            arm,
            {"baseline": 5.0, "dense": 3.0, "support": 4.0}[arm],
        )
        for endpoint in endpoints
        for state_id in range(10)
        for arm in ("baseline", "dense", "support")
    ]

    decision = evaluate_endpoint_evidence(
        gradient_cases,
        response_records,
        zero_gradients_by_state={
            state_id: zero_gradient.copy() for state_id in range(10)
        },
        support_mask=np.asarray([True, False]),
    )

    assert decision.audit_valid
    assert np.isclose(decision.aggregate_endpoint_gradient_cosine, 1.0)
    assert np.allclose(decision.per_state_endpoint_gradient_cosines, 1.0)
    assert len(decision.endpoint_metrics) == 2
    for metrics in decision.endpoint_metrics:
        assert metrics.zero_retention_mean == 0.5
        assert metrics.terminal_retention_mean == 0.8
        assert np.allclose(metrics.retention_differences, 0.3)
        assert metrics.dense_improvement == 2.0
        assert metrics.support_improvement == 1.0
        assert metrics.dense_advantage == 1.0
        assert metrics.dense_better_state_count == 10
        assert metrics.support_better_state_count == 0
        assert metrics.exact_equal_state_count == 0

    invalid_records = list(response_records)
    invalid_records[0] = replace(
        invalid_records[0],
        state_fingerprint="f" * 64,
    )
    invalid = evaluate_endpoint_evidence(
        gradient_cases,
        invalid_records,
        zero_gradients_by_state={
            state_id: zero_gradient.copy() for state_id in range(10)
        },
        support_mask=np.asarray([True, False]),
    )
    assert not invalid.audit_valid
    assert invalid.endpoint_metrics == ()
    assert any("fingerprint" in failure for failure in invalid.failures)

    duplicate_state_gradients = [
        replace(case, state_fingerprint=f"{1:064x}")
        if case.state_id == 1
        else case
        for case in gradient_cases
    ]
    duplicate_state_responses = [
        replace(record, state_fingerprint=f"{1:064x}")
        if record.state_id == 1
        else record
        for record in response_records
    ]
    duplicate_state = evaluate_endpoint_evidence(
        duplicate_state_gradients,
        duplicate_state_responses,
        zero_gradients_by_state={
            state_id: zero_gradient.copy() for state_id in range(10)
        },
        support_mask=np.asarray([True, False]),
    )
    assert not duplicate_state.audit_valid
    assert any("彼此唯一" in failure for failure in duplicate_state.failures)


def test_complete_evidence_marks_zero_aggregate_gradient_invalid() -> None:
    endpoints = ("action_spectral", "action_only_control")
    gradient = np.asarray(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gradient_cases = [
        EndpointGradientEvidence(
            endpoint=endpoint,
            state_id=state_id,
            state_fingerprint=f"{state_id + 1:064x}",
            realized_endpoint_sha256=("d" if endpoint_index == 0 else "e")
            * 64,
            dense_surface_gradient=gradient.copy(),
        )
        for endpoint_index, endpoint in enumerate(endpoints)
        for state_id in range(10)
    ]
    response_records = [
        _response_with_loss(endpoint, state_id, arm, 1.0)
        for endpoint in endpoints
        for state_id in range(10)
        for arm in ("baseline", "dense", "support")
    ]

    decision = evaluate_endpoint_evidence(
        gradient_cases,
        response_records,
        zero_gradients_by_state={
            state_id: (gradient if state_id % 2 == 0 else -gradient).copy()
            for state_id in range(10)
        },
        support_mask=np.asarray([True, False]),
    )

    assert not decision.audit_valid
    assert any("zero aggregate" in failure for failure in decision.failures)
