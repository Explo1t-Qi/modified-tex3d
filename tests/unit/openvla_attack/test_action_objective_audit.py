"""新 Action Objective GPU Audit 的纯 CPU 证据判定测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from openvla.experiments.robot.libero.openvla_attack.action_objective_audit import (
    ACTION_OBJECTIVE_NAME,
    ACTION_OBJECTIVE_SCHEMA_VERSION,
    ActionObjectiveAuditEvidence,
    evaluate_action_objective_evidence,
    load_action_objective_evidence,
    summarize_action_objective_evidence,
    write_action_objective_evidence,
)
from openvla.experiments.robot.libero.openvla_attack.deployment_backward_audit import (
    GradientEvidence,
)


def _gradient(*shape: int) -> GradientEvidence:
    return GradientEvidence.from_tensor(torch.ones(shape))


def _passing_evidence(state_id: int) -> ActionObjectiveAuditEvidence:
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
        source_rgb_gradient=_gradient(1, 3, 512, 512),
        pre_crop_gradient=_gradient(1, 3, 224, 224),
        effective_view_gradient=_gradient(1, 3, 224, 224),
        render_surface_delta_gradient=_gradient(6, 3),
        dense_geometry_gradient=_gradient(4, 3),
        dense_geometry_gradient_sha256="b" * 64,
    )


def test_objective_audit_accepts_zero_delta_margin_and_gradients() -> None:
    decision = evaluate_action_objective_evidence(_passing_evidence(0))

    assert decision.gate_pass
    assert decision.failures == ()
    assert decision.tie_indices == (1,)
    assert decision.negative_margin_indices == ()


def test_objective_audit_rejects_negative_margin_only_at_zero_delta() -> None:
    evidence = replace(
        _passing_evidence(0),
        margins=[2.0, -0.5, 1.0],
        hinge_values=[2.0, 0.0, 1.0],
    )

    decision = evaluate_action_objective_evidence(evidence)

    assert not decision.gate_pass
    assert decision.negative_margin_indices == (1,)
    assert "零 Surface Delta 出现负 clean-action margin" in decision.failures


def test_objective_audit_rejects_wrong_hinge_loss_and_zero_gradient() -> None:
    evidence = replace(
        _passing_evidence(0),
        hinge_values=[2.0, 0.5, 1.0],
        action_loss=2.0,
        dense_geometry_gradient=GradientEvidence.from_tensor(
            torch.zeros((4, 3))
        ),
    )

    decision = evaluate_action_objective_evidence(evidence)

    assert not decision.gate_pass
    assert "hinge_values 与 relu(margins) 不一致" in decision.failures
    assert "Action loss 不等于逐 token hinge 均值" in decision.failures
    assert "Dense Geometry 梯度不是有限非零" in decision.failures


def test_objective_audit_requires_clean_teacher_prefix_and_action_tokens() -> None:
    evidence = replace(
        _passing_evidence(0),
        teacher_forced_clean_prefix=False,
        clean_action_token_ids=[],
        clean_classes=[],
        margins=[],
        hinge_values=[],
    )

    decision = evaluate_action_objective_evidence(evidence)

    assert not decision.gate_pass
    assert "未使用固定 teacher-forced clean prefix" in decision.failures
    assert "Action token 证据为空" in decision.failures


def test_objective_audit_summary_requires_exact_states_zero_to_nine() -> None:
    complete = [_passing_evidence(state_id) for state_id in range(10)]

    summary = summarize_action_objective_evidence(complete)

    assert summary.gate_pass
    assert summary.state_ids == tuple(range(10))
    assert summary.valid_state_count == 10
    assert summary.action_token_count == 30
    assert summary.tie_locations == ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                                    (5, 1), (6, 1), (7, 1), (8, 1), (9, 1))

    incomplete = summarize_action_objective_evidence(complete[:-1])
    assert not incomplete.gate_pass
    assert "state IDs 必须精确等于 0-9" in incomplete.failures


def test_objective_audit_jsonl_round_trip(tmp_path: Path) -> None:
    evidence = [_passing_evidence(state_id) for state_id in range(10)]
    path = write_action_objective_evidence(
        tmp_path / "action-objective.jsonl",
        evidence,
    )

    loaded = load_action_objective_evidence(path)

    assert loaded == evidence
