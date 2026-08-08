"""Gate 2E 纯 CPU 证据判定测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.deployment_backward_audit import (  # noqa: E402
    SCHEMA_VERSION,
    DeploymentBackwardEvidence,
    GradientEvidence,
    evaluate_deployment_backward_evidence,
    load_deployment_backward_evidence,
    write_deployment_backward_evidence,
)


def _gradient(*shape: int) -> GradientEvidence:
    return GradientEvidence.from_tensor(torch.ones(shape))


def _passing_evidence() -> DeploymentBackwardEvidence:
    return DeploymentBackwardEvidence(
        schema_version=SCHEMA_VERSION,
        code_commit="a" * 40,
        state_id=0,
        objective="source_openvla_action_only",
        action_loss=1.0,
        parameter_shape=(256, 3),
        parameter_changed_count=768,
        parameter_linf_change=0.01,
        source_rgb_gradient=_gradient(1, 3, 512, 512),
        pre_crop_gradient=_gradient(1, 3, 224, 224),
        effective_view_gradient=_gradient(1, 3, 224, 224),
        surface_delta_gradient=_gradient(100, 3),
        parameter_gradient=_gradient(256, 3),
        actual_surface_step=2.0 / 255.0,
        max_surface_delta=2.0 / 255.0,
        baked_texture_path="texture.png",
        baked_texture_sha256="baked",
        active_texture_sha256="baked",
        rollout_completed=True,
        rollout_success=True,
        xml_sha256_before="xml",
        xml_sha256_after_restore="xml",
        real_texture_sha256_before="texture",
        real_texture_sha256_after_restore="texture",
        backup_paths_removed=True,
        provenance={"checkpoint": "/checkpoint"},
    )


def test_gate_2e_accepts_complete_single_update_evidence(tmp_path: Path) -> None:
    evidence = _passing_evidence()
    decision = evaluate_deployment_backward_evidence(evidence)
    assert decision.gate_pass
    assert decision.failures == ()

    path = write_deployment_backward_evidence(
        tmp_path / "gate2e.json",
        evidence,
    )
    loaded = load_deployment_backward_evidence(path)
    assert loaded == evidence


def test_gate_2e_rejects_zero_stage_gradient_and_unrestored_asset() -> None:
    evidence = replace(
        _passing_evidence(),
        pre_crop_gradient=GradientEvidence.from_tensor(
            torch.zeros((1, 3, 224, 224))
        ),
        xml_sha256_after_restore="changed",
    )
    decision = evaluate_deployment_backward_evidence(evidence)
    assert not decision.gate_pass
    assert "Pre-Crop 梯度不是有限非零" in decision.failures
    assert "XML 未恢复到事务前内容" in decision.failures


def test_gate_2e_rejects_wrong_parameter_shape_and_surface_step() -> None:
    evidence = replace(
        _passing_evidence(),
        parameter_shape=(128, 3),
        parameter_changed_count=384,
        actual_surface_step=3.0 / 255.0,
    )
    decision = evaluate_deployment_backward_evidence(evidence)
    assert not decision.gate_pass
    assert "谱系数 shape 不是 [256,3]" in decision.failures
    assert "Actual Surface Step 超限或非正" in decision.failures
