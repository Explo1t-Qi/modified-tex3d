"""正式 source gate 的成对 Clean/Adversarial 回归测试。"""

from __future__ import annotations

import json

import pytest

from openvla.experiments.robot.libero.openvla_attack.paired_source_gate import (
    EXPECTED_EVAL_STATE_IDS,
    PairedSourceGateError,
    StateRolloutOutcome,
    evaluate_paired_source_gate_artifact,
    evaluate_paired_source_gate,
    write_paired_source_gate_artifact,
)


def _outcomes(
    successes: tuple[bool, ...],
) -> tuple[StateRolloutOutcome, ...]:
    return tuple(
        StateRolloutOutcome(
            state_id=state_id,
            initial_state_sha256=f"{state_id:064x}",
            success=success,
        )
        for state_id, success in zip(EXPECTED_EVAL_STATE_IDS, successes)
    )


def test_gate_counts_only_failures_introduced_from_clean_success() -> None:
    clean = _outcomes((False, True, True, True, True, True, True, True, True, True))
    adversarial = _outcomes(
        (False, False, False, False, True, True, True, True, True, True)
    )

    decision = evaluate_paired_source_gate(clean, adversarial)

    assert decision.adversarial_failures == 4
    assert decision.preexisting_clean_failures == 1
    assert decision.attack_induced_failures == 3
    assert decision.clean_control_perfect is False
    assert decision.gate_pass is True


def test_clean_failure_cannot_create_false_gate_pass() -> None:
    clean = _outcomes((False, False, False, True, True, True, True, True, True, True))
    adversarial = _outcomes(
        (False, False, False, False, False, True, True, True, True, True)
    )

    decision = evaluate_paired_source_gate(clean, adversarial)

    assert decision.adversarial_failures == 5
    assert decision.attack_induced_failures == 2
    assert decision.gate_pass is False


def test_gate_rejects_misaligned_or_incomplete_pairs() -> None:
    clean = _outcomes((True,) * 10)
    adversarial = list(_outcomes((False,) * 10))
    adversarial[0] = StateRolloutOutcome(
        state_id=10,
        initial_state_sha256="f" * 64,
        success=False,
    )
    with pytest.raises(PairedSourceGateError, match="fingerprint不一致"):
        evaluate_paired_source_gate(clean, adversarial)

    with pytest.raises(PairedSourceGateError, match="按10-19完整"):
        evaluate_paired_source_gate(clean[:-1], adversarial)


def test_gate_artifact_binds_training_and_bake(tmp_path) -> None:
    decision = evaluate_paired_source_gate(
        _outcomes((True,) * 10),
        _outcomes((False, False, False, True, True, True, True, True, True, True)),
    )
    path = write_paired_source_gate_artifact(
        tmp_path / "paired_source_gate.json",
        decision=decision,
        training_manifest_sha256="a" * 64,
        baked_texture_sha256="b" * 64,
        evaluation_code_commit="c" * 40,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate_pass"] is True
    assert payload["attack_induced_failures"] == 3
    assert payload["training_manifest_sha256"] == "a" * 64
    assert [pair["state_id"] for pair in payload["pairs"]] == list(range(10, 20))


def test_paired_artifact_is_recomputed_and_binds_local_training_bundle(
    tmp_path,
) -> None:
    baked_path = tmp_path / "final.png"
    baked_path.write_bytes(b"baked")
    training_manifest_path = tmp_path / "formal_source_training_manifest.json"
    training_manifest_path.write_text(
        json.dumps({"baked_texture_relative_path": baked_path.name}),
        encoding="utf-8",
    )
    from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
        file_sha256,
    )

    decision = evaluate_paired_source_gate(
        _outcomes((True,) * 10),
        _outcomes((False, False, False, True, True, True, True, True, True, True)),
    )
    path = write_paired_source_gate_artifact(
        tmp_path / "paired_source_gate.json",
        decision=decision,
        training_manifest_sha256=file_sha256(training_manifest_path),
        baked_texture_sha256=file_sha256(baked_path),
        evaluation_code_commit="c" * 40,
    )

    artifact_decision = evaluate_paired_source_gate_artifact(path)

    assert artifact_decision.gate_pass is True
    assert artifact_decision.failures == ()
