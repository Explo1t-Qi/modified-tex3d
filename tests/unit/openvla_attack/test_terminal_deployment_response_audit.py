"""Gate 6g Terminal Deployment Response Audit 的纯 CPU 行为测试。"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_deployment_response_audit import (  # noqa: E402
    ActionPathEvidence,
    TERMINAL_DEPLOYMENT_RESPONSE_BUNDLE_SCHEMA_VERSION,
    TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION,
    TERMINAL_RESPONSE_AUTHORITY_CONTRACT,
    TerminalActionResponseDecision,
    TerminalDeploymentResponseEvidence,
    classify_terminal_action_response,
    evaluate_terminal_response_bundle,
    evaluate_terminal_response_evidence,
    evaluate_terminal_response_npz,
    evaluate_terminal_response_smoke_bundle,
    evaluate_final_processor_equivalence,
    evaluate_terminal_bake_pairing,
    evaluate_visibility_equivalence,
    publish_terminal_response_smoke_manifest,
    summarize_terminal_action_responses,
    write_audit_failure_record,
    write_json_atomically,
    write_terminal_response_npz,
)


def _unique_logits(classes: list[int], num_classes: int = 6) -> np.ndarray:
    logits = np.zeros((len(classes), num_classes), dtype=np.float32)
    for index, class_id in enumerate(classes):
        logits[index, class_id] = 2.0
    return logits


def _terminal_evidence(
    *,
    variant: str = "action_spectral",
    state_id: int = 0,
) -> TerminalDeploymentResponseEvidence:
    clean_classes = np.asarray([0, 1, 2], dtype=np.int64)
    training_classes = np.asarray([0, 3, 4], dtype=np.int64)
    deployment_classes = np.asarray([0, 3, 5], dtype=np.int64)
    generated_classes = np.stack(
        (clean_classes, training_classes, deployment_classes)
    )
    teacher_logits = np.stack(
        (
            _unique_logits(clean_classes.tolist()),
            _unique_logits(training_classes.tolist()),
            _unique_logits(deployment_classes.tolist()),
        )
    )
    effective_rgb = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    effective_rgb[1, 0, 0, 0] = 2
    effective_rgb[2, 0, 0, 0] = 1
    rgb_delta = (
        effective_rgb[1:].astype(np.float32)
        - effective_rgb[0:1].astype(np.float32)
    ) / np.float32(255.0)
    segmentation = np.zeros((2, 2, 2), dtype=np.int32)
    alpha = np.asarray([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    processor_bits = np.zeros((3, 1, 6, 2, 2), dtype=np.uint16)
    processor_float32 = np.zeros((3, 1, 6, 2, 2), dtype=np.float32)
    generated_token_ids = generated_classes + 100
    # 真实OpenVLA有256个action token、255个连续action bin center；最远端
    # token按正式codec语义裁剪到最后一个center。
    bin_centers = np.asarray([4, 3, 2, 1, 0], dtype=np.float64)
    bin_indices = np.clip(
        106 - generated_token_ids - 1,
        a_min=0,
        a_max=bin_centers.size - 1,
    )
    decoded_actions = bin_centers[bin_indices]
    return TerminalDeploymentResponseEvidence(
        variant=variant,
        state_id=state_id,
        effective_rgb=effective_rgb,
        rgb_delta=rgb_delta,
        clean_oriented_segmentation=segmentation,
        deployment_oriented_segmentation=segmentation.copy(),
        clean_instance_alpha=alpha,
        deployment_instance_alpha=alpha.copy(),
        training_exact_processor_bf16_bits=processor_bits,
        official_processor_bf16_bits=processor_bits.copy(),
        training_exact_processor_float32=processor_float32,
        official_processor_float32=processor_float32.copy(),
        teacher_logits=teacher_logits,
        generation_logits=teacher_logits.copy(),
        generated_classes=generated_classes,
        generated_token_ids=generated_token_ids,
        decoded_actions=decoded_actions,
        prompt_input_ids=np.asarray([7, 8], dtype=np.int64),
        teacher_input_ids=np.asarray([7, 8, 0, 1, 2], dtype=np.int64),
        action_token_start=100,
        action_token_end=106,
        vocab_size=106,
        bin_centers=bin_centers,
        action_low=np.zeros(3, dtype=np.float64),
        action_high=np.ones(3, dtype=np.float64),
        action_unnormalize_mask=np.zeros(3, dtype=np.bool_),
    )


def _bundle_input_sha256() -> dict[str, str]:
    return {
        "production_support": "a" * 64,
        "rebake_preflight_manifest": "b" * 64,
        "action_spectral_formal_manifest": "c" * 64,
        "action_spectral_parameter": "d" * 64,
        "action_spectral_bake": "e" * 64,
        "action_only_control_formal_manifest": "f" * 64,
        "action_only_control_parameter": "1" * 64,
        "action_only_control_bake": "2" * 64,
    }


def _terminal_pairing() -> dict[str, dict[str, object]]:
    inputs = _bundle_input_sha256()
    return {
        variant: {
            "gate_pass": True,
            "formal_manifest_sha256": inputs[f"{variant}_formal_manifest"],
            "parameter_sha256": inputs[f"{variant}_parameter"],
            "bound_png_sha256": inputs[f"{variant}_bake"],
        }
        for variant in ("action_spectral", "action_only_control")
    }


def _bundle_provenance() -> dict[str, object]:
    return {
        "input_sha256": _bundle_input_sha256(),
        "checkpoint_fingerprints": {"config.json": "3" * 64},
        "processor_specification": {"output_size": [2, 2]},
        "policy_view_specification": {"source_resolution": 2},
        "asset_restore_status": {"xml": True, "texture": True},
        "runtime_asset_backup_paths_removed": True,
    }


def _action_path(
    classes: list[int],
    *,
    generation_logits: np.ndarray | None = None,
    teacher_logits: np.ndarray | None = None,
) -> ActionPathEvidence:
    default_logits = _unique_logits(classes)
    return ActionPathEvidence(
        generated_classes=np.asarray(classes, dtype=np.int64),
        generation_logits=(
            default_logits if generation_logits is None else generation_logits
        ),
        teacher_logits=(
            default_logits if teacher_logits is None else teacher_logits
        ),
    )


def test_same_unique_first_divergence_is_strictly_preserved() -> None:
    """后续序列可分叉；相同的首次唯一响应仍应判为严格保留。"""

    clean = _action_path([0, 1, 2])
    training = _action_path([0, 3, 4])
    deployment = _action_path([0, 3, 5])

    result = classify_terminal_action_response(
        clean=clean,
        training=training,
        deployment=deployment,
    )

    assert result.classification == "deployment_preserved_strict"
    assert result.training_first_divergence_index == 1
    assert result.deployment_first_divergence_index == 1
    assert result.training_first_divergence_class == 3
    assert result.deployment_first_divergence_class == 3


def test_same_tied_first_divergence_is_marked_tie_sensitive() -> None:
    clean = _action_path([0, 1])
    tied_logits = _unique_logits([0, 3])
    tied_logits[1, 1] = tied_logits[1, 3]
    path = _action_path([0, 3], generation_logits=tied_logits)

    result = classify_terminal_action_response(
        clean=clean,
        training=path,
        deployment=path,
    )

    assert result.classification == "deployment_preserved_tie_sensitive"


def test_deployment_without_any_divergence_is_lost() -> None:
    clean = _action_path([0, 1])
    training = _action_path([0, 3])

    result = classify_terminal_action_response(
        clean=clean,
        training=training,
        deployment=clean,
    )

    assert result.classification == "deployment_lost"


def test_different_deployment_first_response_is_altered() -> None:
    clean = _action_path([0, 1])
    training = _action_path([0, 3])
    deployment = _action_path([0, 4])

    result = classify_terminal_action_response(
        clean=clean,
        training=training,
        deployment=deployment,
    )

    assert result.classification == "deployment_response_altered"


def test_teacher_generation_disagreement_is_diagnostic_only() -> None:
    """固定clean-prefix teacher不是部署生成行为的有效性oracle。"""

    clean = _action_path([0, 1])
    inconsistent_logits = _unique_logits([0, 3])
    inconsistent = _action_path(
        [0, 1],
        teacher_logits=inconsistent_logits,
    )

    result = classify_terminal_action_response(
        clean=clean,
        training=inconsistent,
        deployment=clean,
    )

    assert result.classification == "no_training_response"
    assert result.failures == ()
    training_diagnostic = result.teacher_generation_alignment[1]
    assert training_diagnostic.path_name == "training"
    assert training_diagnostic.mismatch_token_indices == (1,)
    assert training_diagnostic.all_comparable_match is False


def test_generation_class_outside_own_score_argmax_is_invalid() -> None:
    """只有生成token与其对应autoregressive score自相矛盾才使case无效。"""

    clean = _action_path([0, 1])
    inconsistent_generation = _action_path(
        [0, 1],
        generation_logits=_unique_logits([0, 3]),
    )

    result = classify_terminal_action_response(
        clean=clean,
        training=inconsistent_generation,
        deployment=clean,
    )

    assert result.classification == "invalid_response_alignment"
    assert any("generation score argmax" in failure for failure in result.failures)


def test_summary_uses_null_when_no_training_response_exists() -> None:
    """没有A路径离散响应时，条件保留率没有定义，不能伪装成零。"""

    decision = TerminalActionResponseDecision(
        classification="no_training_response",
        training_first_divergence_index=None,
        deployment_first_divergence_index=None,
        training_first_divergence_class=None,
        deployment_first_divergence_class=None,
    )

    summary = summarize_terminal_action_responses((decision,))

    assert summary["training_response_count"] == 0
    assert summary["preserved_first_response_count"] == 0
    assert summary["preserved_given_training_response"] is None


def test_processor_equivalence_uses_final_bfloat16_bits() -> None:
    """一个最终BF16 bit漂移也必须保留为精确的输入路径失败证据。"""

    training_bits = np.asarray([[[[0x3F80, 0x4000]]]], dtype=np.uint16)
    official_bits = training_bits.copy()
    official_bits[0, 0, 0, 1] = np.uint16(0x4001)

    decision = evaluate_final_processor_equivalence(
        training_exact_bf16_bits=training_bits,
        official_bf16_bits=official_bits,
    )

    assert decision.bitwise_equal is False
    assert decision.mismatch_count == 1
    assert decision.total_value_count == 2


def test_visibility_equivalence_rejects_one_target_alpha_pixel_change() -> None:
    """B路径的一像素hard-alpha漂移也不能混入终态响应比较。"""

    clean_segmentation = np.zeros((2, 2, 2), dtype=np.int32)
    deployment_segmentation = clean_segmentation.copy()
    clean_alpha = np.asarray([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    deployment_alpha = clean_alpha.copy()
    deployment_alpha[0, 0, 0, 1] = 1.0

    decision = evaluate_visibility_equivalence(
        clean_oriented_segmentation=clean_segmentation,
        deployment_oriented_segmentation=deployment_segmentation,
        clean_instance_alpha=clean_alpha,
        deployment_instance_alpha=deployment_alpha,
    )

    assert decision.segmentation_bitwise_equal is True
    assert decision.alpha_bitwise_equal is False
    assert decision.alpha_mismatch_count == 1
    assert decision.gate_pass is False


def test_terminal_bake_pairing_requires_repeat_and_bound_pixel_equality() -> None:
    """当前重放可重复但不等于历史bound PNG时，终态配对仍必须失败。"""

    rebaked = np.zeros((2, 2, 3), dtype=np.uint8)
    bound = rebaked.copy()
    bound[1, 1, 2] = 1

    decision = evaluate_terminal_bake_pairing(
        first_rebaked_rgb=rebaked,
        second_rebaked_rgb=rebaked.copy(),
        bound_png_rgb=bound,
    )

    assert decision.repeat_bitwise_equal is True
    assert decision.bound_bitwise_equal is False
    assert decision.bound_mismatch_count == 1
    assert decision.gate_pass is False


def test_npz_round_trip_independently_recomputes_terminal_response(
    tmp_path: Path,
) -> None:
    """GPU侧只保存原始数组；CPU从NPZ重新导出分类与完整性。"""

    output_path = tmp_path / "state_00.npz"

    artifact_sha256 = write_terminal_response_npz(
        _terminal_evidence(),
        output_path=output_path,
    )
    result = evaluate_terminal_response_npz(output_path)

    assert len(artifact_sha256) == 64
    assert result.evidence_valid is True
    assert result.action_decision.classification == (
        "deployment_preserved_strict"
    )
    assert result.processor_equivalence.bitwise_equal is True
    assert result.visibility_equivalence.gate_pass is True


def test_v2_evaluator_rejects_legacy_v1_evidence(tmp_path: Path) -> None:
    current_path = tmp_path / "current.npz"
    legacy_path = tmp_path / "legacy.npz"
    write_terminal_response_npz(
        _terminal_evidence(),
        output_path=current_path,
    )
    with np.load(current_path, allow_pickle=False) as archive:
        legacy_payload = {name: archive[name].copy() for name in archive.files}
    legacy_payload["schema_version"] = np.asarray(
        "openvla-terminal-deployment-response-v1"
    )
    np.savez_compressed(legacy_path, **legacy_payload)

    with pytest.raises(ValueError, match="schema不匹配"):
        evaluate_terminal_response_npz(legacy_path)


def test_evaluator_recomputes_decoded_actions_from_codec_evidence() -> None:
    """runner写入的连续动作被篡改时，CPU不能继续信任该case。"""

    evidence = _terminal_evidence()
    tampered_actions = evidence.decoded_actions.copy()
    tampered_actions[2, 1] += 0.25

    result = evaluate_terminal_response_evidence(
        replace(evidence, decoded_actions=tampered_actions)
    )

    assert result.evidence_valid is False
    assert result.decoded_actions_exact is False
    assert any("decoded action" in failure for failure in result.failures)


def test_evaluator_keeps_teacher_disagreement_as_non_blocking_diagnostic() -> None:
    evidence = _terminal_evidence()
    teacher_logits = evidence.teacher_logits.copy()
    teacher_logits[1, 1] = _unique_logits([0, 4, 4])[1]

    result = evaluate_terminal_response_evidence(
        replace(evidence, teacher_logits=teacher_logits)
    )

    assert result.evidence_valid is True
    assert result.generation_self_alignment_pass is True
    training_diagnostic = result.action_decision.teacher_generation_alignment[1]
    assert training_diagnostic.mismatch_token_indices == (1,)


def test_bundle_rejects_missing_variant_state_case(tmp_path: Path) -> None:
    """Formal inventory少一个case也必须整体无效，不能把19行当成完整证据。"""

    cases: list[dict[str, object]] = []
    for variant in ("action_spectral", "action_only_control"):
        for state_id in range(10):
            if variant == "action_only_control" and state_id == 9:
                continue
            relative_path = Path("arrays") / variant / f"state_{state_id:02d}.npz"
            artifact_sha256 = write_terminal_response_npz(
                _terminal_evidence(variant=variant, state_id=state_id),
                output_path=tmp_path / relative_path,
            )
            cases.append(
                {
                    "variant": variant,
                    "state_id": state_id,
                    "npz_relative_path": str(relative_path),
                    "npz_sha256": artifact_sha256,
                    "initial_state_sha256": f"{state_id + 1:064x}",
                    "clean_static_scene_sha256": f"{state_id + 11:064x}",
                    "deployment_static_scene_sha256": f"{state_id + 11:064x}",
                    "transaction_verified": True,
                    "asset_restore_verified": True,
                }
            )
    manifest = {
        "schema_version": TERMINAL_DEPLOYMENT_RESPONSE_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "expected_variants": ["action_spectral", "action_only_control"],
        "expected_state_ids": list(range(10)),
        "state_fingerprints": [f"{state_id + 1:064x}" for state_id in range(10)],
        "terminal_pairing": _terminal_pairing(),
        "response_authority": dict(TERMINAL_RESPONSE_AUTHORITY_CONTRACT),
        "cases": cases,
        "provenance": _bundle_provenance(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = evaluate_terminal_response_bundle(manifest_path)

    assert decision.audit_valid is False
    assert decision.case_count == 19
    assert any("缺失" in failure for failure in decision.failures)


def test_smoke_bundle_requires_exactly_two_state_zero_cases(
    tmp_path: Path,
) -> None:
    cases: list[dict[str, object]] = []
    for variant in ("action_spectral", "action_only_control"):
        relative_path = Path("arrays") / variant / "state_00.npz"
        artifact_sha256 = write_terminal_response_npz(
            _terminal_evidence(variant=variant, state_id=0),
            output_path=tmp_path / relative_path,
        )
        cases.append(
            {
                "variant": variant,
                "state_id": 0,
                "npz_relative_path": str(relative_path),
                "npz_sha256": artifact_sha256,
                "initial_state_sha256": "1" * 64,
                "clean_static_scene_sha256": "2" * 64,
                "deployment_static_scene_sha256": "2" * 64,
                "transaction_verified": True,
                "asset_restore_verified": True,
            }
        )
    manifest = {
        "schema_version": TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "expected_variants": ["action_spectral", "action_only_control"],
        "expected_state_ids": [0],
        "state_fingerprints": ["1" * 64],
        "terminal_pairing": _terminal_pairing(),
        "response_authority": dict(TERMINAL_RESPONSE_AUTHORITY_CONTRACT),
        "cases": cases,
        "provenance": _bundle_provenance(),
    }
    manifest_path = tmp_path / "smoke_manifest.json"
    published_sha256 = publish_terminal_response_smoke_manifest(
        manifest,
        output_path=manifest_path,
    )

    decision = evaluate_terminal_response_smoke_bundle(manifest_path)

    assert len(published_sha256) == 64
    assert decision.audit_valid is True
    assert decision.case_count == 2
    assert decision.per_variant_summary["action_spectral"]["row_count"] == 1

    legacy_manifest = dict(manifest)
    legacy_manifest["schema_version"] = (
        "openvla-terminal-deployment-response-smoke-bundle-v1"
    )
    legacy_path = tmp_path / "legacy_manifest.json"
    legacy_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    legacy_decision = evaluate_terminal_response_smoke_bundle(legacy_path)
    assert legacy_decision.audit_valid is False
    assert any("schema_version" in failure for failure in legacy_decision.failures)

    missing_authority_manifest = dict(manifest)
    del missing_authority_manifest["response_authority"]
    missing_authority_path = tmp_path / "missing_authority_manifest.json"
    missing_authority_path.write_text(
        json.dumps(missing_authority_manifest), encoding="utf-8"
    )
    missing_authority_decision = evaluate_terminal_response_smoke_bundle(
        missing_authority_path
    )
    assert missing_authority_decision.audit_valid is False
    assert any(
        "response_authority" in failure
        for failure in missing_authority_decision.failures
    )


def test_smoke_bundle_rejects_variant_specific_clean_evidence(
    tmp_path: Path,
) -> None:
    cases: list[dict[str, object]] = []
    for index, variant in enumerate(
        ("action_spectral", "action_only_control")
    ):
        evidence = _terminal_evidence(variant=variant, state_id=0)
        if index:
            altered_rgb = evidence.effective_rgb.copy()
            altered_rgb[0, 1, 1, 1] = 1
            altered_delta = (
                altered_rgb[1:].astype(np.float32)
                - altered_rgb[0:1].astype(np.float32)
            ) / np.float32(255.0)
            evidence = replace(
                evidence,
                effective_rgb=altered_rgb,
                rgb_delta=altered_delta,
            )
        relative_path = Path("arrays") / variant / "state_00.npz"
        artifact_sha256 = write_terminal_response_npz(
            evidence,
            output_path=tmp_path / relative_path,
        )
        cases.append(
            {
                "variant": variant,
                "state_id": 0,
                "npz_relative_path": str(relative_path),
                "npz_sha256": artifact_sha256,
                "initial_state_sha256": "1" * 64,
                "clean_static_scene_sha256": "2" * 64,
                "deployment_static_scene_sha256": "2" * 64,
                "transaction_verified": True,
                "asset_restore_verified": True,
            }
        )
    manifest = {
        "schema_version": TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "expected_variants": ["action_spectral", "action_only_control"],
        "expected_state_ids": [0],
        "state_fingerprints": ["1" * 64],
        "terminal_pairing": _terminal_pairing(),
        "response_authority": dict(TERMINAL_RESPONSE_AUTHORITY_CONTRACT),
        "cases": cases,
        "provenance": _bundle_provenance(),
    }
    manifest_path = tmp_path / "tampered_clean_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = evaluate_terminal_response_smoke_bundle(manifest_path)

    assert decision.audit_valid is False
    assert any("Clean证据不一致" in failure for failure in decision.failures)


def test_success_manifest_is_published_atomically(tmp_path: Path) -> None:
    """成功标记只能以最终文件出现，不能遗留可误读的临时JSON。"""

    output_path = tmp_path / "manifest.json"

    artifact_sha256 = write_json_atomically(
        {"schema_version": "test", "status": "complete"},
        output_path=output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == (
        "complete"
    )
    assert len(artifact_sha256) == 64
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_failure_record_cannot_be_mistaken_for_success_manifest(
    tmp_path: Path,
) -> None:
    """异常目录保留可审计上下文，但绝不创建正式成功manifest。"""

    failure_path = write_audit_failure_record(
        output_directory=tmp_path,
        failed_stage="deployment_state_0",
        error=RuntimeError("synthetic failure"),
        completed_keys=(("action_spectral", 0),),
        input_sha256={"support": "a" * 64},
        asset_restore_status={"xml": True, "texture": True},
    )

    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["status"] == "audit_invalid"
    assert payload["failed_stage"] == "deployment_state_0"
    assert not (tmp_path / "manifest.json").exists()
