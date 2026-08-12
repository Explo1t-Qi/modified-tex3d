"""Gate 6g numerical inference replay 的纯 CPU evidence 测试。"""

from __future__ import annotations

import sys
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_numerical_inference_audit import (  # noqa: E402
    ATTRIBUTION_COMPARISON_NAMES,
    NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION,
    NumericalAttributionEvidence,
    NumericalFidelityEvidence,
    evaluate_numerical_attribution,
    evaluate_numerical_fidelity,
    evaluate_numerical_inference_bundle,
    load_numerical_attribution_npz,
    load_numerical_fidelity_npz,
    numerical_decision_payload,
    publish_numerical_inference_manifest,
    write_numerical_attribution_npz,
    write_numerical_fidelity_npz,
)


VARIANTS = ("action_spectral", "action_only_control")
PATHS = ("clean", "training", "deployment")


def _fidelity_evidence() -> NumericalFidelityEvidence:
    variants = len(VARIANTS)
    paths = len(PATHS)
    repeats = 3
    action_dim = 2
    classes = 3
    prompt = np.asarray([[10, 11], [10, 11]], dtype=np.int64)
    teacher = np.asarray(
        [[10, 11, 100, 101], [10, 11, 100, 101]], dtype=np.int64
    )
    source_tokens = np.full(
        (variants, paths, action_dim),
        np.asarray([100, 101], dtype=np.int64),
        dtype=np.int64,
    )
    source_classes = np.full(
        (variants, paths, action_dim),
        np.asarray([0, 1], dtype=np.int64),
        dtype=np.int64,
    )
    generation_logits = np.zeros(
        (variants, paths, action_dim, classes), dtype=np.float32
    )
    teacher_logits = np.zeros_like(generation_logits)
    generation_logits[..., 0, 0] = 3.0
    generation_logits[..., 1, 1] = 3.0
    teacher_logits[...] = generation_logits
    # 保留当前真实失败形态：Action-only training token index 1 的 teacher
    # argmax 为 class 2，而 generation class 为 1。
    teacher_logits[1, 1, 1] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    pixel_bits = np.arange(
        variants * paths * 4, dtype=np.uint16
    ).reshape(variants, paths, 1, 1, 2, 2)
    return NumericalFidelityEvidence(
        variant_names=VARIANTS,
        path_names=PATHS,
        repeat_count=repeats,
        action_token_start=100,
        source_prompt_input_ids=prompt,
        source_teacher_input_ids=teacher,
        reconstructed_prompt_input_ids=prompt.copy(),
        reconstructed_teacher_input_ids=teacher.copy(),
        reconstructed_attention_mask=np.ones_like(prompt),
        source_pixel_bf16_bits=pixel_bits,
        reconstructed_pixel_bf16_bits=pixel_bits.copy(),
        source_generated_token_ids=source_tokens,
        source_generated_classes=source_classes,
        source_generation_logits=generation_logits,
        source_teacher_logits=teacher_logits,
        replay_generated_token_ids=np.repeat(
            source_tokens[:, :, None, :], repeats, axis=2
        ),
        replay_generated_classes=np.repeat(
            source_classes[:, :, None, :], repeats, axis=2
        ),
        replay_generation_logits=np.repeat(
            generation_logits[:, :, None, :, :], repeats, axis=2
        ),
        replay_teacher_logits=np.repeat(
            teacher_logits[:, :, None, :, :], repeats, axis=2
        ),
    )


def _attribution_evidence(
    fidelity: NumericalFidelityEvidence,
) -> NumericalAttributionEvidence:
    cached_logits = fidelity.replay_generation_logits
    teacher_logits = fidelity.replay_teacher_logits
    return NumericalAttributionEvidence(
        variant_names=fidelity.variant_names,
        path_names=fidelity.path_names,
        repeat_count=fidelity.repeat_count,
        no_cache_generated_token_ids=(
            fidelity.replay_generated_token_ids.copy()
        ),
        no_cache_generation_logits=cached_logits.copy(),
        generation_prefix_no_cache_logits=cached_logits.copy(),
        clean_prefix_no_cache_logits=teacher_logits.copy(),
        full_teacher_no_cache_logits=teacher_logits.copy(),
    )


def test_fidelity_passes_per_input_and_retains_original_mismatch() -> None:
    decision = evaluate_numerical_fidelity(_fidelity_evidence())

    assert decision.fidelity_pass
    assert decision.causal_attribution_allowed
    assert len(decision.per_input) == 6
    assert {item.status for item in decision.per_input} == {"pass"}
    assert len(decision.original_mismatches) == 1
    mismatch = decision.original_mismatches[0]
    assert mismatch.variant == "action_only_control"
    assert mismatch.path == "training"
    assert mismatch.token_index == 1
    assert mismatch.generation_class == 1
    assert mismatch.teacher_argmax_classes == (2,)
    assert mismatch.evidence_scope == "invalid_bundle_diagnostic_observation"


def test_stable_replay_drift_is_per_input_and_blocks_run_attribution() -> None:
    evidence = _fidelity_evidence()
    replay = evidence.replay_generation_logits.copy()
    replay[1, 1, :, 1, 1] += np.float32(0.5)

    decision = evaluate_numerical_fidelity(
        replace(evidence, replay_generation_logits=replay)
    )

    assert not decision.fidelity_pass
    assert not decision.causal_attribution_allowed
    failed = [item for item in decision.per_input if not item.fidelity_pass]
    assert [(item.variant, item.path) for item in failed] == [
        ("action_only_control", "training")
    ]
    assert failed[0].status == "replay_provenance_unresolved"
    # Replay失败不能撤销失败bundle里已保存的原始跨路径观察。
    assert len(decision.original_mismatches) == 1


def test_nondeterministic_replay_is_not_mislabeled_as_environment_drift() -> None:
    evidence = _fidelity_evidence()
    replay = evidence.replay_teacher_logits.copy()
    replay[0, 2, 2, 0, 0] += np.float32(0.25)

    decision = evaluate_numerical_fidelity(
        replace(evidence, replay_teacher_logits=replay)
    )

    failed = [item for item in decision.per_input if not item.fidelity_pass]
    assert len(failed) == 1
    assert failed[0].status == "runtime_nondeterministic"
    assert not failed[0].teacher_repeat_exact
    assert not failed[0].teacher_matches_source


def test_input_reconstruction_failure_is_reported_for_each_affected_path() -> None:
    evidence = _fidelity_evidence()
    reconstructed = evidence.reconstructed_pixel_bf16_bits.copy()
    reconstructed[0, 1, 0, 0, 0, 0] ^= np.uint16(1)

    decision = evaluate_numerical_fidelity(
        replace(evidence, reconstructed_pixel_bf16_bits=reconstructed)
    )

    failed = [item for item in decision.per_input if not item.fidelity_pass]
    assert [(item.variant, item.path, item.status) for item in failed] == [
        ("action_spectral", "training", "input_reconstruction_invalid")
    ]


def test_attribution_uses_two_same_prefix_comparisons() -> None:
    fidelity = _fidelity_evidence()
    attribution = _attribution_evidence(fidelity)
    generation_prefix = attribution.generation_prefix_no_cache_logits.copy()
    generation_prefix[1, 1, :, 1] = np.asarray(
        [0.0, 2.0, 2.5], dtype=np.float32
    )
    attribution = replace(
        attribution,
        generation_prefix_no_cache_logits=generation_prefix,
    )

    decision = evaluate_numerical_attribution(
        fidelity=fidelity,
        evidence=attribution,
    )

    assert decision.attribution_valid
    target = next(
        item
        for item in decision.per_input
        if item.variant == "action_only_control" and item.path == "training"
    )
    comparisons = {item.name: item for item in target.comparisons}
    assert set(comparisons) == set(ATTRIBUTION_COMPARISON_NAMES)
    cache_comparison = comparisons[
        "cached_vs_generation_prefix_no_cache"
    ]
    assert cache_comparison.argmax_equal == (True, False)
    assert cache_comparison.linf == (0.0, 2.5)
    shape_comparison = comparisons[
        "full_teacher_no_cache_vs_clean_prefix_no_cache"
    ]
    assert shape_comparison.argmax_equal == (True, True)
    assert shape_comparison.linf == (0.0, 0.0)


def test_attribution_nondeterminism_blocks_causal_interpretation() -> None:
    fidelity = _fidelity_evidence()
    attribution = _attribution_evidence(fidelity)
    prefix = attribution.clean_prefix_no_cache_logits.copy()
    prefix[0, 0, 2, 0, 0] += np.float32(0.125)

    decision = evaluate_numerical_attribution(
        fidelity=fidelity,
        evidence=replace(
            attribution,
            clean_prefix_no_cache_logits=prefix,
        ),
    )

    assert not decision.attribution_valid
    assert not decision.causal_attribution_allowed
    failed = [item for item in decision.per_input if not item.paths_stable]
    assert [(item.variant, item.path) for item in failed] == [
        ("action_spectral", "clean")
    ]


def test_full_generation_logits_are_only_comparable_before_divergence() -> None:
    fidelity = _fidelity_evidence()
    attribution = _attribution_evidence(fidelity)
    no_cache_ids = attribution.no_cache_generated_token_ids.copy()
    no_cache_ids[0, 0, :, 0] = 102

    decision = evaluate_numerical_attribution(
        fidelity=fidelity,
        evidence=replace(
            attribution,
            no_cache_generated_token_ids=no_cache_ids,
        ),
    )

    target = decision.per_input[0]
    assert target.cached_vs_no_cache_first_divergence_index == 0
    comparison = next(
        item
        for item in target.comparisons
        if item.name == "cached_vs_no_cache_generation"
    )
    assert tuple(
        token.causal_prefix_comparable for token in comparison.tokens
    ) == (True, False)


def test_numerical_npz_round_trip_is_pickle_free(tmp_path: Path) -> None:
    fidelity = _fidelity_evidence()
    attribution = _attribution_evidence(fidelity)
    fidelity_path = tmp_path / "fidelity.npz"
    attribution_path = tmp_path / "attribution.npz"

    write_numerical_fidelity_npz(fidelity, output_path=fidelity_path)
    write_numerical_attribution_npz(attribution, output_path=attribution_path)

    loaded_fidelity = load_numerical_fidelity_npz(fidelity_path)
    loaded_attribution = load_numerical_attribution_npz(attribution_path)
    assert np.array_equal(
        loaded_fidelity.replay_generation_logits,
        fidelity.replay_generation_logits,
    )
    assert np.array_equal(
        loaded_attribution.clean_prefix_no_cache_logits,
        attribution.clean_prefix_no_cache_logits,
    )


def test_bundle_reloads_artifacts_and_recomputes_both_decisions(
    tmp_path: Path,
) -> None:
    fidelity = _fidelity_evidence()
    attribution = _attribution_evidence(fidelity)
    fidelity_path = tmp_path / "arrays" / "fidelity.npz"
    attribution_path = tmp_path / "arrays" / "attribution.npz"
    fidelity_sha = write_numerical_fidelity_npz(
        fidelity, output_path=fidelity_path
    )
    attribution_sha = write_numerical_attribution_npz(
        attribution, output_path=attribution_path
    )
    fidelity_decision = evaluate_numerical_fidelity(fidelity)
    attribution_decision = evaluate_numerical_attribution(
        fidelity=fidelity,
        evidence=attribution,
    )
    manifest = {
        "schema_version": NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "gate6g_contract_status": "invalid",
        "repeat_count": 3,
        "source_invalid_bundle": {
            "audit_failed_sha256": "a" * 64,
            "action_spectral": "b" * 64,
            "action_only_control": "c" * 64,
        },
        "artifacts": {
            "fidelity": {
                "relative_path": "arrays/fidelity.npz",
                "sha256": fidelity_sha,
            },
            "attribution": {
                "relative_path": "arrays/attribution.npz",
                "sha256": attribution_sha,
            },
        },
        "fidelity_decision": numerical_decision_payload(fidelity_decision),
        "attribution_decision": numerical_decision_payload(
            attribution_decision
        ),
        "causal_attribution_allowed": True,
    }
    manifest_path = tmp_path / "manifest.json"

    published_sha = publish_numerical_inference_manifest(
        manifest,
        output_path=manifest_path,
    )

    assert published_sha == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    decision = evaluate_numerical_inference_bundle(manifest_path)
    assert decision.bundle_valid
    assert decision.fidelity_decision is not None
    assert decision.attribution_decision is not None


def test_bundle_rejects_tampered_fidelity_artifact(tmp_path: Path) -> None:
    fidelity = _fidelity_evidence()
    fidelity_path = tmp_path / "fidelity.npz"
    fidelity_sha = write_numerical_fidelity_npz(
        fidelity,
        output_path=fidelity_path,
    )
    decision = evaluate_numerical_fidelity(fidelity)
    manifest = {
        "schema_version": NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "gate6g_contract_status": "invalid",
        "repeat_count": 3,
        "source_invalid_bundle": {
            "audit_failed_sha256": "a" * 64,
            "action_spectral": "b" * 64,
            "action_only_control": "c" * 64,
        },
        "artifacts": {
            "fidelity": {
                "relative_path": "fidelity.npz",
                "sha256": fidelity_sha,
            },
            "attribution": None,
        },
        "fidelity_decision": numerical_decision_payload(decision),
        "attribution_decision": None,
        "causal_attribution_allowed": False,
    }
    manifest_path = tmp_path / "manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fidelity_path.write_bytes(fidelity_path.read_bytes() + b"tamper")

    bundle = evaluate_numerical_inference_bundle(manifest_path)

    assert not bundle.bundle_valid
    assert "SHA漂移" in bundle.failures[0]
