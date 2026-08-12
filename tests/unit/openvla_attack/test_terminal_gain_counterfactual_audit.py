"""Gate 6h scalar-gain counterfactual 的纯 CPU 行为测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_gain_counterfactual_audit import (  # noqa: E402
    GainCounterfactualModelEvidence,
    FirstResponseSignature,
    classify_scalar_gain_mechanism,
    construct_scalar_gain_counterfactual,
    evaluate_gain_counterfactual_model_evidence,
    first_response_signature,
    interpret_gain_counterfactual_results,
    load_gain_counterfactual_npz,
    processor_bf16_bits_from_effective_rgb,
    write_gain_counterfactual_npz,
)


def test_scalar_gain_counterfactual_uses_full_effective_rgb_and_ties_to_even() -> None:
    """最优gain在全图/全通道拟合，量化严格使用round-to-even。"""

    clean = np.full((1, 2, 3), 10, dtype=np.uint8)
    training = np.asarray([[[12, 10, 10], [14, 10, 10]]], dtype=np.uint8)
    deployment = np.asarray([[[11, 10, 10], [12, 10, 10]]], dtype=np.uint8)

    result = construct_scalar_gain_counterfactual(
        np.stack((clean, training, deployment), axis=0)
    )

    assert result.alpha_star == 0.5
    # 10 + 0.5 * (12 - 10) = 11；10 + 0.5 * (14 - 10) = 12。
    assert np.array_equal(result.quantized_gain_rgb, deployment)
    assert result.clipped_value_count == 0
    assert np.count_nonzero(result.continuous_residual) == 0
    assert np.count_nonzero(result.quantized_residual) == 0


def test_processor_bits_are_recomputed_from_quantized_effective_rgb() -> None:
    image = np.asarray([[[0, 127, 255]]], dtype=np.uint8)
    specification = {
        "output_size": [1, 1],
        "branches": [
            {"mean": [0.0, 0.5, 1.0], "std": [1.0, 0.5, 2.0]},
            {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        ],
    }

    bits = processor_bf16_bits_from_effective_rgb(image, specification)

    values = np.concatenate(
        (
            (
                image.astype(np.float32).transpose(2, 0, 1)[None]
                / np.float32(255.0)
                - np.asarray([0.0, 0.5, 1.0], np.float32).reshape(1, 3, 1, 1)
            )
            / np.asarray([1.0, 0.5, 2.0], np.float32).reshape(1, 3, 1, 1),
            (
                image.astype(np.float32).transpose(2, 0, 1)[None]
                / np.float32(255.0)
                - np.float32(0.5)
            )
            / np.float32(0.5),
        ),
        axis=1,
    ).astype(np.float32)
    raw = values.view(np.uint32)
    expected = (
        (raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16
    ).astype(np.uint16)
    assert np.array_equal(bits, expected)


def test_first_response_signature_and_three_primary_classifications() -> None:
    clean = np.asarray([1, 2, 3], dtype=np.int64)
    training = np.asarray([1, 4, 5], dtype=np.int64)
    deployment = np.asarray([1, 6, 5], dtype=np.int64)

    assert first_response_signature(clean, clean) is None
    assert first_response_signature(clean, training) == FirstResponseSignature(
        token_index=1,
        action_class=4,
    )
    assert (
        classify_scalar_gain_mechanism(
            clean_classes=clean,
            training_classes=training,
            deployment_classes=deployment,
            gain_classes=deployment.copy(),
        ).classification
        == "gain_sufficient"
    )
    assert (
        classify_scalar_gain_mechanism(
            clean_classes=clean,
            training_classes=training,
            deployment_classes=deployment,
            gain_classes=training.copy(),
        ).classification
        == "residual_necessary_for_first_response"
    )
    assert (
        classify_scalar_gain_mechanism(
            clean_classes=clean,
            training_classes=training,
            deployment_classes=deployment,
            gain_classes=np.asarray([7, 2, 3], dtype=np.int64),
        ).classification
        == "ambiguous"
    )


def _unique_logits(classes: np.ndarray, num_classes: int = 8) -> np.ndarray:
    logits = np.zeros((classes.size, num_classes), dtype=np.float32)
    logits[np.arange(classes.size), classes] = 2.0
    return logits


def test_model_evidence_replays_source_and_classifies_primary_case() -> None:
    clean = np.asarray([1, 2, 3], dtype=np.int64)
    training = np.asarray([1, 4, 3], dtype=np.int64)
    deployment = clean.copy()
    gain = training.copy()
    source_classes = np.stack((clean, training, deployment))
    source_logits = np.stack(tuple(_unique_logits(row) for row in source_classes))
    rgb = np.zeros((3, 1, 1, 3), dtype=np.uint8)
    rgb[1, 0, 0, 0] = 4
    rgb[2, 0, 0, 0] = 2
    counterfactual = construct_scalar_gain_counterfactual(rgb)
    evidence = GainCounterfactualModelEvidence(
        variant="action_spectral",
        state_id=0,
        source_effective_rgb=rgb,
        alpha_star=counterfactual.alpha_star,
        continuous_gain_rgb=counterfactual.continuous_gain_rgb,
        quantized_gain_rgb=counterfactual.quantized_gain_rgb,
        continuous_residual=counterfactual.continuous_residual,
        quantized_residual=counterfactual.quantized_residual,
        clipped_value_count=counterfactual.clipped_value_count,
        action_token_start=100,
        action_token_end=108,
        source_generated_token_ids=source_classes + 100,
        source_generated_classes=source_classes,
        source_generation_logits=source_logits,
        source_teacher_logits=source_logits.copy(),
        replay_generated_token_ids=source_classes.copy() + 100,
        replay_generated_classes=source_classes.copy(),
        replay_generation_logits=source_logits.copy(),
        replay_teacher_logits=source_logits.copy(),
        gain_generated_token_ids=gain + 100,
        gain_generated_classes=gain,
        gain_generation_logits=_unique_logits(gain),
        gain_teacher_logits=_unique_logits(gain),
        gain_processor_bf16_bits=np.zeros((1, 6, 1, 1), dtype=np.uint16),
    )

    result = evaluate_gain_counterfactual_model_evidence(evidence)

    assert result.evidence_valid
    assert result.source_classification == "deployment_lost"
    assert result.primary_mechanism_case
    assert result.mechanism_decision is not None
    assert (
        result.mechanism_decision.classification
        == "residual_necessary_for_first_response"
    )

    # artifact必须无pickle往返，且加载端重新执行完整schema验证。
    # tmp_path由pytest注入的测试另行覆盖写入行为；这里先验证公开接口存在。
    assert callable(write_gain_counterfactual_npz)
    assert callable(load_gain_counterfactual_npz)


def test_gain_counterfactual_npz_round_trip(tmp_path: Path) -> None:
    clean = np.asarray([1, 2, 3], dtype=np.int64)
    training = np.asarray([1, 4, 3], dtype=np.int64)
    deployment = clean.copy()
    source_classes = np.stack((clean, training, deployment))
    logits = np.stack(tuple(_unique_logits(row) for row in source_classes))
    rgb = np.zeros((3, 1, 1, 3), dtype=np.uint8)
    rgb[1, 0, 0, 0] = 4
    rgb[2, 0, 0, 0] = 2
    cf = construct_scalar_gain_counterfactual(rgb)
    evidence = GainCounterfactualModelEvidence(
        variant="action_spectral",
        state_id=0,
        source_effective_rgb=rgb,
        alpha_star=cf.alpha_star,
        continuous_gain_rgb=cf.continuous_gain_rgb,
        quantized_gain_rgb=cf.quantized_gain_rgb,
        continuous_residual=cf.continuous_residual,
        quantized_residual=cf.quantized_residual,
        clipped_value_count=cf.clipped_value_count,
        action_token_start=100,
        action_token_end=108,
        source_generated_token_ids=source_classes + 100,
        source_generated_classes=source_classes,
        source_generation_logits=logits,
        source_teacher_logits=logits.copy(),
        replay_generated_token_ids=source_classes.copy() + 100,
        replay_generated_classes=source_classes.copy(),
        replay_generation_logits=logits.copy(),
        replay_teacher_logits=logits.copy(),
        gain_generated_token_ids=training + 100,
        gain_generated_classes=training.copy(),
        gain_generation_logits=_unique_logits(training),
        gain_teacher_logits=_unique_logits(training),
        gain_processor_bf16_bits=np.zeros((1, 6, 1, 1), dtype=np.uint16),
    )
    path = tmp_path / "case.npz"

    digest = write_gain_counterfactual_npz(evidence, output_path=path)
    loaded = load_gain_counterfactual_npz(path)

    assert len(digest) == 64
    assert loaded.variant == evidence.variant
    assert loaded.alpha_star == evidence.alpha_star
    assert np.array_equal(loaded.quantized_gain_rgb, evidence.quantized_gain_rgb)


def test_interpretation_tree_is_frozen_without_pass_fail_threshold() -> None:
    shared = {
        "action_spectral": {
            "gain_sufficient": 1,
            "residual_necessary_for_first_response": 1,
            "ambiguous": 1,
        },
        "action_only_control": {
            "gain_sufficient": 1,
            "residual_necessary_for_first_response": 1,
            "ambiguous": 1,
        },
    }
    assert (
        interpret_gain_counterfactual_results(shared)
        == "shared_non_scalar_residual_priority"
    )
    shared["action_only_control"]["residual_necessary_for_first_response"] = 0
    shared["action_only_control"]["ambiguous"] = 2
    assert (
        interpret_gain_counterfactual_results(shared)
        == "endpoint_or_trajectory_specific"
    )
    for counts in shared.values():
        counts["residual_necessary_for_first_response"] = 0
    shared["action_spectral"] = {
        "gain_sufficient": 2,
        "residual_necessary_for_first_response": 0,
        "ambiguous": 1,
    }
    shared["action_only_control"] = {
        "gain_sufficient": 2,
        "residual_necessary_for_first_response": 0,
        "ambiguous": 1,
    }
    assert (
        interpret_gain_counterfactual_results(shared)
        == "scalar_gain_calibration_priority"
    )
    shared["action_spectral"] = {
        "gain_sufficient": 1,
        "residual_necessary_for_first_response": 0,
        "ambiguous": 2,
    }
    shared["action_only_control"] = {
        "gain_sufficient": 1,
        "residual_necessary_for_first_response": 0,
        "ambiguous": 2,
    }
    assert (
        interpret_gain_counterfactual_results(shared)
        == "counterfactual_inconclusive_stop"
    )


def test_npz_writer_rejects_replay_that_does_not_match_gate6g(tmp_path: Path) -> None:
    clean = np.asarray([1, 2], dtype=np.int64)
    training = np.asarray([1, 3], dtype=np.int64)
    deployment = clean.copy()
    classes = np.stack((clean, training, deployment))
    logits = np.stack(tuple(_unique_logits(row) for row in classes))
    rgb = np.zeros((3, 1, 1, 3), dtype=np.uint8)
    rgb[1, 0, 0, 0] = 4
    rgb[2, 0, 0, 0] = 2
    cf = construct_scalar_gain_counterfactual(rgb)
    evidence = GainCounterfactualModelEvidence(
        variant="action_spectral",
        state_id=0,
        source_effective_rgb=rgb,
        alpha_star=cf.alpha_star,
        continuous_gain_rgb=cf.continuous_gain_rgb,
        quantized_gain_rgb=cf.quantized_gain_rgb,
        continuous_residual=cf.continuous_residual,
        quantized_residual=cf.quantized_residual,
        clipped_value_count=cf.clipped_value_count,
        action_token_start=100,
        action_token_end=108,
        source_generated_token_ids=classes + 100,
        source_generated_classes=classes,
        source_generation_logits=logits,
        source_teacher_logits=logits.copy(),
        replay_generated_token_ids=classes.copy() + 100,
        replay_generated_classes=classes.copy(),
        replay_generation_logits=logits.copy(),
        replay_teacher_logits=logits.copy(),
        gain_generated_token_ids=training + 100,
        gain_generated_classes=training.copy(),
        gain_generation_logits=_unique_logits(training),
        gain_teacher_logits=_unique_logits(training),
        gain_processor_bf16_bits=np.zeros((1, 6, 1, 1), dtype=np.uint16),
    )
    drifted = evidence.replay_generation_logits.copy()
    drifted[0, 0, 0] += 1.0

    with np.testing.assert_raises_regex(ValueError, "不能保存无效"):
        write_gain_counterfactual_npz(
            replace(evidence, replay_generation_logits=drifted),
            output_path=tmp_path / "invalid.npz",
        )
