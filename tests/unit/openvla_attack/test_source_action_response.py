"""源 OpenVLA 单步动作响应诊断的纯 CPU 测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.source_action_response import (
    SourceActionResponseResult,
    compute_action_response_sample,
)


def test_response_sample_distinguishes_ce_improvement_from_decision_crossing() -> None:
    """target logit 上升但仍非 argmax 时，margin 应保持负值。"""
    clean_logits = torch.zeros((2, 256), dtype=torch.float32)
    adversarial_logits = torch.zeros((2, 256), dtype=torch.float32)
    clean_classes = torch.tensor([10, 20], dtype=torch.long)
    target_classes = torch.tensor([245, 235], dtype=torch.long)

    clean_logits[0, 10] = 8.0
    clean_logits[0, 245] = 1.0
    clean_logits[1, 20] = 6.0
    clean_logits[1, 235] = 0.0
    # 第0维 target 明显提升但仍落后 clean；第1维真正成为 argmax。
    adversarial_logits[0, 10] = 8.0
    adversarial_logits[0, 245] = 7.0
    adversarial_logits[1, 20] = 2.0
    adversarial_logits[1, 235] = 5.0

    sample = compute_action_response_sample(
        clean_action_logits=clean_logits,
        adversarial_action_logits=adversarial_logits,
        processor_action_logits=clean_logits,
        clean_classes=clean_classes,
        symmetric_target_classes=target_classes,
        clean_generated_token_ids=torch.tensor([31754, 31764]),
        adversarial_generated_token_ids=torch.tensor([31754, 31979]),
        processor_generated_token_ids=torch.tensor([31754, 31764]),
        clean_actions=np.array([0.0, 0.5]),
        adversarial_actions=np.array([0.0, -0.5]),
        processor_pixel_mae=0.001,
        processor_pixel_linf=0.01,
    )

    assert sample.adversarial_target_ce < sample.clean_target_ce
    assert sample.adversarial_target_minus_best_other_logit[0] == -1.0
    assert sample.adversarial_target_minus_best_other_logit[1] == 3.0
    assert sample.token_hamming_count == 1
    assert sample.action_l2 == 1.0
    assert sample.action_linf == 1.0
    assert sample.clean_decision_margin.tolist() == [7.0, 6.0]
    assert sample.adversarial_decision_margin.tolist() == [1.0, 3.0]
    assert sample.processor_decision_margin.tolist() == [7.0, 6.0]


def test_response_result_saves_machine_readable_arrays_and_summary(
    tmp_path: Path,
) -> None:
    logits = torch.zeros((1, 256), dtype=torch.float32)
    logits[0, 255] = 2.0
    sample = compute_action_response_sample(
        clean_action_logits=torch.zeros_like(logits),
        adversarial_action_logits=logits,
        processor_action_logits=torch.zeros_like(logits),
        clean_classes=torch.tensor([0]),
        symmetric_target_classes=torch.tensor([255]),
        clean_generated_token_ids=torch.tensor([31744]),
        adversarial_generated_token_ids=torch.tensor([31999]),
        processor_generated_token_ids=torch.tensor([31744]),
        clean_actions=np.array([0.0]),
        adversarial_actions=np.array([0.25]),
        processor_pixel_mae=0.0,
        processor_pixel_linf=0.0,
    )
    result = SourceActionResponseResult(
        state_ids=np.array([10], dtype=np.int64),
        step_indices=np.array([0], dtype=np.int64),
        samples=(sample,),
        reference_path="/tmp/reference.pt",
        reference_sha256="abc123",
    )

    paths = result.save(output_directory=tmp_path, task_id=3)

    assert paths.npz_path.exists()
    assert paths.csv_path.exists()
    summary = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert summary["num_samples"] == 1
    assert summary["state_ids"] == [10]
    assert (
        summary["processor_equivalence"]["exact_token_match_fraction"]
        == 1.0
    )
    assert (
        summary["teacher_forced_first_token_consistency"]
        ["clean_match_fraction"]
        == 1.0
    )
    assert (
        summary["greedy_generation"]
        ["states_with_any_token_change_fraction"]
        == 1.0
    )
    assert (
        summary["teacher_forced_proxy"]
        ["adversarial_symmetric_target_argmax_fraction"]
        == 1.0
    )
    with np.load(paths.npz_path) as archive:
        assert archive["clean_actions"].shape == (1, 1)
        assert archive["adversarial_target_minus_best_other_logit"].shape == (
            1,
            1,
        )
        assert archive["processor_decision_margin"].shape == (1, 1)


def test_summary_reports_processor_first_divergence_margin(
    tmp_path: Path,
) -> None:
    """processor 序列分叉时应保存共享前缀位置的两侧决策 margin。"""
    clean_logits = torch.zeros((2, 256), dtype=torch.float32)
    processor_logits = torch.zeros((2, 256), dtype=torch.float32)
    clean_logits[0, 10] = 4.0
    clean_logits[1, 20] = 3.0
    clean_logits[1, 21] = 2.5
    processor_logits[0, 10] = 4.0
    processor_logits[1, 21] = 2.0
    processor_logits[1, 20] = 1.75
    sample = compute_action_response_sample(
        clean_action_logits=clean_logits,
        adversarial_action_logits=clean_logits,
        processor_action_logits=processor_logits,
        clean_classes=torch.tensor([10, 20]),
        symmetric_target_classes=torch.tensor([245, 235]),
        clean_generated_token_ids=torch.tensor([31754, 31764]),
        adversarial_generated_token_ids=torch.tensor([31754, 31764]),
        processor_generated_token_ids=torch.tensor([31754, 31765]),
        clean_actions=np.array([0.0, 0.0]),
        adversarial_actions=np.array([0.0, 0.0]),
        processor_pixel_mae=0.001,
        processor_pixel_linf=0.01,
    )
    result = SourceActionResponseResult(
        state_ids=np.array([2], dtype=np.int64),
        step_indices=np.array([0], dtype=np.int64),
        samples=(sample,),
        reference_path="/tmp/reference.pt",
        reference_sha256="abc123",
    )

    paths = result.save(output_directory=tmp_path, task_id=0)
    summary = json.loads(paths.json_path.read_text(encoding="utf-8"))
    stability = summary["processor_equivalence"]["first_divergence"]
    assert stability["num_mismatched_samples"] == 1
    assert stability["mean_token_index"] == 1.0
    assert stability["clean_margin_mean"] == 0.5
    assert stability["processor_margin_mean"] == 0.25
    assert stability["teacher_matches_generated_fraction"] == 1.0
