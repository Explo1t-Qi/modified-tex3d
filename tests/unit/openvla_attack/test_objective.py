"""OpenVLA 攻击目标模块的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.objective import (
    extract_action_token_logits,
    get_attack_loss,
)


def test_attack_loss_targets_the_opposite_action_bins() -> None:
    """当前 untargeted objective 应把每个 action bin 推向其对称 bin。"""
    logits = torch.zeros((1, 3, 32000), dtype=torch.float32)
    clean_labels = torch.tensor([[0, 31744, 31999]], dtype=torch.long)
    logits[0, 0, 31999] = 4.0
    logits[0, 1, 31744] = 2.0

    loss = get_attack_loss(logits, clean_labels)

    action_logits = torch.stack(
        [logits[0, 0, 31744:32000], logits[0, 1, 31744:32000]]
    )
    expected_loss = F.cross_entropy(action_logits, torch.tensor([255, 0]))
    torch.testing.assert_close(loss, expected_loss)


def test_attack_loss_is_a_differentiable_zero_without_action_tokens() -> None:
    """没有 action token 时仍应允许训练循环无条件调用 backward。"""
    logits = torch.zeros((1, 3, 32000), dtype=torch.float32, requires_grad=True)
    clean_labels = torch.tensor([[1, 2, 3]], dtype=torch.long)

    loss = get_attack_loss(logits, clean_labels)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()


def test_action_logit_extraction_reuses_tail_alignment_and_causal_shift() -> None:
    """诊断读取的位置必须与实际攻击损失完全一致。"""
    labels = torch.tensor([[5, 31744, 31999]], dtype=torch.long)
    # 比 labels 多两个前缀位置，公共函数应从尾部对齐后再 causal shift。
    logits = torch.zeros((1, 5, 32000), dtype=torch.float32)
    logits[0, 2, 31744:32000] = torch.arange(256)
    logits[0, 3, 31744:32000] = torch.arange(256) + 1000

    action_tokens = extract_action_token_logits(logits, labels)

    assert action_tokens.logits.shape == (2, 256)
    torch.testing.assert_close(
        action_tokens.logits[0],
        torch.arange(256, dtype=torch.float32),
    )
    torch.testing.assert_close(
        action_tokens.logits[1],
        torch.arange(256, dtype=torch.float32) + 1000,
    )
    assert action_tokens.clean_classes.tolist() == [0, 255]
    assert action_tokens.symmetric_target_classes.tolist() == [255, 0]
