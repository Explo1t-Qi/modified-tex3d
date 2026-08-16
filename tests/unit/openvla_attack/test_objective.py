"""OpenVLA 攻击目标模块的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.objective import (
    NoActionTokensError,
    extract_action_token_logits,
    legacy_symmetric_target_cross_entropy,
    untargeted_clean_action_margin_hinge,
)


def test_legacy_attack_loss_targets_the_opposite_action_bins() -> None:
    """Legacy objective 应继续把每个 action bin 推向其对称 bin。"""
    logits = torch.zeros((1, 3, 32000), dtype=torch.float32)
    clean_labels = torch.tensor([[0, 31744, 31999]], dtype=torch.long)
    logits[0, 0, 31999] = 4.0
    logits[0, 1, 31744] = 2.0

    loss = legacy_symmetric_target_cross_entropy(logits, clean_labels)

    action_logits = torch.stack(
        [logits[0, 0, 31744:32000], logits[0, 1, 31744:32000]]
    )
    expected_loss = F.cross_entropy(action_logits, torch.tensor([255, 0]))
    torch.testing.assert_close(loss, expected_loss)


def test_legacy_attack_loss_is_a_differentiable_zero_without_action_tokens(
) -> None:
    """Legacy 路径保留无 action token 时的可反传零值。"""
    logits = torch.zeros((1, 3, 32000), dtype=torch.float32, requires_grad=True)
    clean_labels = torch.tensor([[1, 2, 3]], dtype=torch.long)

    loss = legacy_symmetric_target_cross_entropy(logits, clean_labels)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()


def test_margin_hinge_uses_clean_class_against_best_other_class() -> None:
    """新目标应逐 action token 计算 clean-minus-best-other margin。"""
    logits = torch.zeros((1, 4, 32000), dtype=torch.float32)
    clean_labels = torch.tensor(
        [[7, 31744, 31745, 31999]],
        dtype=torch.long,
    )
    # causal shift 后三个 action 位置分别读取 logits 的 0、1、2 行。
    logits[0, 0, 31744] = 3.0
    logits[0, 0, 31745] = 1.0
    logits[0, 1, 31745] = 2.0
    logits[0, 1, 31746] = 4.0
    logits[0, 2, 31999] = 5.0
    logits[0, 2, 31750] = 5.0

    result = untargeted_clean_action_margin_hinge(logits, clean_labels)

    torch.testing.assert_close(
        result.margins,
        torch.tensor([2.0, -2.0, 0.0]),
    )
    torch.testing.assert_close(
        result.hinge_values,
        torch.tensor([2.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(result.loss, torch.tensor(2.0 / 3.0))
    assert result.clean_classes.tolist() == [0, 1, 255]


def test_margin_hinge_gradient_pushes_only_active_clean_margin() -> None:
    """最小化 hinge 应压低 clean logit、抬高当前 best-other logit。"""
    logits = torch.zeros(
        (1, 3, 32000),
        dtype=torch.float32,
        requires_grad=True,
    )
    clean_labels = torch.tensor([[9, 31744, 31745]], dtype=torch.long)
    with torch.no_grad():
        # 第一个位置 margin=2，第二个位置 margin=-2。
        logits[0, 0, 31744] = 3.0
        logits[0, 0, 31746] = 1.0
        logits[0, 1, 31745] = 1.0
        logits[0, 1, 31747] = 3.0

    result = untargeted_clean_action_margin_hinge(logits, clean_labels)
    result.loss.backward()

    assert logits.grad is not None
    # 两个 token 取 mean，因此激活位置的导数绝对值为 1/2。
    torch.testing.assert_close(logits.grad[0, 0, 31744], torch.tensor(0.5))
    torch.testing.assert_close(logits.grad[0, 0, 31746], torch.tensor(-0.5))
    assert float(logits.grad[0, 1, 31745]) == 0.0
    assert float(logits.grad[0, 1, 31747]) == 0.0


def test_zero_confidence_margin_is_exactly_backward_compatible() -> None:
    """显式 κ=0 必须逐值复现冻结的零置信度目标及其梯度。"""

    base_logits = torch.zeros((1, 4, 32000), dtype=torch.float32)
    clean_labels = torch.tensor(
        [[7, 31744, 31745, 31746]],
        dtype=torch.long,
    )
    base_logits[0, 0, 31744] = 3.0
    base_logits[0, 0, 31750] = 1.0
    base_logits[0, 1, 31745] = 1.0
    base_logits[0, 1, 31751] = 3.0
    base_logits[0, 2, 31746] = 2.0
    base_logits[0, 2, 31752] = 2.0
    legacy_logits = base_logits.clone().requires_grad_(True)
    zero_kappa_logits = base_logits.clone().requires_grad_(True)

    legacy = untargeted_clean_action_margin_hinge(
        legacy_logits,
        clean_labels,
    )
    zero_kappa = untargeted_clean_action_margin_hinge(
        zero_kappa_logits,
        clean_labels,
        action_margin_kappa=0.0,
    )
    legacy.loss.backward()
    zero_kappa.loss.backward()

    assert torch.equal(zero_kappa.loss, legacy.loss)
    assert torch.equal(zero_kappa.margins, legacy.margins)
    assert torch.equal(zero_kappa.hinge_values, legacy.hinge_values)
    assert torch.equal(
        zero_kappa.hinge_values > 0.0,
        legacy.hinge_values > 0.0,
    )
    assert torch.equal(zero_kappa.clean_classes, legacy.clean_classes)
    assert legacy_logits.grad is not None
    assert zero_kappa_logits.grad is not None
    assert torch.equal(zero_kappa_logits.grad, legacy_logits.grad)


def test_positive_confidence_margin_reactivates_only_shallow_crossings() -> None:
    """κ hinge 的三段语义及边界梯度必须与冻结公式一致。"""

    kappa = 4.375
    logits = torch.zeros(
        (1, 6, 32000),
        dtype=torch.float32,
        requires_grad=True,
    )
    clean_labels = torch.tensor(
        [[7, 31744, 31745, 31746, 31747, 31748]],
        dtype=torch.long,
    )
    # 五个margin依次为 2、0、-2、-κ、-(κ+1)。每行使用不同other class，
    # 避免无关的最大值tie影响被检查位置的梯度。
    margins = (2.0, 0.0, -2.0, -kappa, -(kappa + 1.0))
    with torch.no_grad():
        for position, margin in enumerate(margins):
            clean_class = position
            other_class = position + 20
            logits[0, position, 31744 + clean_class] = margin
            logits[0, position, 31744 + other_class] = 0.0

    result = untargeted_clean_action_margin_hinge(
        logits,
        clean_labels,
        action_margin_kappa=kappa,
    )
    result.loss.backward()

    torch.testing.assert_close(result.margins, torch.tensor(margins))
    torch.testing.assert_close(
        result.hinge_values,
        torch.tensor([6.375, 4.375, 2.375, 0.0, 0.0]),
    )
    assert logits.grad is not None
    # m>0、m=0 与 -κ<m<0 均激活；m=-κ 及更深 crossing 不激活。
    for position in range(3):
        assert float(logits.grad[0, position, 31744 + position]) == pytest.approx(
            0.2
        )
    for position in (3, 4):
        assert float(logits.grad[0, position, 31744 + position]) == 0.0


def test_margin_hinge_accepts_exact_tie_without_applying_pressure() -> None:
    """精确 tie 合法，ReLU 在零点不产生训练梯度。"""
    logits = torch.zeros(
        (1, 2, 32000),
        dtype=torch.float32,
        requires_grad=True,
    )
    clean_labels = torch.tensor([[0, 31744]], dtype=torch.long)
    with torch.no_grad():
        logits[0, 0, 31744] = 2.0
        logits[0, 0, 31745] = 2.0

    result = untargeted_clean_action_margin_hinge(logits, clean_labels)
    result.loss.backward()

    assert float(result.margins[0]) == 0.0
    assert float(result.loss) == 0.0
    assert logits.grad is not None
    assert int(torch.count_nonzero(logits.grad)) == 0


def test_margin_hinge_fails_without_action_tokens() -> None:
    """新目标不得继承 legacy 的静默可反传零损失。"""
    logits = torch.zeros((1, 3, 32000), dtype=torch.float32)
    clean_labels = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(NoActionTokensError):
        untargeted_clean_action_margin_hinge(logits, clean_labels)


def test_margin_hinge_promotes_bfloat16_logits_to_float32() -> None:
    """真实 autocast logits 应用 float32 计算 margin，同时保留梯度。"""
    logits = torch.zeros(
        (1, 2, 32000),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    clean_labels = torch.tensor([[0, 31744]], dtype=torch.long)
    with torch.no_grad():
        logits[0, 0, 31744] = 1.0

    result = untargeted_clean_action_margin_hinge(logits, clean_labels)
    result.loss.backward()

    assert result.margins.dtype == torch.float32
    assert result.loss.dtype == torch.float32
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16


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
