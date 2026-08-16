"""OpenVLA action token 的攻击目标函数。

本模块显式隔离两套不能混用的目标语义：

- ``legacy_symmetric_target_cross_entropy`` 只用于复现历史实验，把干净
  action bin ``i`` 推向 ``255-i``；
- ``untargeted_clean_action_margin_hinge`` 是新 Fixed-Support 候选唯一允许的
  Action objective，在固定 clean teacher-forced prefix 下压低 clean token
  相对最佳其他类别的优势。

二者共享唯一的尾部对齐、causal shift 与 action-token 提取函数。模块不感知
图像、renderer、LIBERO 环境或训练循环；零扰动 ``margin >= 0`` 的契约检查
属于审计层，不能错误地让正式训练在负 margin 时失败。
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple, TypeAlias

import torch
import torch.nn.functional as F


Tensor: TypeAlias = torch.Tensor

# OpenVLA 词表中 action token 使用半开区间 [31744, 32000)，共 256 类。
ACTION_TOKEN_START: Final[int] = 31_744
ACTION_TOKEN_END: Final[int] = 32_000
NUM_ACTION_BINS: Final[int] = ACTION_TOKEN_END - ACTION_TOKEN_START


class NoActionTokensError(ValueError):
    """clean labels 中没有可用于 Action loss 的 OpenVLA action token。"""


class ActionTokenLogits(NamedTuple):
    """从 causal LM 输出中抽出的 action 子词表分类数据。

    两个 tensor 的第一维都是有效 action token 数 ``A``。``logits`` shape 为
    ``[A, 256]``，``clean_classes`` shape 为 ``[A]``。当前 LIBERO/OpenVLA
    通常有 ``A=7``，但这里不硬编码动作维数。对称类别只作为 legacy 派生属性
    保留，避免把历史目标字段混入公共 causal alignment 数据。
    """

    logits: Tensor
    clean_classes: Tensor

    @property
    def symmetric_target_classes(self) -> Tensor:
        """返回 legacy ``255-clean_class`` 目标，shape ``[A]``。"""

        return NUM_ACTION_BINS - 1 - self.clean_classes


class UntargetedCleanActionMarginHinge(NamedTuple):
    """新 Action objective 的可微逐 token 结果。

    ``loss`` 是标量；其余 tensor shape 均为 ``[A]``。``margins`` 定义为
    ``clean_logit - best_other_logit``，``hinge_values`` 为其逐位置 ReLU。
    负 margin 在正式训练中是合法的成功状态，绝不能在本函数内 fail-fast。
    """

    loss: Tensor
    margins: Tensor
    hinge_values: Tensor
    clean_classes: Tensor


def extract_action_token_logits(
    logits: Tensor,
    clean_generated_token_ids: Tensor,
) -> ActionTokenLogits:
    """按训练损失的完全相同对齐规则抽取 action logits 与类别。

    该函数把容易出错的“尾部序列对齐 + causal shift + action mask”变为公共
    interface。攻击损失和动作响应诊断必须共同调用它，避免诊断看见的 logit
    位置与实际优化位置不同。

    Raises:
        ValueError: 当前序列中没有有效 action token，或词表不足以覆盖 256 个
            OpenVLA action token。
    """
    if logits.ndim != 3:
        raise ValueError(
            "logits 必须为 [batch, sequence, vocab]，收到 "
            f"{tuple(logits.shape)}"
        )
    if clean_generated_token_ids.ndim != 2:
        raise ValueError(
            "clean_generated_token_ids 必须为 [batch, sequence]，收到 "
            f"{tuple(clean_generated_token_ids.shape)}"
        )
    if logits.shape[0] != clean_generated_token_ids.shape[0]:
        raise ValueError("logits 与 labels 的 batch size 不一致")
    if logits.shape[2] < ACTION_TOKEN_END:
        raise ValueError(
            f"模型词表大小 {logits.shape[2]} 小于 {ACTION_TOKEN_END}"
        )

    aligned_logits: Tensor = logits
    if aligned_logits.shape[1] > clean_generated_token_ids.shape[1]:
        aligned_logits = aligned_logits[
            :, -clean_generated_token_ids.shape[1] :, :
        ]
    if aligned_logits.shape[1] != clean_generated_token_ids.shape[1]:
        raise ValueError(
            "模型 logits 序列不能短于 clean labels："
            f"{aligned_logits.shape[1]} != "
            f"{clean_generated_token_ids.shape[1]}"
        )

    shifted_logits: Tensor = aligned_logits[:, :-1, :].contiguous()
    shifted_labels: Tensor = (
        clean_generated_token_ids[:, 1:].contiguous().to(logits.device)
    )
    action_mask: Tensor = (
        (shifted_labels >= ACTION_TOKEN_START)
        & (shifted_labels < ACTION_TOKEN_END)
        & (shifted_labels != -100)
    )
    if not bool(action_mask.any().item()):
        raise NoActionTokensError(
            "clean labels 中没有有效 OpenVLA action token"
        )

    # 布尔索引合并 batch/sequence 维：[A, vocab] -> [A, 256]。
    action_logits: Tensor = shifted_logits[action_mask][
        :, ACTION_TOKEN_START:ACTION_TOKEN_END
    ]
    clean_classes: Tensor = shifted_labels[action_mask] - ACTION_TOKEN_START
    return ActionTokenLogits(
        logits=action_logits,
        clean_classes=clean_classes,
    )


def legacy_symmetric_target_cross_entropy(
    logits: Tensor,
    clean_generated_token_ids: Tensor,
) -> Tensor:
    """计算历史“对称 action bin”交叉熵，仅供 legacy 路径使用。

    Args:
        logits: 模型未归一化输出，形状为
            ``[batch_size, model_sequence_length, vocab_size]``，浮点类型。
            ``vocab_size`` 是 OpenVLA 总词表大小，必须至少覆盖
            ``ACTION_TOKEN_END``。
        clean_generated_token_ids: 干净图像对应的完整生成序列，形状为
            ``[batch_size, label_sequence_length]``，整数类型。非 action token
            以及值为 ``-100`` 的忽略位置不会参与损失。

    Returns:
        标量张量，形状为 ``[]``。存在 action token 时，它是所有有效位置的
        mean cross entropy；不存在 action token 时，保持现有行为，返回一个
        ``requires_grad=True`` 的独立零张量，使调用方仍可执行 ``backward()``。

    Notes:
        OpenVLA 使用 causal language modeling。位置 ``t`` 的 logits 用于预测
        位置 ``t + 1`` 的 token，因此计算前分别移除 logits 的最后一个位置和
        labels 的第一个位置。当模型输出序列比标签更长时，只保留末尾与标签
        等长的 logits；这与重构前实现保持一致。
    """
    try:
        action_tokens: ActionTokenLogits = extract_action_token_logits(
            logits,
            clean_generated_token_ids,
        )
    except NoActionTokensError:
        # 保留历史约定：唯一允许静默退化为可反传零值的情况是 labels 中完全
        # 没有 action token。shape/词表错误属于实现错误，必须继续抛出。
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(
        action_tokens.logits,
        action_tokens.symmetric_target_classes,
    )


def untargeted_clean_action_margin_hinge(
    logits: Tensor,
    clean_generated_token_ids: Tensor,
    *,
    action_margin_kappa: float = 0.0,
) -> UntargetedCleanActionMarginHinge:
    """计算带固定置信度 ``κ`` 的 Untargeted Clean-Action Margin hinge。

    对每个有效 action token 位置 ``a`` 计算：

    ``m_a = z[a,y_a] - max_{j != y_a} z[a,j]``

    并返回 ``mean(relu(m_a + κ))``。``κ=0`` 走原有表达式，逐值保留冻结的
    零置信度目标。正 ``κ`` 只延后 hinge 的停止边界：浅度 crossing
    ``-κ < m_a <= 0`` 继续提供梯度，``m_a <= -κ`` 停止提供梯度。
    ``clean_generated_token_ids`` 必须是同一部署输入生成后固定的完整 clean
    sequence；调用方必须把它同时作为模型 teacher-forced ``input_ids``，不得
    用 adversarial 自回归 token 改写后续前缀。

    与 legacy 路径不同，没有 action token 时直接抛
    :class:`NoActionTokensError`。本函数接受负 margin；只有零 Surface Delta
    审计才应把负值解释为输入或 causal alignment 失败。

    Args:
        logits: 浮点 ``[batch_size,model_sequence_length,vocab_size]`` 模型
            logits。
        clean_generated_token_ids: 整数
            ``[batch_size,label_sequence_length]`` clean 完整生成序列。
        action_margin_kappa: 非负有限标量 ``κ``。默认 ``0.0``，严格保持旧目标；
            当前正式干预值由实验契约在调用边界冻结，公共数学函数不硬编码该值。

    Returns:
        标量 loss、逐 token margin、逐 token hinge 和 clean classes。所有结果
        保留 autograd 图，供 Dense Seed Audit 和正式训练使用。

    Raises:
        NoActionTokensError: clean sequence 中没有有效 action token。
        ValueError: 输入 shape、词表范围、action logits 或 ``κ`` 不合法。
    """

    resolved_kappa = float(action_margin_kappa)
    if not math.isfinite(resolved_kappa) or resolved_kappa < 0.0:
        raise ValueError("action_margin_kappa必须为非负有限标量")

    action_tokens: ActionTokenLogits = extract_action_token_logits(
        logits,
        clean_generated_token_ids,
    )
    # 模型 forward 通常处于 bfloat16 autocast。margin 的减法、max 与跨 token
    # mean 统一提升到 float32，避免小优势在低精度归约中丢失；cast 保留到原始
    # logits 的 autograd 链。
    action_logits: Tensor = action_tokens.logits.float()
    if not bool(torch.isfinite(action_logits).all().item()):
        raise ValueError("action logits 包含 NaN/Inf")

    clean_classes: Tensor = action_tokens.clean_classes
    clean_logits: Tensor = action_logits.gather(
        dim=1,
        index=clean_classes.unsqueeze(1),
    ).squeeze(1)
    # masked_fill 只排除每行 clean class；amax 对并列最佳其他类使用合法的
    # 分布式子梯度，避免依赖标量 argmax 的任意 tie-break。
    clean_class_mask: Tensor = torch.zeros_like(
        action_logits,
        dtype=torch.bool,
    ).scatter(1, clean_classes.unsqueeze(1), True)
    best_other_logits: Tensor = action_logits.masked_fill(
        clean_class_mask,
        float("-inf"),
    ).amax(dim=1)
    margins: Tensor = clean_logits - best_other_logits
    # κ=0保留旧计算图与运算顺序，使旧调用和显式零值在loss、hinge及梯度上
    # 都能逐值回归，而不是仅数值近似相等。
    hinge_inputs: Tensor = (
        margins if resolved_kappa == 0.0 else margins + resolved_kappa
    )
    hinge_values: Tensor = torch.relu(hinge_inputs)
    return UntargetedCleanActionMarginHinge(
        loss=hinge_values.mean(),
        margins=margins,
        hinge_values=hinge_values,
        clean_classes=clean_classes,
    )
