"""OpenVLA action token 的攻击目标函数。

OpenVLA 将连续机器人动作离散化为 256 个 token。当前 Tex3D 实现使用
“对称 action bin”作为攻击目标：干净动作属于第 ``i`` 个 bin 时，攻击目标
就是第 ``255 - i`` 个 bin。本模块只负责这一数学规则，不感知图像、renderer、
LIBERO 环境或训练循环。
"""

from __future__ import annotations

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

    三个 tensor 的第一维都是有效 action token 数 ``A``。``logits`` shape 为
    ``[A, 256]``，两个 class tensor shape 为 ``[A]``。当前 LIBERO/OpenVLA
    通常有 ``A=7``，但这里不硬编码动作维数。
    """

    logits: Tensor
    clean_classes: Tensor
    symmetric_target_classes: Tensor


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
    target_classes: Tensor = NUM_ACTION_BINS - 1 - clean_classes
    return ActionTokenLogits(
        logits=action_logits,
        clean_classes=clean_classes,
        symmetric_target_classes=target_classes,
    )


def get_attack_loss(logits: Tensor, clean_generated_token_ids: Tensor) -> Tensor:
    """计算把干净动作推向对称 action bin 的交叉熵损失。

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
