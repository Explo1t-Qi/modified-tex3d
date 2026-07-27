"""OpenVLA action token 的攻击目标函数。

OpenVLA 将连续机器人动作离散化为 256 个 token。当前 Tex3D 实现使用
“对称 action bin”作为攻击目标：干净动作属于第 ``i`` 个 bin 时，攻击目标
就是第 ``255 - i`` 个 bin。本模块只负责这一数学规则，不感知图像、renderer、
LIBERO 环境或训练循环。
"""

from __future__ import annotations

from typing import Final, TypeAlias

import torch
import torch.nn.functional as F


Tensor: TypeAlias = torch.Tensor

# OpenVLA 词表中 action token 使用半开区间 [31744, 32000)，共 256 类。
ACTION_TOKEN_START: Final[int] = 31_744
ACTION_TOKEN_END: Final[int] = 32_000
NUM_ACTION_BINS: Final[int] = ACTION_TOKEN_END - ACTION_TOKEN_START


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
    # 某些生成路径会在标签前保留额外模型输出。此处从尾部对齐 token 序列。
    aligned_logits: Tensor = logits
    if aligned_logits.shape[1] > clean_generated_token_ids.shape[1]:
        aligned_logits = aligned_logits[
            :, -clean_generated_token_ids.shape[1] :, :
        ]

    # 自回归模型中位置 t 的 logits 预测位置 t+1 的 token。
    # causal shift 后：
    # shifted_logits: [batch_size, sequence_length - 1, vocab_size]
    # shifted_labels: [batch_size, sequence_length - 1]
    shifted_logits: Tensor = aligned_logits[:, :-1, :].contiguous()
    shifted_labels: Tensor = (
        clean_generated_token_ids[:, 1:].contiguous().to(logits.device)
    )

    # action_mask: [batch_size, sequence_length - 1]，True 表示该标签是动作 token。
    action_mask: Tensor = (
        (shifted_labels >= ACTION_TOKEN_START)
        & (shifted_labels < ACTION_TOKEN_END)
        & (shifted_labels != -100)
    )
    if not action_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # 布尔索引会合并 batch 与 sequence 两个维度：
    # valid_logits: [num_action_tokens, vocab_size]
    # valid_labels: [num_action_tokens]
    valid_logits: Tensor = shifted_logits[action_mask]
    valid_labels: Tensor = shifted_labels[action_mask]

    # 只保留 action 子词表，避免普通语言 token 参与这项分类损失。
    # action_logits: [num_action_tokens, 256]
    action_logits: Tensor = valid_logits[
        :, ACTION_TOKEN_START:ACTION_TOKEN_END
    ]
    clean_token_classes: Tensor = valid_labels - ACTION_TOKEN_START
    target_token_classes: Tensor = (
        NUM_ACTION_BINS - 1 - clean_token_classes
    )

    return F.cross_entropy(action_logits, target_token_classes)
