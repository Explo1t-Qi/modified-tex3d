"""OpenVLA action token 与连续机器人动作之间的解码逻辑。

OpenVLA 的生成结果仍是语言模型词表中的 token ID。本模块集中处理 token ID、
离散 action bin、归一化动作以及数据集尺度动作之间的转换，使训练和 rollout
不需要重复了解这些编码细节。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray


Tensor: TypeAlias = torch.Tensor
FloatingArray: TypeAlias = NDArray[np.floating[Any]]
IntegerArray: TypeAlias = NDArray[np.integer[Any]]
BooleanArray: TypeAlias = NDArray[np.bool_]


class _RequiredActionNormalizationStats(TypedDict):
    """反归一化必需的分位数统计量。"""

    q01: FloatingArray
    q99: FloatingArray


class ActionNormalizationStats(_RequiredActionNormalizationStats, total=False):
    """一个数据集 action space 的归一化统计量。

    ``q01``、``q99`` 和 ``mask`` 的形状均为 ``[action_dim]``。``mask`` 缺省
    时表示所有动作维度都需要反归一化。
    """

    mask: BooleanArray


class OpenVLAActionModel(Protocol):
    """Action codec 使用的最小 OpenVLA 模型 interface。

    这里使用结构化类型：真实模型不需要继承该类，只要提供相同的属性和方法
    即可。这样 codec 不依赖具体 Transformers 模型类，也无需加载模型权重。
    """

    vocab_size: int
    bin_centers: FloatingArray  # [num_action_bins]

    def get_action_dim(self, unnorm_key: Optional[str]) -> int:
        """返回指定数据集 action space 的维数。"""
        ...

    def get_action_stats(
        self, unnorm_key: Optional[str]
    ) -> ActionNormalizationStats:
        """返回指定数据集 action space 的反归一化统计量。"""
        ...


def decode_action_from_generated_ids(
    model: OpenVLAActionModel,
    generated_ids: Tensor,
    unnorm_key: Optional[str] = None,
) -> FloatingArray:
    """把 OpenVLA 生成的 token ID 解码为连续机器人动作。

    Args:
        model: 满足 :class:`OpenVLAActionModel` interface 的模型。这里只读取词表
            大小、action bin centers、动作维数和反归一化统计量。
        generated_ids: 模型生成的 token ID，形状为
            ``[batch_size, generated_sequence_length]``，整数类型。保持重构前
            行为，只解码 batch 中第 0 个样本末尾的 ``action_dim`` 个 token。
        unnorm_key: OpenVLA 中用于选择数据集 action statistics 的键；为
            ``None`` 时沿用模型自身的缺省选择规则。

    Returns:
        连续动作数组，形状为 ``[action_dim]``，浮点类型。``mask=True`` 的
        维度被映射到 ``[q01, q99]``，其他维度保持 ``[-1, 1]`` 归一化尺度。

    Notes:
        OpenVLA 从词表尾部反向分配 action token，因此 token 到 bin 的转换为
        ``bin_index = vocab_size - token_id - 1``。越界结果会截断到有效的
        ``bin_centers`` 范围；此规则与重构前实现完全一致。
    """
    action_dim: int = model.get_action_dim(unnorm_key)

    # predicted_token_ids: [action_dim]。detach/cpu 明确切断生成图并移回 CPU，
    # 因为后续动作解码和 LIBERO env.step 使用 NumPy 数组，不参与反向传播。
    predicted_token_ids: IntegerArray = (
        generated_ids[0, -action_dim:].detach().cpu().numpy()
    )

    # discretized_action_bins: [action_dim]，每个元素都是 bin_centers 的索引。
    discretized_action_bins: IntegerArray = model.vocab_size - predicted_token_ids
    discretized_action_bins = np.clip(
        discretized_action_bins - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )

    # normalized_actions: [action_dim]，各维度处于 OpenVLA 的 [-1, 1] 尺度。
    normalized_actions: FloatingArray = model.bin_centers[discretized_action_bins]

    action_stats: ActionNormalizationStats = model.get_action_stats(unnorm_key)
    action_low: FloatingArray = np.asarray(action_stats["q01"])
    action_high: FloatingArray = np.asarray(action_stats["q99"])
    default_mask: BooleanArray = np.ones_like(action_low, dtype=np.bool_)
    unnormalize_mask: BooleanArray = np.asarray(
        action_stats.get("mask", default_mask), dtype=np.bool_
    )

    # 从 [-1, 1] 仿射映射到 [q01, q99]：
    # scaled_actions: [action_dim]
    scaled_actions: FloatingArray = (
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
    )
    decoded_actions: FloatingArray = np.where(
        unnormalize_mask, scaled_actions, normalized_actions
    )
    return decoded_actions
