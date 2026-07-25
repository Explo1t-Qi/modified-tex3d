"""LIBERO 初始状态的显式训练/评估划分。

谱方法 MVP 需要区分“参与纹理优化的状态”和“只用于评价攻击效果的状态”。
本模块只处理索引，不依赖 LIBERO 状态的具体数组类型，因此可以用 CPU 单测
保护数据泄漏约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class InitialStatePartition:
    """一个 task 的训练与评估状态及其原始索引。"""

    train_state_ids: tuple[int, ...]
    eval_state_ids: tuple[int, ...]
    train_states: tuple[Any, ...]
    eval_states: tuple[Any, ...]


def parse_state_ids(
    specification: Optional[str],
    *,
    field_name: str,
) -> Optional[tuple[int, ...]]:
    """解析 ``"0-3,7,9"`` 形式的非负状态 ID。

    ``None`` 表示让调用方使用默认连续划分；空字符串、倒序区间、负数和重复
    ID 均直接报错，避免实验悄悄使用与记录不一致的状态集合。
    """
    if specification is None:
        return None
    stripped_specification: str = specification.strip()
    if not stripped_specification:
        raise ValueError(f"{field_name} 不能为空字符串")

    parsed_ids: list[int] = []
    for raw_token in stripped_specification.split(","):
        token: str = raw_token.strip()
        if not token:
            raise ValueError(f"{field_name} 包含空 token")
        if "-" in token:
            range_fields: list[str] = token.split("-")
            if len(range_fields) != 2:
                raise ValueError(
                    f"{field_name} 的区间 {token!r} 无效"
                )
            start_id: int = int(range_fields[0])
            end_id: int = int(range_fields[1])
            if start_id < 0 or end_id < start_id:
                raise ValueError(
                    f"{field_name} 的区间 {token!r} 必须非负且递增"
                )
            parsed_ids.extend(range(start_id, end_id + 1))
        else:
            state_id: int = int(token)
            if state_id < 0:
                raise ValueError(f"{field_name} 不能包含负数")
            parsed_ids.append(state_id)

    if len(parsed_ids) != len(set(parsed_ids)):
        raise ValueError(f"{field_name} 包含重复状态 ID")
    return tuple(parsed_ids)


def select_initial_state_partition(
    initial_states: Sequence[Any],
    *,
    num_train_states: int,
    num_eval_states: int,
    train_state_specification: Optional[str] = None,
    eval_state_specification: Optional[str] = None,
    reserve_training_states: bool = True,
    require_disjoint: bool = True,
) -> InitialStatePartition:
    """建立可复查的训练/评估状态划分。

    默认攻击划分为前 ``num_train_states`` 个训练状态，以及紧随其后的最多
    ``num_eval_states`` 个评估状态。例如 LIBERO 50 个状态、训练 10 个、评估
    请求 50 个时，最终自然得到 train=0..9、eval=10..49。
    """
    total_states: int = len(initial_states)
    if total_states <= 0:
        raise ValueError("当前 task 没有可用初始状态")
    if num_train_states < 0:
        raise ValueError("num_train_states 不能为负数")
    if num_eval_states <= 0:
        raise ValueError("num_eval_states 必须为正数")

    explicit_train_ids: Optional[tuple[int, ...]] = parse_state_ids(
        train_state_specification,
        field_name="train_init_state_ids",
    )
    explicit_eval_ids: Optional[tuple[int, ...]] = parse_state_ids(
        eval_state_specification,
        field_name="eval_init_state_ids",
    )

    if not reserve_training_states:
        train_ids: tuple[int, ...] = ()
    elif explicit_train_ids is not None:
        if len(explicit_train_ids) > num_train_states:
            raise ValueError(
                "train_init_state_ids 数量超过 num_train_init_states；"
                "请同步增大后者"
            )
        train_ids = explicit_train_ids
    else:
        train_ids = tuple(range(min(num_train_states, total_states)))

    if explicit_eval_ids is not None:
        eval_ids: tuple[int, ...] = explicit_eval_ids[:num_eval_states]
    else:
        excluded_ids: set[int] = set(train_ids) if require_disjoint else set()
        eval_ids = tuple(
            state_id
            for state_id in range(total_states)
            if state_id not in excluded_ids
        )[:num_eval_states]

    all_selected_ids: tuple[int, ...] = train_ids + eval_ids
    out_of_range_ids: list[int] = [
        state_id
        for state_id in all_selected_ids
        if state_id >= total_states
    ]
    if out_of_range_ids:
        raise ValueError(
            f"初始状态 ID 越界（总数 {total_states}）: "
            f"{out_of_range_ids}"
        )
    overlap: set[int] = set(train_ids).intersection(eval_ids)
    if require_disjoint and overlap:
        raise ValueError(
            f"训练与评估初始状态重叠: {sorted(overlap)}"
        )
    if reserve_training_states and not train_ids:
        raise ValueError("攻击训练至少需要一个训练初始状态")
    if not eval_ids:
        raise ValueError(
            "没有剩余评估状态；请减少训练状态或显式指定评估状态"
        )

    return InitialStatePartition(
        train_state_ids=train_ids,
        eval_state_ids=eval_ids,
        train_states=tuple(initial_states[index] for index in train_ids),
        eval_states=tuple(initial_states[index] for index in eval_ids),
    )
