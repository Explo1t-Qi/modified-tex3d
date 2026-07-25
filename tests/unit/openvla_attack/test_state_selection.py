"""训练/评估初始状态隔离测试。"""

from __future__ import annotations

import pytest

from openvla.experiments.robot.libero.openvla_attack.state_selection import (
    parse_state_ids,
    select_initial_state_partition,
)


def test_parse_state_ids_supports_ranges_and_single_ids() -> None:
    assert parse_state_ids(
        "0-3,7,9",
        field_name="states",
    ) == (0, 1, 2, 3, 7, 9)


@pytest.mark.parametrize("specification", ["", "3-1", "-1", "1,1"])
def test_parse_state_ids_rejects_ambiguous_values(
    specification: str,
) -> None:
    with pytest.raises(ValueError):
        parse_state_ids(specification, field_name="states")


def test_default_attack_partition_uses_held_out_states() -> None:
    states = list(range(50))

    partition = select_initial_state_partition(
        states,
        num_train_states=10,
        num_eval_states=50,
        reserve_training_states=True,
    )

    assert partition.train_state_ids == tuple(range(10))
    assert partition.eval_state_ids == tuple(range(10, 50))
    assert partition.train_states == tuple(range(10))
    assert partition.eval_states == tuple(range(10, 50))


def test_explicit_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="重叠"):
        select_initial_state_partition(
            list(range(20)),
            num_train_states=10,
            num_eval_states=10,
            train_state_specification="0-4",
            eval_state_specification="4-9",
            require_disjoint=True,
        )


def test_clean_control_does_not_reserve_training_states() -> None:
    partition = select_initial_state_partition(
        list(range(5)),
        num_train_states=10,
        num_eval_states=3,
        reserve_training_states=False,
    )

    assert partition.train_state_ids == ()
    assert partition.eval_state_ids == (0, 1, 2)
