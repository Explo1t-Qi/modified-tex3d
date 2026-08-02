"""双视角谱梯度冲突统计的纯 NumPy 测试。"""

from __future__ import annotations

import numpy as np

from scripts.analyze_dual_view_spectral_gradients import (
    summarize_dual_view_gradients,
)


def test_summary_separates_direction_conflict_from_weighted_norm_dominance(
) -> None:
    # 两个 states、一个谱模态、RGB 三维。
    # State 10: primary 与 Action 同向、wrist 反向，二者均值为零。
    # State 11: 两个 Feature 都与 Action 正交，均值保留完整 Feature 范数。
    action = np.asarray(
        [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    primary = np.asarray(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    wrist = np.asarray(
        [[[-1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    combined = (primary + wrist) / 2.0

    summary = summarize_dual_view_gradients(
        state_ids=np.asarray([10, 11], dtype=np.int64),
        action_gradients=action,
        combined_feature_gradients=combined,
        primary_feature_gradients=primary,
        wrist_feature_gradients=wrist,
        action_weight=0.1,
        feature_weight=4.0,
    )

    assert summary["combined_feature_identity_max_abs_error"] == 0.0
    mean_cosines = summary["mean_cosines"]
    assert isinstance(mean_cosines, dict)
    np.testing.assert_allclose(
        [
            mean_cosines["action_to_primary_feature"],
            mean_cosines["action_to_wrist_feature"],
            mean_cosines["primary_feature_to_wrist_feature"],
        ],
        [0.5, -0.5, 0.0],
    )
    # 两个 state 的加权 Feature/Action norm 比分别为0和40，均值为20。
    np.testing.assert_allclose(
        summary["mean_weighted_feature_to_action_norm_ratio"],
        20.0,
    )
    per_state = summary["per_state"]
    assert isinstance(per_state, list)
    assert [row["state_id"] for row in per_state] == [10, 11]
