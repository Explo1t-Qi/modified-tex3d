"""谱系数空间跨模型梯度统计的纯 CPU 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.vla_spectral_gradient_projection import (
    SpectralGradientProjectionArtifact,
    SpectralProjectionError,
    summarize_spectral_projection,
)


def _artifact() -> SpectralGradientProjectionArtifact:
    # base: float64 [S=2,K=4,RGB=3]。source/target 主视角方向完全相同；
    # wrist 为主视角的2倍，因此双视角求和后方向仍保持一致。
    base = np.arange(1, 25, dtype=np.float64).reshape(2, 4, 3)
    renderer_masks = np.asarray(
        [
            [[True, True], [False, False]],
            [[True, False], [True, False]],
        ],
        dtype=np.bool_,
    )
    observed_masks = np.asarray(
        [
            [[True, False], [False, False]],
            [[True, False], [False, False]],
        ],
        dtype=np.bool_,
    )
    return SpectralGradientProjectionArtifact(
        state_ids=np.asarray([10, 11], dtype=np.int64),
        source_primary_feature=base,
        source_primary_action=base * 2.0,
        target_primary_feature=base * 3.0,
        target_primary_action=base * 4.0,
        target_wrist_feature=base * 6.0,
        target_wrist_action=base * 8.0,
        primary_renderer_masks=renderer_masks,
        wrist_renderer_masks=renderer_masks,
        primary_observed_masks=observed_masks,
        wrist_observed_masks=observed_masks,
        metadata={"target_gradient_role": "diagnostic_only"},
    )


def test_projection_artifact_round_trip_and_summary(tmp_path: Path) -> None:
    saved_path = _artifact().save(tmp_path / "projection.npz")
    loaded = SpectralGradientProjectionArtifact.load(saved_path)
    summary = summarize_spectral_projection(
        loaded,
        source_action_weight=0.1,
        source_feature_weight=4.0,
    )

    assert loaded.num_basis == 4
    assert summary["target_gradient_role"] == (
        "diagnostic_only_not_training_or_selection"
    )
    assert summary["aggregate"]["cross_model_primary_feature_cosine"][
        "mean"
    ] == pytest.approx(1.0)
    assert summary["aggregate"][
        "source_weighted_target_combined_action_cosine"
    ]["mean"] == pytest.approx(1.0)
    assert summary["aggregate"][
        "target_wrist_primary_action_norm_ratio"
    ]["mean"] == pytest.approx(2.0)
    assert summary["aggregate"]["primary_mask_observed_recall"][
        "mean"
    ] == pytest.approx(1.0)
    assert summary["mode_energy"]["source_feature"][
        "effective_basis_count"
    ] > 1.0


def test_projection_rejects_mismatched_gradient_shape() -> None:
    artifact = _artifact()
    invalid = SpectralGradientProjectionArtifact(
        **{
            **artifact.__dict__,
            "target_wrist_action": np.ones((2, 3, 3), dtype=np.float64),
        }
    )

    with pytest.raises(SpectralProjectionError, match="相同.*shape"):
        invalid.validate()


def test_projection_rejects_zero_source_weights() -> None:
    with pytest.raises(SpectralProjectionError, match="不能同时为0"):
        summarize_spectral_projection(
            _artifact(),
            source_action_weight=0.0,
            source_feature_weight=0.0,
        )
