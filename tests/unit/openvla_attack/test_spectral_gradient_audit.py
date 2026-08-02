"""Source-only 谱基梯度审计的纯数值与产物测试。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.frame_collection import TrainingFrame
from openvla_attack.spectral_gradient_audit import (
    ObjectiveParameterGradients,
    SpectralGradientAuditor,
    summarize_spectral_gradients,
)


def test_summary_normalizes_surface_scale_and_measures_state_consistency(
    tmp_path: Path,
) -> None:
    # basis: [num_geometry_vertices=2, num_basis=2]。第 0 个模态的
    # surface L∞ 为 2，第 1 个为 1。
    basis = torch.tensor(
        [
            [2.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    # gradients: [num_samples=2, num_basis=2, rgb=3]。
    # Action 的 mode 0 跨状态同向，mode 1 完全抵消；Feature 则相反。
    action_gradients = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    feature_gradients = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    result = summarize_spectral_gradients(
        state_ids=[10, 11],
        step_indices=[0, 0],
        action_losses=[4.0, 6.0],
        feature_losses=[-1.0, -3.0],
        action_gradients=action_gradients,
        feature_gradients=feature_gradients,
        basis=basis,
        eigenvalues=torch.tensor([0.25, 0.5]),
        requested_top_k=1,
        primary_feature_losses=[-0.5, -1.5],
        wrist_feature_losses=[-1.5, -4.5],
        primary_feature_gradients=feature_gradients * 0.5,
        wrist_feature_gradients=feature_gradients * 1.5,
    )

    np.testing.assert_allclose(result.basis_linf, [2.0, 1.0])
    np.testing.assert_allclose(result.action_mean_norm, [2.0, 1.0])
    np.testing.assert_allclose(result.action_consistency, [1.0, 0.0])
    np.testing.assert_allclose(result.action_surface_score, [1.0, 1.0])
    np.testing.assert_allclose(result.action_stable_score, [1.0, 0.0])
    np.testing.assert_array_equal(result.action_ranked_indices, [0, 1])
    np.testing.assert_array_equal(result.top_action_indices, [0])

    np.testing.assert_allclose(result.feature_consistency, [0.0, 1.0])
    np.testing.assert_allclose(result.feature_stable_score, [0.0, 3.0])
    np.testing.assert_array_equal(result.feature_ranked_indices, [1, 0])
    np.testing.assert_array_equal(result.top_feature_indices, [1])
    np.testing.assert_allclose(
        result.action_cumulative_energy,
        [0.8, 1.0],
    )
    np.testing.assert_allclose(
        result.feature_cumulative_energy,
        [8.0 / 26.0, 1.0],
    )

    paths = result.save(output_directory=tmp_path, task_id=3)
    with np.load(paths.npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["feature_ranked_indices"],
            [1, 0],
        )
        assert archive["feature_gradients"].shape == (2, 2, 3)
        assert archive["primary_feature_gradients"].shape == (2, 2, 3)
        np.testing.assert_allclose(
            archive["wrist_feature_losses"],
            [-1.5, -4.5],
        )
    with paths.csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 2
    assert csv_rows[1]["feature_rank"] == "0"
    summary = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert summary["selection_scope"] == "source_openvla_only"
    assert summary["state_ids"] == [10, 11]
    assert summary["top_feature_indices"] == [1]
    assert summary["mean_action_loss"] == 5.0
    assert summary["mean_feature_loss"] == -2.0
    assert summary["dual_view_diagnostic_available"] is True
    assert summary["mean_primary_feature_loss"] == -1.0
    assert summary["mean_wrist_feature_loss"] == -3.0


class _FakeGradientProvider:
    """按 frame metadata 返回固定梯度，并跳过一个无效帧。"""

    def compute_objective_parameter_gradients(
        self,
        frame: TrainingFrame,
    ) -> Optional[ObjectiveParameterGradients]:
        if frame["mvp"] is None:
            return None
        state_scale: float = float(frame["initial_state_id"])
        gradient = torch.full((2, 3), state_scale)
        return ObjectiveParameterGradients(
            action_loss=state_scale,
            feature_loss=-state_scale,
            action_gradient=gradient,
            feature_gradient=-gradient,
        )


class _FakeSpectralRenderer:
    """记录 auditor 是否在每个样本前后都恢复零参数。"""

    def __init__(self) -> None:
        self.parameter = nn.Parameter(torch.ones((2, 3)))
        self.reset_count: int = 0

    def reset_texture(self) -> None:
        self.reset_count += 1
        with torch.no_grad():
            self.parameter.zero_()

    def get_texture_param(self) -> nn.Parameter:
        return self.parameter

    def get_texture_parameterization_name(self) -> str:
        return "spectral"

    def get_spectral_basis_and_eigenvalues(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.eye(2, dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
        )


def _audit_frame(
    *,
    state_id: int,
    step_index: int,
    has_mvp: bool,
) -> TrainingFrame:
    """构造 auditor 只读取 metadata/MVP 的最小 TypedDict。"""
    return cast(
        TrainingFrame,
        {
            "initial_state_id": state_id,
            "collection_step_index": step_index,
            "mvp": torch.eye(4) if has_mvp else None,
        },
    )


def test_auditor_resets_each_frame_skips_invalid_mvp_and_preserves_state_ids(
) -> None:
    renderer = _FakeSpectralRenderer()
    auditor = SpectralGradientAuditor(
        gradient_provider=_FakeGradientProvider(),
        renderer=renderer,
        requested_top_k=1,
    )
    result = auditor.run(
        [
            _audit_frame(state_id=7, step_index=2, has_mvp=True),
            _audit_frame(state_id=8, step_index=3, has_mvp=False),
        ]
    )

    np.testing.assert_array_equal(result.state_ids, [7])
    np.testing.assert_array_equal(result.step_indices, [2])
    assert result.action_gradients.shape == (1, 2, 3)
    # 两个输入帧各 reset 一次，finally 再 reset 一次。
    assert renderer.reset_count == 3
    torch.testing.assert_close(
        renderer.parameter,
        torch.zeros((2, 3)),
    )
