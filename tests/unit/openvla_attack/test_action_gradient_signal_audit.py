"""Fixed Support 参数量与 Action 梯度信号只读审计测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.action_gradient_signal_audit import (  # noqa: E402
    GradientSignalCase,
    analyze_gradient_signal_cases,
)


def _case(
    *,
    endpoint: str,
    state_id: int,
    gradient: np.ndarray,
    realized_endpoint: np.ndarray | None = None,
) -> GradientSignalCase:
    endpoint_digit = "1" if endpoint == "action_spectral" else "2"
    if realized_endpoint is None:
        realized_endpoint = np.zeros_like(gradient)
    return GradientSignalCase(
        endpoint=endpoint,
        state_id=state_id,
        state_fingerprint=f"{state_id + 1:064x}",
        gradient_artifact_sha256=(endpoint_digit * 63) + str(state_id),
        dense_surface_gradient=np.asarray(gradient, dtype=np.float32),
        realized_endpoint=np.asarray(realized_endpoint, dtype=np.float32),
    )


def _complete_cases() -> list[GradientSignalCase]:
    cases: list[GradientSignalCase] = []
    for endpoint in ("action_spectral", "action_only_control"):
        scale = 1.0 if endpoint == "action_spectral" else 2.0
        cases.extend(
            [
                _case(
                    endpoint=endpoint,
                    state_id=0,
                    gradient=np.asarray(
                        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                        dtype=np.float32,
                    )
                    * np.float32(scale),
                ),
                _case(
                    endpoint=endpoint,
                    state_id=1,
                    gradient=np.asarray(
                        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                        dtype=np.float32,
                    )
                    * np.float32(scale),
                ),
            ]
        )
    return cases


def test_signal_audit_separates_total_norm_from_per_trainable_coordinate() -> None:
    """Support统计分母只能含可训练坐标，不能用补零后的全向量稀释RMS。"""

    decision = analyze_gradient_signal_cases(
        _complete_cases(),
        support_mask=np.asarray([True, False], dtype=np.bool_),
        epsilon=0.5,
        surface_step=0.1,
        expected_state_ids=(0, 1),
    )

    assert decision.audit_valid, decision.failures
    row = next(
        value
        for value in decision.rows
        if value.endpoint == "action_spectral" and value.state_id == 0
    )
    assert row.dense.vertex_count == 2
    assert row.dense.coordinate_count == 6
    assert row.support.vertex_count == 1
    assert row.support.coordinate_count == 3
    assert row.outside_support.vertex_count == 1
    assert row.dense.l2_norm == np.sqrt(5.0)
    assert row.support.l2_norm == 1.0
    assert row.support.mean_absolute == 1.0 / 3.0
    assert np.isclose(row.support.rms, 1.0 / np.sqrt(3.0))
    assert np.isclose(row.outside_support.rms, 2.0 / np.sqrt(3.0))
    assert row.gradient_energy_retention == 1.0 / 5.0
    # 原始梯度幅值不同，但两臂经同一surface normalization后Linf都达到step。
    assert np.isclose(row.dense_raw_update.linf_norm, 0.1)
    assert np.isclose(row.support_raw_update.linf_norm, 0.1)
    assert np.isclose(row.dense_executed_update.linf_norm, 0.1)
    assert np.isclose(row.support_executed_update.linf_norm, 0.1)


def test_signal_audit_reports_float32_aggregate_and_per_state_distributions() -> None:
    """aggregate必须使用正式float32 state mean，且不跨endpoint混合。"""

    decision = analyze_gradient_signal_cases(
        _complete_cases(),
        support_mask=np.asarray([True, False], dtype=np.bool_),
        epsilon=0.5,
        surface_step=0.1,
        expected_state_ids=(0, 1),
    )

    summary = next(
        value
        for value in decision.summaries
        if value.endpoint == "action_spectral"
    )
    assert summary.state_count == 2
    assert summary.aggregate.dense.l2_norm == np.sqrt(13.0)
    assert summary.aggregate.support.l2_norm == 2.0
    assert summary.aggregate.gradient_energy_retention == 4.0 / 13.0
    distributions = {
        value.name: value for value in summary.per_state_distributions
    }
    assert distributions["dense_l2_norm"].mean == (
        np.sqrt(5.0) + 5.0
    ) / 2.0
    assert np.isclose(
        distributions["support_rms"].median, 2.0 / np.sqrt(3.0)
    )
    assert "gradient_energy_retention" in distributions
    assert "support_executed_update_linf_norm" in distributions


def test_signal_audit_rejects_incomplete_inventory_and_zero_support() -> None:
    """缺state或没有可训练坐标时不得输出看似有效的描述统计。"""

    incomplete = analyze_gradient_signal_cases(
        _complete_cases()[:-1],
        support_mask=np.asarray([True, False], dtype=np.bool_),
        epsilon=0.5,
        surface_step=0.1,
        expected_state_ids=(0, 1),
    )
    assert not incomplete.audit_valid
    assert any("inventory" in failure for failure in incomplete.failures)

    empty_support = analyze_gradient_signal_cases(
        _complete_cases(),
        support_mask=np.asarray([False, False], dtype=np.bool_),
        epsilon=0.5,
        surface_step=0.1,
        expected_state_ids=(0, 1),
    )
    assert not empty_support.audit_valid
    assert any("Support" in failure for failure in empty_support.failures)
