"""Gate 6j GPU runner 的静态科学边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[3]
    / "openvla/experiments/robot/libero/openvla_attack"
    / "diagnose_terminal_projection_counterfactual.py"
)


def test_runner_is_response_only_and_publishes_frozen_three_arm_bundle() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "openvla_attack.optimization" not in imports
    assert "capture_surface_delta_gradients" not in source
    assert "surface_normalized_step_" not in source
    assert ".backward(" not in source
    assert "evaluate_terminal_endpoint_bundle" in source
    assert "build_matched_box_endpoint_evidence" in source
    assert "for endpoint in ENDPOINT_NAMES" in source
    assert "for state_id in EXPECTED_STATE_IDS" in source
    assert "for arm in PROJECTION_RESPONSE_ARMS" in source
    assert "write_projection_response_npz" in source
    assert "publish_terminal_projection_bundle" in source
    assert '"gradient_recomputed": False' in source
    assert '"training_or_rollout_run": False' in source


def test_runner_copies_parent_bundle_before_collecting_child_evidence() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "shutil.copytree" in source
    assert "parent_gate6i" in source
    assert "copied_parent_sha" in source
    assert "original_parent_sha" in source
    assert "strict parent copy SHA漂移" in source
