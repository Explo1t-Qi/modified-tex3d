"""Support Construction runners 的 CPU-only 边界回归测试。"""

from __future__ import annotations

import ast
from pathlib import Path


MODULE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "openvla/experiments/robot/libero/openvla_attack"
)


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_support_runner_replays_accepted_artifacts_without_gpu_or_optimizer() -> None:
    path = MODULE_ROOT / "diagnose_support_construction.py"
    source = path.read_text(encoding="utf-8")
    called = _called_names(path)

    assert "load_seed_score_artifact_arrays" in called
    assert "load_projection_coverage_evidence_npz" in called
    assert "construct_akita_support_candidates" in called
    assert "evaluate_support_artifacts" in called
    assert "get_model" not in called
    assert "DifferentiableRenderer" not in called
    assert "get_libero_env" not in called
    assert "backward" not in called
    assert "from openvla_attack.optimization import" not in source
    assert "nvdiffrast" in source  # 只出现在禁止导入的模块说明中。
    assert '"fixed_support_frozen": False' in source
    assert '"production_support_constructed": False' in source


def test_repeat_runner_only_compares_existing_candidate_artifacts() -> None:
    path = MODULE_ROOT / "compare_support_construction_repeats.py"
    source = path.read_text(encoding="utf-8")
    called = _called_names(path)

    assert "compare_support_repeat_artifacts" in called
    assert "construct_akita_support_candidates" not in called
    assert "load_seed_score_artifact_arrays" not in called
    assert '"threshold_registered": False' in source
    assert '"fixed_support_frozen": False' in source
