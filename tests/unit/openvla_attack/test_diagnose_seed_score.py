"""Seed Score CPU runner 的静态依赖护栏。"""

from __future__ import annotations

import ast
from pathlib import Path


def test_seed_score_runner_is_cpu_only_and_does_not_construct_support() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    runner_path = (
        repository_root
        / "openvla/experiments/robot/libero/openvla_attack"
        / "diagnose_seed_score.py"
    )
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)

    forbidden_fragments = ("optimization", "feature", "oft", "libero.libero")
    assert not any(
        fragment in module
        for module in imported_modules
        for fragment in forbidden_fragments
    )
    assert "load_dense_gradient_stack" in called_names
    assert "compute_seed_score_arrays" in called_names
    assert "write_seed_score_artifact" in called_names
    assert not any("support" in name.lower() for name in called_names)


def test_repeat_runner_only_compares_existing_score_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    runner_path = (
        repository_root
        / "openvla/experiments/robot/libero/openvla_attack"
        / "compare_seed_score_repeats.py"
    )
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "load_seed_score_artifact_arrays" in called_names
    assert "compare_seed_score_stability" in called_names
    assert not any("support" in name.lower() for name in called_names)
