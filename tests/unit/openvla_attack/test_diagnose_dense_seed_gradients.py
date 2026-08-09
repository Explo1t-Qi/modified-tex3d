"""Dense Seed GPU runner 的静态依赖护栏测试。"""

from __future__ import annotations

import ast
from pathlib import Path


def test_dense_seed_runner_reuses_new_capture_without_legacy_optimizer() -> None:
    """Runner 不得导入 optimization 或另建 legacy loss 路径。"""

    repository_root = Path(__file__).resolve().parents[3]
    runner_path = (
        repository_root
        / "openvla/experiments/robot/libero/openvla_attack"
        / "diagnose_dense_seed_gradients.py"
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

    assert not any(
        module == "optimization" or module.endswith(".optimization")
        for module in imported_modules
    )
    assert "collect_action_objective_state_capture" in called_names
    assert "write_dense_seed_state_artifact" in called_names
