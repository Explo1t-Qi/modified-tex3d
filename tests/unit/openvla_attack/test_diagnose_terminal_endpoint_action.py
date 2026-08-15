"""Gate 6i GPU runner 的静态科学边界测试。

本地 CPU 环境可能没有 nvdiffrast/LIBERO，因此这里解析源码 AST，不导入 GPU
runner。数值 artifact 和 bundle 的独立复算已由相邻 CPU 单元测试覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[3]
    / "openvla/experiments/robot/libero/openvla_attack"
    / "diagnose_terminal_endpoint_action.py"
)


def _source_and_tree() -> tuple[str, ast.Module]:
    source = RUNNER.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else node.module
            names.add(module)
            names.update(alias.name for alias in node.names)
    return names


def test_runner_excludes_legacy_and_target_gradient_paths() -> None:
    source, tree = _source_and_tree()
    imports = _imported_names(tree)
    assert "openvla_attack.optimization" not in imports
    assert "feature_gradient" in source
    assert '"feature_gradient": False' in source
    assert '"wrist_gradient": False' in source
    assert '"oft_gradient": False' in source
    assert '"training_or_rollout_run": False' in source


def test_raw_gradient_uses_surface_hook_and_strict_scatter_add() -> None:
    source, _ = _source_and_tree()
    assert "capture_surface_delta_gradients" in source
    assert "capture.summed_gradient()" in source
    assert "get_render_to_geometry_mapping" in source
    assert "geometry_gradient.index_add_" in source
    # 参数 .grad 只允许清空，不能作为 raw G_s 的数据源。
    assert "parameter.grad.detach" not in source
    assert "parameter.grad.cpu" not in source


def test_counterfactual_step_reuses_formal_trainer_primitive() -> None:
    source, _ = _source_and_tree()
    assert "surface_normalized_step_(" in source
    assert "FORMAL_SURFACE_STEP" in source
    assert "GeometryVertexTextureParameterization" in source
    assert "aggregate_state_gradients" in source
    assert "support_gradient[~support.support_mask]" in source
    assert "evaluate_endpoint_step_evidence" in source


def test_runner_publishes_complete_frozen_inventory_atomically() -> None:
    source, _ = _source_and_tree()
    assert "for endpoint in ENDPOINT_NAMES" in source
    assert "for state_id in EXPECTED_STATE_IDS" in source
    assert "for arm in RESPONSE_ARMS" in source
    assert "write_endpoint_gradient_npz" in source
    assert "write_endpoint_step_npz" in source
    assert "write_endpoint_response_npz" in source
    assert "publish_terminal_endpoint_bundle" in source
    assert 'weights_only=True' in source
