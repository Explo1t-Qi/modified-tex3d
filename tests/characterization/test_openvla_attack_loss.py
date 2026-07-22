"""尚未从 OpenVLA 单体入口拆出的行为刻画测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACK_SCRIPT = REPO_ROOT / "openvla/experiments/robot/libero/attack_openvla.py"

# Importing the current monolithic script also imports Robosuite.  Its Numba
# decorators attempt to create caches next to the installed package, which is
# unrelated to the behavior under characterization and may be read-only.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


@cache
def _load_attack_module():
    spec = importlib.util.spec_from_file_location("tex3d_openvla_attack_current", ATTACK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OpenVLA attack module from {ATTACK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_composite_uses_foreground_mask_and_clamps_rgb() -> None:
    attack = _load_attack_module()
    foreground = torch.tensor([[[[-0.2, 0.5, 1.2], [1.0, 1.0, 1.0]]]])
    mask = torch.tensor([[[[1.0], [0.0]]]])
    background = torch.tensor([[[[0.9, 0.25]], [[0.9, 0.25]], [[0.9, 0.25]]]])

    composited = attack._composite(foreground, mask, background)

    expected = torch.tensor([[[[0.0, 0.25]], [[0.5, 0.25]], [[1.0, 0.25]]]])
    torch.testing.assert_close(composited, expected)


def test_openvla_rollout_binarizes_then_inverts_the_gripper_action() -> None:
    attack = _load_attack_module()
    action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.75], dtype=np.float32)

    normalized = attack.normalize_gripper_action(action.copy(), binarize=True)
    openvla_action = attack.invert_gripper_action(normalized)

    np.testing.assert_allclose(openvla_action[:-1], action[:-1])
    assert openvla_action[-1] == -1.0


def test_adv_sample_builder_uses_one_view_and_the_background_without_target() -> None:
    attack = _load_attack_module()

    class FakeRenderer:
        def __init__(self) -> None:
            self.calls = []

        def render(self, mvp, resolution, model_rot=None):
            self.calls.append((mvp, resolution, model_rot))
            foreground = torch.full((1, 1, 1, 3), 0.8)
            mask = torch.ones((1, 1, 1, 1))
            return foreground, mask

    renderer = FakeRenderer()
    mvp = object()
    model_rot = object()
    regular_background = torch.full((1, 3, 1, 1), 0.1)
    background_without_target = torch.full((1, 3, 1, 1), 0.2)

    samples = attack._build_adv_samples(
        renderer,
        {
            "mvp": mvp,
            "model_rot": model_rot,
            "bg_tensor": regular_background,
            "bg_tensor_no_obj": background_without_target,
        },
        RENDER_RES=64,
    )

    assert renderer.calls == [(mvp, (64, 64), model_rot)]
    assert len(samples) == 1
    torch.testing.assert_close(samples[0], torch.full((1, 3, 1, 1), 0.8))


def test_mesh_scale_parsing_preserves_current_scalar_vector_and_default_rules(
    tmp_path: Path,
) -> None:
    attack = _load_attack_module()
    cases = {
        "vector.xml": ('<mujoco><asset><mesh scale="1 2 3"/></asset></mujoco>', [1.0, 2.0, 3.0]),
        "scalar.xml": ('<mujoco><asset><mesh scale="0.5"/></asset></mujoco>', [0.5, 0.5, 0.5]),
        "default.xml": ("<mujoco><asset><mesh/></asset></mujoco>", [1.0, 1.0, 1.0]),
    }

    for filename, (xml, expected) in cases.items():
        xml_path = tmp_path / filename
        xml_path.write_text(xml)
        assert attack.parse_mesh_scale(str(xml_path)) == expected
