"""尚未从 OpenVLA 单体入口拆出的行为刻画测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np


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


def test_openvla_rollout_binarizes_then_inverts_the_gripper_action() -> None:
    attack = _load_attack_module()
    action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.75], dtype=np.float32)

    normalized = attack.normalize_gripper_action(action.copy(), binarize=True)
    openvla_action = attack.invert_gripper_action(normalized)

    np.testing.assert_allclose(openvla_action[:-1], action[:-1])
    assert openvla_action[-1] == -1.0
