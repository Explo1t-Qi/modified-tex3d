"""尚未从 OpenVLA 单体入口拆出的行为刻画测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

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


def test_openvla_rollout_binarizes_then_inverts_the_gripper_action() -> None:
    attack = _load_attack_module()
    action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.75], dtype=np.float32)

    normalized = attack.normalize_gripper_action(action.copy(), binarize=True)
    openvla_action = attack.invert_gripper_action(normalized)

    np.testing.assert_allclose(openvla_action[:-1], action[:-1])
    assert openvla_action[-1] == -1.0


def test_attack_training_returns_loss_history_after_frame_collector_owns_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """训练环境移入 collector 后，训练函数不应再返回已不存在的 env。"""
    attack = _load_attack_module()

    class EmptyFrameCollector:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def collect(self, **kwargs: object) -> list[object]:
            del kwargs
            return []

    monkeypatch.setattr(attack, "TrainingFrameCollector", EmptyFrameCollector)
    cfg = SimpleNamespace(
        model_family="openvla",
        num_frames_to_attack=1,
        attack_lr=0.05,
        save_attack_artifacts=False,
    )
    model = SimpleNamespace(device=torch.device("cpu"))

    loss_history = attack.train_adversarial_texture(
        cfg=cfg,
        model=model,
        processor=object(),
        renderer=object(),
        initial_obs_state=object(),
        task=object(),
        task_description="pick up the bowl",
        save_dir=str(tmp_path),
        episode_idx=0,
        search_keywords_list=[["akita", "bowl"]],
        xml_path=tmp_path / "asset.xml",
        num_iters=0,
        init_states=[],
    )

    assert loss_history == []
