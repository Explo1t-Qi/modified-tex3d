"""OpenVLA LIBERO episode runner 的行为测试。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

# Robosuite 的 Numba decorator 默认尝试在只读 site-packages 中创建 cache；
# 这里测试的是 episode 数据流，不需要 JIT。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import openvla_attack.evaluation as evaluation
from openvla_attack.evaluation import LiberoEpisodeRunner
from openvla_attack.scene import TargetBodyPose


@dataclass
class FakeRolloutConfig:
    """覆盖 robot_utils.get_action 所需配置字段。"""

    model_family: str = "openvla"
    num_steps_wait: int = 1
    pretrained_checkpoint: str = "fake-openvla"
    unnorm_key: Optional[str] = "fake"
    center_crop: bool = False


class FakeEnvironment:
    """记录 reset、等待动作、策略动作与 close 顺序的环境 fake。"""

    def __init__(
        self,
        observation: dict[str, np.ndarray],
        *,
        successful_action_number: int = 2,
    ) -> None:
        self.observation = observation
        self.actions: list[list[float]] = []
        self.successful_action_number = successful_action_number
        self.closed: bool = False
        self.forward_calls: int = 0
        self.env = SimpleNamespace(
            sim=SimpleNamespace(forward=self._record_forward)
        )

    def _record_forward(self) -> None:
        self.forward_calls += 1

    def reset(self) -> None:
        return None

    def set_init_state(self, initial_state: object) -> dict[str, np.ndarray]:
        del initial_state
        return self.observation

    def step(
        self,
        action: list[float],
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        self.actions.append(action)
        # 第一次是等待动作；第一次策略动作立即成功。
        done = len(self.actions) == self.successful_action_number
        return self.observation, 0.0, done, {}

    def close(self) -> None:
        self.closed = True


def test_episode_runner_preserves_wait_observation_action_and_cleanup_flow(
    monkeypatch,
) -> None:
    observation = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }
    env = FakeEnvironment(observation)
    camera_image = np.full((2, 2, 3), 127, dtype=np.uint8)
    policy_observations: list[dict[str, np.ndarray]] = []

    monkeypatch.setattr(
        evaluation,
        "get_libero_env",
        lambda task, model_family, resolution: (env, "pick up the bowl"),
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_dummy_action",
        lambda model_family: [0.0] * 7,
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        lambda current_observation, resolution: camera_image,
    )
    monkeypatch.setattr(evaluation, "get_image_resize_size", lambda cfg: 2)
    monkeypatch.setattr(
        evaluation,
        "quat2axisangle",
        lambda quaternion: np.array([4.0, 5.0, 6.0], dtype=np.float32),
    )

    def fake_get_action(
        cfg: FakeRolloutConfig,
        model: object,
        policy_observation: dict[str, np.ndarray],
        task_description: str,
        *,
        processor: object,
    ) -> np.ndarray:
        del cfg, model, task_description, processor
        policy_observations.append(policy_observation)
        return np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
            dtype=np.float32,
        )

    monkeypatch.setattr(evaluation, "get_action", fake_get_action)

    runner = LiberoEpisodeRunner(
        cfg=FakeRolloutConfig(),
        model=SimpleNamespace(device=torch.device("cpu")),
        processor=object(),
        renderer=None,
        search_keywords=(("akita", "bowl"),),
        video_resolution=2,
        max_steps=3,
    )
    result = runner.run(
        task=object(),
        initial_state=object(),
        task_id=0,
        episode_index=0,
    )

    assert result.success is True
    assert result.task_description == "pick up the bowl"
    assert len(result.replay_images) == 1
    assert result.replay_images[0] is camera_image
    assert env.forward_calls == 1
    assert env.closed is True
    assert env.actions[0] == [0.0] * 7
    # OpenVLA gripper action: [0, 1] -> [-1, 1] 二值化，再执行符号反转。
    np.testing.assert_allclose(
        env.actions[1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
    )
    np.testing.assert_allclose(
        policy_observations[0]["state"],
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2]),
    )
    assert policy_observations[0]["full_image"].shape == (2, 2, 3)


def test_episode_runner_sends_composite_to_policy_but_records_camera_image(
    monkeypatch,
) -> None:
    """攻击视图只进入策略，rollout 视频仍保存原始 MuJoCo 相机帧。"""
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    env = FakeEnvironment(observation, successful_action_number=1)
    camera_image = np.zeros((2, 2, 3), dtype=np.uint8)
    policy_images: list[np.ndarray] = []

    class OpaqueForegroundRenderer:
        def render(
            self,
            mvp: torch.Tensor,
            resolution: tuple[int, int],
            *,
            model_rot: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del mvp, model_rot
            height, width = resolution
            foreground = torch.ones((1, height, width, 3))
            mask = torch.ones((1, height, width, 1))
            return foreground, mask

    monkeypatch.setattr(
        evaluation,
        "get_libero_env",
        lambda task, model_family, resolution: (env, "pick up the bowl"),
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        lambda current_observation, resolution: camera_image,
    )
    monkeypatch.setattr(evaluation, "get_image_resize_size", lambda cfg: 2)
    monkeypatch.setattr(
        evaluation,
        "quat2axisangle",
        lambda quaternion: np.zeros(3, dtype=np.float32),
    )
    monkeypatch.setattr(
        evaluation,
        "find_target_body_pose",
        lambda current_env, keywords, device: TargetBodyPose(
            model_matrix=torch.eye(4),
            body_id=2,
            body_name="akita_black_bowl",
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "compute_render_mvp",
        lambda current_env, model_matrix, resolution: torch.eye(4),
    )
    monkeypatch.setattr(
        evaluation,
        "render_background_without_target",
        lambda current_env, body_id, resolution: np.zeros(
            (resolution, resolution, 3),
            dtype=np.uint8,
        ),
    )

    def fake_get_action(
        cfg: FakeRolloutConfig,
        model: object,
        policy_observation: dict[str, np.ndarray],
        task_description: str,
        *,
        processor: object,
    ) -> np.ndarray:
        del cfg, model, task_description, processor
        policy_images.append(policy_observation["full_image"])
        return np.zeros(7, dtype=np.float32)

    monkeypatch.setattr(evaluation, "get_action", fake_get_action)

    cfg = FakeRolloutConfig(num_steps_wait=0)
    result = LiberoEpisodeRunner(
        cfg=cfg,
        model=SimpleNamespace(device=torch.device("cpu")),
        processor=object(),
        renderer=OpaqueForegroundRenderer(),
        search_keywords=(("akita", "bowl"),),
        video_resolution=2,
        max_steps=1,
    ).run(
        task=object(),
        initial_state=object(),
        task_id=0,
        episode_index=0,
    )

    assert result.success is True
    assert result.replay_images[0] is camera_image
    np.testing.assert_array_equal(
        policy_images[0],
        np.full((2, 2, 3), 255, dtype=np.uint8),
    )
