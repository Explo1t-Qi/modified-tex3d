"""OpenVLA LIBERO episode runner 的行为测试。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

# Robosuite 的 Numba decorator 默认尝试在只读 site-packages 中创建 cache；
# 这里测试的是 episode 数据流，不需要 JIT。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import openvla_attack.evaluation as evaluation
from openvla_attack.evaluation import LiberoEpisodeRunner


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
    policy_source_image = np.full((512, 512, 3), 127, dtype=np.uint8)
    policy_observations: list[dict[str, np.ndarray]] = []
    environment_resolutions: list[int] = []

    def fake_get_libero_env(
        task: object,
        model_family: str,
        resolution: int,
    ) -> tuple[FakeEnvironment, str]:
        del task, model_family
        environment_resolutions.append(resolution)
        return env, "pick up the bowl"

    monkeypatch.setattr(
        evaluation,
        "get_libero_env",
        fake_get_libero_env,
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_dummy_action",
        lambda model_family: [0.0] * 7,
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        lambda current_observation, resolution: (
            policy_source_image if resolution == 512 else camera_image
        ),
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
        model=SimpleNamespace(),
        processor=object(),
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
    assert environment_resolutions == [512]
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


def test_policy_canvas_is_independent_from_replay_resolution(
    monkeypatch,
) -> None:
    """录像分辨率变化不能改变固定 512 policy-source 语义。"""
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    replay_image = np.zeros((4, 4, 3), dtype=np.uint8)
    policy_source_image = np.arange(
        512 * 512 * 3,
        dtype=np.uint8,
    ).reshape(512, 512, 3)
    requested_resolutions: list[int] = []

    def fake_get_libero_image(
        current_observation: object,
        resolution: int,
    ) -> np.ndarray:
        del current_observation
        requested_resolutions.append(resolution)
        if resolution == 4:
            return replay_image
        if resolution == 512:
            return policy_source_image
        raise AssertionError(f"unexpected resolution: {resolution}")

    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        fake_get_libero_image,
    )
    monkeypatch.setattr(evaluation, "get_image_resize_size", lambda cfg: 2)

    runner = LiberoEpisodeRunner(
        cfg=FakeRolloutConfig(num_steps_wait=0),
        model=SimpleNamespace(),
        processor=object(),
        video_resolution=4,
        max_steps=1,
    )
    replay_image, policy_image = runner._build_policy_image(observation)

    assert replay_image.shape == (4, 4, 3)
    assert requested_resolutions == [4, 512]
    np.testing.assert_array_equal(
        policy_image,
        np.asarray(
            Image.fromarray(policy_source_image).resize(
                (2, 2),
                resample=Image.Resampling.BICUBIC,
            )
        ),
    )


def test_policy_canvas_reuses_replay_array_at_reference_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """录像恰为 512 时可以缓存复用，但仍输出独立 policy canvas。"""
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    policy_source_image = np.full((512, 512, 3), 91, dtype=np.uint8)
    requested_resolutions: list[int] = []

    def fake_get_libero_image(
        current_observation: object,
        resolution: int,
    ) -> np.ndarray:
        del current_observation
        requested_resolutions.append(resolution)
        assert resolution == 512
        return policy_source_image

    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        fake_get_libero_image,
    )
    monkeypatch.setattr(evaluation, "get_image_resize_size", lambda cfg: 2)

    runner = LiberoEpisodeRunner(
        cfg=FakeRolloutConfig(num_steps_wait=0),
        model=SimpleNamespace(),
        processor=object(),
        video_resolution=512,
        max_steps=1,
    )
    replay_image, policy_image = runner._build_policy_image(observation)

    assert replay_image is policy_source_image
    assert requested_resolutions == [512]
    assert policy_image.shape == (2, 2, 3)


def test_episode_runner_propagates_policy_error_and_closes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行异常不能伪装成可被计入攻击成功率的普通任务失败。"""
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    env = FakeEnvironment(observation, successful_action_number=99)
    camera_image = np.zeros((2, 2, 3), dtype=np.uint8)
    policy_source_image = np.zeros((512, 512, 3), dtype=np.uint8)

    monkeypatch.setattr(
        evaluation,
        "get_libero_env",
        lambda task, model_family, resolution: (env, "pick up the bowl"),
    )
    monkeypatch.setattr(
        evaluation,
        "get_libero_image",
        lambda current_observation, resolution: (
            policy_source_image if resolution == 512 else camera_image
        ),
    )
    monkeypatch.setattr(evaluation, "get_image_resize_size", lambda cfg: 2)
    monkeypatch.setattr(
        evaluation,
        "quat2axisangle",
        lambda quaternion: np.zeros(3, dtype=np.float32),
    )

    def raise_policy_error(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise RuntimeError("synthetic policy failure")

    monkeypatch.setattr(evaluation, "get_action", raise_policy_error)

    runner = LiberoEpisodeRunner(
        cfg=FakeRolloutConfig(num_steps_wait=0),
        model=SimpleNamespace(),
        processor=object(),
        video_resolution=2,
        max_steps=1,
    )

    with pytest.raises(
        RuntimeError,
        match=r"task=0, episode=0, step=0",
    ) as error_info:
        runner.run(
            task=object(),
            initial_state=object(),
            task_id=0,
            episode_index=0,
        )

    assert str(error_info.value.__cause__) == "synthetic policy failure"
    assert env.closed is True
