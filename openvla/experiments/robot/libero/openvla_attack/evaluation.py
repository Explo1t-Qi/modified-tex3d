"""OpenVLA 在单个 LIBERO episode 上的 rollout 状态机。

主实验入口只需要遍历 task、记录统计和保存视频；本模块隐藏一个 episode 内部
的环境生命周期、等待步、相机图像转换、可选对抗纹理合成、策略调用与动作后
处理。这样评估数据流可以使用 fake 环境在无 GPU 条件下通过同一个 interface
验证，而不必导入整个 ``attack_openvla.py``。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict

import numpy as np
import torch
from PIL import Image


# 直接执行 attack_openvla.py 与从测试导入本 module 时，Python 提供的搜索路径
# 不同。显式加入 OpenVLA 根目录和 robot 工具目录，保持两种调用方式一致。
ROBOT_EXPERIMENT_DIR: str = str(Path(__file__).resolve().parents[2])
OPENVLA_ROOT: str = str(Path(__file__).resolve().parents[4])
if ROBOT_EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_EXPERIMENT_DIR)
if OPENVLA_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_ROOT)

from libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)

from .compositing import ForegroundRenderer, render_and_composite
from .scene import (
    SearchKeywords,
    compute_render_mvp,
    find_target_body_pose,
    render_background_without_target,
)


class RolloutConfig(Protocol):
    """episode runner 从 OpenVLA 实验配置读取的字段。"""

    model_family: str
    num_steps_wait: int
    pretrained_checkpoint: str | Path
    unnorm_key: Optional[str]
    center_crop: bool


class PolicyModel(Protocol):
    """rollout 图像合成所需的最小策略模型 interface。"""

    device: torch.device


class LiberoObservation(TypedDict):
    """构造 OpenVLA observation 所需的 LIBERO 状态字段。"""

    robot0_eef_pos: np.ndarray
    robot0_eef_quat: np.ndarray
    robot0_gripper_qpos: np.ndarray


class PolicyObservation(TypedDict):
    """传给 ``robot_utils.get_action`` 的策略输入。"""

    # uint8 HWC RGB，shape [model_height, model_width, 3]。
    full_image: np.ndarray
    # float array，shape [8] = xyz(3) + axis-angle(3) + gripper qpos(2)。
    state: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    """单个 LIBERO episode 的行为结果与视频输入。"""

    success: bool
    task_description: str
    # 原始 LIBERO 相机帧，而不是策略实际看到的对抗合成帧；保持现有视频语义。
    replay_images: list[np.ndarray]


def _rgb_numpy_to_nchw_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """把 uint8 HWC RGB 转成 device 上 ``[1, 3, H, W]`` 的 float tensor。"""
    return (
        torch.from_numpy(image)
        .float()
        .to(device)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


class LiberoEpisodeRunner:
    """运行一个 OpenVLA LIBERO episode 的深 module。

    构造函数接收同一批 task 共享的模型、processor、renderer 和目标关键词；
    :meth:`run` 只接收每个 episode 真正变化的 task、初始状态和日志上下文。
    """

    def __init__(
        self,
        *,
        cfg: RolloutConfig,
        model: PolicyModel,
        processor: Optional[Any],
        renderer: Optional[ForegroundRenderer],
        search_keywords: SearchKeywords,
        video_resolution: int = 512,
        max_steps: int = 300,
    ) -> None:
        self._cfg: RolloutConfig = cfg
        self._model: PolicyModel = model
        self._processor: Optional[Any] = processor
        self._renderer: Optional[ForegroundRenderer] = renderer
        self._search_keywords: SearchKeywords = search_keywords
        self._video_resolution: int = video_resolution
        self._max_steps: int = max_steps

    def _build_policy_image(
        self,
        env: Any,
        observation: LiberoObservation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回原始录像帧与模型实际接收的 RGB 图像。

        两个返回值均为 uint8 HWC。没有 renderer 时仅执行 resize；启用攻击时，
        先在录像分辨率合成对抗物体，再 resize 到 OpenVLA 输入分辨率。
        """
        # camera_image: uint8 HWC, [video_resolution, video_resolution, 3]。
        camera_image: np.ndarray = get_libero_image(
            observation,
            self._video_resolution,
        )
        model_input_size: int = get_image_resize_size(self._cfg)

        if self._renderer is None:
            policy_image: np.ndarray = np.asarray(
                Image.fromarray(camera_image).resize(
                    (model_input_size, model_input_size)
                )
            )
            return camera_image, policy_image

        # regular_background: float32 NCHW,
        # [1, 3, video_resolution, video_resolution]。
        regular_background: torch.Tensor = _rgb_numpy_to_nchw_tensor(
            camera_image,
            self._model.device,
        )
        target_pose = find_target_body_pose(
            env,
            self._search_keywords,
            self._model.device,
        )
        mvp: Optional[torch.Tensor] = (
            compute_render_mvp(
                env,
                target_pose.model_matrix,
                resolution=(
                    self._video_resolution,
                    self._video_resolution,
                ),
            )
            if target_pose.body_id != -1
            else None
        )
        background_without_target_numpy: Optional[np.ndarray] = (
            render_background_without_target(
                env,
                target_pose.body_id,
                self._video_resolution,
            )
            if target_pose.body_id != -1
            else None
        )
        composition_background: torch.Tensor = (
            _rgb_numpy_to_nchw_tensor(
                background_without_target_numpy.copy(),
                self._model.device,
            )
            if background_without_target_numpy is not None
            else regular_background
        )

        # composited: float NCHW, [1, 3, video_resolution, video_resolution]。
        with torch.no_grad():
            composited: torch.Tensor = render_and_composite(
                self._renderer,
                composition_background,
                mvp,
                resolution=(
                    self._video_resolution,
                    self._video_resolution,
                ),
                model_rotation=target_pose.model_matrix[:3, :3],
            )
        perceived_image: np.ndarray = (
            composited[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        )
        perceived_image = perceived_image.astype(np.uint8)
        policy_image = np.asarray(
            Image.fromarray(perceived_image).resize(
                (model_input_size, model_input_size)
            )
        )
        return camera_image, policy_image

    def run(
        self,
        *,
        task: Any,
        initial_state: Any,
        task_id: int,
        episode_index: int,
    ) -> RolloutResult:
        """从指定初始状态运行一个 episode，并把异常收敛为失败结果。

        当前行为约束：

        1. 先执行 ``cfg.num_steps_wait`` 个 dummy action。
        2. 每个策略步使用当前 observation 构造图像和机器人状态。
        3. OpenVLA gripper action 先二值化到 ``{-1, +1}``，再反转符号。
        4. 单步异常打印 task/episode/step 上下文，并像原实现一样结束为失败。
        5. 无论成功、失败或异常，环境都由本 module 关闭。
        """
        env: Any
        task_description: str
        env, task_description = get_libero_env(
            task,
            self._cfg.model_family,
            resolution=self._video_resolution,
        )

        success: bool = False
        replay_images: list[np.ndarray] = []
        step_index: int = 0
        try:
            env.reset()
            observation: LiberoObservation = env.set_init_state(initial_state)
            env.env.sim.forward()

            while step_index < self._max_steps + self._cfg.num_steps_wait:
                try:
                    if step_index < self._cfg.num_steps_wait:
                        observation, _, _, _ = env.step(
                            get_libero_dummy_action(self._cfg.model_family)
                        )
                        step_index += 1
                        continue

                    camera_image: np.ndarray
                    policy_image: np.ndarray
                    camera_image, policy_image = self._build_policy_image(
                        env,
                        observation,
                    )
                    replay_images.append(camera_image)

                    # state: float array [8] = position(3) + axis-angle(3)
                    # + gripper qpos(2)。
                    robot_state: np.ndarray = np.concatenate(
                        (
                            observation["robot0_eef_pos"],
                            quat2axisangle(
                                observation["robot0_eef_quat"]
                            ),
                            observation["robot0_gripper_qpos"],
                        )
                    )
                    policy_observation: PolicyObservation = {
                        "full_image": policy_image,
                        "state": robot_state,
                    }
                    action: np.ndarray = get_action(
                        self._cfg,
                        self._model,
                        policy_observation,
                        task_description,
                        processor=self._processor,
                    )
                    action = normalize_gripper_action(
                        action,
                        binarize=True,
                    )
                    if self._cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    observation, _, success, _ = env.step(action.tolist())
                    if success:
                        break
                    step_index += 1
                except Exception as error:
                    print(
                        f"[ERROR] Task {task_id} Ep {episode_index} "
                        f"step {step_index}: {error}"
                    )
                    traceback.print_exc()
                    break
        finally:
            env.close()

        return RolloutResult(
            success=success,
            task_description=task_description,
            replay_images=replay_images,
        )
