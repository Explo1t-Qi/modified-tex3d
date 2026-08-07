"""OpenVLA 在单个 LIBERO episode 上的 rollout 状态机。

主实验入口只需要遍历 task、记录统计和保存视频；本模块隐藏一个 episode 内部
的环境生命周期、等待步、MuJoCo 相机图像转换、策略调用与动作后处理。最终
对抗纹理在 rollout 前已经安装到 MuJoCo 资产，因此评估阶段直接使用环境真实
渲染，不再用训练用 nvdiffrast renderer 重画物体。这样策略、录像与物理场景
共享同一个像素来源，也避免把几何错位或缺失深度遮挡混入纹理鲁棒性结果。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict

import numpy as np


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

from .policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    PolicyPreCropSpecification,
    resize_policy_pre_crop_canvas,
)


class RolloutConfig(Protocol):
    """episode runner 从 OpenVLA 实验配置读取的字段。"""

    model_family: str
    num_steps_wait: int
    pretrained_checkpoint: str | Path
    unnorm_key: Optional[str]
    center_crop: bool


class PolicyModel(Protocol):
    """传给 ``robot_utils.get_action`` 的策略模型占位 interface。"""


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
    # uint8 HWC，高分辨率 MuJoCo 相机帧；它不定义 policy 输入分辨率。
    replay_images: list[np.ndarray]


class LiberoEpisodeRunner:
    """运行一个 OpenVLA LIBERO episode 的深 module。

    构造函数接收同一批 task 共享的模型和 processor；:meth:`run` 只接收每个
    episode 真正变化的 task、初始状态和日志上下文。可微 renderer 严格属于
    Attack Training，不进入该评估边界。
    """

    def __init__(
        self,
        *,
        cfg: RolloutConfig,
        model: PolicyModel,
        processor: Optional[Any],
        video_resolution: int = 512,
        max_steps: int = 300,
    ) -> None:
        self._cfg: RolloutConfig = cfg
        self._model: PolicyModel = model
        self._processor: Optional[Any] = processor
        self._video_resolution: int = video_resolution
        self._max_steps: int = max_steps

    def _build_policy_image(
        self,
        observation: LiberoObservation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """从同一 observation 独立构造录像图像与 Policy Pre-Crop Canvas。

        Returns:
            ``(camera_image, policy_image)``，两者均为 uint8 HWC。
            ``camera_image`` shape 为
            ``[video_resolution, video_resolution, 3]``，直接用于录像；
            ``policy_image`` shape 为
            ``[model_input_size, model_input_size, 3]``，固定从 512 policy-source
            RGB 显式 bicubic resize 得到，直接写入 OpenVLA observation。
            当录像也为 512 时只复用同一 source 数组作为缓存；改变录像分辨率
            不得改变 policy canvas。
        """
        # camera_image: uint8 HWC, [video_resolution, video_resolution, 3]。
        camera_image: np.ndarray = get_libero_image(
            observation,
            self._video_resolution,
        )
        policy_source_image: np.ndarray = (
            camera_image
            if self._video_resolution == POLICY_SOURCE_RESOLUTION
            else get_libero_image(
                observation,
                POLICY_SOURCE_RESOLUTION,
            )
        )
        model_input_size: int = get_image_resize_size(self._cfg)
        policy_image: np.ndarray = resize_policy_pre_crop_canvas(
            policy_source_image,
            specification=PolicyPreCropSpecification(
                source_resolution=POLICY_SOURCE_RESOLUTION,
                canvas_resolution=model_input_size,
            ),
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
        """从指定初始状态运行一个 episode，运行异常直接向上层传播。

        当前行为约束：

        1. 先执行 ``cfg.num_steps_wait`` 个 dummy action。
        2. 每个策略步使用当前 observation 构造图像和机器人状态。
        3. OpenVLA gripper action 先二值化到 ``{-1, +1}``，再反转符号。
        4. 正常达到最大步数才记作任务失败；策略、环境或预处理异常不得伪装成
           攻击成功，必须使整个实验以非零状态失败。
        5. 无论成功、任务失败或异常，环境都由本 module 关闭。
        """
        env: Any
        task_description: str
        env, task_description = get_libero_env(
            task,
            self._cfg.model_family,
            # 环境 observation 分辨率属于固定 policy-source 契约；录像只从
            # 同一 observation 派生，不得反向改变模型看到的像素来源。
            resolution=POLICY_SOURCE_RESOLUTION,
        )

        success: bool = False
        replay_images: list[np.ndarray] = []
        step_index: int = 0
        try:
            env.reset()
            observation: LiberoObservation = env.set_init_state(initial_state)
            env.env.sim.forward()

            while step_index < self._max_steps + self._cfg.num_steps_wait:
                if step_index < self._cfg.num_steps_wait:
                    observation, _, _, _ = env.step(
                        get_libero_dummy_action(self._cfg.model_family)
                    )
                    step_index += 1
                    continue

                camera_image: np.ndarray
                policy_image: np.ndarray
                camera_image, policy_image = self._build_policy_image(
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
            raise RuntimeError(
                f"LIBERO rollout 运行异常：task={task_id}, "
                f"episode={episode_index}, step={step_index}"
            ) from error
        finally:
            env.close()

        return RolloutResult(
            success=success,
            task_description=task_description,
            replay_images=replay_images,
        )
