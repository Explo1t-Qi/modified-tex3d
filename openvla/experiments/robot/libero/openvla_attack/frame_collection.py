"""OpenVLA 对抗纹理优化所需的训练帧采集。

训练帧不是一张普通 RGB 图像：它同时保存 MuJoCo 场景姿态、去除目标物体后的
背景、clean action token、连续动作、视觉 hidden state 和双视觉编码器的归一化
参数。原实现把这些数据的构造、环境推进与后续优化循环放在同一个函数中。

本模块通过 :class:`TrainingFrameCollector` 集中采集实现。调用方只接收结构明确
的 :class:`TrainingFrame` 列表，不需要了解等待动作、抓取窗口、光照校准或
OpenVLA clean forward 的执行顺序。
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch.cuda.amp import autocast


# 与 evaluation.py 相同，兼容脚本直接执行和测试按 package 导入两种方式。
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
)
from robot_utils import (  # noqa: E402
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)

from .action_codec import (
    FloatingArray,
    OpenVLAActionModel,
    decode_action_from_generated_ids,
)
from .configuration import FeatureObjectiveKind
from .scene import (
    SearchKeywords,
    compute_render_mvp,
    find_target_body_pose,
    render_background_without_target,
)
from .vision_features import (
    SigLIPFeatureModel,
    extract_siglip_patch_features,
)


class FrameCollectionConfig(Protocol):
    """训练帧采集从实验配置读取的最小字段集合。"""

    model_family: str
    num_steps_wait: int
    num_train_init_states: int
    train_frames_per_state: int
    frame_collect_with_policy: bool
    collect_grasp_frames: bool
    grasp_pre_frames: int
    grasp_post_frames: int
    grasp_max_steps: int
    grasp_qpos_threshold: float
    photometric_calib_frames: int
    unnorm_key: Optional[str]


class TrainingModel(
    OpenVLAActionModel,
    SigLIPFeatureModel,
    Protocol,
):
    """采集 clean token/action/feature 所需的 OpenVLA 模型 interface。"""

    device: torch.device

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """返回生成 token，shape ``[batch_size, sequence_length]``。"""
        ...

    def __call__(self, **kwargs: Any) -> Any:
        """返回至少含 ``hidden_states`` 的 Transformers 模型输出。"""
        ...


class TrainingProcessor(Protocol):
    """训练帧采集所需的 Hugging Face processor interface。"""

    tokenizer: Any

    def __call__(self, prompt: str, *, images: Image.Image) -> Any:
        """返回支持 mapping 与 ``.to(device)`` 的模型输入容器。"""
        ...


class LightingCalibrator(Protocol):
    """采集阶段使用的 renderer 光照校准 interface。"""

    def calibrate_lighting(
        self,
        mvp: torch.Tensor,
        camera_rgb: torch.Tensor,
        *,
        ema: float,
        model_rot: torch.Tensor,
    ) -> None:
        """根据当前相机帧更新 renderer 光照参数。"""
        ...


class TrainingFrame(TypedDict):
    """一次 OpenVLA 对抗优化所需的完整训练帧。

    Tensor 字段的 dtype、device 和 shape：

    - ``bg_tensor`` / ``bg_tensor_no_obj``：float32 NCHW，
      ``[1, 3, render_height, render_width]``，位于模型 device。
    - ``mvp``：float32 ``[4, 4]``；未定位到目标 body 时为 ``None``。
    - ``model_rot``：float32 ``[3, 3]``。
    - ``clean_output_ids``：整数 token，``[1, generated_sequence_length]``。
    - ``prompt_ids``：整数 token，``[1, prompt_sequence_length]``。
    - ``clean_hidden``：模型最后一层 hidden state，
      ``[1, generated_sequence_length, hidden_size]``。
    - ``clean_siglip_features``：共享 SigLIP 模式下的干净 patch features，
      ``[1, num_patches, siglip_feature_dim]``；默认 last-hidden 模式为
      ``None``。
    - ``initial_state_id`` / ``collection_step_index``：该帧对应的原始 LIBERO
      state ID 与状态内采集步，用于跨状态梯度审计，不能用局部列表下标替代。
    - 四个 mean/std：float32 ``[1, 3, 1, 1]``。

    NumPy action 字段 shape 均为 ``[action_dim]``。只有策略驱动采帧时
    ``executed_action`` 才非 ``None``。
    """

    bg_tensor: torch.Tensor
    bg_tensor_no_obj: Optional[torch.Tensor]
    mvp: Optional[torch.Tensor]
    model_rot: torch.Tensor
    clean_output_ids: torch.Tensor
    prompt_ids: torch.Tensor
    clean_action: FloatingArray
    executed_action: Optional[FloatingArray]
    clean_hidden: torch.Tensor
    clean_siglip_features: Optional[torch.Tensor]
    initial_state_id: int
    collection_step_index: int
    siglip_mean: torch.Tensor
    siglip_std: torch.Tensor
    dino_mean: torch.Tensor
    dino_std: torch.Tensor
    model_input_size: int


class _LiberoObservation(TypedDict, total=False):
    """帧采集实际读取的 LIBERO observation 字段。"""

    robot0_gripper_qpos: NDArray[np.floating[Any]]


def _rgb_numpy_to_nchw_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """把 uint8 HWC RGB 转成 device 上 ``[1, 3, H, W]`` float32 tensor。"""
    return (
        torch.from_numpy(image)
        .float()
        .to(device)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


class TrainingFrameCollector:
    """采集多个初始状态上的 OpenVLA 对抗训练帧。

    构造参数在一次 task 训练期间保持不变；:meth:`collect` 的 interface 只暴露
    task、语言描述和候选初始状态。光照校准计数在单次 ``collect`` 内跨初始状态
    共享，与重构前行为一致。
    """

    def __init__(
        self,
        *,
        cfg: FrameCollectionConfig,
        model: TrainingModel,
        processor: TrainingProcessor,
        renderer: LightingCalibrator,
        search_keywords: SearchKeywords,
        feature_objective: FeatureObjectiveKind,
        render_resolution: int = 256,
    ) -> None:
        self._cfg: FrameCollectionConfig = cfg
        self._model: TrainingModel = model
        self._processor: TrainingProcessor = processor
        self._renderer: LightingCalibrator = renderer
        self._search_keywords: SearchKeywords = search_keywords
        self._feature_objective: FeatureObjectiveKind = feature_objective
        self._render_resolution: int = render_resolution
        self._model_input_size: int = get_image_resize_size(cfg)

        device: torch.device = model.device
        # OpenVLA 当前拼接 SigLIP 与 DINOv2 两路视觉输入。四个 tensor 在所有帧
        # 间共享，只读使用；shape 均为 [1, 3, 1, 1]。
        self._siglip_mean: torch.Tensor = torch.tensor(
            [0.5, 0.5, 0.5],
            device=device,
        ).view(1, 3, 1, 1)
        self._siglip_std: torch.Tensor = torch.tensor(
            [0.5, 0.5, 0.5],
            device=device,
        ).view(1, 3, 1, 1)
        self._dino_mean: torch.Tensor = torch.tensor(
            [0.485, 0.456, 0.406],
            device=device,
        ).view(1, 3, 1, 1)
        self._dino_std: torch.Tensor = torch.tensor(
            [0.229, 0.224, 0.225],
            device=device,
        ).view(1, 3, 1, 1)

    def _build_frame(
        self,
        *,
        env: Any,
        observation: _LiberoObservation,
        task_description: str,
        state_index: int,
        initial_state_id: int,
        step_index: int,
        calibration_count: int,
    ) -> tuple[TrainingFrame, int]:
        """从当前 observation 构造一帧，并返回更新后的光照校准计数。"""
        # camera_image: uint8 HWC,
        # [render_resolution, render_resolution, 3]。
        camera_image: np.ndarray = get_libero_image(
            observation,
            self._render_resolution,
        )
        background: torch.Tensor = _rgb_numpy_to_nchw_tensor(
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
                    self._render_resolution,
                    self._render_resolution,
                ),
            )
            if target_pose.body_id != -1
            else None
        )
        if target_pose.body_id != -1:
            print(
                f"  [状态{state_index} 步{step_index}] "
                f"目标 body: '{target_pose.body_name}'"
            )
        else:
            print(f"  [状态{state_index} 步{step_index}] 未找到目标 body")

        background_without_target_numpy: Optional[np.ndarray] = (
            render_background_without_target(
                env,
                target_pose.body_id,
                self._render_resolution,
            )
            if target_pose.body_id != -1
            else None
        )
        background_without_target: Optional[torch.Tensor] = (
            _rgb_numpy_to_nchw_tensor(
                background_without_target_numpy.copy(),
                self._model.device,
            )
            if background_without_target_numpy is not None
            else None
        )

        if (
            mvp is not None
            and calibration_count < self._cfg.photometric_calib_frames
        ):
            calibration_ema: float = 0.8 if calibration_count > 0 else 0.0
            self._renderer.calibrate_lighting(
                mvp,
                background,
                ema=calibration_ema,
                model_rot=target_pose.model_matrix[:3, :3],
            )
            calibration_count += 1

        resized_image: Image.Image = Image.fromarray(camera_image).resize(
            (self._model_input_size, self._model_input_size)
        )
        prompt: str = (
            "In: What action should the robot take to "
            f"{task_description.lower()}?\nOut:"
        )
        clean_inputs: Any = self._processor(
            prompt,
            images=resized_image,
        ).to(self._model.device)
        if "pixel_values" in clean_inputs:
            clean_inputs["pixel_values"] = clean_inputs["pixel_values"].to(
                torch.bfloat16
            )

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                # clean_output_ids: [1, generated_sequence_length]。保留 prompt
                # token 与末尾 7 个 action token，供攻击目标和 hidden 对齐使用。
                clean_output_ids: torch.Tensor = self._model.generate(
                    **clean_inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )
                clean_resized: torch.Tensor = F.interpolate(
                    background,
                    size=(self._model_input_size, self._model_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # clean_pixel_values: bfloat16 [1, 6, H, W]。保留历史
                # SigLIP→DINOv2 拼接顺序，只用于复现 action/last_hidden
                # 基线；共享 SigLIP feature 由下方独立分支正确提取。
                clean_pixel_values: torch.Tensor = torch.cat(
                    (
                        (clean_resized - self._siglip_mean)
                        / self._siglip_std,
                        (clean_resized - self._dino_mean) / self._dino_std,
                    ),
                    dim=1,
                ).to(torch.bfloat16)
                clean_forward: Any = self._model(
                    input_ids=clean_output_ids,
                    attention_mask=torch.ones_like(clean_output_ids),
                    pixel_values=clean_pixel_values,
                    output_hidden_states=True,
                )
                clean_hidden: torch.Tensor = clean_forward.hidden_states[
                    -1
                ].detach()
                clean_siglip_features: Optional[torch.Tensor] = None
                if self._feature_objective == "siglip_patch":
                    # 共享目标直接进入 checkpoint 配置标识的 SigLIP 分支，不走
                    # 历史 6 通道手工拼接，避免 DINO/SigLIP 顺序错误。
                    normalized_clean_siglip: torch.Tensor = (
                        (clean_resized - self._siglip_mean)
                        / self._siglip_std
                    ).to(torch.bfloat16)
                    clean_siglip_features = (
                        extract_siglip_patch_features(
                            self._model,
                            normalized_clean_siglip,
                        ).detach()
                    )
                clean_action: FloatingArray = (
                    decode_action_from_generated_ids(
                        self._model,
                        clean_output_ids,
                        self._cfg.unnorm_key,
                    )
                )

        executed_action: Optional[FloatingArray] = None
        if self._cfg.frame_collect_with_policy:
            executed_action = normalize_gripper_action(
                clean_action.copy(),
                binarize=True,
            )
            if self._cfg.model_family == "openvla":
                executed_action = invert_gripper_action(executed_action)

        frame: TrainingFrame = {
            "bg_tensor": background,
            "bg_tensor_no_obj": background_without_target,
            "mvp": mvp,
            "model_rot": target_pose.model_matrix[:3, :3],
            "clean_output_ids": clean_output_ids,
            "prompt_ids": clean_inputs["input_ids"],
            "clean_action": clean_action,
            "executed_action": executed_action,
            "clean_hidden": clean_hidden,
            "clean_siglip_features": clean_siglip_features,
            "initial_state_id": initial_state_id,
            "collection_step_index": step_index,
            "siglip_mean": self._siglip_mean,
            "siglip_std": self._siglip_std,
            "dino_mean": self._dino_mean,
            "dino_std": self._dino_std,
            "model_input_size": self._model_input_size,
        }
        return frame, calibration_count

    def _collect_from_state(
        self,
        *,
        task: Any,
        task_description: str,
        initial_state: Any,
        state_index: int,
        initial_state_id: int,
        calibration_count: int,
    ) -> tuple[list[TrainingFrame], int]:
        """在一个初始状态上按当前采样策略收集帧。"""
        env: Any
        env, _ = get_libero_env(
            task,
            self._cfg.model_family,
            resolution=self._render_resolution,
        )
        try:
            env.reset()
            observation: _LiberoObservation = env.set_init_state(initial_state)
            env.env.sim.forward()

            if self._cfg.frame_collect_with_policy:
                for _ in range(self._cfg.num_steps_wait):
                    observation, _, _, _ = env.step(
                        get_libero_dummy_action(self._cfg.model_family)
                    )

            state_frames: deque[TrainingFrame] | list[TrainingFrame]
            loop_limit: int
            if self._cfg.collect_grasp_frames:
                state_frames = deque(maxlen=self._cfg.grasp_pre_frames)
                loop_limit = self._cfg.grasp_max_steps
            else:
                state_frames = []
                loop_limit = max(1, self._cfg.train_frames_per_state)

            grasp_detected: bool = False
            post_grasp_count: int = 0

            for step_index in range(loop_limit):
                frame: TrainingFrame
                frame, calibration_count = self._build_frame(
                    env=env,
                    observation=observation,
                    task_description=task_description,
                    state_index=state_index,
                    initial_state_id=initial_state_id,
                    step_index=step_index,
                    calibration_count=calibration_count,
                )

                if self._cfg.collect_grasp_frames:
                    gripper_qpos: Optional[np.ndarray] = observation.get(
                        "robot0_gripper_qpos"
                    )
                    gripper_closed: bool = (
                        self._cfg.frame_collect_with_policy
                        and gripper_qpos is not None
                        and float(np.max(gripper_qpos))
                        < self._cfg.grasp_qpos_threshold
                    )
                    if not grasp_detected:
                        state_frames.append(frame)
                        if (
                            gripper_closed
                            and len(state_frames) >= self._cfg.grasp_pre_frames
                        ):
                            grasp_detected = True
                    else:
                        if post_grasp_count < self._cfg.grasp_post_frames:
                            state_frames.append(frame)
                            post_grasp_count += 1
                        if post_grasp_count >= self._cfg.grasp_post_frames:
                            break
                else:
                    state_frames.append(frame)
                    executed_action: Optional[FloatingArray] = frame[
                        "executed_action"
                    ]
                    if (
                        self._cfg.frame_collect_with_policy
                        and executed_action is not None
                        and float(executed_action[-1]) > 0
                    ):
                        print(
                            f"  [状态{state_index} 步{step_index}] "
                            f"夹爪闭合，停止采帧（{step_index + 1} 帧）"
                        )
                        break

                if self._cfg.frame_collect_with_policy:
                    executed_action = frame["executed_action"]
                    if executed_action is None:
                        raise RuntimeError(
                            "策略驱动采帧缺少 executed_action"
                        )
                    observation, _, _, _ = env.step(
                        executed_action.tolist()
                    )
                else:
                    observation, _, _, _ = env.step(
                        get_libero_dummy_action(self._cfg.model_family)
                    )

            collected_frames: list[TrainingFrame] = list(state_frames)
            if self._cfg.collect_grasp_frames:
                if not grasp_detected:
                    print(
                        f"[ATTACK] 状态{state_index}: 未检测到夹爪闭合，"
                        f"使用末尾 {len(collected_frames)} 帧"
                    )
                print(
                    f"[ATTACK] 状态{state_index} grasp-window "
                    f"帧数 = {len(collected_frames)}"
                )
            return collected_frames, calibration_count
        finally:
            env.close()

    def collect(
        self,
        *,
        task: Any,
        task_description: str,
        fallback_initial_state: Any,
        initial_states: Optional[Sequence[Any]] = None,
        initial_state_ids: Optional[Sequence[int]] = None,
    ) -> list[TrainingFrame]:
        """从配置指定数量的初始状态中采集训练帧。

        ``initial_states`` 缺失或为空时，使用 ``fallback_initial_state``。候选
        状态按原顺序取前 ``cfg.num_train_init_states`` 个，不进行随机采样。
        """
        available_states: Sequence[Any] = (
            initial_states
            if initial_states is not None and len(initial_states) > 0
            else (fallback_initial_state,)
        )
        number_of_states: int = min(
            self._cfg.num_train_init_states,
            len(available_states),
        )
        if (
            initial_state_ids is not None
            and len(initial_state_ids) < number_of_states
        ):
            raise ValueError(
                "initial_state_ids 数量少于待采集状态数量："
                f"{len(initial_state_ids)} < {number_of_states}"
            )
        frames: list[TrainingFrame] = []
        calibration_count: int = 0

        for state_index in range(number_of_states):
            state_frames: list[TrainingFrame]
            state_frames, calibration_count = self._collect_from_state(
                task=task,
                task_description=task_description,
                initial_state=available_states[state_index],
                state_index=state_index,
                initial_state_id=(
                    int(initial_state_ids[state_index])
                    if initial_state_ids is not None
                    else state_index
                ),
                calibration_count=calibration_count,
            )
            frames.extend(state_frames)
        return frames
