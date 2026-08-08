"""OpenVLA 对抗纹理优化所需的训练帧采集。

训练帧不是一张普通 RGB 图像：它同时保存 MuJoCo 场景姿态、去除目标物体后的
背景、clean action token、连续动作、视觉 hidden state 和真实 processor 对照
输入。原实现把这些数据的构造、环境推进与后续优化循环放在同一个函数中。

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
    get_libero_wrist_image,
)
from robot_utils import (  # noqa: E402
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    ensure_trailing_empty_token,
)

from .action_codec import (
    FloatingArray,
    OpenVLAActionModel,
    decode_action_from_generated_ids,
)
from .compositing import MultiInstanceViewFrame, TextureRenderInstance
from .configuration import FeatureObjectiveKind, FeatureViewModeKind
from .image_preprocessing import DifferentiableOpenVLAImageProcessor
from .policy_view import (
    DifferentiableDeploymentViewStages,
    DifferentiablePolicyViewTransform,
    build_policy_view_transform,
)
from .scene import (
    SearchKeywords,
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_pose,
    find_target_body_poses,
    render_background_without_target,
    render_background_without_targets,
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
    image_processor: Any

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
    - ``processor_pixel_values``：真实 checkpoint processor 在同一 clean RGB
      上产生的 bfloat16 fused 输入，``[1, 6, model_height, model_width]``；仅作
      一致性诊断，不进入攻击反向传播。

    NumPy action 字段 shape 均为 ``[action_dim]``。只有策略驱动采帧时
    ``executed_action`` 才非 ``None``。
    """

    bg_tensor: torch.Tensor  # float32 [1, 3, H, W]，原始相机画面
    # float32 [1, 3, H, W]，移除目标物体后的合成背景，避免前景重影。
    bg_tensor_no_obj: Optional[torch.Tensor]
    mvp: Optional[torch.Tensor]  # float32 [4, 4]，mesh 到相机裁剪空间
    model_rot: torch.Tensor  # float32 [3, 3]，用于旋转法线并计算光照
    # int64 [1, prompt_length + action_dim]
    clean_output_ids: torch.Tensor
    prompt_ids: torch.Tensor  # int64 [1, prompt_length]
    clean_action: FloatingArray  # float [action_dim]
    executed_action: Optional[FloatingArray]
    # [1, sequence_length, hidden_dim]，last-hidden objective 的干净参照。
    clean_hidden: torch.Tensor
    # [1, num_patches, feature_dim]，SigLIP objective 的干净参照。
    clean_siglip_features: Optional[torch.Tensor]
    # 双视角模式下依次保存 primary/wrist；默认单视角模式为空 tuple。
    shared_texture_views: tuple[MultiInstanceViewFrame, ...]
    initial_state_id: int
    collection_step_index: int
    processor_pixel_values: torch.Tensor


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
        image_preprocessor: DifferentiableOpenVLAImageProcessor,
        search_keywords: SearchKeywords,
        feature_objective: FeatureObjectiveKind,
        feature_view_mode: FeatureViewModeKind = "primary",
        render_resolution: int = 512,
        policy_view_transform: Optional[
            DifferentiablePolicyViewTransform
        ] = None,
    ) -> None:
        self._cfg: FrameCollectionConfig = cfg
        self._model: TrainingModel = model
        self._processor: TrainingProcessor = processor
        self._renderer: LightingCalibrator = renderer
        self._image_preprocessor: DifferentiableOpenVLAImageProcessor = (
            image_preprocessor
        )
        self._search_keywords: SearchKeywords = search_keywords
        self._feature_objective: FeatureObjectiveKind = feature_objective
        self._feature_view_mode: FeatureViewModeKind = feature_view_mode
        if (
            feature_view_mode == "primary_wrist"
            and feature_objective != "siglip_patch"
        ):
            raise ValueError(
                "primary_wrist 只支持 feature_objective='siglip_patch'"
            )
        self._render_resolution: int = render_resolution
        expected_input_size: int = get_image_resize_size(cfg)
        if self._image_preprocessor.output_size != (
            expected_input_size,
            expected_input_size,
        ):
            raise ValueError(
                "checkpoint processor 输出尺寸与 OpenVLA policy 配置不一致："
                f"{self._image_preprocessor.output_size} != "
                f"{(expected_input_size, expected_input_size)}"
            )
        self._policy_view_transform = (
            policy_view_transform
            if policy_view_transform is not None
            else build_policy_view_transform(
                source_resolution=render_resolution,
                model_input_resolution=expected_input_size,
            )
        )
        if (
            self._policy_view_transform.specification.policy_canvas
            .source_resolution
            != render_resolution
        ):
            raise ValueError(
                "collector resolution 与 Policy Source specification 不一致"
            )

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

        wrist_image: Optional[np.ndarray] = None
        wrist_background: Optional[torch.Tensor] = None
        if self._feature_view_mode == "primary_wrist":
            wrist_image = get_libero_wrist_image(
                observation,
                self._render_resolution,
            )
            wrist_background = _rgb_numpy_to_nchw_tensor(
                wrist_image,
                self._model.device,
            )

        shared_target_poses: tuple[TargetBodyPose, ...] = ()
        if self._feature_view_mode == "primary_wrist":
            shared_target_poses = find_target_body_poses(
                env,
                self._search_keywords,
                self._model.device,
            )
        target_pose: TargetBodyPose = (
            shared_target_poses[0]
            if shared_target_poses
            else find_target_body_pose(
                env,
                self._search_keywords,
                self._model.device,
            )
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
            if self._feature_view_mode == "primary_wrist":
                print(
                    f"  [状态{state_index} 步{step_index}] 共享纹理实例: "
                    f"{[pose.body_name for pose in shared_target_poses]}"
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

        # 双视角训练必须移除所有共享 PNG 的物体实例，否则未被当前 renderer
        # 覆盖的原碗会残留在背景中，形成 clean/adversarial 重影。
        shared_primary_background_without: Optional[torch.Tensor] = None
        shared_wrist_background_without: Optional[torch.Tensor] = None
        if shared_target_poses:
            shared_body_ids: tuple[int, ...] = tuple(
                pose.body_id for pose in shared_target_poses
            )
            shared_primary_numpy: Optional[np.ndarray] = (
                render_background_without_targets(
                    env,
                    shared_body_ids,
                    self._render_resolution,
                    camera_name="agentview",
                )
            )
            shared_wrist_numpy: Optional[np.ndarray] = (
                render_background_without_targets(
                    env,
                    shared_body_ids,
                    self._render_resolution,
                    camera_name="robot0_eye_in_hand",
                )
            )
            if shared_primary_numpy is not None:
                shared_primary_background_without = (
                    _rgb_numpy_to_nchw_tensor(
                        shared_primary_numpy.copy(),
                        self._model.device,
                    )
                )
            if shared_wrist_numpy is not None:
                shared_wrist_background_without = (
                    _rgb_numpy_to_nchw_tensor(
                        shared_wrist_numpy.copy(),
                        self._model.device,
                    )
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

        # clean label 与攻击路径共享完整 Deployment Effective View。processor
        # 接收 exact uint8 effective RGB；可微实现接收同一 stage 的 BPDA tensor。
        clean_deployment_stages: DifferentiableDeploymentViewStages = (
            self._policy_view_transform.build_stages(background)
        )
        clean_effective_view: torch.Tensor = (
            clean_deployment_stages.effective_view
        )
        clean_effective_uint8: np.ndarray = (
            clean_effective_view.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)[0]
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        clean_image: Image.Image = Image.fromarray(clean_effective_uint8)
        prompt: str = (
            "In: What action should the robot take to "
            f"{task_description.lower()}?\nOut:"
        )
        clean_inputs: Any = self._processor(
            prompt,
            images=clean_image,
        ).to(self._model.device)
        ensure_trailing_empty_token(clean_inputs)
        if "pixel_values" not in clean_inputs:
            raise RuntimeError("OpenVLA processor 输出缺少 pixel_values")
        processor_pixel_values: torch.Tensor = clean_inputs[
            "pixel_values"
        ].to(torch.bfloat16).detach()
        clean_pixel_values: torch.Tensor = (
            self._image_preprocessor.build_fused_pixel_values(
                clean_effective_view
            )
            .to(torch.bfloat16)
        )
        clean_inputs["pixel_values"] = clean_pixel_values

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                # clean_output_ids: [1, generated_sequence_length]。保留 prompt
                # token 与末尾 7 个 action token，供攻击目标和 hidden 对齐使用。
                clean_output_ids: torch.Tensor = self._model.generate(
                    **clean_inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )  # int64 [1, prompt_sequence_length + action_dim]
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
                wrist_clean_siglip_features: Optional[torch.Tensor] = None
                if self._feature_objective == "siglip_patch":
                    # 共享目标直接进入 checkpoint 配置标识的 SigLIP 分支，不走
                    # 历史 6 通道手工拼接，避免 DINO/SigLIP 顺序错误。
                    siglip_channel_start: int = (
                        3 * self._image_preprocessor.siglip_index
                    )
                    normalized_clean_siglip: torch.Tensor = (
                        clean_pixel_values[
                            :,
                            siglip_channel_start : siglip_channel_start + 3,
                        ].to(torch.bfloat16)
                    )
                    clean_siglip_features = (
                        extract_siglip_patch_features(
                            self._model,
                            normalized_clean_siglip,
                        ).detach()
                    )
                    if self._feature_view_mode == "primary_wrist":
                        if wrist_background is None:
                            raise RuntimeError(
                                "双视角采集缺少 wrist_background"
                            )
                        normalized_wrist_siglip: torch.Tensor = (
                            self._image_preprocessor
                            .build_siglip_pixel_values(
                                self._policy_view_transform
                                .build_effective_view(wrist_background)
                            )
                            .to(torch.bfloat16)
                        )
                        wrist_clean_siglip_features = (
                            extract_siglip_patch_features(
                                self._model,
                                normalized_wrist_siglip,
                            ).detach()
                        )
                clean_action: FloatingArray = (
                    decode_action_from_generated_ids(
                        self._model,
                        clean_output_ids,
                        self._cfg.unnorm_key,
                    )
                )

        shared_texture_views: tuple[MultiInstanceViewFrame, ...] = ()
        if shared_target_poses:
            if (
                clean_siglip_features is None
                or wrist_clean_siglip_features is None
                or wrist_background is None
            ):
                raise RuntimeError("双视角 Shared-SigLIP 干净参照不完整")

            def build_instances(camera_name: str) -> tuple[
                TextureRenderInstance, ...
            ]:
                return tuple(
                    {
                        "mvp": compute_render_mvp(
                            env,
                            pose.model_matrix,
                            resolution=(
                                self._render_resolution,
                                self._render_resolution,
                            ),
                            camera_name=camera_name,
                        ),
                        "model_rot": pose.model_matrix[:3, :3],
                    }
                    for pose in shared_target_poses
                )

            shared_texture_views = (
                {
                    "view_name": "primary",
                    "bg_tensor": background,
                    "bg_tensor_no_obj": shared_primary_background_without,
                    "instances": build_instances("agentview"),
                    "clean_siglip_features": clean_siglip_features,
                },
                {
                    "view_name": "wrist",
                    "bg_tensor": wrist_background,
                    "bg_tensor_no_obj": shared_wrist_background_without,
                    "instances": build_instances("robot0_eye_in_hand"),
                    "clean_siglip_features": wrist_clean_siglip_features,
                },
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
            "shared_texture_views": shared_texture_views,
            "initial_state_id": initial_state_id,
            "collection_step_index": step_index,
            "processor_pixel_values": processor_pixel_values,
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
            # 三种帧采集策略：
            # A. 默认固定帧：不运行策略，每个 state 保存固定数量的初始帧；
            # B. 策略驱动：等待稳定后，用 clean action 推进并持续采帧；
            # C. 抓取窗口：用 deque 保留抓取发生前后的局部轨迹。
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
