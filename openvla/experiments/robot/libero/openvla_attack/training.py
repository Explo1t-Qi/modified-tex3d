"""OpenVLA 对抗纹理训练的 task 级编排。

一次攻击训练不是单独的优化循环，而是以下完整数据流：

``TrainingFrameCollector -> AttackOptimizer -> optional live rollout
-> AttackArtifactStore``。

:class:`AttackTrainer` 把这条数据流隐藏在 :meth:`train` interface 后。正式评估
和训练期间的 live-test 共同复用 :class:`LiberoEpisodeRunner`，避免两份环境
推进、MuJoCo 相机预处理、动作后处理状态机逐渐产生行为差异。可微图像合成只
用于 Attack Training，不进入 live-test/final evaluation。

本 module 不遍历 benchmark task，也不统计最终成功率；这些仍属于
``attack_openvla.py`` 的实验编排职责。

编排关系为：

``attack_openvla.py → AttackTrainer
→ TrainingFrameCollector / AttackOptimizer / live-test / AttackArtifactStore``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Optional, Protocol, Sequence

import numpy as np
import torch

from .artifacts import (
    ArtifactRenderer,
    AttackArtifactStore,
    LiveSnapshotPaths,
)
from .configuration import FeatureObjectiveKind, FeatureViewModeKind
from .evaluation import LiberoEpisodeRunner, RolloutConfig, RolloutResult
from .frame_collection import (
    FrameCollectionConfig,
    LightingCalibrator,
    TrainingFrame,
    TrainingFrameCollector,
    TrainingModel,
    TrainingProcessor,
)
from .image_preprocessing import DifferentiableOpenVLAImageProcessor
from .optimization import (
    AttackOptimizer,
    OptimizationConfig,
    OptimizationRenderer,
)
from .runtime_assets import RuntimeAssetTransaction
from .scene import SearchKeywords
from .spectral_gradient_audit import (
    SpectralAuditRenderer,
    SpectralGradientAuditPaths,
    SpectralGradientAuditResult,
    SpectralGradientAuditor,
)
from .source_action_response import (
    SourceActionResponseAuditor,
    SourceActionResponsePaths,
    SourceActionResponseResult,
)


DEFAULT_RENDER_RESOLUTION: Final[int] = 256


class AttackTrainingConfig(
    FrameCollectionConfig,
    OptimizationConfig,
    RolloutConfig,
    Protocol,
):
    """AttackTrainer 从实验配置读取的字段并集。"""

    live_test_resolution: int
    live_test_max_steps: int
    save_attack_artifacts: bool
    spectral_gradient_audit_enabled: bool
    spectral_gradient_audit_only: bool
    spectral_gradient_audit_top_k: int
    spectral_gradient_audit_reference_path: Optional[str]
    source_action_response_audit_enabled: bool
    source_action_response_reference_path: Optional[str]


class AttackTrainingModel(TrainingModel, Protocol):
    """采帧、优化和 live rollout 共同需要的 OpenVLA 模型 interface。"""


class AttackTrainingRenderer(
    ArtifactRenderer,
    LightingCalibrator,
    OptimizationRenderer,
    SpectralAuditRenderer,
    Protocol,
):
    """训练、合成和产物保存共同需要的 renderer interface。"""


class AttackTrainer:
    """在一个 LIBERO task 上训练 OpenVLA 对抗纹理。

    构造参数在多个 task 间共享；:meth:`train` 只接收当前 task 的描述、候选初始
    状态与迭代次数。collector 每个 task 重新创建，以保持光照校准计数彼此独立；
    optimizer 和 live episode runner 则安全复用。
    """

    def __init__(
        self,
        *,
        cfg: AttackTrainingConfig,
        model: AttackTrainingModel,
        processor: TrainingProcessor,
        renderer: AttackTrainingRenderer,
        artifact_store: AttackArtifactStore,
        runtime_assets: RuntimeAssetTransaction,
        search_keywords: SearchKeywords,
        feature_objective: FeatureObjectiveKind,
        feature_view_mode: FeatureViewModeKind = "primary",
        render_resolution: int = DEFAULT_RENDER_RESOLUTION,
        image_preprocessor: Optional[
            DifferentiableOpenVLAImageProcessor
        ] = None,
    ) -> None:
        self._cfg: AttackTrainingConfig = cfg
        self._model: AttackTrainingModel = model
        self._processor: TrainingProcessor = processor
        self._renderer: AttackTrainingRenderer = renderer
        self._artifact_store: AttackArtifactStore = artifact_store
        self._runtime_assets: RuntimeAssetTransaction = runtime_assets
        self._search_keywords: SearchKeywords = search_keywords
        self._feature_objective: FeatureObjectiveKind = feature_objective
        self._feature_view_mode: FeatureViewModeKind = feature_view_mode
        self._render_resolution: int = render_resolution
        self._image_preprocessor: DifferentiableOpenVLAImageProcessor = (
            image_preprocessor
            if image_preprocessor is not None
            else DifferentiableOpenVLAImageProcessor.from_checkpoint(
                model=model,
                processor=processor,
            )
        )

        self._optimizer: AttackOptimizer = AttackOptimizer(
            cfg=cfg,
            model=model,
            renderer=renderer,
            image_preprocessor=self._image_preprocessor,
            feature_objective=feature_objective,
            feature_view_mode=feature_view_mode,
            render_resolution=render_resolution,
        )
        self._live_episode_runner: LiberoEpisodeRunner = (
            LiberoEpisodeRunner(
                cfg=cfg,
                model=model,
                processor=processor,
                video_resolution=cfg.live_test_resolution,
                max_steps=cfg.live_test_max_steps,
            )
        )

    @staticmethod
    def _choose_live_initial_state(
        *,
        fallback_initial_state: Any,
        initial_states: Optional[Sequence[Any]],
    ) -> Any:
        """按现有随机规则选择 live-test 初始状态。"""
        if initial_states is not None and len(initial_states) > 1:
            random_index: int = int(
                np.random.randint(len(initial_states))
            )
            return initial_states[random_index]
        return fallback_initial_state

    def train(
        self,
        *,
        task: Any,
        task_description: str,
        fallback_initial_state: Any,
        task_id: int,
        num_iters: int,
        initial_states: Optional[Sequence[Any]] = None,
        initial_state_ids: Optional[Sequence[int]] = None,
    ) -> list[float]:
        """采集训练帧、优化纹理并保存 task 级 Attack Artifact。

        返回的 ``loss_history`` 长度等于实际执行的优化轮数；正常完成时为
        ``num_iters``，若 live-test 触发提前停止则可能更短。
        """
        print(
            f"[ATTACK] Training Ep {task_id} | "
            f"{self._cfg.num_frames_to_attack}-Frame Optimization..."
        )
        # 即使不保存终态产物，optimizer 的梯度日志仍需要攻击目录。
        self._artifact_store.ensure_attack_directory()

        frame_collector: TrainingFrameCollector = TrainingFrameCollector(
            cfg=self._cfg,
            model=self._model,
            processor=self._processor,
            renderer=self._renderer,
            image_preprocessor=self._image_preprocessor,
            search_keywords=self._search_keywords,
            feature_objective=self._feature_objective,
            feature_view_mode=self._feature_view_mode,
            render_resolution=self._render_resolution,
        )
        frame_pool: list[TrainingFrame] = frame_collector.collect(
            task=task,
            task_description=task_description,
            fallback_initial_state=fallback_initial_state,
            initial_states=initial_states,
            initial_state_ids=initial_state_ids,
        )
        if self._cfg.source_action_response_audit_enabled:
            reference_path: Optional[str] = (
                self._cfg.source_action_response_reference_path
            )
            if reference_path is None:
                raise ValueError("源动作响应诊断缺少参考谱系数路径")
            tokenizer: Any = self._processor.tokenizer
            response_auditor = SourceActionResponseAuditor(
                model=self._model,
                renderer=self._renderer,
                image_preprocessor=self._image_preprocessor,
                reference_parameter_path=reference_path,
                unnorm_key=self._cfg.unnorm_key,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                render_resolution=self._render_resolution,
            )
            response_result: SourceActionResponseResult = (
                response_auditor.run(frame_pool)
            )
            response_paths: SourceActionResponsePaths = response_result.save(
                output_directory=self._artifact_store.attack_directory,
                task_id=task_id,
            )
            mean_hamming: float = float(
                np.mean(
                    [
                        sample.token_hamming_count
                        for sample in response_result.samples
                    ]
                )
            )
            mean_action_l2: float = float(
                np.mean(
                    [sample.action_l2 for sample in response_result.samples]
                )
            )
            print(
                "[ACTION-RESPONSE] "
                f"samples={response_result.num_samples}, "
                f"mean_token_hamming={mean_hamming:.3f}, "
                f"mean_action_l2={mean_action_l2:.6f}"
            )
            print(
                "[ACTION-RESPONSE] "
                f"NPZ={response_paths.npz_path} | "
                f"CSV={response_paths.csv_path} | "
                f"JSON={response_paths.json_path}"
            )
            # 该诊断的定义就是 fixed-reference forward-only。保存完结果后不得
            # 意外继续5000轮优化或 held-out rollout。
            return []
        if self._cfg.spectral_gradient_audit_enabled:
            if self._feature_objective != "siglip_patch":
                raise ValueError(
                    "谱基梯度审计要求 feature_objective='siglip_patch'"
                )
            gradient_auditor = SpectralGradientAuditor(
                gradient_provider=self._optimizer,
                renderer=self._renderer,
                requested_top_k=(
                    self._cfg.spectral_gradient_audit_top_k
                ),
                reference_parameter_path=(
                    self._cfg.spectral_gradient_audit_reference_path
                ),
            )
            audit_result: SpectralGradientAuditResult = (
                gradient_auditor.run(frame_pool)
            )
            audit_paths: SpectralGradientAuditPaths = audit_result.save(
                output_directory=self._artifact_store.attack_directory,
                task_id=task_id,
            )
            energy_index: int = (
                self._cfg.spectral_gradient_audit_top_k - 1
            )
            print(
                "[SPECTRAL-AUDIT] "
                f"samples={audit_result.num_samples}, "
                f"basis={audit_result.num_basis}, "
                f"low-K feature energy="
                f"{audit_result.feature_cumulative_energy[energy_index]:.2%}"
            )
            print(
                "[SPECTRAL-AUDIT] "
                f"NPZ={audit_paths.npz_path} | "
                f"CSV={audit_paths.csv_path} | "
                f"JSON={audit_paths.json_path}"
            )
            if self._cfg.spectral_gradient_audit_only:
                return []
        gradient_log_path: Path = (
            self._artifact_store.gradient_log_path(
                episode_index=task_id
            )
        )

        def run_live_test(iteration: int) -> bool:
            """保存/激活当前纹理并运行一次共享的 episode 状态机。

            ``live_test_enabled=False`` 时 optimizer 不会调用该闭包。
            """
            snapshot_paths: LiveSnapshotPaths = (
                self._artifact_store.save_live_snapshot(
                    episode_index=task_id,
                    iteration=iteration,
                    renderer=self._renderer,
                )
            )
            self._runtime_assets.activate_texture(
                snapshot_paths.texture_path,
                mirror_real_texture=False,
            )
            live_initial_state: Any = self._choose_live_initial_state(
                fallback_initial_state=fallback_initial_state,
                initial_states=initial_states,
            )
            rollout_result: RolloutResult = (
                self._live_episode_runner.run(
                    task=task,
                    initial_state=live_initial_state,
                    task_id=task_id,
                    episode_index=iteration,
                )
            )
            video_path: Path = (
                self._artifact_store.save_live_test_video(
                    episode_index=task_id,
                    iteration=iteration,
                    success=rollout_result.success,
                    frames=rollout_result.replay_images,
                )
            )
            print(
                f"[LIVE-TEST] iter={iteration} "
                f"success={rollout_result.success} | video={video_path}"
            )
            return rollout_result.success

        loss_history: list[float] = self._optimizer.optimize(
            frames=frame_pool,
            num_iters=num_iters,
            gradient_log_path=gradient_log_path,
            iteration_callback=run_live_test,
        )
        if self._cfg.save_attack_artifacts:
            self._artifact_store.save_optimization_result(
                episode_index=task_id,
                renderer=self._renderer,
                loss_history=loss_history,
            )
        return loss_history
