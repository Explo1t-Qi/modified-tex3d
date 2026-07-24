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
from .evaluation import LiberoEpisodeRunner, RolloutConfig, RolloutResult
from .frame_collection import (
    FrameCollectionConfig,
    LightingCalibrator,
    TrainingFrame,
    TrainingFrameCollector,
    TrainingProcessor,
)
from .optimization import (
    AttackOptimizer,
    OptimizationConfig,
    OptimizationRenderer,
)
from .runtime_assets import RuntimeAssetTransaction
from .scene import SearchKeywords


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


class AttackTrainingModel(Protocol):
    """采帧、优化和 live rollout 共同需要的 OpenVLA 模型 interface。"""

    device: torch.device

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """返回 token IDs，shape ``[batch_size, sequence_length]``。"""
        ...

    def __call__(self, **kwargs: Any) -> Any:
        """执行 OpenVLA forward。"""
        ...


class AttackTrainingRenderer(
    ArtifactRenderer,
    LightingCalibrator,
    OptimizationRenderer,
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
        render_resolution: int = DEFAULT_RENDER_RESOLUTION,
    ) -> None:
        self._cfg: AttackTrainingConfig = cfg
        self._model: AttackTrainingModel = model
        self._processor: TrainingProcessor = processor
        self._renderer: AttackTrainingRenderer = renderer
        self._artifact_store: AttackArtifactStore = artifact_store
        self._runtime_assets: RuntimeAssetTransaction = runtime_assets
        self._search_keywords: SearchKeywords = search_keywords
        self._render_resolution: int = render_resolution

        self._optimizer: AttackOptimizer = AttackOptimizer(
            cfg=cfg,
            model=model,
            renderer=renderer,
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
    ) -> list[float]:
        """采集训练帧、优化纹理并保存 task 级 Attack Artifact。"""
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
            search_keywords=self._search_keywords,
            render_resolution=self._render_resolution,
        )
        frame_pool: list[TrainingFrame] = frame_collector.collect(
            task=task,
            task_description=task_description,
            fallback_initial_state=fallback_initial_state,
            initial_states=initial_states,
        )
        gradient_log_path: Path = (
            self._artifact_store.gradient_log_path(
                episode_index=task_id
            )
        )

        def run_live_test(iteration: int) -> bool:
            """保存/激活当前纹理并运行一次共享的 episode 状态机。"""
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
