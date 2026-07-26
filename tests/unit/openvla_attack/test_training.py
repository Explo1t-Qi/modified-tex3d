"""OpenVLA 对抗纹理训练编排的单元测试。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

# Robosuite 默认尝试向只读 site-packages 写 Numba cache。该测试只覆盖 trainer
# 编排，不需要 JIT；在导入 training/evaluation 之前固定关闭，避免依赖 pytest
# 恰好先收集其他测试文件。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import openvla_attack.training as training
from openvla_attack.artifacts import LiveSnapshotPaths
from openvla_attack.evaluation import RolloutResult
from openvla_attack.training import AttackTrainer


@dataclass
class FakeTrainingConfig:
    """覆盖 AttackTrainer 本身直接读取的配置字段。"""

    num_frames_to_attack: int = 1
    live_test_resolution: int = 64
    live_test_max_steps: int = 8
    save_attack_artifacts: bool = False


class FakeArtifactStore:
    """记录 trainer 产生的路径请求和 live-test 视频。"""

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.directory_prepared: bool = False
        self.saved_video: Optional[dict[str, Any]] = None

    def ensure_attack_directory(self) -> None:
        self.directory_prepared = True

    def gradient_log_path(self, *, episode_index: int) -> Path:
        return self.root / f"Ep{episode_index}_gradient_log.txt"

    def save_live_snapshot(
        self,
        *,
        episode_index: int,
        iteration: int,
        renderer: object,
    ) -> LiveSnapshotPaths:
        del renderer
        return LiveSnapshotPaths(
            texture_path=self.root
            / f"Ep{episode_index}_LiveTexture_iter{iteration:04d}.png",
            noise_path=self.root
            / f"Ep{episode_index}_noise_iter{iteration:04d}.pt",
        )

    def save_live_test_video(
        self,
        *,
        episode_index: int,
        iteration: int,
        success: bool,
        frames: Sequence[np.ndarray],
        fps: int = 30,
    ) -> Path:
        self.saved_video = {
            "episode_index": episode_index,
            "iteration": iteration,
            "success": success,
            "frames": list(frames),
            "fps": fps,
        }
        return self.root / "live.mp4"


class FakeRuntimeAssets:
    """记录 live-test 前激活的 Active Texture。"""

    def __init__(self) -> None:
        self.activations: list[tuple[Path, bool]] = []

    def activate_texture(
        self,
        texture_path: Path,
        *,
        mirror_real_texture: bool,
    ) -> bool:
        self.activations.append(
            (texture_path, mirror_real_texture)
        )
        return False


def test_trainer_collects_optimizes_and_reuses_episode_runner_for_live_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []
    camera_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeFrameCollector:
        def __init__(self, **kwargs: object) -> None:
            events.append(("collector_init", kwargs))

        def collect(self, **kwargs: object) -> list[object]:
            events.append(("collect", kwargs))
            return [{"frame": 1}]

    class FakeEpisodeRunner:
        def __init__(self, **kwargs: object) -> None:
            events.append(("runner_init", kwargs))

        def run(self, **kwargs: object) -> RolloutResult:
            events.append(("live_run", kwargs))
            return RolloutResult(
                success=True,
                task_description="pick up the bowl",
                replay_images=[camera_frame],
            )

    class FakeOptimizer:
        def __init__(self, **kwargs: object) -> None:
            events.append(("optimizer_init", kwargs))

        def optimize(
            self,
            *,
            frames: Sequence[object],
            num_iters: int,
            gradient_log_path: Path,
            iteration_callback: Any,
        ) -> list[float]:
            events.append(
                (
                    "optimize",
                    {
                        "frames": frames,
                        "num_iters": num_iters,
                        "gradient_log_path": gradient_log_path,
                    },
                )
            )
            assert iteration_callback(20) is True
            return [0.25]

    monkeypatch.setattr(
        training,
        "TrainingFrameCollector",
        FakeFrameCollector,
    )
    monkeypatch.setattr(
        training,
        "LiberoEpisodeRunner",
        FakeEpisodeRunner,
    )
    monkeypatch.setattr(training, "AttackOptimizer", FakeOptimizer)
    monkeypatch.setattr(training.np.random, "randint", lambda size: 1)

    artifact_store = FakeArtifactStore(tmp_path)
    runtime_assets = FakeRuntimeAssets()
    renderer = SimpleNamespace()
    model = SimpleNamespace(device=torch.device("cpu"))
    trainer = AttackTrainer(
        cfg=FakeTrainingConfig(),
        model=model,
        processor=object(),
        renderer=renderer,
        artifact_store=artifact_store,
        runtime_assets=runtime_assets,
        search_keywords=[["akita", "bowl"]],
        feature_objective="last_hidden",
        render_resolution=32,
    )
    fallback_state = object()
    first_state = object()
    second_state = object()

    loss_history: list[float] = trainer.train(
        task=object(),
        task_description="pick up the bowl",
        fallback_initial_state=fallback_state,
        task_id=3,
        num_iters=1,
        initial_states=[first_state, second_state],
    )

    assert loss_history == [0.25]
    assert artifact_store.directory_prepared is True
    assert runtime_assets.activations == [
        (
            tmp_path / "Ep3_LiveTexture_iter0020.png",
            False,
        )
    ]
    runner_init_event: dict[str, object] = next(
        value
        for name, value in events
        if name == "runner_init"
    )
    # live-test 必须从已激活纹理的 MuJoCo 环境读取图像，不得把训练 renderer
    # 重新注入 episode runner 后重画目标物体。
    assert "renderer" not in runner_init_event
    assert "search_keywords" not in runner_init_event
    live_event: dict[str, object] = next(
        value
        for name, value in events
        if name == "live_run"
    )
    assert live_event["initial_state"] is second_state
    assert live_event["task_id"] == 3
    assert live_event["episode_index"] == 20
    assert artifact_store.saved_video is not None
    assert artifact_store.saved_video["success"] is True
    assert artifact_store.saved_video["frames"] == [camera_frame]
