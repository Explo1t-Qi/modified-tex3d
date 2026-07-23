"""OpenVLA 攻击实验产物管理的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

import openvla_attack.artifacts as artifacts
from openvla_attack.artifacts import (
    AttackArtifactStore,
    LiveSnapshotPaths,
    OptimizationArtifactPaths,
)


class FakeArtifactRenderer:
    """提供产物落盘所需的最小 renderer interface。"""

    def __init__(self) -> None:
        # adv_noise: float32 [num_vertices, rgb_channels]。
        self.adv_noise = nn.Parameter(
            torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
        )

    def get_texture_param(self) -> nn.Parameter:
        """返回需要保存的可学习纹理参数。"""
        return self.adv_noise

    def get_baked_adv_texture(self) -> torch.Tensor:
        """返回 NHWC [1, texture_height, texture_width, 3] 的测试纹理。"""
        return torch.tensor(
            [[[[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]]]],
            dtype=torch.float32,
        )


def test_store_preserves_run_paths_and_optimization_artifact_formats(
    tmp_path: Path,
) -> None:
    store: AttackArtifactStore = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="smoke-EVAL-libero_spatial-test",
        create_attack_directory=True,
    )
    renderer: FakeArtifactRenderer = FakeArtifactRenderer()

    assert store.run_log_path == (
        tmp_path / "smoke-EVAL-libero_spatial-test.txt"
    )
    assert store.attack_directory == (
        tmp_path
        / "attack_artifacts"
        / "smoke-EVAL-libero_spatial-test"
    )
    assert store.attack_directory.is_dir()
    assert store.gradient_log_path(episode_index=3).name == (
        "Ep3_gradient_log.txt"
    )

    with store.open_run_log() as log_file:
        log_file.write("Task: 3 | Ep: 0 | Success: True\n")
    assert "Success: True" in store.run_log_path.read_text()

    saved: OptimizationArtifactPaths = store.save_optimization_result(
        episode_index=3,
        renderer=renderer,
        loss_history=[1.25, -0.5],
    )

    assert saved.noise_path.name == "Ep3_Vertex_Noise.pt"
    assert saved.texture_path.name == "Ep3_UV_Map.png"
    assert saved.loss_history_path.name == "Ep3_loss_history.npy"
    torch.testing.assert_close(
        torch.load(saved.noise_path, weights_only=False),
        renderer.adv_noise.detach(),
    )
    np.testing.assert_allclose(
        np.load(saved.loss_history_path),
        np.array([1.25, -0.5]),
    )
    saved_pixels: np.ndarray = np.array(Image.open(saved.texture_path))
    np.testing.assert_array_equal(
        saved_pixels,
        np.array([[[0, 127, 255], [255, 63, 0]]], dtype=np.uint8),
    )


def test_store_preserves_snapshot_texture_and_video_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store: AttackArtifactStore = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="run",
        create_attack_directory=True,
    )
    renderer: FakeArtifactRenderer = FakeArtifactRenderer()

    snapshot: LiveSnapshotPaths = store.save_live_snapshot(
        episode_index=2,
        iteration=40,
        renderer=renderer,
    )
    assert snapshot.texture_path.name == "Ep2_LiveTexture_iter0040.png"
    assert snapshot.noise_path.name == "Ep2_noise_iter0040.pt"
    torch.testing.assert_close(
        torch.load(snapshot.noise_path, weights_only=False),
        renderer.adv_noise.detach(),
    )

    loaded_path: Path = store.save_loaded_texture(
        task_id=2,
        renderer=renderer,
    )
    trained_path: Path = store.save_trained_texture(
        task_id=2,
        timestamp="2026_07_23-12_00_00",
        renderer=renderer,
    )
    assert loaded_path.name == "task_2_adv_tex_loaded.png"
    assert trained_path.name == (
        "task_2_adv_texture_2026_07_23-12_00_00.png"
    )

    appended_frames: list[np.ndarray] = []
    writer_arguments: dict[str, object] = {}

    class FakeVideoWriter:
        """记录 video writer 收到的帧，避免测试依赖 ffmpeg。"""

        def append_data(self, frame: np.ndarray) -> None:
            appended_frames.append(frame)

        def close(self) -> None:
            writer_arguments["closed"] = True

    def fake_get_writer(path: Path, *, fps: int) -> FakeVideoWriter:
        writer_arguments.update(path=path, fps=fps)
        return FakeVideoWriter()

    monkeypatch.setattr(artifacts.imageio, "get_writer", fake_get_writer)
    frames: list[np.ndarray] = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3, 3), dtype=np.uint8),
    ]
    video_path: Path = store.save_live_test_video(
        episode_index=2,
        iteration=40,
        success=False,
        frames=frames,
    )

    assert video_path.name == "Ep2_LiveTest_iter0040_success=False.mp4"
    assert writer_arguments == {
        "path": video_path,
        "fps": 30,
        "closed": True,
    }
    assert appended_frames == frames
