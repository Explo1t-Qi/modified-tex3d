"""OpenVLA 对抗纹理优化循环的单元测试。"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

import openvla_attack.optimization as optimization
from openvla_attack.compositing import SingleViewFrame
from openvla_attack.frame_collection import TrainingFrame
from openvla_attack.optimization import (
    AttackOptimizer,
    WeightedTrainingFrame,
    sample_uniform_frame_batch,
)


@dataclass
class FakeOptimizationConfig:
    """覆盖 optimizer 读取的配置字段。"""

    num_frames_to_attack: int = 1
    attack_lr: float = 0.1
    alpha_action: float = 1.0
    alpha_feature: float = 0.0
    live_test_enabled: bool = True
    live_test_every_n_iters: int = 1


class FakeOptimizationRenderer:
    """仅暴露优化参数；图像由注入的 view sampler 构造。"""

    def __init__(self) -> None:
        self.adv_noise = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))


class FakeOptimizationModel:
    """让 logits 与 hidden state 都保持到对抗图像的梯度。"""

    device: torch.device = torch.device("cpu")

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        pixel_values = kwargs["pixel_values"].float()
        scalar = pixel_values.mean()
        logits = scalar.reshape(1, 1, 1)
        hidden = scalar.reshape(1, 1, 1)
        return SimpleNamespace(logits=logits, hidden_states=[hidden])


def _training_frame() -> TrainingFrame:
    """构造一个具有有效 MVP 的最小训练帧。"""
    zeros = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
    ones = torch.ones((1, 3, 1, 1), dtype=torch.float32)
    return {
        "bg_tensor": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "bg_tensor_no_obj": None,
        "mvp": torch.eye(4),
        "model_rot": torch.eye(3),
        "clean_output_ids": torch.tensor([[7]], dtype=torch.long),
        "prompt_ids": torch.tensor([[3]], dtype=torch.long),
        "clean_action": np.zeros(7, dtype=np.float32),
        "executed_action": None,
        "clean_hidden": torch.zeros((1, 1, 1), dtype=torch.float32),
        "siglip_mean": zeros,
        "siglip_std": ones,
        "dino_mean": zeros,
        "dino_std": ones,
        "model_input_size": 2,
    }


def test_default_frame_sampler_is_random_without_replacement_and_uniform(
    monkeypatch,
) -> None:
    frames: list[TrainingFrame] = [_training_frame(), _training_frame()]
    choice_arguments: dict[str, object] = {}

    def fake_choice(
        frame_pool_size: int,
        batch_size: int,
        *,
        replace: bool,
    ) -> np.ndarray:
        choice_arguments.update(
            frame_pool_size=frame_pool_size,
            batch_size=batch_size,
            replace=replace,
        )
        return np.array([1, 0])

    monkeypatch.setattr(optimization.np.random, "choice", fake_choice)
    selected: list[WeightedTrainingFrame] = list(
        sample_uniform_frame_batch(frames, batch_size=2)
    )

    assert choice_arguments == {
        "frame_pool_size": 2,
        "batch_size": 2,
        "replace": False,
    }
    assert selected[0].frame is frames[1]
    assert selected[1].frame is frames[0]
    assert [item.weight for item in selected] == [0.5, 0.5]


def test_optimizer_uses_view_sampler_updates_texture_logs_and_schedules_callback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    renderer = FakeOptimizationRenderer()
    sampler_frames: list[SingleViewFrame] = []
    sampled_batch_sizes: list[int] = []
    callback_iterations: list[int] = []

    def fake_frame_batch_sampler(
        frames: list[TrainingFrame],
        batch_size: int,
    ) -> list[WeightedTrainingFrame]:
        sampled_batch_sizes.append(batch_size)
        return [
            WeightedTrainingFrame(
                frame=frames[0],
                weight=1.0,
            )
        ]

    def fake_view_sampler(
        current_renderer: FakeOptimizationRenderer,
        frame: SingleViewFrame,
        render_resolution: int,
    ) -> list[torch.Tensor]:
        assert render_resolution == 2
        sampler_frames.append(frame)
        # adversarial_image: float32 NCHW [1, 3, 2, 2]。
        adversarial_image = current_renderer.adv_noise.reshape(
            1, 1, 1, 1
        ).expand(1, 3, 2, 2)
        return [adversarial_image]

    monkeypatch.setattr(
        optimization,
        "get_attack_loss",
        lambda logits, clean_ids: logits.mean(),
    )
    monkeypatch.setattr(
        optimization,
        "autocast",
        lambda **kwargs: nullcontext(),
    )

    gradient_log_path = tmp_path / "gradient.txt"
    optimizer = AttackOptimizer(
        cfg=FakeOptimizationConfig(),
        model=FakeOptimizationModel(),
        renderer=renderer,
        view_sampler=fake_view_sampler,
        frame_batch_sampler=fake_frame_batch_sampler,
        render_resolution=2,
    )
    loss_history = optimizer.optimize(
        frames=[_training_frame()],
        num_iters=1,
        gradient_log_path=gradient_log_path,
        iteration_callback=callback_iterations.append,
    )

    assert len(sampler_frames) == 1
    assert sampled_batch_sizes == [1]
    torch.testing.assert_close(sampler_frames[0]["mvp"], torch.eye(4))
    # optimizer 会在模型前向前转为 bfloat16，因此允许对应的量化误差。
    np.testing.assert_allclose(loss_history, [0.2], rtol=2e-3)
    torch.testing.assert_close(
        renderer.adv_noise.detach(),
        torch.tensor([0.1]),
    )
    assert callback_iterations == [1]

    gradient_log = gradient_log_path.read_text()
    assert gradient_log.startswith(
        "Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | LR\n"
    )
    assert "0.200195" in gradient_log
    assert "1.000000e-01" in gradient_log
