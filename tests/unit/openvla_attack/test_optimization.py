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
from openvla_attack.texture_parameterization import SurfaceStepStats
from openvla_attack.texture_parameterization import (
    GeometryVertexTextureParameterization,
    surface_normalized_step_,
)


@dataclass
class FakeOptimizationConfig:
    """覆盖 optimizer 读取的配置字段。"""

    num_frames_to_attack: int = 1
    attack_lr: float = 0.1
    attack_surface_step: float = 0.02
    alpha_action: float = 1.0
    alpha_feature: float = 0.0
    live_test_enabled: bool = True
    live_test_every_n_iters: int = 1


class FakeOptimizationRenderer:
    """仅暴露优化参数；图像由注入的 view sampler 构造。"""

    def __init__(self) -> None:
        self.adv_noise = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))

    def get_texture_param(self) -> nn.Parameter:
        return self.adv_noise

    def get_texture_parameterization_name(self) -> str:
        return "legacy_vertex"

    def get_surface_delta(self) -> torch.Tensor:
        return self.adv_noise

    def step_surface_parameterization_(
        self,
        gradient: torch.Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        del gradient, surface_step
        raise AssertionError("legacy 测试不应调用曲面更新")


class FakeOptimizationModel:
    """让 logits 与 hidden state 都保持到对抗图像的梯度。"""

    device: torch.device = torch.device("cpu")

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        pixel_values = kwargs["pixel_values"].float()
        scalar = pixel_values.mean()
        logits = scalar.reshape(1, 1, 1)
        hidden = scalar.reshape(1, 1, 1)
        return SimpleNamespace(logits=logits, hidden_states=[hidden])


class FakeSharedSigLIPFeaturizer(nn.Module):
    """把三通道均值作为一个 patch，并保留输入梯度。"""

    def __init__(self) -> None:
        super().__init__()
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(pixel_values.shape))
        # 转成 float32，避免 CPU bfloat16 MSE 的版本差异；cast 仍保留梯度。
        return pixel_values.float().mean(dim=(2, 3)).unsqueeze(1)


class FakeSharedFeatureModel(FakeOptimizationModel):
    """同时提供历史 VLA forward 与独立 SigLIP 分支。"""

    def __init__(self) -> None:
        self.siglip = FakeSharedSigLIPFeaturizer()
        self.config = SimpleNamespace(
            timm_model_ids=[
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            ]
        )
        self.vision_backbone = SimpleNamespace(
            featurizer=nn.Identity(),
            fused_featurizer=self.siglip,
        )


class FakeSurfaceOptimizationRenderer:
    """使用真实 Geometry adapter 覆盖 surface-normalized optimizer 分支。"""

    def __init__(self) -> None:
        self.parameterization = GeometryVertexTextureParameterization(
            render_to_geometry=torch.tensor([0]),
            num_geometry_vertices=1,
            epsilon=0.5,
            device="cpu",
        )

    def get_texture_param(self) -> nn.Parameter:
        return self.parameterization.coefficients

    def get_texture_parameterization_name(self) -> str:
        return "geometry_vertex"

    def get_surface_delta(self) -> torch.Tensor:
        return self.parameterization.render_delta()

    def step_surface_parameterization_(
        self,
        gradient: torch.Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        return surface_normalized_step_(
            self.parameterization,
            gradient,
            surface_step,
        )


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
        "clean_siglip_features": None,
        "shared_texture_views": (),
        "initial_state_id": 0,
        "collection_step_index": 0,
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
        feature_objective="last_hidden",
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
        "Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | "
        "Update Rule | Actual Surface Step | Max Surface Delta\n"
    )
    assert "0.200195" in gradient_log
    assert "legacy_sign" in gradient_log
    assert "1.000000e-01" in gradient_log


def test_optimizer_uses_surface_normalized_update_for_new_adapter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    renderer = FakeSurfaceOptimizationRenderer()

    def surface_view_sampler(
        current_renderer: FakeSurfaceOptimizationRenderer,
        frame: SingleViewFrame,
        render_resolution: int,
    ) -> list[torch.Tensor]:
        del frame, render_resolution
        scalar = current_renderer.get_surface_delta().mean()
        return [scalar.reshape(1, 1, 1, 1).expand(1, 3, 2, 2)]

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
    gradient_log_path = tmp_path / "surface-gradient.txt"
    optimizer = AttackOptimizer(
        cfg=FakeOptimizationConfig(attack_surface_step=0.02),
        model=FakeOptimizationModel(),
        renderer=renderer,
        feature_objective="last_hidden",
        view_sampler=surface_view_sampler,
        render_resolution=2,
    )

    optimizer.optimize(
        frames=[_training_frame()],
        num_iters=1,
        gradient_log_path=gradient_log_path,
    )

    assert abs(renderer.parameterization.max_abs_delta() - 0.02) < 1e-6
    gradient_log = gradient_log_path.read_text()
    assert "surface_normalized" in gradient_log
    assert "2.000000e-02" in gradient_log


def test_siglip_objective_uses_three_channel_shared_features_and_backpropagates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    renderer = FakeOptimizationRenderer()
    model = FakeSharedFeatureModel()
    frame = _training_frame()
    frame["clean_siglip_features"] = torch.zeros(
        (1, 1, 3),
        dtype=torch.float32,
    )

    def fake_view_sampler(
        current_renderer: FakeOptimizationRenderer,
        current_frame: SingleViewFrame,
        render_resolution: int,
    ) -> list[torch.Tensor]:
        del current_frame, render_resolution
        image = current_renderer.adv_noise.reshape(
            1,
            1,
            1,
            1,
        ).expand(1, 3, 2, 2)
        return [image]

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

    optimizer = AttackOptimizer(
        cfg=FakeOptimizationConfig(
            alpha_action=0.0,
            alpha_feature=1.0,
        ),
        model=model,
        renderer=renderer,
        feature_objective="siglip_patch",
        view_sampler=fake_view_sampler,
        render_resolution=2,
    )
    losses = optimizer.optimize(
        frames=[frame],
        num_iters=1,
        gradient_log_path=tmp_path / "siglip-gradient.txt",
    )

    # 负 MSE 的梯度下降会增大 feature 距离，因此参数从 0.2 增至 0.3。
    np.testing.assert_allclose(losses, [-0.04], rtol=2e-3)
    torch.testing.assert_close(
        renderer.adv_noise.detach(),
        torch.tensor([0.3]),
    )
    assert model.siglip.input_shapes == [(1, 3, 2, 2)]


def test_dual_view_siglip_uses_primary_action_and_two_feature_views(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeDualRenderer(FakeOptimizationRenderer):
        def render(
            self,
            mvp: torch.Tensor,
            resolution: tuple[int, int],
            *,
            model_rot: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del mvp, model_rot
            height, width = resolution
            rgb = self.adv_noise.reshape(1, 1, 1, 1).expand(
                1, height, width, 3
            )
            mask = torch.ones((1, height, width, 1))
            return rgb, mask

    renderer = FakeDualRenderer()
    model = FakeSharedFeatureModel()
    frame = _training_frame()
    instance = {"mvp": torch.eye(4), "model_rot": torch.eye(3)}
    frame["clean_siglip_features"] = torch.zeros((1, 1, 3))
    frame["shared_texture_views"] = (
        {
            "view_name": "primary",
            "bg_tensor": torch.zeros((1, 3, 2, 2)),
            "bg_tensor_no_obj": torch.zeros((1, 3, 2, 2)),
            "instances": (instance,),
            "clean_siglip_features": torch.zeros((1, 1, 3)),
        },
        {
            "view_name": "wrist",
            "bg_tensor": torch.zeros((1, 3, 2, 2)),
            "bg_tensor_no_obj": torch.zeros((1, 3, 2, 2)),
            "instances": (instance,),
            "clean_siglip_features": torch.full((1, 1, 3), 0.1),
        },
    )
    action_calls: list[torch.Tensor] = []

    def fake_action_loss(
        logits: torch.Tensor,
        clean_ids: torch.Tensor,
    ) -> torch.Tensor:
        del clean_ids
        action_calls.append(logits)
        return logits.mean()

    monkeypatch.setattr(optimization, "get_attack_loss", fake_action_loss)
    monkeypatch.setattr(
        optimization,
        "autocast",
        lambda **kwargs: nullcontext(),
    )
    optimizer = AttackOptimizer(
        cfg=FakeOptimizationConfig(alpha_action=0.0, alpha_feature=1.0),
        model=model,
        renderer=renderer,
        feature_objective="siglip_patch",
        feature_view_mode="primary_wrist",
        render_resolution=2,
    )

    losses = optimizer.optimize(
        frames=[frame],
        num_iters=1,
        gradient_log_path=tmp_path / "dual-view-gradient.txt",
    )

    # Feature = mean(-(.2-0)^2, -(.2-.1)^2) = -0.025。
    np.testing.assert_allclose(losses, [-0.025], rtol=4e-3)
    assert len(action_calls) == 1
    assert model.siglip.input_shapes == [(1, 3, 2, 2), (1, 3, 2, 2)]
    torch.testing.assert_close(
        renderer.adv_noise.detach(),
        torch.tensor([0.3]),
    )


def test_objective_gradient_audit_returns_unweighted_independent_gradients(
    monkeypatch,
) -> None:
    renderer = FakeOptimizationRenderer()
    model = FakeSharedFeatureModel()
    frame = _training_frame()
    frame["initial_state_id"] = 9
    frame["clean_siglip_features"] = torch.zeros(
        (1, 1, 3),
        dtype=torch.float32,
    )

    def fake_view_sampler(
        current_renderer: FakeOptimizationRenderer,
        current_frame: SingleViewFrame,
        render_resolution: int,
    ) -> list[torch.Tensor]:
        del current_frame, render_resolution
        return [
            current_renderer.adv_noise.reshape(1, 1, 1, 1).expand(
                1,
                3,
                2,
                2,
            )
        ]

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
    optimizer = AttackOptimizer(
        cfg=FakeOptimizationConfig(
            alpha_action=123.0,
            alpha_feature=456.0,
        ),
        model=model,
        renderer=renderer,
        feature_objective="siglip_patch",
        view_sampler=fake_view_sampler,
        render_resolution=2,
    )

    sample = optimizer.compute_objective_parameter_gradients(frame)

    assert sample is not None
    np.testing.assert_allclose(sample.action_loss, 0.2, rtol=2e-3)
    np.testing.assert_allclose(sample.feature_loss, -0.04, rtol=2e-3)
    # 返回的是未乘 alpha 的独立梯度：d(0.2)/dp=1，
    # d(-p²)/dp=-2p=-0.4。
    torch.testing.assert_close(
        sample.action_gradient,
        torch.tensor([1.0]),
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        sample.feature_gradient,
        torch.tensor([-0.4]),
        rtol=2e-3,
        atol=2e-3,
    )
    assert renderer.adv_noise.grad is None
    torch.testing.assert_close(
        renderer.adv_noise.detach(),
        torch.tensor([0.2]),
    )
