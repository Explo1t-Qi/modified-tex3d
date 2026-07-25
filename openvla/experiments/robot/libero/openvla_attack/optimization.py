"""OpenVLA 对抗纹理的迭代优化循环。

本模块集中管理训练帧采样、对抗视图构造、action/feature loss、反向传播、
SignSGD 参数更新、梯度日志和 live-test 调度。调用方只需提供已经采集好的
``TrainingFrame`` 列表和迭代次数，不需要了解每个 loss 如何按 frame/view
聚合。

``ViewSampler`` 是为后续 EoT/多视角生成明确保留的 seam。当前唯一 adapter 是
``build_single_view_samples``，负责未来 EoT/多视角生成。``FrameBatchSampler``
是独立的 trajectory-aware seam，默认仍执行随机等权采样。这里只预留替换点，
不实现论文中缺失的 latent dynamics、criticality scoring 或多视图算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.cuda.amp import autocast

from .compositing import (
    ForegroundRenderer,
    SingleViewFrame,
    build_single_view_samples,
)
from .frame_collection import TrainingFrame
from .objective import get_attack_loss
from .texture_parameterization import SurfaceStepStats


class OptimizationConfig(Protocol):
    """攻击优化循环读取的最小配置字段集合。"""

    num_frames_to_attack: int
    attack_lr: float
    attack_surface_step: float
    alpha_action: float
    alpha_feature: float
    live_test_enabled: bool
    live_test_every_n_iters: int


class OptimizationModel(Protocol):
    """攻击优化所需的最小 OpenVLA 前向 interface。"""

    device: torch.device

    def __call__(self, **kwargs: torch.Tensor) -> Any:
        """返回包含 ``logits`` 和 ``hidden_states`` 的模型输出。"""
        ...


class OptimizationRenderer(ForegroundRenderer, Protocol):
    """攻击优化需要读取和更新的 renderer interface。"""

    def get_texture_param(self) -> nn.Parameter:
        """返回唯一可学习参数，shape 为 ``[P, 3]`` 或 legacy ``[P]``。"""
        ...

    def get_texture_parameterization_name(self) -> str:
        """返回 legacy_vertex、geometry_vertex 或 spectral。"""
        ...

    def get_surface_delta(self) -> torch.Tensor:
        """返回渲染顶点域 float ``[V, 3]`` Surface Delta。"""
        ...

    def step_surface_parameterization_(
        self,
        gradient: torch.Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        """执行 Geometry/Spectral 的曲面归一化更新。"""
        ...


class ViewSampler(Protocol):
    """从一个有效训练帧构造对抗视图的可替换 interface。

    当前 single-view adapter 返回一个 NCHW RGB tensor；未来 EoT adapter
    可以返回同一帧的多个视图，但必须保持每个 tensor 为浮点 NCHW，
    shape ``[1, 3, render_height, render_width]``，并保留到 renderer 参数的
    梯度。
    """

    def __call__(
        self,
        renderer: ForegroundRenderer,
        frame: SingleViewFrame,
        render_resolution: int,
    ) -> Sequence[torch.Tensor]:
        ...


@dataclass(frozen=True)
class WeightedTrainingFrame:
    """optimizer 本轮选中的训练帧及其 loss 权重。"""

    frame: TrainingFrame
    weight: float


class FrameBatchSampler(Protocol):
    """从完整训练帧池选择并赋权一个 optimizer batch。

    未来 TAAO adapter 可以根据 trajectory latent dynamics 返回非均匀权重。
    当前训练帧尚未显式保存 trajectory 边界；真正实现 TAAO 时应同步扩展
    ``TrainingFrame`` schema，而不是在该默认 adapter 中猜测边界。
    """

    def __call__(
        self,
        frames: Sequence[TrainingFrame],
        batch_size: int,
    ) -> Sequence[WeightedTrainingFrame]:
        ...


def sample_uniform_frame_batch(
    frames: Sequence[TrainingFrame],
    batch_size: int,
) -> Sequence[WeightedTrainingFrame]:
    """保持现有随机无放回、按抽取 batch 大小等权的 frame 采样。"""
    selected_indices: np.ndarray = np.random.choice(
        len(frames),
        batch_size,
        replace=False,
    )
    uniform_weight: float = 1.0 / batch_size
    return [
        WeightedTrainingFrame(
            frame=frames[int(frame_index)],
            weight=uniform_weight,
        )
        for frame_index in selected_indices
    ]


IterationCallback = Callable[[int], Any]


class AttackOptimizer:
    """对一组 OpenVLA 训练帧执行对抗纹理优化。

    :meth:`optimize` 是唯一行为 interface。实现内部隐藏随机 frame batch、
    多视图 loss 平均、frame 权重、梯度日志和 SignSGD 更新顺序。
    """

    def __init__(
        self,
        *,
        cfg: OptimizationConfig,
        model: OptimizationModel,
        renderer: OptimizationRenderer,
        view_sampler: ViewSampler = build_single_view_samples,
        frame_batch_sampler: FrameBatchSampler = (
            sample_uniform_frame_batch
        ),
        render_resolution: int = 256,
    ) -> None:
        self._cfg: OptimizationConfig = cfg
        self._model: OptimizationModel = model
        self._renderer: OptimizationRenderer = renderer
        self._view_sampler: ViewSampler = view_sampler
        self._frame_batch_sampler: FrameBatchSampler = (
            frame_batch_sampler
        )
        self._render_resolution: int = render_resolution

    @staticmethod
    def _as_single_view_frame(
        frame: TrainingFrame,
        mvp: torch.Tensor,
    ) -> SingleViewFrame:
        """把 Optional MVP 已收窄的训练帧转换为 view sampler interface。"""
        return {
            "mvp": mvp,
            "bg_tensor": frame["bg_tensor"],
            "bg_tensor_no_obj": frame["bg_tensor_no_obj"],
            "model_rot": frame["model_rot"],
        }

    def _compute_view_losses(
        self,
        frame: TrainingFrame,
        adversarial_views: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算一个 frame 上所有视图的平均 action/feature loss。"""
        action_losses: list[torch.Tensor] = []
        feature_losses: list[torch.Tensor] = []

        adversarial_image: torch.Tensor
        for adversarial_image in adversarial_views:
            # resized_image: float NCHW [1, 3, model_height, model_width]。
            resized_image: torch.Tensor = F.interpolate(
                adversarial_image,
                size=(
                    frame["model_input_size"],
                    frame["model_input_size"],
                ),
                mode="bilinear",
                align_corners=False,
            )
            # pixel_values: float/bfloat16 NCHW
            # [1, 6, model_height, model_width]，前三通道为 SigLIP，后三通道
            # 为 DINOv2。
            pixel_values: torch.Tensor = torch.cat(
                (
                    (resized_image - frame["siglip_mean"])
                    / frame["siglip_std"],
                    (resized_image - frame["dino_mean"])
                    / frame["dino_std"],
                ),
                dim=1,
            )

            with autocast(dtype=torch.bfloat16):
                outputs: Any = self._model(
                    input_ids=frame["clean_output_ids"],
                    attention_mask=torch.ones_like(
                        frame["clean_output_ids"]
                    ),
                    pixel_values=pixel_values.to(torch.bfloat16),
                    output_hidden_states=True,
                )
            action_losses.append(
                get_attack_loss(
                    outputs.logits,
                    frame["clean_output_ids"],
                )
            )
            feature_losses.append(
                -F.mse_loss(
                    outputs.hidden_states[-1],
                    frame["clean_hidden"],
                )
            )

        # 当前 single-view adapter 保证至少一个视图。未来 adapter 也必须遵守
        # 这一约束，否则 stack 会明确报错而不是产生无意义的零 loss。
        mean_action_loss: torch.Tensor = torch.stack(action_losses).mean()
        mean_feature_loss: torch.Tensor = torch.stack(feature_losses).mean()
        return mean_action_loss, mean_feature_loss

    def optimize(
        self,
        *,
        frames: Sequence[TrainingFrame],
        num_iters: int,
        gradient_log_path: str | Path,
        iteration_callback: Optional[IterationCallback] = None,
    ) -> list[float]:
        """执行 SignSGD 优化并返回每轮平均 total loss。

        Args:
            frames: collector 生成的训练帧池。每轮从中无放回抽取最多
                ``cfg.num_frames_to_attack`` 帧。
            num_iters: 优化迭代次数。
            gradient_log_path: 梯度日志输出路径；父目录必须已经存在。
            iteration_callback: 可选的一基迭代回调。只在配置启用 live-test
                且到达 ``live_test_every_n_iters`` 时调用。

        Returns:
            Python float 列表，长度等于 ``num_iters``。
        """
        frame_pool_size: int = len(frames)
        frame_batch_size: int = min(
            self._cfg.num_frames_to_attack,
            frame_pool_size,
        )
        legacy_parameter_step: float = self._cfg.attack_lr
        surface_step: float = self._cfg.attack_surface_step
        parameterization_name: str = (
            self._renderer.get_texture_parameterization_name()
        )
        update_rule: Literal["legacy_sign", "surface_normalized"] = (
            "legacy_sign"
            if parameterization_name == "legacy_vertex"
            else "surface_normalized"
        )
        loss_history: list[float] = []
        resolved_log_path = Path(gradient_log_path)
        resolved_log_path.write_text(
            "Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | "
            "Update Rule | Actual Surface Step | Max Surface Delta\n"
        )

        iterator = tqdm.tqdm(
            range(num_iters),
            desc="Optimizing",
            leave=False,
        )
        for iteration_index in iterator:
            texture_parameter: nn.Parameter = (
                self._renderer.get_texture_param()
            )
            texture_parameter.grad = None
            average_total_loss: float = 0.0
            average_action_loss: float = 0.0
            average_feature_loss: float = 0.0
            valid_frame_count: int = 0

            selected_frames: Sequence[WeightedTrainingFrame] = (
                self._frame_batch_sampler(
                    frames,
                    frame_batch_size,
                )
            )
            selected_frame: WeightedTrainingFrame
            for selected_frame in selected_frames:
                frame: TrainingFrame = selected_frame.frame
                frame_mvp: Optional[torch.Tensor] = frame["mvp"]
                if frame_mvp is None:
                    continue

                # 默认 adapter 与重构前一致：每帧按抽取 batch 大小加权，随后
                # 统计日志时再除以有效帧数。未来 TAAO adapter 可以替换 weight。
                frame_weight: torch.Tensor = torch.tensor(
                    selected_frame.weight,
                    device=self._model.device,
                )
                view_frame: SingleViewFrame = self._as_single_view_frame(
                    frame,
                    frame_mvp,
                )
                adversarial_views: Sequence[torch.Tensor] = (
                    self._view_sampler(
                        self._renderer,
                        view_frame,
                        self._render_resolution,
                    )
                )
                action_loss: torch.Tensor
                feature_loss: torch.Tensor
                action_loss, feature_loss = self._compute_view_losses(
                    frame,
                    adversarial_views,
                )
                frame_loss: torch.Tensor = frame_weight * (
                    self._cfg.alpha_action * action_loss
                    + self._cfg.alpha_feature * feature_loss
                )
                frame_loss.backward()

                average_total_loss += frame_loss.item()
                average_action_loss += action_loss.item()
                average_feature_loss += feature_loss.item()
                valid_frame_count += 1

            if valid_frame_count > 0:
                average_total_loss /= valid_frame_count
                average_action_loss /= valid_frame_count
                average_feature_loss /= valid_frame_count
            loss_history.append(average_total_loss)

            gradient: Optional[torch.Tensor] = texture_parameter.grad
            gradient_norm: float = (
                gradient.norm().item() if gradient is not None else 0.0
            )

            if gradient is None:
                raise RuntimeError(
                    "攻击优化没有得到纹理梯度；请检查训练帧 MVP 和 view sampler"
                )
            before_surface_delta: torch.Tensor = (
                self._renderer.get_surface_delta().detach().clone()
            )
            with torch.no_grad():
                if update_rule == "legacy_sign":
                    texture_parameter.data -= (
                        legacy_parameter_step * gradient.sign()
                    )
                    after_surface_delta: torch.Tensor = (
                        self._renderer.get_surface_delta().detach()
                    )
                    actual_surface_step: float = float(
                        (
                            after_surface_delta - before_surface_delta
                        ).abs().amax().item()
                    )
                    max_surface_delta: float = float(
                        after_surface_delta.abs().amax().item()
                    )
                else:
                    step_stats: SurfaceStepStats = (
                        self._renderer.step_surface_parameterization_(
                            gradient,
                            surface_step,
                        )
                    )
                    actual_surface_step = step_stats.actual_surface_step
                    max_surface_delta = step_stats.max_abs_delta

            with resolved_log_path.open("a") as log_file:
                log_file.write(
                    f"{iteration_index:02d} | "
                    f"{average_total_loss:.6f} | "
                    f"{average_action_loss:.6f} | "
                    f"{average_feature_loss:.6f} | "
                    f"{gradient_norm:.6e} | "
                    f"{update_rule} | "
                    f"{actual_surface_step:.6e} | "
                    f"{max_surface_delta:.6e}\n"
                )
            iterator.set_postfix(
                act=f"{average_action_loss:.4f}",
                feat=f"{average_feature_loss:.4f}",
                gnorm=f"{gradient_norm:.4f}",
                step=f"{actual_surface_step:.4f}",
            )

            one_based_iteration: int = iteration_index + 1
            should_run_callback: bool = (
                iteration_callback is not None
                and self._cfg.live_test_enabled
                and self._cfg.live_test_every_n_iters > 0
                and one_based_iteration
                % self._cfg.live_test_every_n_iters
                == 0
            )
            if should_run_callback and iteration_callback is not None:
                iteration_callback(one_based_iteration)

        return loss_history
