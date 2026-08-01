"""OpenVLA 对抗纹理的迭代优化循环。

本模块集中管理训练帧采样、对抗视图构造、action/feature loss、反向传播、
SignSGD 参数更新、梯度日志和 live-test 调度。调用方只需提供已经采集好的
``TrainingFrame`` 列表和迭代次数，不需要了解每个 loss 如何按 frame/view
聚合。

``ViewSampler`` 是为后续 EoT/多视角生成明确保留的 seam。当前唯一 adapter 是
``build_single_view_samples``，负责未来 EoT/多视角生成。``FrameBatchSampler``
是独立的 trajectory-aware seam，默认仍执行随机等权采样。这里只预留替换点，
不实现论文中缺失的 latent dynamics、criticality scoring 或多视图算法。

优化路径为：

``TrainingFrame 池 → FrameBatchSampler → 构造对抗视图 → OpenVLA forward
→ action/feature loss → 加权 total loss → backward → 纹理参数更新``。
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
    MultiInstanceViewFrame,
    SharedTextureViewName,
    SingleViewFrame,
    build_multi_instance_view_sample,
    build_single_view_samples,
)
from .configuration import FeatureObjectiveKind, FeatureViewModeKind
from .frame_collection import TrainingFrame
from .objective import get_attack_loss
from .spectral_gradient_audit import ObjectiveParameterGradients
from .texture_parameterization import SurfaceStepStats
from .vision_features import (
    SigLIPFeatureModel,
    extract_siglip_patch_features,
)


class OptimizationConfig(Protocol):
    """攻击优化循环读取的最小配置字段集合。"""

    num_frames_to_attack: int
    attack_lr: float
    attack_surface_step: float
    alpha_action: float
    alpha_feature: float
    live_test_enabled: bool
    live_test_every_n_iters: int


class OptimizationModel(SigLIPFeatureModel, Protocol):
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


@dataclass(frozen=True)
class FrameObjectiveLosses:
    """一个训练帧的模型目标及可选分视角诊断。

    ``action`` 与 ``feature`` 是实际进入总目标的标量 tensor。双视角模式额外
    保留主视角/腕部各自的 Feature loss，供梯度日志持久化；该诊断字典不会
    再次参与 loss 聚合或反向传播。
    """

    action: torch.Tensor
    feature: torch.Tensor
    feature_by_view: Optional[
        dict[SharedTextureViewName, torch.Tensor]
    ] = None


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
        feature_objective: FeatureObjectiveKind,
        feature_view_mode: FeatureViewModeKind = "primary",
        view_sampler: ViewSampler = build_single_view_samples,
        frame_batch_sampler: FrameBatchSampler = (
            sample_uniform_frame_batch
        ),
        render_resolution: int = 256,
    ) -> None:
        self._cfg: OptimizationConfig = cfg
        self._model: OptimizationModel = model
        self._renderer: OptimizationRenderer = renderer
        self._feature_objective: FeatureObjectiveKind = feature_objective
        self._feature_view_mode: FeatureViewModeKind = feature_view_mode
        if (
            feature_view_mode == "primary_wrist"
            and feature_objective != "siglip_patch"
        ):
            raise ValueError(
                "primary_wrist 只支持 feature_objective='siglip_patch'"
            )
        self._view_sampler: ViewSampler = view_sampler
        self._frame_batch_sampler: FrameBatchSampler = (
            frame_batch_sampler
        )
        self._render_resolution: int = render_resolution
        # 双视角真实运行时只打印一次分视角诊断，既能确认 wrist 分支确实接入
        # loss，又避免数千次优化迭代把日志淹没。该状态不参与任何数值计算。
        self._dual_view_diagnostics_logged: bool = False

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
    ) -> FrameObjectiveLosses:
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
            # [1, 6, model_height, model_width]。这里故意保留历史
            # SigLIP→DINOv2 拼接顺序，使 last_hidden/action 基线行为不变；
            # 新 siglip_patch 目标不复用这条路径。
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
                self._compute_feature_loss(
                    frame=frame,
                    resized_image=resized_image,
                    model_outputs=outputs,
                )
            )

        # 当前 single-view adapter 保证至少一个视图。未来 adapter 也必须遵守
        # 这一约束，否则 stack 会明确报错而不是产生无意义的零 loss。
        mean_action_loss: torch.Tensor = torch.stack(action_losses).mean()
        mean_feature_loss: torch.Tensor = torch.stack(feature_losses).mean()
        return FrameObjectiveLosses(
            action=mean_action_loss,
            feature=mean_feature_loss,
        )

    def _compute_feature_loss(
        self,
        *,
        frame: TrainingFrame,
        resized_image: torch.Tensor,
        model_outputs: Any,
    ) -> torch.Tensor:
        """计算当前配置指定的负特征距离。

        optimizer 执行梯度下降；使用负 MSE 会主动增大对抗图像与干净图像的
        feature 距离。``last_hidden`` 完整保留历史行为。``siglip_patch`` 只
        读取正确归一化的三通道 SigLIP 输入，并直接调用共享视觉分支。
        """
        if self._feature_objective == "last_hidden":
            return -F.mse_loss(
                model_outputs.hidden_states[-1],
                frame["clean_hidden"],
            )

        clean_siglip_features: Optional[torch.Tensor] = frame[
            "clean_siglip_features"
        ]
        if clean_siglip_features is None:
            raise RuntimeError(
                "siglip_patch objective 缺少 clean_siglip_features；"
                "请使用相同 feature objective 重新采集训练帧"
            )
        return self._compute_siglip_feature_loss(
            frame=frame,
            resized_image=resized_image,
            clean_siglip_features=clean_siglip_features,
        )

    def _compute_siglip_feature_loss(
        self,
        *,
        frame: TrainingFrame,
        resized_image: torch.Tensor,
        clean_siglip_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算一个相机视角的 Shared-SigLIP 负 MSE。"""
        normalized_adversarial_siglip: torch.Tensor = (
            (resized_image - frame["siglip_mean"])
            / frame["siglip_std"]
        ).to(torch.bfloat16)
        with autocast(dtype=torch.bfloat16):
            adversarial_siglip_features: torch.Tensor = (
                extract_siglip_patch_features(
                    self._model,
                    normalized_adversarial_siglip,
                )
            )
        if (
            adversarial_siglip_features.shape
            != clean_siglip_features.shape
        ):
            raise RuntimeError(
                "干净/对抗 SigLIP feature shape 不一致："
                f"{tuple(clean_siglip_features.shape)} != "
                f"{tuple(adversarial_siglip_features.shape)}"
            )
        return -F.mse_loss(
            adversarial_siglip_features,
            clean_siglip_features,
        )

    def _compute_dual_view_losses(
        self,
        frame: TrainingFrame,
    ) -> FrameObjectiveLosses:
        """Action仅用主视角，Feature在主视角与腕部等权平均。

        每个相机视角先把所有共享纹理 body 实例合成为一张 RGB。腕部图像不会
        进入 OpenVLA language/action forward，避免把单主视角策略强行当成腕部
        动作模型；它只进入两模型共有的 SigLIP encoder。
        """
        views: tuple[MultiInstanceViewFrame, ...] = frame[
            "shared_texture_views"
        ]
        view_by_name: dict[str, MultiInstanceViewFrame] = {
            view["view_name"]: view for view in views
        }
        if set(view_by_name) != {"primary", "wrist"}:
            raise RuntimeError(
                "primary_wrist 模式要求且只允许 primary/wrist 两个视角"
            )

        resized_by_name: dict[str, torch.Tensor] = {}
        for view_name in ("primary", "wrist"):
            adversarial_image: torch.Tensor = (
                build_multi_instance_view_sample(
                    self._renderer,
                    view_by_name[view_name],
                    self._render_resolution,
                )
            )
            resized_by_name[view_name] = F.interpolate(
                adversarial_image,
                size=(
                    frame["model_input_size"],
                    frame["model_input_size"],
                ),
                mode="bilinear",
                align_corners=False,
            )

        primary_resized: torch.Tensor = resized_by_name["primary"]
        primary_pixel_values: torch.Tensor = torch.cat(
            (
                (primary_resized - frame["siglip_mean"])
                / frame["siglip_std"],
                (primary_resized - frame["dino_mean"])
                / frame["dino_std"],
            ),
            dim=1,
        )
        with autocast(dtype=torch.bfloat16):
            primary_outputs: Any = self._model(
                input_ids=frame["clean_output_ids"],
                attention_mask=torch.ones_like(frame["clean_output_ids"]),
                pixel_values=primary_pixel_values.to(torch.bfloat16),
                output_hidden_states=True,
            )
        action_loss: torch.Tensor = get_attack_loss(
            primary_outputs.logits,
            frame["clean_output_ids"],
        )
        feature_loss_by_name: dict[
            SharedTextureViewName, torch.Tensor
        ] = {}
        feature_view_name: SharedTextureViewName
        for feature_view_name in ("primary", "wrist"):
            feature_loss_by_name[feature_view_name] = (
                self._compute_siglip_feature_loss(
                    frame=frame,
                    resized_image=resized_by_name[feature_view_name],
                    clean_siglip_features=view_by_name[feature_view_name][
                        "clean_siglip_features"
                    ],
                )
            )
        feature_loss: torch.Tensor = torch.stack(
            tuple(feature_loss_by_name.values())
        ).mean()

        # 这是 GPU smoke 的运行时证据：两个相机都产生有限的 Feature loss，
        # 同时明确 Action 仍只读取主视角。独立的反向传播路径由单元测试覆盖；
        # 此处不额外调用 autograd，避免改变正式训练的梯度或显存占用。
        all_losses: tuple[torch.Tensor, ...] = (
            action_loss,
            feature_loss_by_name["primary"],
            feature_loss_by_name["wrist"],
        )
        if not all(bool(torch.isfinite(loss).item()) for loss in all_losses):
            raise RuntimeError("双视角 Action/Feature loss 出现非有限值")
        if not self._dual_view_diagnostics_logged:
            primary_view: MultiInstanceViewFrame = view_by_name["primary"]
            wrist_view: MultiInstanceViewFrame = view_by_name["wrist"]
            print(
                "[DUAL-VIEW] "
                "action_scope=primary_only, "
                f"instances=primary:{len(primary_view['instances'])}/"
                f"wrist:{len(wrist_view['instances'])}, "
                "feature_loss="
                f"primary:{feature_loss_by_name['primary'].detach().item():.6f}/"
                f"wrist:{feature_loss_by_name['wrist'].detach().item():.6f}"
            )
            self._dual_view_diagnostics_logged = True
        return FrameObjectiveLosses(
            action=action_loss,
            feature=feature_loss,
            feature_by_view=feature_loss_by_name,
        )

    def _compute_frame_losses(
        self,
        frame: TrainingFrame,
        frame_mvp: torch.Tensor,
    ) -> FrameObjectiveLosses:
        """按 feature view mode 分派，集中保持两个梯度入口行为一致。"""
        if self._feature_view_mode == "primary_wrist":
            return self._compute_dual_view_losses(frame)
        view_frame: SingleViewFrame = self._as_single_view_frame(
            frame,
            frame_mvp,
        )
        adversarial_views: Sequence[torch.Tensor] = self._view_sampler(
            self._renderer,
            view_frame,
            self._render_resolution,
        )
        return self._compute_view_losses(frame, adversarial_views)

    def compute_objective_parameter_gradients(
        self,
        frame: TrainingFrame,
    ) -> Optional[ObjectiveParameterGradients]:
        """在当前纹理处求单帧 Action/Feature 的独立参数梯度。

        该方法不读取 ``alpha``、frame weight，也不写入 ``parameter.grad``。
        谱基审计会在每帧调用前把系数清零，因此返回的两个浮点 tensor 均表示
        零 Surface Delta 参考点上的 ``[num_basis, 3]`` 梯度。Action 与
        Feature 共用一次图像构造/模型前向；第一次 autograd 保留 graph，第二次
        完成后立即释放。
        """
        frame_mvp: Optional[torch.Tensor] = frame["mvp"]
        if frame_mvp is None:
            return None
        frame_losses: FrameObjectiveLosses = self._compute_frame_losses(
            frame,
            frame_mvp,
        )
        texture_parameter: nn.Parameter = (
            self._renderer.get_texture_param()
        )

        def objective_gradient(
            loss: torch.Tensor,
            *,
            retain_graph: bool,
        ) -> torch.Tensor:
            gradient: Optional[torch.Tensor] = torch.autograd.grad(
                loss,
                texture_parameter,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True,
            )[0]
            resolved_gradient: torch.Tensor = (
                gradient
                if gradient is not None
                else torch.zeros_like(texture_parameter)
            )
            if not torch.isfinite(resolved_gradient).all():
                raise RuntimeError(
                    "独立目标关于纹理参数的梯度包含 NaN/Inf"
                )
            return resolved_gradient.detach()

        action_gradient: torch.Tensor = objective_gradient(
            frame_losses.action,
            retain_graph=True,
        )
        feature_gradient: torch.Tensor = objective_gradient(
            frame_losses.feature,
            retain_graph=False,
        )
        return ObjectiveParameterGradients(
            action_loss=float(frame_losses.action.detach().item()),
            feature_loss=float(frame_losses.feature.detach().item()),
            action_gradient=action_gradient,
            feature_gradient=feature_gradient,
        )

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
        log_header: str = (
            "Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | "
            "Update Rule | Actual Surface Step | Max Surface Delta"
        )
        if self._feature_view_mode == "primary_wrist":
            log_header += " | Primary Feature Loss | Wrist Feature Loss"
        resolved_log_path.write_text(log_header + "\n")

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
            average_primary_feature_loss: float = 0.0
            average_wrist_feature_loss: float = 0.0
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
                frame_losses: FrameObjectiveLosses = (
                    self._compute_frame_losses(
                        frame,
                        frame_mvp,
                    )
                )
                frame_loss: torch.Tensor = frame_weight * (
                    self._cfg.alpha_action * frame_losses.action
                    + self._cfg.alpha_feature * frame_losses.feature
                )
                frame_loss.backward()

                average_total_loss += frame_loss.item()
                average_action_loss += frame_losses.action.item()
                average_feature_loss += frame_losses.feature.item()
                if frame_losses.feature_by_view is not None:
                    average_primary_feature_loss += (
                        frame_losses.feature_by_view["primary"].item()
                    )
                    average_wrist_feature_loss += (
                        frame_losses.feature_by_view["wrist"].item()
                    )
                valid_frame_count += 1

            if valid_frame_count > 0:
                average_total_loss /= valid_frame_count
                average_action_loss /= valid_frame_count
                average_feature_loss /= valid_frame_count
                average_primary_feature_loss /= valid_frame_count
                average_wrist_feature_loss /= valid_frame_count
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
                log_line: str = (
                    f"{iteration_index:02d} | "
                    f"{average_total_loss:.6f} | "
                    f"{average_action_loss:.6f} | "
                    f"{average_feature_loss:.6f} | "
                    f"{gradient_norm:.6e} | "
                    f"{update_rule} | "
                    f"{actual_surface_step:.6e} | "
                    f"{max_surface_delta:.6e}"
                )
                if self._feature_view_mode == "primary_wrist":
                    log_line += (
                        f" | {average_primary_feature_loss:.6f}"
                        f" | {average_wrist_feature_loss:.6f}"
                    )
                log_file.write(log_line + "\n")
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
