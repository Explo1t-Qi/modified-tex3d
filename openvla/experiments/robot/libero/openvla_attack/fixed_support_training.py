"""正式Fixed-Support Action+Spectral trainer与校准共享的更新核心。

该核心只定义一次更新的数值语义：校准可直接提交Action梯度；正式训练先在紧凑
``delta_S`` 中形成 ``g_A + lambda_spec*g_S``，再调用一次renderer的
``surface_normalized_step_``，由底层执行相同步长与Surface-L∞ projection。
两条路径必须持有同一个类，禁止复制更新公式或先后执行两次更新。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

import torch
from torch import Tensor

from .texture_parameterization import SurfaceStepStats


class FixedSupportTrainingError(RuntimeError):
    """Fixed-Support trainer更新配置或renderer接口不满足契约。"""


class FixedSupportTrainingRenderer(Protocol):
    """正式trainer与校准共同依赖的最小renderer接口。"""

    epsilon: float

    def get_texture_parameterization_name(self) -> str: ...

    def get_texture_param(self) -> torch.nn.Parameter: ...

    def get_geometry_surface_delta(self) -> Tensor: ...

    def step_surface_parameterization_(
        self,
        gradient: Tensor,
        surface_step: float,
    ) -> SurfaceStepStats: ...


@dataclass(frozen=True)
class CombinedGradientUpdate:
    """一次联合梯度及其唯一Surface update的可审计结果。

    梯度tensor均位于紧凑Fixed-Support参数空间，dtype/device/shape与renderer的
    ``delta_S [N_support,3]`` 一致。它们只用于当前训练步和证据落盘，不持有
    需要继续反传的计算图。
    """

    weighted_spectral_gradient: Tensor
    total_gradient: Tensor
    action_gradient_l2: float
    spectral_gradient_l2: float
    weighted_spectral_gradient_l2: float
    total_gradient_l2: float
    action_spectral_cosine: Optional[float]
    weighted_spectral_action_ratio: Optional[float]
    combination_residual_linf: float
    surface_step_stats: SurfaceStepStats


class FixedSupportTrainerCore:
    """正式Action+Spectral训练与校准共用的唯一Surface更新器。"""

    def __init__(
        self,
        renderer: FixedSupportTrainingRenderer,
        *,
        surface_step: float,
    ) -> None:
        if renderer.get_texture_parameterization_name() != "fixed_support":
            raise FixedSupportTrainingError("正式更新器只接受fixed_support renderer")
        if not math.isfinite(surface_step) or surface_step <= 0.0:
            raise FixedSupportTrainingError("surface_step必须为有限正数")
        if surface_step > float(renderer.epsilon):
            raise FixedSupportTrainingError("surface_step不得大于Surface-Linf预算")
        self._renderer = renderer
        self.surface_step: float = float(surface_step)
        self.update_count: int = 0

    @property
    def texture_parameter(self) -> torch.nn.Parameter:
        return self._renderer.get_texture_param()

    def geometry_delta(self) -> Tensor:
        """返回谱能量使用的原始OBJ几何顶点域 ``[N_v,3]`` delta。"""

        return self._renderer.get_geometry_surface_delta()

    def _validate_gradient(self, gradient: Tensor, *, name: str) -> None:
        if gradient.shape != self.texture_parameter.shape:
            raise FixedSupportTrainingError(
                f"{name}梯度shape与紧凑参数不匹配"
            )
        if not gradient.is_floating_point():
            raise FixedSupportTrainingError(f"{name}梯度必须为浮点tensor")
        if not bool(torch.isfinite(gradient).all()):
            raise FixedSupportTrainingError(f"{name}梯度包含NaN/Inf")

    def _apply_gradient(self, gradient: Tensor) -> SurfaceStepStats:
        self._validate_gradient(gradient, name="训练")
        stats = self._renderer.step_surface_parameterization_(
            gradient,
            self.surface_step,
        )
        self.update_count += 1
        return stats

    def apply_action_gradient(self, gradient: Tensor) -> SurfaceStepStats:
        """执行与正式训练完全相同的一次Action梯度下降和L∞投影。"""

        self._validate_gradient(gradient, name="Action")
        return self._apply_gradient(gradient)

    def apply_action_spectral_gradients(
        self,
        action_gradient: Tensor,
        spectral_gradient: Tensor,
        *,
        lambda_spec: float,
    ) -> CombinedGradientUpdate:
        """先合成 ``g_A + lambda*g_S``，再且仅再执行一次Surface update。"""

        self._validate_gradient(action_gradient, name="Action")
        self._validate_gradient(spectral_gradient, name="Spectral")
        resolved_lambda = float(lambda_spec)
        if not math.isfinite(resolved_lambda) or not 0.0 <= resolved_lambda <= 1.0:
            raise FixedSupportTrainingError("lambda_spec必须是[0,1]内有限数")

        # weighted/total: float [N_support,3]，与参数保持相同dtype/device。
        weighted_spectral = spectral_gradient * resolved_lambda
        total_gradient = action_gradient + weighted_spectral
        if not bool(torch.isfinite(total_gradient).all()):
            raise FixedSupportTrainingError("联合训练梯度包含NaN/Inf")

        action_norm_tensor = torch.linalg.vector_norm(action_gradient)
        spectral_norm_tensor = torch.linalg.vector_norm(spectral_gradient)
        weighted_norm_tensor = torch.linalg.vector_norm(weighted_spectral)
        total_norm_tensor = torch.linalg.vector_norm(total_gradient)
        action_norm = float(action_norm_tensor.item())
        spectral_norm = float(spectral_norm_tensor.item())
        weighted_norm = float(weighted_norm_tensor.item())
        total_norm = float(total_norm_tensor.item())
        cosine: Optional[float]
        if action_norm == 0.0 or spectral_norm == 0.0:
            cosine = None
        else:
            cosine = float(
                torch.dot(
                    action_gradient.reshape(-1),
                    spectral_gradient.reshape(-1),
                ).item()
                / (action_norm * spectral_norm)
            )
        recombined = action_gradient + weighted_spectral
        residual_linf = float(
            (total_gradient - recombined).abs().amax().item()
        )
        step_stats = self._apply_gradient(total_gradient)
        return CombinedGradientUpdate(
            weighted_spectral_gradient=weighted_spectral.detach(),
            total_gradient=total_gradient.detach(),
            action_gradient_l2=action_norm,
            spectral_gradient_l2=spectral_norm,
            weighted_spectral_gradient_l2=weighted_norm,
            total_gradient_l2=total_norm,
            action_spectral_cosine=cosine,
            weighted_spectral_action_ratio=(
                None if action_norm == 0.0 else weighted_norm / action_norm
            ),
            combination_residual_linf=residual_linf,
            surface_step_stats=step_stats,
        )

    def state_dict(self) -> Mapping[str, Any]:
        """供Spectral Guard事务恢复更新计数与冻结步长。"""

        return {
            "surface_step": self.surface_step,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        loaded_step = float(state_dict["surface_step"])
        if loaded_step != self.surface_step:
            raise FixedSupportTrainingError("恢复的surface_step与冻结配置不一致")
        loaded_count = int(state_dict["update_count"])
        if loaded_count < 0:
            raise FixedSupportTrainingError("update_count不得为负数")
        self.update_count = loaded_count


# 兼容校准实现基线的旧导入名；新代码必须使用能表达联合目标的正式名称。
FixedSupportActionTrainerCore = FixedSupportTrainerCore
