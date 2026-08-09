"""正式 Fixed-Support Action trainer 与校准共享的更新核心。

该核心只定义一次更新的数值语义：对紧凑 ``delta_S`` 的 Action 梯度调用
renderer 的 ``surface_normalized_step_``，使用同一 ``surface_step`` 并由底层
执行 Surface-L∞ projection。Spectral Guard Calibration 与后续正式 trainer
必须持有同一个类，禁止各自复制更新公式。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol

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


class FixedSupportActionTrainerCore:
    """Action-only正式训练与Spectral Guard共用的Surface更新器。"""

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

    def apply_action_gradient(self, gradient: Tensor) -> SurfaceStepStats:
        """执行与正式训练完全相同的一次Action梯度下降和L∞投影。"""

        if gradient.shape != self.texture_parameter.shape:
            raise FixedSupportTrainingError("Action梯度shape与紧凑参数不匹配")
        if not bool(torch.isfinite(gradient).all()):
            raise FixedSupportTrainingError("Action梯度包含NaN/Inf")
        stats = self._renderer.step_surface_parameterization_(
            gradient,
            self.surface_step,
        )
        self.update_count += 1
        return stats

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
