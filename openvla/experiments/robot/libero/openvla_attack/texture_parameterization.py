"""将低维可学习参数映射为曲面 RGB 增量。

``SurfaceTextureParameterization`` 是纹理优化的核心 seam。它把具体参数空间
隐藏在统一 interface 后：

- ``GeometryVertexTextureParameterization``：每个几何顶点三个自由参数，
  用作与谱方法公平比较的高维基线；
- ``FixedSupportTextureParameterization``：只为外部冻结 Support 中的顶点
  分配紧凑 ``[|S|,3]`` 参数，再散射到完整几何顶点域；
- ``SpectralTextureParameterization``：仅优化 ``[K, 3]`` 谱系数，通过
  ``Phi @ coefficients`` 得到 ``[N, 3]`` 曲面增量。

两种 adapter 使用相同的曲面 L∞ 预算、相同的 UV 保留渲染路径和相同的
surface-normalized 更新，因此实验差异只来自参数空间本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypeAlias

import numpy as np
import torch
import torch.nn as nn


Tensor: TypeAlias = torch.Tensor


class TextureParameterizationError(ValueError):
    """纹理参数 shape、数值或预算不满足约束。"""


class SurfaceTextureParameterization(Protocol):
    """优化器与 renderer 共同依赖的最小参数化 interface。"""

    epsilon: float
    coefficients: nn.Parameter

    def reset_parameters(self) -> None:
        """把可学习参数原地清零。"""
        ...

    def geometry_delta(self) -> Tensor:
        """返回 float ``[N, 3]``、满足 L∞ 预算的几何曲面增量。"""
        ...

    def render_delta(self) -> Tensor:
        """返回 float ``[V, 3]``、与渲染 mesh 顶点对齐的曲面增量。"""
        ...

    def parameter_direction_to_geometry(self, direction: Tensor) -> Tensor:
        """把参数空间方向映射为 float ``[N, 3]`` 曲面方向。"""
        ...

    def project_coefficients_(self) -> float:
        """在参数空间内原地投影，返回全局缩放因子。"""
        ...

    def max_abs_delta(self) -> float:
        """返回当前曲面增量最大绝对值。"""
        ...


def _validate_epsilon(epsilon: float) -> float:
    resolved_epsilon: float = float(epsilon)
    if not np.isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise TextureParameterizationError("epsilon 必须为有限正数")
    return resolved_epsilon


def _functional_projection_scale(delta: Tensor, epsilon: float) -> Tensor:
    """计算保持参数子空间不变的全局 L∞ 投影比例。"""
    maximum_absolute_delta: Tensor = delta.abs().amax()
    epsilon_tensor: Tensor = delta.new_tensor(epsilon)
    numeric_tiny: float = torch.finfo(delta.dtype).tiny
    return torch.clamp(
        epsilon_tensor / maximum_absolute_delta.clamp_min(numeric_tiny),
        max=1.0,
    )


class SpectralTextureParameterization(nn.Module):
    """``[K, 3]`` 谱系数定义的低维 RGB 曲面增量。"""

    def __init__(
        self,
        basis: np.ndarray | Tensor,
        render_to_geometry: np.ndarray | Tensor,
        epsilon: float,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # basis_tensor: float [num_geometry_vertices, num_basis]。
        basis_tensor: Tensor = torch.as_tensor(
            basis,
            dtype=dtype,
            device=device,
        )
        # mapping_tensor: int64 [num_render_vertices]。
        mapping_tensor: Tensor = torch.as_tensor(
            render_to_geometry,
            dtype=torch.long,
            device=device,
        )
        if basis_tensor.ndim != 2 or basis_tensor.numel() == 0:
            raise TextureParameterizationError(
                "basis 必须为非空 [num_geometry_vertices, num_basis]"
            )
        if not torch.isfinite(basis_tensor).all():
            raise TextureParameterizationError("basis 包含非有限值")
        if mapping_tensor.ndim != 1 or mapping_tensor.numel() == 0:
            raise TextureParameterizationError(
                "render_to_geometry 必须为非空一维向量"
            )
        if (
            mapping_tensor.min().item() < 0
            or mapping_tensor.max().item() >= basis_tensor.shape[0]
        ):
            raise TextureParameterizationError(
                "render_to_geometry 包含越界索引"
            )

        self.epsilon: float = _validate_epsilon(epsilon)
        self.register_buffer("basis", basis_tensor.contiguous())
        self.register_buffer(
            "render_to_geometry",
            mapping_tensor.contiguous(),
        )
        # coefficients: float [num_basis, 3]。
        self.coefficients: nn.Parameter = nn.Parameter(
            torch.zeros(
                basis_tensor.shape[1],
                3,
                dtype=dtype,
                device=basis_tensor.device,
            )
        )

    @property
    def num_geometry_vertices(self) -> int:
        return int(self.basis.shape[0])

    @property
    def num_render_vertices(self) -> int:
        return int(self.render_to_geometry.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.basis.shape[1])

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.coefficients.zero_()

    def unprojected_geometry_delta(self) -> Tensor:
        """返回投影前 float ``[N, 3]`` 的谱重建结果。"""
        return self.basis @ self.coefficients

    def geometry_delta(self) -> Tensor:
        """返回仍在谱子空间内且满足 L∞ 预算的 ``[N, 3]`` 增量。"""
        raw_delta: Tensor = self.unprojected_geometry_delta()
        return raw_delta * _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )

    def render_delta(self) -> Tensor:
        return self.geometry_delta()[self.render_to_geometry]

    def parameter_direction_to_geometry(self, direction: Tensor) -> Tensor:
        if direction.shape != self.coefficients.shape:
            raise TextureParameterizationError(
                f"谱系数方向 shape {tuple(direction.shape)} 与参数 "
                f"{tuple(self.coefficients.shape)} 不一致"
            )
        return self.basis @ direction

    @torch.no_grad()
    def project_coefficients_(self) -> float:
        raw_delta: Tensor = self.unprojected_geometry_delta()
        scale: Tensor = _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )
        self.coefficients.mul_(scale)
        return float(scale.item())

    @torch.no_grad()
    def max_abs_delta(self) -> float:
        return float(self.geometry_delta().abs().amax().item())


class GeometryVertexTextureParameterization(nn.Module):
    """每个 OBJ 几何顶点独立优化 RGB，但 UV seam 副本共享同一增量。"""

    def __init__(
        self,
        render_to_geometry: np.ndarray | Tensor,
        num_geometry_vertices: int,
        epsilon: float,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        mapping_tensor: Tensor = torch.as_tensor(
            render_to_geometry,
            dtype=torch.long,
            device=device,
        )
        if num_geometry_vertices <= 0:
            raise TextureParameterizationError(
                "num_geometry_vertices 必须为正数"
            )
        if mapping_tensor.ndim != 1 or mapping_tensor.numel() == 0:
            raise TextureParameterizationError(
                "render_to_geometry 必须为非空一维向量"
            )
        if (
            mapping_tensor.min().item() < 0
            or mapping_tensor.max().item() >= num_geometry_vertices
        ):
            raise TextureParameterizationError(
                "render_to_geometry 包含越界索引"
            )

        self.epsilon: float = _validate_epsilon(epsilon)
        self.register_buffer(
            "render_to_geometry",
            mapping_tensor.contiguous(),
        )
        # coefficients: float [num_geometry_vertices, 3]。
        self.coefficients: nn.Parameter = nn.Parameter(
            torch.zeros(
                num_geometry_vertices,
                3,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def num_geometry_vertices(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def num_render_vertices(self) -> int:
        return int(self.render_to_geometry.shape[0])

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.coefficients.zero_()

    def unprojected_geometry_delta(self) -> Tensor:
        return self.coefficients

    def geometry_delta(self) -> Tensor:
        raw_delta: Tensor = self.unprojected_geometry_delta()
        return raw_delta * _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )

    def render_delta(self) -> Tensor:
        return self.geometry_delta()[self.render_to_geometry]

    def parameter_direction_to_geometry(self, direction: Tensor) -> Tensor:
        if direction.shape != self.coefficients.shape:
            raise TextureParameterizationError(
                f"顶点方向 shape {tuple(direction.shape)} 与参数 "
                f"{tuple(self.coefficients.shape)} 不一致"
            )
        return direction

    @torch.no_grad()
    def project_coefficients_(self) -> float:
        raw_delta: Tensor = self.unprojected_geometry_delta()
        scale: Tensor = _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )
        self.coefficients.mul_(scale)
        return float(scale.item())

    @torch.no_grad()
    def max_abs_delta(self) -> float:
        return float(self.geometry_delta().abs().amax().item())


class FixedSupportTextureParameterization(nn.Module):
    """外部 Fixed Vertex Support 上的紧凑独立 RGB 参数化。

    本类只消费 Support，不生成、扩张或修改它。``support_vertex_indices`` 必须
    来自 Support Construction 的冻结产物；在正式产物存在前，测试只能注入
    合成索引。可学习 ``coefficients`` shape 为 ``[num_support_vertices,3]``，
    :meth:`geometry_delta` 再按索引散射成 ``[num_geometry_vertices,3]``，因此
    Support 外不但 Surface Delta 恒为零，也不存在伪可学习参数。
    """

    def __init__(
        self,
        render_to_geometry: np.ndarray | Tensor,
        num_geometry_vertices: int,
        support_vertex_indices: np.ndarray | Tensor | Sequence[int],
        epsilon: float,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        mapping_tensor: Tensor = torch.as_tensor(
            render_to_geometry,
            dtype=torch.long,
            device=device,
        )
        raw_support_tensor: Tensor = torch.as_tensor(
            support_vertex_indices,
            device=device,
        )
        if num_geometry_vertices <= 0:
            raise TextureParameterizationError(
                "num_geometry_vertices 必须为正数"
            )
        if mapping_tensor.ndim != 1 or mapping_tensor.numel() == 0:
            raise TextureParameterizationError(
                "render_to_geometry 必须为非空一维向量"
            )
        if (
            mapping_tensor.min().item() < 0
            or mapping_tensor.max().item() >= num_geometry_vertices
        ):
            raise TextureParameterizationError(
                "render_to_geometry 包含越界索引"
            )
        if raw_support_tensor.ndim != 1:
            raise TextureParameterizationError(
                "support_vertex_indices 必须为一维向量"
            )
        if raw_support_tensor.numel() == 0:
            raise TextureParameterizationError(
                "support_vertex_indices 不得为空"
            )
        if raw_support_tensor.dtype == torch.bool or (
            raw_support_tensor.is_floating_point()
            or raw_support_tensor.is_complex()
        ):
            raise TextureParameterizationError(
                "support_vertex_indices 必须使用整数 dtype"
            )
        support_tensor: Tensor = raw_support_tensor.to(dtype=torch.long)
        if (
            support_tensor.min().item() < 0
            or support_tensor.max().item() >= num_geometry_vertices
        ):
            raise TextureParameterizationError(
                "support_vertex_indices 包含越界索引"
            )
        if int(torch.unique(support_tensor).numel()) != int(
            support_tensor.numel()
        ):
            raise TextureParameterizationError(
                "support_vertex_indices 包含重复顶点"
            )

        self.epsilon: float = _validate_epsilon(epsilon)
        self._num_geometry_vertices: int = int(num_geometry_vertices)
        self.register_buffer(
            "render_to_geometry",
            mapping_tensor.contiguous(),
        )
        self.register_buffer(
            "support_vertex_indices",
            support_tensor.contiguous(),
        )
        # coefficients: float [num_support_vertices, 3]，即冻结决策中的
        # 实际可学习 RGB 参数空间 delta_S。
        self.coefficients: nn.Parameter = nn.Parameter(
            torch.zeros(
                support_tensor.numel(),
                3,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def num_geometry_vertices(self) -> int:
        return self._num_geometry_vertices

    @property
    def num_render_vertices(self) -> int:
        return int(self.render_to_geometry.shape[0])

    @property
    def num_support_vertices(self) -> int:
        return int(self.support_vertex_indices.shape[0])

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.coefficients.zero_()

    def unprojected_geometry_delta(self) -> Tensor:
        """把紧凑 ``delta_S`` 散射为未投影的完整 ``[N,3]`` 增量。"""

        full_delta: Tensor = self.coefficients.new_zeros(
            (self.num_geometry_vertices, 3)
        )
        return full_delta.index_copy(
            0,
            self.support_vertex_indices,
            self.coefficients,
        )

    def geometry_delta(self) -> Tensor:
        raw_delta: Tensor = self.unprojected_geometry_delta()
        return raw_delta * _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )

    def render_delta(self) -> Tensor:
        return self.geometry_delta()[self.render_to_geometry]

    def parameter_direction_to_geometry(self, direction: Tensor) -> Tensor:
        if direction.shape != self.coefficients.shape:
            raise TextureParameterizationError(
                f"Fixed Support方向 shape {tuple(direction.shape)} 与参数 "
                f"{tuple(self.coefficients.shape)} 不一致"
            )
        full_direction: Tensor = direction.new_zeros(
            (self.num_geometry_vertices, 3)
        )
        return full_direction.index_copy(
            0,
            self.support_vertex_indices,
            direction,
        )

    @torch.no_grad()
    def project_coefficients_(self) -> float:
        raw_delta: Tensor = self.unprojected_geometry_delta()
        scale: Tensor = _functional_projection_scale(
            raw_delta,
            self.epsilon,
        )
        self.coefficients.mul_(scale)
        return float(scale.item())

    @torch.no_grad()
    def max_abs_delta(self) -> float:
        return float(self.geometry_delta().abs().amax().item())


@dataclass(frozen=True)
class SurfaceStepStats:
    """一次 surface-normalized 更新的可记录统计量。"""

    direction_surface_max: float
    parameter_scale: float
    projection_scale: float
    step_cap_scale: float
    actual_surface_step: float
    max_abs_delta: float


@torch.no_grad()
def surface_normalized_step_(
    parameterization: SurfaceTextureParameterization,
    gradient: Tensor,
    surface_step: float,
) -> SurfaceStepStats:
    """执行最大 RGB 曲面变化相同的下降更新。

    参数梯度先映射到几何曲面，再做全局缩放，使本轮最大绝对曲面变化不超过
    ``surface_step``。随后使用同样的全局缩放执行 L∞ 投影，因此谱方法不会
    离开选定子空间，Geometry Vertex 与 Spectral 也获得相同的曲面步长语义。
    """
    if gradient.shape != parameterization.coefficients.shape:
        raise TextureParameterizationError(
            f"梯度 shape {tuple(gradient.shape)} 与参数 "
            f"{tuple(parameterization.coefficients.shape)} 不一致"
        )
    if not torch.isfinite(gradient).all():
        raise TextureParameterizationError("纹理梯度包含非有限值")
    resolved_surface_step: float = float(surface_step)
    if not np.isfinite(resolved_surface_step) or resolved_surface_step <= 0:
        raise TextureParameterizationError("surface_step 必须为有限正数")

    # 先固化已有 functional projection。两个端点都在 L∞ 球内时，后续凸插值
    # 也必然在球内。
    parameterization.project_coefficients_()
    before_coefficients: Tensor = parameterization.coefficients.clone()
    before_delta: Tensor = parameterization.geometry_delta().clone()
    parameter_direction: Tensor = -gradient
    surface_direction: Tensor = (
        parameterization.parameter_direction_to_geometry(
            parameter_direction
        )
    )
    direction_surface_max: Tensor = surface_direction.abs().amax()
    numeric_tiny: float = torch.finfo(surface_direction.dtype).tiny

    if float(direction_surface_max.item()) <= numeric_tiny:
        return SurfaceStepStats(
            direction_surface_max=0.0,
            parameter_scale=0.0,
            projection_scale=1.0,
            step_cap_scale=1.0,
            actual_surface_step=0.0,
            max_abs_delta=parameterization.max_abs_delta(),
        )

    parameter_scale: Tensor = (
        direction_surface_max.new_tensor(resolved_surface_step)
        / direction_surface_max
    )
    parameterization.coefficients.add_(
        parameter_scale * parameter_direction
    )
    projection_scale: float = parameterization.project_coefficients_()
    after_delta: Tensor = parameterization.geometry_delta()
    actual_surface_step: Tensor = (
        after_delta - before_delta
    ).abs().amax()

    step_cap_scale: float = 1.0
    if float(actual_surface_step.item()) > resolved_surface_step:
        step_cap: Tensor = (
            actual_surface_step.new_tensor(resolved_surface_step)
            / actual_surface_step
        )
        parameterization.coefficients.copy_(
            before_coefficients
            + step_cap
            * (parameterization.coefficients - before_coefficients)
        )
        step_cap_scale = float(step_cap.item())
        after_delta = parameterization.geometry_delta()
        actual_surface_step = (after_delta - before_delta).abs().amax()

    return SurfaceStepStats(
        direction_surface_max=float(direction_surface_max.item()),
        parameter_scale=float(parameter_scale.item()),
        projection_scale=projection_scale,
        step_cap_scale=step_cap_scale,
        actual_surface_step=float(actual_surface_step.item()),
        max_abs_delta=float(after_delta.abs().amax().item()),
    )
