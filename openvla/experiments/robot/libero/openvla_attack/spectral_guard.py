"""Action-only Spectral Guard 的可微正则、校准窗口与状态事务。

本模块不依赖 LIBERO 或 VLA。GPU runner 只需提供当轮全部有效训练帧的平均
Action hinge，以及一次只使用 Action 梯度的 surface-normalized update。本模块
负责谱正则梯度、首个连续5轮稳定激活窗口、``lambda_spec`` 计算和校准前状态的
严格恢复；任何恢复失败都阻止返回可消费的校准结果。
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from .production_support import array_sha256
from .seed_score_audit import file_sha256
from .spectral_geometry import load_spectral_basis
from .spectral_naturalness import (
    NATURALNESS_NUM_LOW_MODES,
    SpectralNaturalnessCalibration,
    SpectralNaturalnessError,
    load_rho_nat_calibration_artifact,
)
from .texture_parameterization import SurfaceStepStats


class SpectralGuardError(RuntimeError):
    """谱正则、校准梯度或状态恢复违反冻结契约。"""


class SpectralGuardTerms(NamedTuple):
    """一次可微谱自然性计算的标量tensor。"""

    total_energy: Tensor
    low_energy: Tensor
    high_energy: Tensor
    high_ratio: Tensor
    diagnostic_high_ratio: Tensor
    hinge: Tensor
    penalty: Tensor


class StatefulComponent(Protocol):
    """校准事务内需要保存和恢复的更新器或sampler。"""

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> Any: ...


class SpectralNaturalnessRegularizer(nn.Module):
    """在完整几何 ``Surface Delta [N,3]`` 上计算谱自然性hinge。"""

    def __init__(
        self,
        mass: np.ndarray | Tensor,
        low_basis: np.ndarray | Tensor,
        *,
        rho_nat: float,
        energy_epsilon: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        mass_tensor = torch.as_tensor(mass, dtype=dtype, device=device)
        basis_tensor = torch.as_tensor(low_basis, dtype=dtype, device=device)
        if mass_tensor.ndim != 1 or torch.any(mass_tensor <= 0):
            raise SpectralGuardError("mass必须为正的一维tensor")
        if basis_tensor.shape != (
            mass_tensor.numel(),
            NATURALNESS_NUM_LOW_MODES,
        ):
            raise SpectralGuardError("low_basis必须为[N,129]")
        if not math.isfinite(rho_nat) or not 0.0 <= rho_nat <= 1.0:
            raise SpectralGuardError("rho_nat必须位于[0,1]")
        if not math.isfinite(energy_epsilon) or energy_epsilon <= 0.0:
            raise SpectralGuardError("energy_epsilon必须为有限正数")
        self.register_buffer("mass", mass_tensor.contiguous())
        self.register_buffer("low_basis", basis_tensor.contiguous())
        self.rho_nat: float = float(rho_nat)
        self.energy_epsilon: float = float(energy_epsilon)

    @classmethod
    def from_artifacts(
        cls,
        calibration_path: str | Path,
        spectral_basis_path: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SpectralNaturalnessRegularizer":
        """验证校准与谱基hash后构造训练期只读正则。"""

        calibration: SpectralNaturalnessCalibration = (
            load_rho_nat_calibration_artifact(calibration_path)
        )
        basis_path = Path(spectral_basis_path)
        if file_sha256(basis_path) != calibration.spectral_basis_artifact_sha256:
            raise SpectralNaturalnessError("训练谱基文件hash与rho_nat校准不匹配")
        basis_data = load_spectral_basis(
            basis_path,
            max_basis=NATURALNESS_NUM_LOW_MODES,
            include_constant=True,
        )
        if basis_data.mesh_sha256 != calibration.mesh_array_sha256:
            raise SpectralNaturalnessError("训练谱基mesh与rho_nat校准不匹配")
        if array_sha256(basis_data.mass) != calibration.mass_sha256:
            raise SpectralNaturalnessError("训练mass与rho_nat校准不匹配")
        if array_sha256(basis_data.basis) != calibration.low_basis_sha256:
            raise SpectralNaturalnessError("训练低频带与rho_nat校准不匹配")
        if array_sha256(basis_data.eigenvalues) != (
            calibration.low_band_eigenvalues_sha256
        ):
            raise SpectralNaturalnessError("训练特征值与rho_nat校准不匹配")
        return cls(
            basis_data.mass,
            basis_data.basis,
            rho_nat=calibration.rho_nat,
            energy_epsilon=calibration.energy_epsilon,
            device=device,
            dtype=dtype,
        )

    def forward(self, geometry_delta: Tensor) -> SpectralGuardTerms:
        """计算stopgrad分母训练比例及无stopgrad同值诊断比例。"""

        if geometry_delta.shape != (self.mass.numel(), 3):
            raise SpectralGuardError("geometry_delta必须为[N,3]")
        if not geometry_delta.is_floating_point():
            raise SpectralGuardError("geometry_delta必须为浮点tensor")
        mass = self.mass.to(dtype=geometry_delta.dtype)
        basis = self.low_basis.to(dtype=geometry_delta.dtype)
        weighted_delta = mass[:, None] * geometry_delta
        total_energy = torch.sum(weighted_delta * geometry_delta)
        coefficients = basis.transpose(0, 1) @ weighted_delta
        low_energy = torch.sum(coefficients.square())
        high_energy = torch.clamp(total_energy - low_energy, min=0.0)
        high_ratio = high_energy / (
            total_energy.detach() + self.energy_epsilon
        )
        diagnostic_high_ratio = high_energy / (
            total_energy + self.energy_epsilon
        )
        hinge = torch.relu(high_ratio - self.rho_nat)
        return SpectralGuardTerms(
            total_energy=total_energy,
            low_energy=low_energy,
            high_energy=high_energy,
            high_ratio=high_ratio,
            diagnostic_high_ratio=diagnostic_high_ratio,
            hinge=hinge,
            penalty=hinge.square(),
        )


@dataclass(frozen=True)
class RngStateSnapshot:
    """Python、NumPy、Torch CPU/CUDA RNG的拥有型快照。"""

    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...]


@dataclass(frozen=True)
class CalibrationStateSnapshot:
    """校准前全部可变状态与参数梯度缓存。"""

    module_state: dict[str, Any]
    component_states: dict[str, Any]
    parameter_gradients: dict[str, Optional[Tensor]]
    rng_state: RngStateSnapshot


@dataclass(frozen=True)
class StateRestoreEvidence:
    """校准后逐类状态的精确恢复结果。"""

    module_state_restored: bool
    component_states_restored: bool
    gradient_cache_restored: bool
    python_rng_restored: bool
    numpy_rng_restored: bool
    torch_cpu_rng_restored: bool
    torch_cuda_rng_restored: bool

    @property
    def all_restored(self) -> bool:
        return all(
            (
                self.module_state_restored,
                self.component_states_restored,
                self.gradient_cache_restored,
                self.python_rng_restored,
                self.numpy_rng_restored,
                self.torch_cpu_rng_restored,
                self.torch_cuda_rng_restored,
            )
        )


@dataclass(frozen=True)
class SpectralGuardIteration:
    """一次Action-only校准更新前的梯度与谱诊断。"""

    iteration: int
    action_loss: float
    num_action_frames: int
    action_state_ids: tuple[int, ...]
    action_state_fingerprints: tuple[str, ...]
    total_energy: float
    low_energy: float
    high_energy: float
    high_ratio: float
    diagnostic_high_ratio: float
    rho_nat: float
    hinge: float
    hinge_active: bool
    action_gradient_l2: float
    spectral_gradient_l2: float
    q_t: Optional[float]
    gradient_cosine: Optional[float]
    configured_surface_step: float
    surface_step_stats: SurfaceStepStats


@dataclass(frozen=True)
class SpectralGuardCalibrationResult:
    """恢复通过后才可消费的lambda_spec校准结果。"""

    iterations: tuple[SpectralGuardIteration, ...]
    selected_window_start: Optional[int]
    selected_window_end: Optional[int]
    q_median: Optional[float]
    lambda_spec: float
    calibration_status: str
    restore_evidence: StateRestoreEvidence


@dataclass(frozen=True)
class MeanActionGradient:
    """当轮全部有效训练帧Action hinge的算术均值及其参数梯度。"""

    loss: float
    gradient: Tensor
    num_frames: int
    state_ids: tuple[int, ...]
    state_fingerprints: tuple[str, ...]


def _clone_state(value: Any) -> Any:
    return copy.deepcopy(value)


def _capture_rng_state() -> RngStateSnapshot:
    return RngStateSnapshot(
        python_state=random.getstate(),
        numpy_state=_clone_state(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        )
        if torch.cuda.is_available()
        else (),
    )


def capture_calibration_state(
    mutable_module: nn.Module,
    *,
    stateful_components: Mapping[str, StatefulComponent],
) -> CalibrationStateSnapshot:
    """在校准更新前捕获参数、更新器、sampler、梯度缓存与RNG。"""

    return CalibrationStateSnapshot(
        module_state=_clone_state(mutable_module.state_dict()),
        component_states={
            name: _clone_state(component.state_dict())
            for name, component in stateful_components.items()
        },
        parameter_gradients={
            name: None if parameter.grad is None else parameter.grad.clone()
            for name, parameter in mutable_module.named_parameters()
        },
        rng_state=_capture_rng_state(),
    )


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and bool(
            torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and bool(
            np.array_equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def restore_calibration_state(
    mutable_module: nn.Module,
    snapshot: CalibrationStateSnapshot,
    *,
    stateful_components: Mapping[str, StatefulComponent],
) -> StateRestoreEvidence:
    """恢复全部状态并立即逐项精确验证。"""

    mutable_module.load_state_dict(_clone_state(snapshot.module_state))
    if set(stateful_components) != set(snapshot.component_states):
        raise SpectralGuardError("校准前后stateful component集合改变")
    for name, component in stateful_components.items():
        component.load_state_dict(_clone_state(snapshot.component_states[name]))
    for name, parameter in mutable_module.named_parameters():
        expected_gradient = snapshot.parameter_gradients[name]
        parameter.grad = (
            None
            if expected_gradient is None
            else expected_gradient.to(parameter.device).clone()
        )

    random.setstate(snapshot.rng_state.python_state)
    np.random.set_state(_clone_state(snapshot.rng_state.numpy_state))
    torch.set_rng_state(snapshot.rng_state.torch_cpu_state.clone())
    if snapshot.rng_state.torch_cuda_states:
        torch.cuda.set_rng_state_all(
            [state.clone() for state in snapshot.rng_state.torch_cuda_states]
        )

    current_rng = _capture_rng_state()
    current_gradients = {
        name: None if parameter.grad is None else parameter.grad
        for name, parameter in mutable_module.named_parameters()
    }
    evidence = StateRestoreEvidence(
        module_state_restored=_values_equal(
            mutable_module.state_dict(), snapshot.module_state
        ),
        component_states_restored=all(
            _values_equal(
                component.state_dict(), snapshot.component_states[name]
            )
            for name, component in stateful_components.items()
        ),
        gradient_cache_restored=_values_equal(
            current_gradients, snapshot.parameter_gradients
        ),
        python_rng_restored=_values_equal(
            current_rng.python_state, snapshot.rng_state.python_state
        ),
        numpy_rng_restored=_values_equal(
            current_rng.numpy_state, snapshot.rng_state.numpy_state
        ),
        torch_cpu_rng_restored=_values_equal(
            current_rng.torch_cpu_state,
            snapshot.rng_state.torch_cpu_state,
        ),
        torch_cuda_rng_restored=_values_equal(
            current_rng.torch_cuda_states,
            snapshot.rng_state.torch_cuda_states,
        ),
    )
    if not evidence.all_restored:
        raise SpectralGuardError(f"谱校准状态恢复失败: {evidence}")
    return evidence


def _gradient_metrics(
    action_gradient: Tensor,
    spectral_gradient: Tensor,
) -> tuple[float, float, Optional[float], Optional[float]]:
    action_norm = float(torch.linalg.vector_norm(action_gradient).item())
    spectral_norm = float(torch.linalg.vector_norm(spectral_gradient).item())
    if not math.isfinite(action_norm) or not math.isfinite(spectral_norm):
        raise SpectralGuardError("校准梯度范数包含NaN/Inf")
    if action_norm == 0.0 or spectral_norm == 0.0:
        return action_norm, spectral_norm, None, None
    q_t = spectral_norm / (action_norm + 1e-12)
    cosine = float(
        torch.sum(action_gradient * spectral_gradient).item()
        / (action_norm * spectral_norm)
    )
    return action_norm, spectral_norm, q_t, cosine


def _stable_window(
    rows: list[SpectralGuardIteration],
    *,
    window_size: int,
) -> Optional[tuple[SpectralGuardIteration, ...]]:
    if len(rows) < window_size:
        return None
    candidate = tuple(rows[-window_size:])
    if all(
        row.hinge_active
        and row.q_t is not None
        and math.isfinite(row.q_t)
        and row.action_gradient_l2 > 0.0
        and row.spectral_gradient_l2 > 0.0
        for row in candidate
    ):
        return candidate
    return None


def calibrate_spectral_guard(
    *,
    mutable_module: nn.Module,
    texture_parameter: nn.Parameter,
    geometry_delta_provider: Callable[[], Tensor],
    mean_action_gradient_provider: Callable[[], MeanActionGradient],
    action_only_update: Callable[[Tensor], SurfaceStepStats],
    regularizer: SpectralNaturalnessRegularizer,
    stateful_components: Mapping[str, StatefulComponent],
    expected_state_fingerprints: Mapping[int, str],
    surface_step: float,
    max_iterations: int = 64,
    stable_window_size: int = 5,
) -> SpectralGuardCalibrationResult:
    """沿Action-only轨迹校准lambda，并无条件恢复校准前状态。"""

    if max_iterations <= 0 or max_iterations > 64:
        raise SpectralGuardError("Spectral Guard最多允许64轮校准")
    if stable_window_size != 5:
        raise SpectralGuardError("第一版稳定激活窗口必须固定为连续5轮")
    expected_state_ids = tuple(sorted(expected_state_fingerprints))
    if expected_state_ids != tuple(range(10)):
        raise SpectralGuardError("正式校准必须固定且完整使用states 0-9")
    if any(
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        for fingerprint in expected_state_fingerprints.values()
    ):
        raise SpectralGuardError("正式校准state fingerprint必须为SHA-256")
    if not math.isfinite(surface_step) or surface_step <= 0.0:
        raise SpectralGuardError("surface_step必须为有限正数")
    initial_delta = geometry_delta_provider().detach()
    if float(initial_delta.abs().amax().item()) != 0.0:
        raise SpectralGuardError("Spectral Guard校准必须从零Surface Delta开始")

    snapshot = capture_calibration_state(
        mutable_module,
        stateful_components=stateful_components,
    )
    rows: list[SpectralGuardIteration] = []
    selected_window: Optional[tuple[SpectralGuardIteration, ...]] = None
    restore_evidence: Optional[StateRestoreEvidence] = None
    try:
        for iteration in range(max_iterations):
            action_sample = mean_action_gradient_provider()
            if not math.isfinite(action_sample.loss):
                raise SpectralGuardError("全部训练帧平均Action loss必须有限")
            if action_sample.num_frames <= 0:
                raise SpectralGuardError("Action梯度均值必须覆盖至少一个训练帧")
            if action_sample.num_frames != 10:
                raise SpectralGuardError("每轮Action梯度必须恰好覆盖10个训练state")
            if action_sample.state_ids != expected_state_ids:
                raise SpectralGuardError(
                    "每轮Action states必须按0-9完整、唯一且各参与一次"
                )
            if len(set(action_sample.state_ids)) != 10:
                raise SpectralGuardError("每轮Action states包含重复ID")
            expected_fingerprints = tuple(
                expected_state_fingerprints[state_id]
                for state_id in expected_state_ids
            )
            if action_sample.state_fingerprints != expected_fingerprints:
                raise SpectralGuardError("每轮Action state fingerprint绑定不一致")
            action_gradient = action_sample.gradient
            if action_gradient.shape != texture_parameter.shape:
                raise SpectralGuardError("Action均值梯度shape与纹理参数不匹配")
            if not bool(torch.isfinite(action_gradient).all()):
                raise SpectralGuardError("Action均值梯度包含NaN/Inf")
            geometry_delta = geometry_delta_provider()
            terms = regularizer(geometry_delta)
            spectral_gradient = torch.autograd.grad(
                terms.penalty,
                texture_parameter,
            )[0]
            action_norm, spectral_norm, q_t, cosine = _gradient_metrics(
                action_gradient,
                spectral_gradient,
            )
            step_stats = action_only_update(action_gradient.detach())
            step_values = (
                step_stats.direction_surface_max,
                step_stats.parameter_scale,
                step_stats.projection_scale,
                step_stats.step_cap_scale,
                step_stats.actual_surface_step,
                step_stats.max_abs_delta,
            )
            if not all(math.isfinite(value) for value in step_values):
                raise SpectralGuardError("SurfaceStepStats包含NaN/Inf")
            if step_stats.actual_surface_step > surface_step + 1e-7:
                raise SpectralGuardError("实际Surface step超过配置上限")
            row = SpectralGuardIteration(
                iteration=iteration,
                action_loss=float(action_sample.loss),
                num_action_frames=action_sample.num_frames,
                action_state_ids=action_sample.state_ids,
                action_state_fingerprints=action_sample.state_fingerprints,
                total_energy=float(terms.total_energy.detach().item()),
                low_energy=float(terms.low_energy.detach().item()),
                high_energy=float(terms.high_energy.detach().item()),
                high_ratio=float(terms.high_ratio.detach().item()),
                diagnostic_high_ratio=float(
                    terms.diagnostic_high_ratio.detach().item()
                ),
                rho_nat=regularizer.rho_nat,
                hinge=float(terms.hinge.detach().item()),
                hinge_active=bool(terms.hinge.detach().item() > 0.0),
                action_gradient_l2=action_norm,
                spectral_gradient_l2=spectral_norm,
                q_t=q_t,
                gradient_cosine=cosine,
                configured_surface_step=surface_step,
                surface_step_stats=step_stats,
            )
            rows.append(row)
            selected_window = _stable_window(
                rows,
                window_size=stable_window_size,
            )
            if selected_window is not None:
                break
    finally:
        restore_evidence = restore_calibration_state(
            mutable_module,
            snapshot,
            stateful_components=stateful_components,
        )

    if selected_window is None:
        q_median = None
        lambda_spec = 1.0
        status = "uncalibrated_no_stable_activation"
        window_start = None
        window_end = None
    else:
        q_values = np.asarray(
            [row.q_t for row in selected_window],
            dtype=np.float64,
        )
        q_median = float(np.median(q_values))
        if not math.isfinite(q_median) or q_median <= 0.0:
            raise SpectralGuardError("稳定窗口q_med必须为有限正数")
        lambda_spec = min(1.0, 0.1 / q_median)
        status = "calibrated_stable_activation"
        window_start = selected_window[0].iteration
        window_end = selected_window[-1].iteration
    assert restore_evidence is not None and restore_evidence.all_restored
    return SpectralGuardCalibrationResult(
        iterations=tuple(rows),
        selected_window_start=window_start,
        selected_window_end=window_end,
        q_median=q_median,
        lambda_spec=lambda_spec,
        calibration_status=status,
        restore_evidence=restore_evidence,
    )
