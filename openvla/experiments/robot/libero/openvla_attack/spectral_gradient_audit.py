"""Source-only 谱基梯度审计与可复查产物。

本模块回答“候选谱基中的哪些方向对当前攻击目标更重要”。它不负责加载 VLA、
创建 LIBERO 环境或执行优化更新，而是接收现有训练数据流产生的逐帧
``Action`` / ``Feature`` 参数梯度，并在状态维度上汇总：

```
TrainingFrame
    │
    ├─ d(Action Loss) / d(C)   [K, 3]
    └─ d(Feature Loss) / d(C)  [K, 3]
                     │
                     ▼
          stack over frames/states [S, K, 3]
                     │
                     ├─ mean gradient norm
                     ├─ cross-sample direction consistency
                     ├─ surface-budget-normalized score
                     └─ cumulative gradient energy
```

其中 ``C`` 是谱系数，``S`` 是有效训练帧数量，``K`` 是候选谱基数量。审计严格
使用源 OpenVLA 训练帧；OFT 等目标模型梯度不得进入本模块的选基统计。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from .compositing import SharedTextureViewName
from .frame_collection import TrainingFrame


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SpectralGradientAuditError(RuntimeError):
    """谱基梯度输入、运行模式或统计量不满足审计约束。"""


@dataclass(frozen=True)
class ObjectiveParameterGradients:
    """一个训练帧对两项攻击目标的独立参数梯度。

    ``action_gradient`` 与 ``feature_gradient`` 均为浮点 tensor
    ``[num_basis, 3]``，已经从 autograd graph detach，但保留原 device。
    两个 loss 是未乘 ``alpha`` 和 frame weight 的标量，便于后续独立标定。
    """

    action_loss: float
    feature_loss: float
    action_gradient: torch.Tensor
    feature_gradient: torch.Tensor
    # 双视角诊断时额外保留 primary/wrist 的未加权标量 loss 与 [K,3] 梯度。
    # 默认单视角审计保持 None，原有选基产物和调用方无需感知新字段。
    feature_losses_by_view: Optional[
        dict[SharedTextureViewName, float]
    ] = None
    feature_gradients_by_view: Optional[
        dict[SharedTextureViewName, torch.Tensor]
    ] = None


class ObjectiveGradientProvider(Protocol):
    """单帧独立目标梯度的最小计算接口。"""

    def compute_objective_parameter_gradients(
        self,
        frame: TrainingFrame,
    ) -> Optional[ObjectiveParameterGradients]:
        """返回有效 MVP 帧的两项梯度；不可渲染帧返回 ``None``。"""
        ...


class SpectralAuditRenderer(Protocol):
    """审计所需的谱 renderer 最小接口。"""

    def reset_texture(self) -> None:
        """把候选谱系数恢复为零扰动。"""
        ...

    def get_texture_param(self) -> nn.Parameter:
        """返回浮点谱系数，shape ``[num_basis, 3]``。"""
        ...

    def get_texture_parameterization_name(self) -> str:
        """返回 ``spectral``。"""
        ...

    def get_spectral_basis_and_eigenvalues(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 basis ``[N, K]`` 与 eigenvalues ``[K]``。"""
        ...


@dataclass(frozen=True)
class SpectralGradientAuditPaths:
    """一次审计生成的机器可读、表格和摘要路径。"""

    npz_path: Path
    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class SpectralGradientAuditResult:
    """跨训练状态聚合后的逐模态统计。

    所有梯度数组使用 float64 CPU NumPy 保存，避免日志分析依赖 CUDA 或模型
    环境。``sample_*`` 第一维为有效训练帧 ``S``；其余逐模态数组 shape 为
    ``[K]``。
    """

    state_ids: IntArray
    step_indices: IntArray
    eigenvalues: FloatArray
    basis_linf: FloatArray
    action_losses: FloatArray
    feature_losses: FloatArray
    action_gradients: FloatArray
    feature_gradients: FloatArray
    action_mean_norm: FloatArray
    feature_mean_norm: FloatArray
    action_consistency: FloatArray
    feature_consistency: FloatArray
    action_surface_score: FloatArray
    feature_surface_score: FloatArray
    action_stable_score: FloatArray
    feature_stable_score: FloatArray
    action_cumulative_energy: FloatArray
    feature_cumulative_energy: FloatArray
    action_ranked_indices: IntArray
    feature_ranked_indices: IntArray
    requested_top_k: int
    # 以下四项只在 primary_wrist Shared-SigLIP 审计中存在。梯度 shape 均为
    # float64 CPU [num_samples, num_basis, 3]。
    primary_feature_losses: Optional[FloatArray] = None
    wrist_feature_losses: Optional[FloatArray] = None
    primary_feature_gradients: Optional[FloatArray] = None
    wrist_feature_gradients: Optional[FloatArray] = None
    reference_kind: str = "zero_surface_delta"
    reference_path: Optional[str] = None
    reference_sha256: Optional[str] = None

    @property
    def num_samples(self) -> int:
        return int(self.action_gradients.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.action_gradients.shape[1])

    @property
    def top_action_indices(self) -> IntArray:
        return self.action_ranked_indices[: self.requested_top_k]

    @property
    def top_feature_indices(self) -> IntArray:
        return self.feature_ranked_indices[: self.requested_top_k]

    def save(
        self,
        *,
        output_directory: str | Path,
        task_id: int,
    ) -> SpectralGradientAuditPaths:
        """保存完整 NPZ、逐模态 CSV 和简短 JSON 摘要。"""
        resolved_directory: Path = Path(output_directory)
        resolved_directory.mkdir(parents=True, exist_ok=True)
        stem: str = f"Ep{task_id}_Spectral_Gradient_Audit"
        paths = SpectralGradientAuditPaths(
            npz_path=resolved_directory / f"{stem}.npz",
            csv_path=resolved_directory / f"{stem}.csv",
            json_path=resolved_directory / f"{stem}.json",
        )

        archive_payload: dict[str, NDArray[np.generic]] = {
            "state_ids": self.state_ids,
            "step_indices": self.step_indices,
            "eigenvalues": self.eigenvalues,
            "basis_linf": self.basis_linf,
            "action_losses": self.action_losses,
            "feature_losses": self.feature_losses,
            "action_gradients": self.action_gradients,
            "feature_gradients": self.feature_gradients,
            "action_mean_norm": self.action_mean_norm,
            "feature_mean_norm": self.feature_mean_norm,
            "action_consistency": self.action_consistency,
            "feature_consistency": self.feature_consistency,
            "action_surface_score": self.action_surface_score,
            "feature_surface_score": self.feature_surface_score,
            "action_stable_score": self.action_stable_score,
            "feature_stable_score": self.feature_stable_score,
            "action_cumulative_energy": self.action_cumulative_energy,
            "feature_cumulative_energy": self.feature_cumulative_energy,
            "action_ranked_indices": self.action_ranked_indices,
            "feature_ranked_indices": self.feature_ranked_indices,
            "reference_kind": np.asarray(self.reference_kind),
        }
        if self.primary_feature_gradients is not None:
            if (
                self.primary_feature_losses is None
                or self.wrist_feature_losses is None
                or self.wrist_feature_gradients is None
            ):
                raise SpectralGradientAuditError(
                    "双视角审计产物字段不完整"
                )
            archive_payload.update(
                primary_feature_losses=self.primary_feature_losses,
                wrist_feature_losses=self.wrist_feature_losses,
                primary_feature_gradients=self.primary_feature_gradients,
                wrist_feature_gradients=self.wrist_feature_gradients,
            )
        np.savez_compressed(paths.npz_path, **archive_payload)
        self._save_csv(paths.csv_path)
        self._save_json(paths.json_path)
        return paths

    def _save_csv(self, path: Path) -> None:
        """按原始低频顺序保存每个候选模态的一行统计。"""
        action_ranks: IntArray = np.empty(
            self.num_basis,
            dtype=np.int64,
        )
        feature_ranks: IntArray = np.empty(
            self.num_basis,
            dtype=np.int64,
        )
        action_ranks[self.action_ranked_indices] = np.arange(
            self.num_basis,
            dtype=np.int64,
        )
        feature_ranks[self.feature_ranked_indices] = np.arange(
            self.num_basis,
            dtype=np.int64,
        )

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "mode_index",
                    "eigenvalue",
                    "basis_linf",
                    "action_mean_norm",
                    "feature_mean_norm",
                    "action_consistency",
                    "feature_consistency",
                    "action_surface_score",
                    "feature_surface_score",
                    "action_stable_score",
                    "feature_stable_score",
                    "action_cumulative_energy",
                    "feature_cumulative_energy",
                    "action_rank",
                    "feature_rank",
                )
            )
            for mode_index in range(self.num_basis):
                writer.writerow(
                    (
                        mode_index,
                        self.eigenvalues[mode_index],
                        self.basis_linf[mode_index],
                        self.action_mean_norm[mode_index],
                        self.feature_mean_norm[mode_index],
                        self.action_consistency[mode_index],
                        self.feature_consistency[mode_index],
                        self.action_surface_score[mode_index],
                        self.feature_surface_score[mode_index],
                        self.action_stable_score[mode_index],
                        self.feature_stable_score[mode_index],
                        self.action_cumulative_energy[mode_index],
                        self.feature_cumulative_energy[mode_index],
                        action_ranks[mode_index],
                        feature_ranks[mode_index],
                    )
                )

    def _save_json(self, path: Path) -> None:
        """保存无需 NumPy 环境即可阅读的 source-only 选基摘要。"""
        low_frequency_cutoff: int = min(
            self.requested_top_k,
            self.num_basis,
        )
        summary: dict[str, object] = {
            "selection_scope": "source_openvla_only",
            "num_samples": self.num_samples,
            "num_basis": self.num_basis,
            "requested_top_k": self.requested_top_k,
            "state_ids": self.state_ids.tolist(),
            "step_indices": self.step_indices.tolist(),
            "top_action_indices": self.top_action_indices.tolist(),
            "top_feature_indices": self.top_feature_indices.tolist(),
            "low_frequency_action_energy_at_top_k": float(
                self.action_cumulative_energy[
                    low_frequency_cutoff - 1
                ]
            ),
            "low_frequency_feature_energy_at_top_k": float(
                self.feature_cumulative_energy[
                    low_frequency_cutoff - 1
                ]
            ),
            "mean_action_loss": float(self.action_losses.mean()),
            "mean_feature_loss": float(self.feature_losses.mean()),
        }
        if self.primary_feature_gradients is not None:
            if (
                self.primary_feature_losses is None
                or self.wrist_feature_losses is None
                or self.wrist_feature_gradients is None
            ):
                raise SpectralGradientAuditError(
                    "双视角审计 JSON 字段不完整"
                )
            summary["dual_view_diagnostic_available"] = True
            summary["mean_primary_feature_loss"] = float(
                self.primary_feature_losses.mean()
            )
            summary["mean_wrist_feature_loss"] = float(
                self.wrist_feature_losses.mean()
            )
        else:
            summary["dual_view_diagnostic_available"] = False
        summary["gradient_reference"] = {
            "kind": self.reference_kind,
            "path": self.reference_path,
            "sha256": self.reference_sha256,
        }
        path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _validate_and_convert_inputs(
    *,
    state_ids: Sequence[int],
    step_indices: Sequence[int],
    action_losses: Sequence[float],
    feature_losses: Sequence[float],
    action_gradients: torch.Tensor,
    feature_gradients: torch.Tensor,
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    requested_top_k: int,
) -> tuple[
    IntArray,
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """集中校验 shape/有限值，并转换为 float64 CPU NumPy。"""
    action_array: FloatArray = (
        action_gradients.detach().float().cpu().numpy().astype(np.float64)
    )
    feature_array: FloatArray = (
        feature_gradients.detach().float().cpu().numpy().astype(np.float64)
    )
    basis_array: FloatArray = (
        basis.detach().float().cpu().numpy().astype(np.float64)
    )
    eigenvalue_array: FloatArray = (
        eigenvalues.detach().double().cpu().numpy().astype(np.float64)
    )
    state_array: IntArray = np.asarray(state_ids, dtype=np.int64)
    step_array: IntArray = np.asarray(step_indices, dtype=np.int64)
    action_loss_array: FloatArray = np.asarray(
        action_losses,
        dtype=np.float64,
    )
    feature_loss_array: FloatArray = np.asarray(
        feature_losses,
        dtype=np.float64,
    )

    if (
        action_array.ndim != 3
        or action_array.shape[2] != 3
        or action_array.shape != feature_array.shape
    ):
        raise SpectralGradientAuditError(
            "Action/Feature gradients 必须具有相同的 [S, K, 3] shape，"
            f"实际为 {action_array.shape} 和 {feature_array.shape}"
        )
    num_samples: int = int(action_array.shape[0])
    num_basis: int = int(action_array.shape[1])
    if num_samples <= 0 or num_basis <= 0:
        raise SpectralGradientAuditError("梯度审计至少需要一个样本和一个谱基")
    if basis_array.ndim != 2 or basis_array.shape[1] != num_basis:
        raise SpectralGradientAuditError(
            f"basis 应为 [N, {num_basis}]，实际为 {basis_array.shape}"
        )
    expected_sample_shape: tuple[int, ...] = (num_samples,)
    sample_arrays: tuple[NDArray[np.generic], ...] = (
        state_array,
        step_array,
        action_loss_array,
        feature_loss_array,
    )
    if any(
        array.shape != expected_sample_shape
        for array in sample_arrays
    ):
        raise SpectralGradientAuditError(
            "state/step/loss 数量必须与梯度样本数一致"
        )
    if eigenvalue_array.shape != (num_basis,):
        raise SpectralGradientAuditError(
            f"eigenvalues 应为 [{num_basis}]，实际为 "
            f"{eigenvalue_array.shape}"
        )
    if requested_top_k <= 0 or requested_top_k > num_basis:
        raise SpectralGradientAuditError(
            f"requested_top_k 必须位于 [1, {num_basis}]，"
            f"实际为 {requested_top_k}"
        )
    floating_arrays: tuple[FloatArray, ...] = (
        action_array,
        feature_array,
        basis_array,
        eigenvalue_array,
        action_loss_array,
        feature_loss_array,
    )
    if not all(np.isfinite(array).all() for array in floating_arrays):
        raise SpectralGradientAuditError("谱基梯度审计输入包含 NaN/Inf")
    return (
        state_array,
        step_array,
        action_loss_array,
        feature_loss_array,
        action_array,
        feature_array,
        basis_array,
        eigenvalue_array,
    )


def summarize_spectral_gradients(
    *,
    state_ids: Sequence[int],
    step_indices: Sequence[int],
    action_losses: Sequence[float],
    feature_losses: Sequence[float],
    action_gradients: torch.Tensor,
    feature_gradients: torch.Tensor,
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    requested_top_k: int,
    primary_feature_losses: Optional[Sequence[float]] = None,
    wrist_feature_losses: Optional[Sequence[float]] = None,
    primary_feature_gradients: Optional[torch.Tensor] = None,
    wrist_feature_gradients: Optional[torch.Tensor] = None,
    reference_kind: str = "zero_surface_delta",
    reference_path: Optional[str] = None,
    reference_sha256: Optional[str] = None,
) -> SpectralGradientAuditResult:
    """计算逐模态强度、状态一致性和曲面预算归一化排名。

    对第 ``k`` 个模态，``mean_norm`` 是样本梯度 RGB 二范数的均值；
    ``consistency`` 为 ``||mean(g)|| / mean(||g||)``，范围 ``[0, 1]``；
    ``surface_score = mean_norm / ||phi_k||_inf``，近似表示相同 L∞ 曲面预算下
    的一阶 loss 改变量；最终 ``stable_score`` 再乘一致性，抑制跨状态互相
    抵消的方向。
    """
    (
        state_array,
        step_array,
        action_loss_array,
        feature_loss_array,
        action_array,
        feature_array,
        basis_array,
        eigenvalue_array,
    ) = _validate_and_convert_inputs(
        state_ids=state_ids,
        step_indices=step_indices,
        action_losses=action_losses,
        feature_losses=feature_losses,
        action_gradients=action_gradients,
        feature_gradients=feature_gradients,
        basis=basis,
        eigenvalues=eigenvalues,
        requested_top_k=requested_top_k,
    )

    basis_linf: FloatArray = np.max(np.abs(basis_array), axis=0)
    numeric_tiny: float = np.finfo(np.float64).tiny
    if np.any(basis_linf <= numeric_tiny):
        raise SpectralGradientAuditError("候选谱基包含零幅值模态")

    def summarize_objective(
        gradients: FloatArray,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        IntArray,
    ]:
        # per_sample_norm: float64 [S, K]。
        per_sample_norm: FloatArray = np.linalg.norm(
            gradients,
            axis=2,
        )
        mean_norm: FloatArray = per_sample_norm.mean(axis=0)
        mean_vector_norm: FloatArray = np.linalg.norm(
            gradients.mean(axis=0),
            axis=1,
        )
        consistency: FloatArray = np.divide(
            mean_vector_norm,
            np.maximum(mean_norm, numeric_tiny),
        )
        consistency = np.clip(consistency, 0.0, 1.0)
        surface_score: FloatArray = mean_norm / basis_linf
        stable_score: FloatArray = surface_score * consistency

        mode_energy: FloatArray = np.sum(gradients**2, axis=(0, 2))
        total_energy: float = float(mode_energy.sum())
        cumulative_energy: FloatArray = (
            np.cumsum(mode_energy) / total_energy
            if total_energy > numeric_tiny
            else np.zeros_like(mode_energy)
        )
        ranked_indices: IntArray = np.argsort(
            -stable_score,
            kind="stable",
        ).astype(np.int64)
        return (
            mean_norm,
            consistency,
            surface_score,
            stable_score,
            cumulative_energy,
            ranked_indices,
        )

    (
        action_mean_norm,
        action_consistency,
        action_surface_score,
        action_stable_score,
        action_cumulative_energy,
        action_ranked_indices,
    ) = summarize_objective(action_array)
    (
        feature_mean_norm,
        feature_consistency,
        feature_surface_score,
        feature_stable_score,
        feature_cumulative_energy,
        feature_ranked_indices,
    ) = summarize_objective(feature_array)

    # 双视角字段必须四项同时存在，并与 combined Feature 使用相同的
    # [S,K,3] 坐标。这里不重新参与选基，只为后续冲突诊断保存原始证据。
    optional_view_values: tuple[object, ...] = (
        primary_feature_losses,
        wrist_feature_losses,
        primary_feature_gradients,
        wrist_feature_gradients,
    )
    present_view_value_count: int = sum(
        value is not None for value in optional_view_values
    )
    if present_view_value_count not in (0, len(optional_view_values)):
        raise SpectralGradientAuditError(
            "双视角 Feature loss/gradient 必须同时提供"
        )
    primary_feature_loss_array: Optional[FloatArray] = None
    wrist_feature_loss_array: Optional[FloatArray] = None
    primary_feature_array: Optional[FloatArray] = None
    wrist_feature_array: Optional[FloatArray] = None
    if present_view_value_count:
        assert primary_feature_losses is not None
        assert wrist_feature_losses is not None
        assert primary_feature_gradients is not None
        assert wrist_feature_gradients is not None
        primary_feature_loss_array = np.asarray(
            primary_feature_losses,
            dtype=np.float64,
        )
        wrist_feature_loss_array = np.asarray(
            wrist_feature_losses,
            dtype=np.float64,
        )
        primary_feature_array = (
            primary_feature_gradients.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        wrist_feature_array = (
            wrist_feature_gradients.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        expected_gradient_shape: tuple[int, ...] = action_array.shape
        expected_loss_shape: tuple[int, ...] = action_loss_array.shape
        if (
            primary_feature_array.shape != expected_gradient_shape
            or wrist_feature_array.shape != expected_gradient_shape
            or primary_feature_loss_array.shape != expected_loss_shape
            or wrist_feature_loss_array.shape != expected_loss_shape
        ):
            raise SpectralGradientAuditError(
                "双视角 Feature 字段与 combined audit shape 不一致"
            )
        if not all(
            np.isfinite(array).all()
            for array in (
                primary_feature_loss_array,
                wrist_feature_loss_array,
                primary_feature_array,
                wrist_feature_array,
            )
        ):
            raise SpectralGradientAuditError(
                "双视角 Feature 审计包含 NaN/Inf"
            )
        if not np.allclose(
            feature_array,
            (primary_feature_array + wrist_feature_array) / 2.0,
            rtol=2e-4,
            atol=2e-6,
        ):
            raise SpectralGradientAuditError(
                "combined Feature 梯度不等于主/腕部梯度均值"
            )

    return SpectralGradientAuditResult(
        state_ids=state_array,
        step_indices=step_array,
        eigenvalues=eigenvalue_array,
        basis_linf=basis_linf,
        action_losses=action_loss_array,
        feature_losses=feature_loss_array,
        action_gradients=action_array,
        feature_gradients=feature_array,
        action_mean_norm=action_mean_norm,
        feature_mean_norm=feature_mean_norm,
        action_consistency=action_consistency,
        feature_consistency=feature_consistency,
        action_surface_score=action_surface_score,
        feature_surface_score=feature_surface_score,
        action_stable_score=action_stable_score,
        feature_stable_score=feature_stable_score,
        action_cumulative_energy=action_cumulative_energy,
        feature_cumulative_energy=feature_cumulative_energy,
        action_ranked_indices=action_ranked_indices,
        feature_ranked_indices=feature_ranked_indices,
        requested_top_k=requested_top_k,
        primary_feature_losses=primary_feature_loss_array,
        wrist_feature_losses=wrist_feature_loss_array,
        primary_feature_gradients=primary_feature_array,
        wrist_feature_gradients=wrist_feature_array,
        reference_kind=reference_kind,
        reference_path=reference_path,
        reference_sha256=reference_sha256,
    )


class SpectralGradientAuditor:
    """在零扰动处采集 source-only 逐帧独立目标梯度。"""

    def __init__(
        self,
        *,
        gradient_provider: ObjectiveGradientProvider,
        renderer: SpectralAuditRenderer,
        requested_top_k: int,
        reference_parameter_path: Optional[str | Path] = None,
    ) -> None:
        self._gradient_provider: ObjectiveGradientProvider = (
            gradient_provider
        )
        self._renderer: SpectralAuditRenderer = renderer
        self._requested_top_k: int = requested_top_k
        self._reference_parameter_path: Optional[Path] = (
            Path(reference_parameter_path).resolve()
            if reference_parameter_path is not None
            else None
        )

    def run(
        self,
        frames: Sequence[TrainingFrame],
    ) -> SpectralGradientAuditResult:
        """逐帧从零谱系数求梯度，并在状态维度汇总。"""
        if (
            self._renderer.get_texture_parameterization_name()
            != "spectral"
        ):
            raise SpectralGradientAuditError(
                "谱基梯度审计只支持 spectral 参数化"
            )
        basis: torch.Tensor
        eigenvalues: torch.Tensor
        basis, eigenvalues = (
            self._renderer.get_spectral_basis_and_eigenvalues()
        )
        texture_parameter: nn.Parameter = (
            self._renderer.get_texture_param()
        )
        expected_parameter_shape: tuple[int, int] = (
            int(basis.shape[1]),
            3,
        )
        if tuple(texture_parameter.shape) != expected_parameter_shape:
            raise SpectralGradientAuditError(
                "谱系数与 basis 不匹配："
                f"{tuple(texture_parameter.shape)} != "
                f"{expected_parameter_shape}"
            )

        reference_parameter: Optional[torch.Tensor] = None
        reference_kind: str = "zero_surface_delta"
        reference_path: Optional[str] = None
        reference_sha256: Optional[str] = None
        if self._reference_parameter_path is not None:
            if not self._reference_parameter_path.is_file():
                raise SpectralGradientAuditError(
                    "谱梯度审计参考参数不存在："
                    f"{self._reference_parameter_path}"
                )
            loaded_reference: object = torch.load(
                self._reference_parameter_path,
                map_location=texture_parameter.device,
                weights_only=True,
            )
            if not isinstance(loaded_reference, torch.Tensor):
                raise SpectralGradientAuditError(
                    "谱梯度审计参考 .pt 必须直接保存 Tensor"
                )
            reference_parameter = loaded_reference.to(
                device=texture_parameter.device,
                dtype=texture_parameter.dtype,
            )
            if tuple(reference_parameter.shape) != expected_parameter_shape:
                raise SpectralGradientAuditError(
                    "谱梯度审计参考参数 shape 不匹配："
                    f"{tuple(reference_parameter.shape)} != "
                    f"{expected_parameter_shape}"
                )
            if not torch.isfinite(reference_parameter).all():
                raise SpectralGradientAuditError(
                    "谱梯度审计参考参数包含 NaN/Inf"
                )
            reference_kind = "spectral_coefficients"
            reference_path = str(self._reference_parameter_path)
            reference_sha256 = hashlib.sha256(
                self._reference_parameter_path.read_bytes()
            ).hexdigest()

        state_ids: list[int] = []
        step_indices: list[int] = []
        action_losses: list[float] = []
        feature_losses: list[float] = []
        action_gradients: list[torch.Tensor] = []
        feature_gradients: list[torch.Tensor] = []
        primary_feature_losses: list[float] = []
        wrist_feature_losses: list[float] = []
        primary_feature_gradients: list[torch.Tensor] = []
        wrist_feature_gradients: list[torch.Tensor] = []
        has_dual_view_diagnostics: Optional[bool] = None
        try:
            frame: TrainingFrame
            for frame in frames:
                # 每个样本恢复到完全相同的固定参考点，避免前一状态改变后一状态
                # 的梯度。默认参考为零 Surface Delta；诊断训练终点时则复制同一
                # 份保存谱系数，不执行任何优化更新。
                if reference_parameter is None:
                    self._renderer.reset_texture()
                else:
                    with torch.no_grad():
                        texture_parameter.copy_(reference_parameter)
                sample: Optional[ObjectiveParameterGradients] = (
                    self._gradient_provider
                    .compute_objective_parameter_gradients(frame)
                )
                if sample is None:
                    continue
                state_ids.append(frame["initial_state_id"])
                step_indices.append(frame["collection_step_index"])
                action_losses.append(sample.action_loss)
                feature_losses.append(sample.feature_loss)
                action_gradients.append(
                    sample.action_gradient.detach().cpu()
                )
                feature_gradients.append(
                    sample.feature_gradient.detach().cpu()
                )
                sample_has_dual_view: bool = (
                    sample.feature_losses_by_view is not None
                    and sample.feature_gradients_by_view is not None
                )
                if has_dual_view_diagnostics is None:
                    has_dual_view_diagnostics = sample_has_dual_view
                elif has_dual_view_diagnostics != sample_has_dual_view:
                    raise SpectralGradientAuditError(
                        "同一次审计不能混合单视角与双视角样本"
                    )
                if sample_has_dual_view:
                    assert sample.feature_losses_by_view is not None
                    assert sample.feature_gradients_by_view is not None
                    if (
                        set(sample.feature_losses_by_view)
                        != {"primary", "wrist"}
                        or set(sample.feature_gradients_by_view)
                        != {"primary", "wrist"}
                    ):
                        raise SpectralGradientAuditError(
                            "双视角样本必须包含 primary/wrist"
                        )
                    primary_feature_losses.append(
                        sample.feature_losses_by_view["primary"]
                    )
                    wrist_feature_losses.append(
                        sample.feature_losses_by_view["wrist"]
                    )
                    primary_feature_gradients.append(
                        sample.feature_gradients_by_view["primary"]
                        .detach()
                        .cpu()
                    )
                    wrist_feature_gradients.append(
                        sample.feature_gradients_by_view["wrist"]
                        .detach()
                        .cpu()
                    )
        finally:
            self._renderer.reset_texture()

        if not action_gradients:
            raise SpectralGradientAuditError(
                "没有具有有效 MVP 的训练帧可用于谱基梯度审计"
            )
        return summarize_spectral_gradients(
            state_ids=state_ids,
            step_indices=step_indices,
            action_losses=action_losses,
            feature_losses=feature_losses,
            action_gradients=torch.stack(action_gradients, dim=0),
            feature_gradients=torch.stack(feature_gradients, dim=0),
            basis=basis,
            eigenvalues=eigenvalues,
            requested_top_k=self._requested_top_k,
            primary_feature_losses=(
                primary_feature_losses
                if has_dual_view_diagnostics
                else None
            ),
            wrist_feature_losses=(
                wrist_feature_losses
                if has_dual_view_diagnostics
                else None
            ),
            primary_feature_gradients=(
                torch.stack(primary_feature_gradients, dim=0)
                if has_dual_view_diagnostics
                else None
            ),
            wrist_feature_gradients=(
                torch.stack(wrist_feature_gradients, dim=0)
                if has_dual_view_diagnostics
                else None
            ),
            reference_kind=reference_kind,
            reference_path=reference_path,
            reference_sha256=reference_sha256,
        )
