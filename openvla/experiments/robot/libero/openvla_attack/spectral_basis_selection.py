"""根据 source-only 梯度审计结果构造非连续谱基产物。

候选谱基文件中的非恒定模态按特征值从低到高排列，而梯度审计输出的是这些
源模态的稳定性排名。本模块只负责将排名前 K 个源模态复制到一个较小、可直接
被 renderer 加载的 NPZ 中，并把完整选择依据写入 metadata：

```
K=512 candidate basis + source OpenVLA audit
                    │
                    ▼
        校验 geometry / eigenvalue / basis L∞
                    │
                    ▼
       按 stable-score rank 选择 source indices
                    │
                    ▼
 [constant, selected modes] + provenance metadata
                    │
                    ▼
       gradient-selected K=128 basis artifact
```

输出 basis 的列顺序就是 stable-score 从高到低的排名顺序，不再是特征值升序。
攻击优化只依赖基向量及其列顺序，不要求特征值有序；元数据中的
``selected_source_mode_indices`` 可把每一列映射回 K=512 候选池。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .spectral_geometry import (
    SpectralBasisData,
    SpectralGeometryError,
    load_spectral_basis,
    save_spectral_basis,
)


SelectionObjective = Literal["action", "feature"]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SpectralBasisSelectionError(RuntimeError):
    """候选谱基、审计结果或选择约束不一致。"""


@dataclass(frozen=True)
class GradientSelectedBasisResult:
    """一次梯度选基生成的可复查摘要。"""

    output_path: Path
    selected_source_mode_indices: IntArray  # [K]
    selected_eigenvalues: FloatArray  # [K]
    selected_stable_scores: FloatArray  # [K]
    source_state_ids: IntArray  # [S]
    mass_orthogonality_error: float
    candidate_sha256: str
    audit_sha256: str

    @property
    def num_selected_basis(self) -> int:
        return int(self.selected_source_mode_indices.shape[0])


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大型候选谱基一次读入 bytes。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audit_arrays(
    audit_path: Path,
    *,
    objective: SelectionObjective,
) -> dict[str, NDArray[np.generic]]:
    """只读取选基所需数组，并拒绝缺失或包含 pickle 的审计产物。"""
    required_keys: set[str] = {
        "state_ids",
        "eigenvalues",
        "basis_linf",
        f"{objective}_gradients",
        f"{objective}_ranked_indices",
        f"{objective}_stable_score",
    }
    with np.load(audit_path, allow_pickle=False) as archive:
        missing_keys: list[str] = sorted(
            required_keys.difference(archive.files)
        )
        if missing_keys:
            raise SpectralBasisSelectionError(
                f"{audit_path} 缺少审计数组: {missing_keys}"
            )
        return {
            key: np.asarray(archive[key])
            for key in required_keys
        }


def _validate_audit_against_candidate(
    *,
    audit_arrays: dict[str, NDArray[np.generic]],
    candidate_with_constant: SpectralBasisData,
    objective: SelectionObjective,
    top_k: int,
    expected_state_ids: Optional[Sequence[int]],
) -> tuple[IntArray, IntArray, FloatArray]:
    """验证审计确实对应当前候选谱基和预期 source states。

    审计中的 eigenvalue / basis L∞ 来自 renderer 的 float32 tensor，而候选
    artifact 保存 float64，因此比较时允许 float32 舍入误差；源模态数量、排名
    permutation 和 state ID 则要求严格一致。
    """
    number_of_candidate_modes: int = (
        candidate_with_constant.num_basis - 1
    )
    if number_of_candidate_modes <= 0:
        raise SpectralBasisSelectionError(
            "候选谱基必须包含常数模态和至少一个非恒定模态"
        )
    if top_k <= 0 or top_k > number_of_candidate_modes:
        raise SpectralBasisSelectionError(
            f"top_k 必须位于 [1, {number_of_candidate_modes}]，"
            f"实际为 {top_k}"
        )

    state_ids: IntArray = np.asarray(
        audit_arrays["state_ids"],
        dtype=np.int64,
    )
    audit_eigenvalues: FloatArray = np.asarray(
        audit_arrays["eigenvalues"],
        dtype=np.float64,
    )
    audit_basis_linf: FloatArray = np.asarray(
        audit_arrays["basis_linf"],
        dtype=np.float64,
    )
    gradients: FloatArray = np.asarray(
        audit_arrays[f"{objective}_gradients"],
        dtype=np.float64,
    )
    ranked_indices: IntArray = np.asarray(
        audit_arrays[f"{objective}_ranked_indices"],
        dtype=np.int64,
    )
    stable_scores: FloatArray = np.asarray(
        audit_arrays[f"{objective}_stable_score"],
        dtype=np.float64,
    )

    if state_ids.ndim != 1 or state_ids.size <= 0:
        raise SpectralBasisSelectionError(
            "审计 state_ids 必须是非空一维数组"
        )
    if expected_state_ids is not None:
        expected_array: IntArray = np.asarray(
            expected_state_ids,
            dtype=np.int64,
        )
        if not np.array_equal(state_ids, expected_array):
            raise SpectralBasisSelectionError(
                "审计 states 与 expected_state_ids 不一致："
                f"{state_ids.tolist()} != {expected_array.tolist()}"
            )
    expected_mode_shape: tuple[int, ...] = (
        number_of_candidate_modes,
    )
    if (
        audit_eigenvalues.shape != expected_mode_shape
        or audit_basis_linf.shape != expected_mode_shape
        or stable_scores.shape != expected_mode_shape
        or ranked_indices.shape != expected_mode_shape
    ):
        raise SpectralBasisSelectionError(
            "审计逐模态数组数量与候选谱基不一致"
        )
    if gradients.shape != (
        state_ids.size,
        number_of_candidate_modes,
        3,
    ):
        raise SpectralBasisSelectionError(
            "审计梯度应为 "
            f"[{state_ids.size}, {number_of_candidate_modes}, 3]，"
            f"实际为 {gradients.shape}"
        )

    floating_arrays: tuple[FloatArray, ...] = (
        audit_eigenvalues,
        audit_basis_linf,
        gradients,
        stable_scores,
    )
    if not all(np.isfinite(array).all() for array in floating_arrays):
        raise SpectralBasisSelectionError("审计选基数组包含 NaN/Inf")
    if not np.array_equal(
        np.sort(ranked_indices),
        np.arange(number_of_candidate_modes, dtype=np.int64),
    ):
        raise SpectralBasisSelectionError(
            "ranked_indices 必须是所有候选源模态的 permutation"
        )

    candidate_eigenvalues: FloatArray = (
        candidate_with_constant.eigenvalues[1:]
    )
    candidate_basis: FloatArray = candidate_with_constant.basis[:, 1:]
    candidate_basis_linf: FloatArray = np.max(
        np.abs(candidate_basis),
        axis=0,
    )
    if not np.allclose(
        audit_eigenvalues,
        candidate_eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise SpectralBasisSelectionError(
            "审计 eigenvalues 与候选谱基不匹配"
        )
    if not np.allclose(
        audit_basis_linf,
        candidate_basis_linf,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise SpectralBasisSelectionError(
            "审计 basis_linf 与候选谱基不匹配"
        )
    return state_ids, ranked_indices, stable_scores


def build_gradient_selected_spectral_basis(
    *,
    candidate_basis_path: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
    objective: SelectionObjective,
    top_k: int,
    expected_state_ids: Optional[Sequence[int]] = None,
) -> GradientSelectedBasisResult:
    """从候选池按 source-only stable score 生成可直接训练的谱基。

    输出的 ``nonconstant_basis[:, j]`` 对应
    ``candidate.nonconstant_basis[:, selected_source_mode_indices[j]]``。
    选择不会重新求特征分解，也不会改变基向量的数值。
    """
    resolved_candidate_path: Path = Path(candidate_basis_path).resolve()
    resolved_audit_path: Path = Path(audit_path).resolve()
    resolved_output_path: Path = Path(output_path).resolve()
    if not resolved_audit_path.exists():
        raise FileNotFoundError(resolved_audit_path)
    if resolved_output_path == resolved_candidate_path:
        raise SpectralBasisSelectionError(
            "输出路径不能覆盖候选谱基"
        )

    # include_constant=True 得到 [N, M+1]，第0列是全局常数模态；其余列的
    # source index 比数组列号小1。
    try:
        candidate_with_constant: SpectralBasisData = load_spectral_basis(
            resolved_candidate_path,
            include_constant=True,
        )
    except SpectralGeometryError as error:
        raise SpectralBasisSelectionError(
            f"候选谱基加载失败: {error}"
        ) from error
    audit_arrays = _load_audit_arrays(
        resolved_audit_path,
        objective=objective,
    )
    state_ids: IntArray
    ranked_indices: IntArray
    stable_scores: FloatArray
    state_ids, ranked_indices, stable_scores = (
        _validate_audit_against_candidate(
            audit_arrays=audit_arrays,
            candidate_with_constant=candidate_with_constant,
            objective=objective,
            top_k=top_k,
            expected_state_ids=expected_state_ids,
        )
    )

    selected_source_indices: IntArray = np.ascontiguousarray(
        ranked_indices[:top_k],
        dtype=np.int64,
    )
    selected_columns: IntArray = selected_source_indices + 1
    # basis_with_constant: float64 [N, K+1]；第0列仍保留常数模态，以兼容
    # 统一的谱基 artifact 格式，但攻击 loader 默认排除它。
    selected_basis_with_constant: FloatArray = np.ascontiguousarray(
        np.concatenate(
            (
                candidate_with_constant.basis[:, :1],
                candidate_with_constant.basis[:, selected_columns],
            ),
            axis=1,
        ),
        dtype=np.float64,
    )
    selected_eigenvalues_with_constant: FloatArray = (
        np.ascontiguousarray(
            np.concatenate(
                (
                    candidate_with_constant.eigenvalues[:1],
                    candidate_with_constant.eigenvalues[
                        selected_columns
                    ],
                )
            ),
            dtype=np.float64,
        )
    )
    selected_basis: FloatArray = selected_basis_with_constant[:, 1:]
    gram: FloatArray = selected_basis.T @ (
        candidate_with_constant.mass[:, None] * selected_basis
    )
    mass_orthogonality_error: float = float(
        np.max(np.abs(gram - np.eye(top_k, dtype=np.float64)))
    )
    candidate_sha256: str = _sha256_file(resolved_candidate_path)
    audit_sha256: str = _sha256_file(resolved_audit_path)

    saved_path: Path = save_spectral_basis(
        resolved_output_path,
        vertices=candidate_with_constant.vertices,
        faces=candidate_with_constant.faces,
        mass=candidate_with_constant.mass,
        eigenvalues_with_constant=(
            selected_eigenvalues_with_constant
        ),
        basis_with_constant=selected_basis_with_constant,
        metadata={
            "selection_scope": "source_openvla_only",
            "selection_method": (
                f"{objective}_surface_stable_score_descending"
            ),
            "basis_column_order": "selection_rank_descending",
            "source_candidate_basis": str(resolved_candidate_path),
            "source_candidate_sha256": candidate_sha256,
            "source_audit": str(resolved_audit_path),
            "source_audit_sha256": audit_sha256,
            "source_state_ids": state_ids.tolist(),
            "selected_source_mode_indices": (
                selected_source_indices.tolist()
            ),
            "selected_stable_scores": (
                stable_scores[selected_source_indices].tolist()
            ),
            "max_m_orthogonality_error": (
                mass_orthogonality_error
            ),
        },
    )

    # 回读一次最终 artifact，确保 loader 看到的列顺序与选中源模态完全一致。
    loaded_output: SpectralBasisData = load_spectral_basis(
        saved_path,
        max_basis=top_k,
        include_constant=False,
    )
    if not np.array_equal(loaded_output.basis, selected_basis):
        raise SpectralBasisSelectionError(
            "选基产物回读后 basis 列发生变化"
        )
    return GradientSelectedBasisResult(
        output_path=saved_path,
        selected_source_mode_indices=selected_source_indices,
        selected_eigenvalues=np.ascontiguousarray(
            selected_eigenvalues_with_constant[1:],
        ),
        selected_stable_scores=np.ascontiguousarray(
            stable_scores[selected_source_indices],
        ),
        source_state_ids=state_ids,
        mass_orthogonality_error=mass_orthogonality_error,
        candidate_sha256=candidate_sha256,
        audit_sha256=audit_sha256,
    )
