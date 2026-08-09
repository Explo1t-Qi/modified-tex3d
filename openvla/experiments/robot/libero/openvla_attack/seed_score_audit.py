"""从正式 Dense Seed gradients 构造可复算的 Seed Score evidence。

本模块只执行 CPU 上的确定性统计与原始 OBJ 几何计算。输入是 source OpenVLA
在零 Surface Delta 处采集的完整 Action-only ``G_s [S,N_v,3]``；输出依次保存
逐 state Surface-L∞ 归一化、跨 state sensitivity、RGB direction consistency、
``q_i``、barycentric lumped mass、单位面积 ``d_i`` 与 mass-aware implicit
Laplacian smoothing ``d_tilde``。它不生成 Fixed Vertex Support。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr

from .action_objective_audit import EXPECTED_STATE_IDS
from .dense_seed_audit import (
    DENSE_SEED_AUDIT_SCHEMA_VERSION,
    DenseSeedGradientEvidence,
    evaluate_dense_seed_gradient_evidence,
)
from .spectral_geometry import (
    build_cotangent_laplacian_and_mass,
    mesh_array_sha256,
)


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

SEED_SCORE_SCHEMA_VERSION: Final[str] = "openvla-seed-score-audit-v1"
AKITA_SMOOTHING_ALPHA: Final[float] = 0.05
SCORE_ARTIFACT_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "code_commit",
    "dense_schema_version",
    "dense_code_commit",
    "dense_metrics_sha256",
    "dense_artifact_relative_paths",
    "dense_artifact_sha256",
    "dense_gradient_sha256",
    "state_ids",
    "mesh_file_sha256",
    "mesh_array_sha256",
    "vertices",
    "faces",
    "alpha",
    "total_surface_area",
    "tau",
    "state_normalization_scales",
    "state_abs_component_p99",
    "state_max_to_p99_ratio",
    "normalized_gradient_norms",
    "mean_normalized_gradient",
    "mean_sensitivity",
    "direction_consistency",
    "raw_score",
    "vertex_mass",
    "raw_density",
    "smoothed_density",
    "ranked_vertex_ids",
    "local_peak_mask",
    "ranked_local_peak_vertex_ids",
    "linear_system_relative_residual",
    "mass_conservation_relative_error",
    "production_support_constructed",
)
FLOAT_RTOL: Final[float] = 1e-10
FLOAT_ATOL: Final[float] = 1e-12


class SeedScoreAuditError(ValueError):
    """Seed Score 输入、几何或数值证据不满足冻结契约。"""


@dataclass(frozen=True)
class SeedScoreProvenance:
    """Seed Score 对 Dense bundle 与原始 OBJ 的完整绑定。"""

    code_commit: str
    dense_schema_version: str
    dense_code_commit: str
    dense_metrics_sha256: str
    dense_artifact_relative_paths: tuple[str, ...]
    dense_artifact_sha256: tuple[str, ...]
    dense_gradient_sha256: tuple[str, ...]
    state_ids: tuple[int, ...]
    mesh_file_sha256: str
    mesh_array_sha256: str


@dataclass(frozen=True)
class SeedScoreArrays:
    """全部可复算中间量；浮点数组统一为 CPU float64。"""

    state_ids: IntArray
    vertices: FloatArray
    faces: IntArray
    alpha: float
    total_surface_area: float
    tau: float
    state_normalization_scales: FloatArray
    state_abs_component_p99: FloatArray
    state_max_to_p99_ratio: FloatArray
    normalized_gradient_norms: FloatArray
    mean_normalized_gradient: FloatArray
    mean_sensitivity: FloatArray
    direction_consistency: FloatArray
    raw_score: FloatArray
    vertex_mass: FloatArray
    raw_density: FloatArray
    smoothed_density: FloatArray
    ranked_vertex_ids: IntArray
    local_peak_mask: BoolArray
    ranked_local_peak_vertex_ids: IntArray
    linear_system_relative_residual: float
    mass_conservation_relative_error: float


@dataclass(frozen=True)
class SeedScoreDiagnostics:
    """不参与构造或 Gate 的分布与尖峰诊断。"""

    num_states: int
    num_vertices: int
    local_peak_count: int
    normalization_max_to_p99_min: float
    normalization_max_to_p99_median: float
    normalization_max_to_p99_max: float
    raw_score_max_to_p99: float
    raw_density_max_to_p99: float
    smoothed_density_max_to_p99: float
    direction_consistency_mean: float
    direction_consistency_median: float
    direction_consistency_max: float
    smoothed_density_min: float
    smoothed_density_max: float


@dataclass(frozen=True)
class SeedScoreArtifactEvidence:
    """一个禁止覆盖的 Seed Score NPZ 及其 provenance。"""

    schema_version: str
    provenance: SeedScoreProvenance
    artifact_relative_path: str
    artifact_sha256: str
    diagnostics: SeedScoreDiagnostics
    production_support_constructed: bool


@dataclass(frozen=True)
class SeedScoreArtifactDecision:
    """NPZ、Dense 输入与重算结果的严格判定。"""

    gate_pass: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class SeedScoreStabilityReport:
    """两个同配置 score artifact 的只读稳定性诊断，不定义通过阈值。"""

    compatible: bool
    failures: tuple[str, ...]
    raw_score_cosine: Optional[float]
    raw_density_cosine: Optional[float]
    smoothed_density_cosine: Optional[float]
    smoothed_density_spearman: Optional[float]
    top_fraction_jaccard: tuple[tuple[float, float], ...]
    compared_local_peak_count: int
    top_local_peak_jaccard: Optional[float]
    threshold_registered: bool


def _is_git_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_to_p99(values: FloatArray) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    maximum = float(absolute.max(initial=0.0))
    percentile = float(np.quantile(absolute, 0.99, method="linear"))
    denominator = max(percentile, np.finfo(np.float64).tiny)
    return maximum / denominator


def _rank_descending(values: FloatArray) -> IntArray:
    vertex_ids = np.arange(len(values), dtype=np.int64)
    return np.lexsort((vertex_ids, -values)).astype(np.int64)


def _local_peak_evidence(
    values: FloatArray,
    faces: IntArray,
) -> tuple[BoolArray, IntArray]:
    """按 face-adjacent 邻域报告非负局部峰；并列按顶点 ID 排序。

    一个顶点必须不低于所有邻居、严格高于至少一个邻居且自身为正。该规则只作
    诊断，不产生 region seed 或 Support。
    """

    num_vertices = len(values)
    neighbor_max = np.full(num_vertices, -np.inf, dtype=np.float64)
    has_lower_neighbor = np.zeros(num_vertices, dtype=np.bool_)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        left = faces[:, first]
        right = faces[:, second]
        np.maximum.at(neighbor_max, left, values[right])
        np.maximum.at(neighbor_max, right, values[left])
        np.logical_or.at(has_lower_neighbor, left, values[left] > values[right])
        np.logical_or.at(has_lower_neighbor, right, values[right] > values[left])
    mask = (
        (values > 0.0)
        & (values >= neighbor_max)
        & has_lower_neighbor
    )
    peak_ids = np.flatnonzero(mask).astype(np.int64)
    ranked = peak_ids[
        np.lexsort((peak_ids, -values[peak_ids]))
    ].astype(np.int64)
    return mask, ranked


def compute_seed_score_arrays(
    *,
    raw_gradients: NDArray[np.floating[Any]],
    state_ids: Sequence[int],
    vertices: NDArray[np.floating[Any]],
    faces: NDArray[np.integer[Any]],
    alpha: float = AKITA_SMOOTHING_ALPHA,
) -> SeedScoreArrays:
    """严格按冻结公式计算 ``q_i``、``d_i`` 与 ``d_tilde_i``。

    ``raw_gradients`` 为完整 Action-only float ``[S,N_v,3]``。每个 state 的
    scale 是其全部顶点/RGB 分量的最大绝对值，不允许逐顶点或逐通道归一化。
    """

    gradients = np.asarray(raw_gradients, dtype=np.float64)
    state_array = np.asarray(state_ids, dtype=np.int64)
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if gradients.ndim != 3 or gradients.shape[2] != 3:
        raise SeedScoreAuditError("raw_gradients 必须为 [S,N_v,3]")
    if state_array.shape != (gradients.shape[0],):
        raise SeedScoreAuditError("state_ids 与 gradient state 维不一致")
    if len(set(int(item) for item in state_array)) != len(state_array):
        raise SeedScoreAuditError("state_ids 包含重复")
    if vertex_array.shape != (gradients.shape[1], 3):
        raise SeedScoreAuditError("OBJ vertices 与 gradient 顶点维不一致")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise SeedScoreAuditError("OBJ faces 必须为 [F,3]")
    if not all(
        np.all(np.isfinite(value))
        for value in (gradients, vertex_array)
    ):
        raise SeedScoreAuditError("gradient/vertices 包含 NaN 或 Inf")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise SeedScoreAuditError("alpha 必须为有限正数")

    absolute = np.abs(gradients)
    scales = absolute.max(axis=(1, 2))
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise SeedScoreAuditError("每个 state 的全局 max-abs scale 必须有限非零")
    state_p99 = np.quantile(
        absolute.reshape(len(gradients), -1),
        0.99,
        axis=1,
        method="linear",
    ).astype(np.float64)
    state_max_to_p99 = np.divide(
        scales,
        np.maximum(state_p99, np.finfo(np.float64).tiny),
    )
    normalized = gradients / scales[:, None, None]
    normalized_norms = np.linalg.norm(normalized, axis=2)
    mean_sensitivity = normalized_norms.mean(axis=0)
    mean_gradient = normalized.mean(axis=0)
    mean_vector_norm = np.linalg.norm(mean_gradient, axis=1)
    consistency = np.zeros_like(mean_sensitivity)
    positive = mean_sensitivity > np.finfo(np.float64).tiny
    consistency[positive] = (
        mean_vector_norm[positive] / mean_sensitivity[positive]
    )
    if np.any(consistency < -FLOAT_ATOL) or np.any(
        consistency > 1.0 + FLOAT_ATOL
    ):
        raise SeedScoreAuditError("direction consistency 超出 [0,1]")
    consistency = np.minimum(np.maximum(consistency, 0.0), 1.0)
    raw_score = mean_sensitivity * consistency

    laplacian, mass, valid_faces = build_cotangent_laplacian_and_mass(
        vertex_array,
        face_array,
    )
    if len(valid_faces) != len(face_array):
        raise SeedScoreAuditError("正式 Seed Score 不接受退化 OBJ face")
    total_area = float(mass.sum())
    tau = float(alpha * alpha * total_area)
    raw_density = raw_score / mass
    system = sparse.diags(mass, format="csr") + tau * laplacian
    right_hand_side = mass * raw_density
    smoothed = np.asarray(
        spsolve(system, right_hand_side),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(smoothed)):
        raise SeedScoreAuditError("implicit smoothing 产生 NaN/Inf")
    residual = system @ smoothed - right_hand_side
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(right_hand_side), np.finfo(np.float64).tiny)
    )
    before_mass_integral = float(np.dot(mass, raw_density))
    after_mass_integral = float(np.dot(mass, smoothed))
    conservation_error = abs(after_mass_integral - before_mass_integral) / max(
        abs(before_mass_integral),
        np.finfo(np.float64).tiny,
    )
    negativity_tolerance = FLOAT_ATOL * max(1.0, float(smoothed.max()))
    if float(smoothed.min()) < -negativity_tolerance:
        raise SeedScoreAuditError(
            "cotangent implicit smoothing 产生实质负 density，禁止静默 clamp"
        )
    peak_mask, ranked_peaks = _local_peak_evidence(smoothed, face_array)
    return SeedScoreArrays(
        state_ids=state_array,
        vertices=np.ascontiguousarray(vertex_array),
        faces=np.ascontiguousarray(face_array),
        alpha=float(alpha),
        total_surface_area=total_area,
        tau=tau,
        state_normalization_scales=scales,
        state_abs_component_p99=state_p99,
        state_max_to_p99_ratio=state_max_to_p99,
        normalized_gradient_norms=normalized_norms,
        mean_normalized_gradient=mean_gradient,
        mean_sensitivity=mean_sensitivity,
        direction_consistency=consistency,
        raw_score=raw_score,
        vertex_mass=mass,
        raw_density=raw_density,
        smoothed_density=smoothed,
        ranked_vertex_ids=_rank_descending(smoothed),
        local_peak_mask=peak_mask,
        ranked_local_peak_vertex_ids=ranked_peaks,
        linear_system_relative_residual=relative_residual,
        mass_conservation_relative_error=float(conservation_error),
    )


def summarize_seed_score_arrays(arrays: SeedScoreArrays) -> SeedScoreDiagnostics:
    """生成不参与排名或 Gate 的尖峰/分布摘要。"""

    ratios = arrays.state_max_to_p99_ratio
    consistency = arrays.direction_consistency
    return SeedScoreDiagnostics(
        num_states=len(arrays.state_ids),
        num_vertices=len(arrays.vertices),
        local_peak_count=len(arrays.ranked_local_peak_vertex_ids),
        normalization_max_to_p99_min=float(ratios.min()),
        normalization_max_to_p99_median=float(np.median(ratios)),
        normalization_max_to_p99_max=float(ratios.max()),
        raw_score_max_to_p99=_max_to_p99(arrays.raw_score),
        raw_density_max_to_p99=_max_to_p99(arrays.raw_density),
        smoothed_density_max_to_p99=_max_to_p99(
            arrays.smoothed_density
        ),
        direction_consistency_mean=float(consistency.mean()),
        direction_consistency_median=float(np.median(consistency)),
        direction_consistency_max=float(consistency.max()),
        smoothed_density_min=float(arrays.smoothed_density.min()),
        smoothed_density_max=float(arrays.smoothed_density.max()),
    )


def _safe_path(root: Path, relative_path: str) -> Path | None:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def load_dense_gradient_stack(
    rows: Sequence[DenseSeedGradientEvidence],
    *,
    artifact_root: str | Path,
) -> NDArray[np.float32]:
    """先复核 Dense evidence，再读取完整 float32 ``[S,N_v,3]``。"""

    root = Path(artifact_root)
    gradients: list[NDArray[np.float32]] = []
    for row in rows:
        decision = evaluate_dense_seed_gradient_evidence(
            row,
            artifact_root=root,
        )
        if not decision.gate_pass:
            raise SeedScoreAuditError(
                f"state {row.objective_evidence.state_id} Dense artifact无效: "
                + "; ".join(decision.failures)
            )
        path = _safe_path(root, row.artifact_relative_path)
        if path is None:
            raise SeedScoreAuditError("Dense artifact path 非法")
        with np.load(path, allow_pickle=False) as archive:
            gradient = np.asarray(
                archive["dense_geometry_gradient"],
                dtype=np.float32,
            )
        gradients.append(np.ascontiguousarray(gradient))
    if not gradients:
        raise SeedScoreAuditError("Dense evidence 不得为空")
    return np.stack(gradients, axis=0)


def build_seed_score_provenance(
    *,
    code_commit: str,
    dense_metrics_path: str | Path,
    dense_rows: Sequence[DenseSeedGradientEvidence],
    mesh_file_sha256: str,
    vertices: FloatArray,
    faces: IntArray,
) -> SeedScoreProvenance:
    """构造并验证 Dense/mesh 输入 provenance。"""

    rows = list(dense_rows)
    state_ids = tuple(row.objective_evidence.state_id for row in rows)
    if state_ids != EXPECTED_STATE_IDS:
        raise SeedScoreAuditError("正式 Seed Score states 必须精确等于0-9")
    dense_commits = {row.objective_evidence.code_commit for row in rows}
    dense_schemas = {row.schema_version for row in rows}
    mesh_hashes = {row.metadata.mesh_sha256 for row in rows}
    if len(dense_commits) != 1 or len(dense_schemas) != 1:
        raise SeedScoreAuditError("Dense rows 的 commit/schema 不一致")
    if dense_schemas != {DENSE_SEED_AUDIT_SCHEMA_VERSION}:
        raise SeedScoreAuditError("Dense schema 不是正式版本")
    if mesh_hashes != {mesh_file_sha256}:
        raise SeedScoreAuditError("OBJ file SHA-256 与 Dense evidence 不一致")
    if not _is_git_commit(code_commit):
        raise SeedScoreAuditError("code_commit 必须是40位小写十六进制 SHA")
    return SeedScoreProvenance(
        code_commit=code_commit,
        dense_schema_version=next(iter(dense_schemas)),
        dense_code_commit=next(iter(dense_commits)),
        dense_metrics_sha256=file_sha256(dense_metrics_path),
        dense_artifact_relative_paths=tuple(
            row.artifact_relative_path for row in rows
        ),
        dense_artifact_sha256=tuple(row.artifact_sha256 for row in rows),
        dense_gradient_sha256=tuple(
            row.dense_geometry_gradient_sha256 for row in rows
        ),
        state_ids=state_ids,
        mesh_file_sha256=mesh_file_sha256,
        mesh_array_sha256=mesh_array_sha256(vertices, faces),
    )


def write_seed_score_artifact(
    *,
    output_root: str | Path,
    provenance: SeedScoreProvenance,
    arrays: SeedScoreArrays,
) -> SeedScoreArtifactEvidence:
    """独占写入包含全部中间量和内嵌OBJ几何的禁止pickle NPZ。"""

    if tuple(int(item) for item in arrays.state_ids) != provenance.state_ids:
        raise SeedScoreAuditError("score state IDs 与 provenance 不一致")
    if mesh_array_sha256(arrays.vertices, arrays.faces) != (
        provenance.mesh_array_sha256
    ):
        raise SeedScoreAuditError("score mesh arrays 与 provenance 不一致")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    relative_path = Path("seed_score_arrays.npz")
    artifact_path = root / relative_path
    if artifact_path.exists():
        raise FileExistsError(f"拒绝覆盖 Seed Score artifact: {artifact_path}")
    payload = {
        "schema_version": np.asarray(SEED_SCORE_SCHEMA_VERSION),
        "code_commit": np.asarray(provenance.code_commit),
        "dense_schema_version": np.asarray(provenance.dense_schema_version),
        "dense_code_commit": np.asarray(provenance.dense_code_commit),
        "dense_metrics_sha256": np.asarray(provenance.dense_metrics_sha256),
        "dense_artifact_relative_paths": np.asarray(
            provenance.dense_artifact_relative_paths,
            dtype=np.str_,
        ),
        "dense_artifact_sha256": np.asarray(
            provenance.dense_artifact_sha256,
            dtype=np.str_,
        ),
        "dense_gradient_sha256": np.asarray(
            provenance.dense_gradient_sha256,
            dtype=np.str_,
        ),
        "state_ids": arrays.state_ids,
        "mesh_file_sha256": np.asarray(provenance.mesh_file_sha256),
        "mesh_array_sha256": np.asarray(provenance.mesh_array_sha256),
        "vertices": arrays.vertices,
        "faces": arrays.faces,
        "alpha": np.asarray(arrays.alpha, dtype=np.float64),
        "total_surface_area": np.asarray(
            arrays.total_surface_area,
            dtype=np.float64,
        ),
        "tau": np.asarray(arrays.tau, dtype=np.float64),
        "state_normalization_scales": arrays.state_normalization_scales,
        "state_abs_component_p99": arrays.state_abs_component_p99,
        "state_max_to_p99_ratio": arrays.state_max_to_p99_ratio,
        "normalized_gradient_norms": arrays.normalized_gradient_norms,
        "mean_normalized_gradient": arrays.mean_normalized_gradient,
        "mean_sensitivity": arrays.mean_sensitivity,
        "direction_consistency": arrays.direction_consistency,
        "raw_score": arrays.raw_score,
        "vertex_mass": arrays.vertex_mass,
        "raw_density": arrays.raw_density,
        "smoothed_density": arrays.smoothed_density,
        "ranked_vertex_ids": arrays.ranked_vertex_ids,
        "local_peak_mask": arrays.local_peak_mask,
        "ranked_local_peak_vertex_ids": arrays.ranked_local_peak_vertex_ids,
        "linear_system_relative_residual": np.asarray(
            arrays.linear_system_relative_residual,
            dtype=np.float64,
        ),
        "mass_conservation_relative_error": np.asarray(
            arrays.mass_conservation_relative_error,
            dtype=np.float64,
        ),
        "production_support_constructed": np.asarray(False),
    }
    with artifact_path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return SeedScoreArtifactEvidence(
        schema_version=SEED_SCORE_SCHEMA_VERSION,
        provenance=provenance,
        artifact_relative_path=str(relative_path),
        artifact_sha256=file_sha256(artifact_path),
        diagnostics=summarize_seed_score_arrays(arrays),
        production_support_constructed=False,
    )


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise SeedScoreAuditError(f"{key} 必须为scalar")
    return value.item()


def _arrays_from_archive(
    archive: Mapping[str, np.ndarray],
) -> SeedScoreArrays:
    return SeedScoreArrays(
        state_ids=np.asarray(archive["state_ids"], dtype=np.int64),
        vertices=np.asarray(archive["vertices"], dtype=np.float64),
        faces=np.asarray(archive["faces"], dtype=np.int64),
        alpha=float(_scalar(archive, "alpha")),
        total_surface_area=float(_scalar(archive, "total_surface_area")),
        tau=float(_scalar(archive, "tau")),
        state_normalization_scales=np.asarray(
            archive["state_normalization_scales"], dtype=np.float64
        ),
        state_abs_component_p99=np.asarray(
            archive["state_abs_component_p99"], dtype=np.float64
        ),
        state_max_to_p99_ratio=np.asarray(
            archive["state_max_to_p99_ratio"], dtype=np.float64
        ),
        normalized_gradient_norms=np.asarray(
            archive["normalized_gradient_norms"], dtype=np.float64
        ),
        mean_normalized_gradient=np.asarray(
            archive["mean_normalized_gradient"], dtype=np.float64
        ),
        mean_sensitivity=np.asarray(
            archive["mean_sensitivity"], dtype=np.float64
        ),
        direction_consistency=np.asarray(
            archive["direction_consistency"], dtype=np.float64
        ),
        raw_score=np.asarray(archive["raw_score"], dtype=np.float64),
        vertex_mass=np.asarray(archive["vertex_mass"], dtype=np.float64),
        raw_density=np.asarray(archive["raw_density"], dtype=np.float64),
        smoothed_density=np.asarray(
            archive["smoothed_density"], dtype=np.float64
        ),
        ranked_vertex_ids=np.asarray(
            archive["ranked_vertex_ids"], dtype=np.int64
        ),
        local_peak_mask=np.asarray(
            archive["local_peak_mask"], dtype=np.bool_
        ),
        ranked_local_peak_vertex_ids=np.asarray(
            archive["ranked_local_peak_vertex_ids"], dtype=np.int64
        ),
        linear_system_relative_residual=float(
            _scalar(archive, "linear_system_relative_residual")
        ),
        mass_conservation_relative_error=float(
            _scalar(archive, "mass_conservation_relative_error")
        ),
    )


def load_seed_score_artifact_arrays(
    path: str | Path,
) -> SeedScoreArrays:
    """禁止pickle并按冻结keys读取一个score NPZ，供repeat诊断使用。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(SCORE_ARTIFACT_KEYS):
            raise SeedScoreAuditError("Seed Score NPZ keys与冻结schema不一致")
        if _scalar(archive, "schema_version") != SEED_SCORE_SCHEMA_VERSION:
            raise SeedScoreAuditError("Seed Score NPZ schema不匹配")
        if bool(_scalar(archive, "production_support_constructed")):
            raise SeedScoreAuditError("Seed Score NPZ不得包含production support")
        return _arrays_from_archive(archive)


def _compare_arrays(
    actual: SeedScoreArrays,
    expected: SeedScoreArrays,
) -> list[str]:
    failures: list[str] = []
    exact_names = (
        "state_ids",
        "faces",
        "ranked_vertex_ids",
        "local_peak_mask",
        "ranked_local_peak_vertex_ids",
    )
    float_names = (
        "vertices",
        "state_normalization_scales",
        "state_abs_component_p99",
        "state_max_to_p99_ratio",
        "normalized_gradient_norms",
        "mean_normalized_gradient",
        "mean_sensitivity",
        "direction_consistency",
        "raw_score",
        "vertex_mass",
        "raw_density",
        "smoothed_density",
    )
    for name in exact_names:
        if not np.array_equal(getattr(actual, name), getattr(expected, name)):
            failures.append(f"artifact {name} 与重算结果不一致")
    for name in float_names:
        if not np.allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
        ):
            failures.append(f"artifact {name} 与重算结果不一致")
    for name in (
        "alpha",
        "total_surface_area",
        "tau",
        "linear_system_relative_residual",
        "mass_conservation_relative_error",
    ):
        if not np.isclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
        ):
            failures.append(f"artifact {name} 与重算结果不一致")
    return failures


def evaluate_seed_score_artifact(
    evidence: SeedScoreArtifactEvidence,
    *,
    artifact_root: str | Path,
    dense_artifact_root: str | Path,
    dense_rows: Sequence[DenseSeedGradientEvidence],
) -> SeedScoreArtifactDecision:
    """从原始 Dense NPZ 与内嵌 OBJ 独立重算整个 score pipeline。"""

    failures: list[str] = []
    if evidence.schema_version != SEED_SCORE_SCHEMA_VERSION:
        failures.append("Seed Score schema 不匹配")
    if evidence.production_support_constructed:
        failures.append("Seed Score artifact 禁止构造 production support")
    path = _safe_path(Path(artifact_root), evidence.artifact_relative_path)
    if path is None or not path.is_file():
        failures.append("Seed Score artifact path无效或文件不存在")
        return SeedScoreArtifactDecision(False, tuple(failures))
    if file_sha256(path) != evidence.artifact_sha256:
        failures.append("Seed Score artifact SHA-256不匹配")
        return SeedScoreArtifactDecision(False, tuple(failures))
    rows = list(dense_rows)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(SCORE_ARTIFACT_KEYS):
                failures.append("Seed Score NPZ keys与冻结schema不一致")
                return SeedScoreArtifactDecision(False, tuple(failures))
            if bool(_scalar(archive, "production_support_constructed")):
                failures.append("NPZ声称已构造production support")
            scalar_expectations = {
                "schema_version": SEED_SCORE_SCHEMA_VERSION,
                "code_commit": evidence.provenance.code_commit,
                "dense_schema_version": evidence.provenance.dense_schema_version,
                "dense_code_commit": evidence.provenance.dense_code_commit,
                "dense_metrics_sha256": evidence.provenance.dense_metrics_sha256,
                "mesh_file_sha256": evidence.provenance.mesh_file_sha256,
                "mesh_array_sha256": evidence.provenance.mesh_array_sha256,
            }
            for name, expected in scalar_expectations.items():
                if _scalar(archive, name) != expected:
                    failures.append(f"artifact {name}与provenance不一致")
            exact_provenance = {
                "dense_artifact_relative_paths": np.asarray(
                    evidence.provenance.dense_artifact_relative_paths,
                    dtype=np.str_,
                ),
                "dense_artifact_sha256": np.asarray(
                    evidence.provenance.dense_artifact_sha256,
                    dtype=np.str_,
                ),
                "dense_gradient_sha256": np.asarray(
                    evidence.provenance.dense_gradient_sha256,
                    dtype=np.str_,
                ),
                "state_ids": np.asarray(
                    evidence.provenance.state_ids,
                    dtype=np.int64,
                ),
            }
            for name, expected in exact_provenance.items():
                if not np.array_equal(archive[name], expected):
                    failures.append(f"artifact {name}与provenance不一致")
            actual = _arrays_from_archive(archive)
        if mesh_array_sha256(actual.vertices, actual.faces) != (
            evidence.provenance.mesh_array_sha256
        ):
            failures.append("内嵌OBJ mesh array SHA-256不匹配")
        row_state_ids = tuple(row.objective_evidence.state_id for row in rows)
        if row_state_ids != evidence.provenance.state_ids:
            failures.append("Dense rows state IDs与score provenance不一致")
        if tuple(row.artifact_sha256 for row in rows) != (
            evidence.provenance.dense_artifact_sha256
        ):
            failures.append("Dense artifact hashes与score provenance不一致")
        if tuple(row.dense_geometry_gradient_sha256 for row in rows) != (
            evidence.provenance.dense_gradient_sha256
        ):
            failures.append("Dense gradient hashes与score provenance不一致")
        gradients = load_dense_gradient_stack(
            rows,
            artifact_root=dense_artifact_root,
        )
        expected_arrays = compute_seed_score_arrays(
            raw_gradients=gradients,
            state_ids=evidence.provenance.state_ids,
            vertices=actual.vertices,
            faces=actual.faces,
            alpha=actual.alpha,
        )
        failures.extend(_compare_arrays(actual, expected_arrays))
        if summarize_seed_score_arrays(actual) != evidence.diagnostics:
            failures.append("Seed Score diagnostics与artifact不一致")
    except (OSError, ValueError, KeyError, SeedScoreAuditError) as error:
        failures.append(f"Seed Score artifact无法严格复核: {error}")
    return SeedScoreArtifactDecision(not failures, tuple(failures))


def _cosine(first: FloatArray, second: FloatArray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / max(denominator, np.finfo(float).tiny))


def compare_seed_score_stability(
    first: SeedScoreArrays,
    second: SeedScoreArrays,
    *,
    top_fractions: Sequence[float] = (0.001, 0.005, 0.01, 0.05),
    local_peak_limit: int = 100,
) -> SeedScoreStabilityReport:
    """比较两个repeat的高分区域/峰值稳定性；只报告，不设置通过阈值。"""

    failures: list[str] = []
    if not np.array_equal(first.state_ids, second.state_ids):
        failures.append("repeat state IDs不一致")
    if mesh_array_sha256(first.vertices, first.faces) != mesh_array_sha256(
        second.vertices, second.faces
    ):
        failures.append("repeat mesh不一致")
    if first.alpha != second.alpha:
        failures.append("repeat alpha不一致")
    if len(first.raw_score) != len(second.raw_score):
        failures.append("repeat vertex count不一致")
    if failures:
        return SeedScoreStabilityReport(
            compatible=False,
            failures=tuple(failures),
            raw_score_cosine=None,
            raw_density_cosine=None,
            smoothed_density_cosine=None,
            smoothed_density_spearman=None,
            top_fraction_jaccard=(),
            compared_local_peak_count=0,
            top_local_peak_jaccard=None,
            threshold_registered=False,
        )
    overlaps: list[tuple[float, float]] = []
    num_vertices = len(first.raw_score)
    for fraction in top_fractions:
        if not 0.0 < fraction <= 1.0:
            raise SeedScoreAuditError("top fraction必须位于(0,1]")
        count = max(1, int(np.ceil(fraction * num_vertices)))
        first_ids = set(first.ranked_vertex_ids[:count].tolist())
        second_ids = set(second.ranked_vertex_ids[:count].tolist())
        overlaps.append(
            (fraction, len(first_ids & second_ids) / len(first_ids | second_ids))
        )
    peak_count = min(
        local_peak_limit,
        len(first.ranked_local_peak_vertex_ids),
        len(second.ranked_local_peak_vertex_ids),
    )
    peak_jaccard: Optional[float] = None
    if peak_count > 0:
        first_peaks = set(
            first.ranked_local_peak_vertex_ids[:peak_count].tolist()
        )
        second_peaks = set(
            second.ranked_local_peak_vertex_ids[:peak_count].tolist()
        )
        peak_jaccard = len(first_peaks & second_peaks) / len(
            first_peaks | second_peaks
        )
    correlation_result = spearmanr(
        first.smoothed_density,
        second.smoothed_density,
    )
    correlation_value = float(
        correlation_result.statistic
        if hasattr(correlation_result, "statistic")
        else correlation_result.correlation
    )
    correlation: Optional[float] = (
        correlation_value if np.isfinite(correlation_value) else None
    )
    return SeedScoreStabilityReport(
        compatible=True,
        failures=(),
        raw_score_cosine=_cosine(first.raw_score, second.raw_score),
        raw_density_cosine=_cosine(first.raw_density, second.raw_density),
        smoothed_density_cosine=_cosine(
            first.smoothed_density,
            second.smoothed_density,
        ),
        smoothed_density_spearman=correlation,
        top_fraction_jaccard=tuple(overlaps),
        compared_local_peak_count=peak_count,
        top_local_peak_jaccard=peak_jaccard,
        threshold_registered=False,
    )


def evidence_to_json(evidence: SeedScoreArtifactEvidence) -> dict[str, Any]:
    """把 evidence 转成 manifest 可序列化字典。"""

    return asdict(evidence)


def seed_score_evidence_from_mapping(
    value: Mapping[str, Any],
) -> SeedScoreArtifactEvidence:
    """从manifest mapping恢复强类型evidence。"""

    provenance_value: Mapping[str, Any] = value["provenance"]
    diagnostics_value: Mapping[str, Any] = value["diagnostics"]
    provenance = SeedScoreProvenance(
        code_commit=str(provenance_value["code_commit"]),
        dense_schema_version=str(
            provenance_value["dense_schema_version"]
        ),
        dense_code_commit=str(provenance_value["dense_code_commit"]),
        dense_metrics_sha256=str(
            provenance_value["dense_metrics_sha256"]
        ),
        dense_artifact_relative_paths=tuple(
            str(item)
            for item in provenance_value["dense_artifact_relative_paths"]
        ),
        dense_artifact_sha256=tuple(
            str(item) for item in provenance_value["dense_artifact_sha256"]
        ),
        dense_gradient_sha256=tuple(
            str(item) for item in provenance_value["dense_gradient_sha256"]
        ),
        state_ids=tuple(
            int(item) for item in provenance_value["state_ids"]
        ),
        mesh_file_sha256=str(provenance_value["mesh_file_sha256"]),
        mesh_array_sha256=str(provenance_value["mesh_array_sha256"]),
    )
    diagnostics = SeedScoreDiagnostics(
        **{
            key: int(raw_value)
            if key in ("num_states", "num_vertices", "local_peak_count")
            else float(raw_value)
            for key, raw_value in diagnostics_value.items()
        }
    )
    return SeedScoreArtifactEvidence(
        schema_version=str(value["schema_version"]),
        provenance=provenance,
        artifact_relative_path=str(value["artifact_relative_path"]),
        artifact_sha256=str(value["artifact_sha256"]),
        diagnostics=diagnostics,
        production_support_constructed=bool(
            value["production_support_constructed"]
        ),
    )
