"""固定面积 Connected Vertex Support 的确定性纯计算契约。

本模块消费已经验收的 Seed Score 与投影 coverage contribution，不读取模型、
MuJoCo 或 nvdiffrast，也不进入攻击训练。Akita MVP 的面积、区域数、coverage
门槛和 seed 分离尺度均为冻结常量；调用方不能通过参数悄悄覆盖它们。

多区域增长采用确定性轮转：按 seed 顺序，每轮每个尚未达到份额的区域加入其
自身最高 Smoothed Seed Density 的合法 face-adjacent frontier 顶点；同分按
顶点 ID。该调度只消除区域之间的执行顺序歧义，不增加优化超参数。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final, Literal, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .visibility_evidence import VisibilityEvidenceStatus


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

AKITA_SUPPORT_AREA_FRACTION: Final[float] = 0.10
AKITA_MAX_REGIONS: Final[int] = 3
AKITA_PRIMARY_COVERAGE_MIN: Final[float] = 0.20
AKITA_SMOOTHING_ALPHA: Final[float] = 0.05
SUPPORT_MARGINAL_GAIN_ATOL: Final[float] = 64.0 * np.finfo(np.float64).eps
SUPPORT_COVERAGE_NUMERATOR_ATOL: Final[float] = 1e-6
SUPPORT_CONSTRUCTION_SCHEMA_VERSION: Final[str] = (
    "openvla-support-construction-v1"
)


class SupportConstructionError(ValueError):
    """输入或中间结果违反冻结的 Support Construction 契约。"""


@dataclass(frozen=True)
class ProjectionCoverageEvidence:
    """一个 ``(state, view)`` 对每个几何顶点的线性 coverage 证据。

    ``*_vertex_numerator`` 为 CPU float64 ``[N_v]``，保存单独控制该顶点时
    ``sum(alpha * w_i)`` 的投影质量；因此任意 support 的分子等于所选
    顶点 contribution 之和。source 表示 512 raw camera，effective 表示
    512→224 area downsample 后的 center-crop 有效视野。
    """

    state_id: int
    view_name: str
    status: VisibilityEvidenceStatus
    source_vertex_numerator: FloatArray
    effective_vertex_numerator: FloatArray
    source_denominator: float
    effective_denominator: float
    source_instance_vertex_numerator: FloatArray
    effective_instance_vertex_numerator: FloatArray
    source_instance_denominators: FloatArray
    effective_instance_denominators: FloatArray

    def __post_init__(self) -> None:
        if self.status not in {
            "not_observable",
            "insufficient_observation",
            "invalid_alignment",
            "valid",
        }:
            raise SupportConstructionError(
                f"未知 Visibility Evidence Status: {self.status}"
            )
        source = np.asarray(self.source_vertex_numerator)
        effective = np.asarray(self.effective_vertex_numerator)
        if source.dtype != np.float64 or effective.dtype != np.float64:
            raise TypeError("coverage vertex numerator 必须使用 float64")
        if source.ndim != 1 or effective.shape != source.shape:
            raise SupportConstructionError(
                "source/effective vertex numerator 必须是同 shape [N_v]"
            )
        if len(source) == 0:
            raise SupportConstructionError("coverage evidence 不得没有顶点")
        if not np.isfinite(source).all() or not np.isfinite(effective).all():
            raise SupportConstructionError("coverage numerator 包含 NaN/Inf")
        # nvdiffrast 允许重心坐标在 1e-6 内越过三角形边界；对应单顶点
        # contribution 也保留同一容差，不做 clamp，以便 artifact 可精确重放。
        if np.any(source < -SUPPORT_COVERAGE_NUMERATOR_ATOL) or np.any(
            effective < -SUPPORT_COVERAGE_NUMERATOR_ATOL
        ):
            raise SupportConstructionError("coverage numerator 不得为负")
        for name, value in (
            ("source_denominator", self.source_denominator),
            ("effective_denominator", self.effective_denominator),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise SupportConstructionError(f"{name} 必须为有限非负数")
        source_instances = np.asarray(self.source_instance_vertex_numerator)
        effective_instances = np.asarray(
            self.effective_instance_vertex_numerator
        )
        source_denominators = np.asarray(self.source_instance_denominators)
        effective_denominators = np.asarray(
            self.effective_instance_denominators
        )
        num_instances = len(source_denominators)
        expected_matrix_shape = (num_instances, len(source))
        if (
            num_instances <= 0
            or source_instances.dtype != np.float64
            or effective_instances.dtype != np.float64
            or source_instances.shape != expected_matrix_shape
            or effective_instances.shape != expected_matrix_shape
            or source_denominators.dtype != np.float64
            or source_denominators.shape != (num_instances,)
            or effective_denominators.dtype != np.float64
            or effective_denominators.shape != (num_instances,)
        ):
            raise SupportConstructionError(
                "per-instance coverage 必须为 float64 [K,N_v] 与 [K]"
            )
        if not (
            np.isfinite(source_instances).all()
            and np.isfinite(effective_instances).all()
            and np.isfinite(source_denominators).all()
            and np.isfinite(effective_denominators).all()
        ):
            raise SupportConstructionError("per-instance coverage 包含 NaN/Inf")
        if np.any(source_instances < -SUPPORT_COVERAGE_NUMERATOR_ATOL) or np.any(
            effective_instances < -SUPPORT_COVERAGE_NUMERATOR_ATOL
        ):
            raise SupportConstructionError("per-instance numerator 不得为负")
        if np.any(source_denominators < 0.0) or np.any(
            effective_denominators < 0.0
        ):
            raise SupportConstructionError("per-instance denominator 不得为负")
        if not np.allclose(
            source_instances.sum(axis=0),
            source,
            rtol=1e-12,
            atol=1e-12,
        ) or not np.allclose(
            effective_instances.sum(axis=0),
            effective,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise SupportConstructionError("union numerator 不等于逐实例之和")
        if not math.isclose(
            float(source_denominators.sum()),
            self.source_denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(effective_denominators.sum()),
            self.effective_denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SupportConstructionError("union denominator 不等于逐实例之和")
        if self.status == "not_observable" and self.effective_denominator != 0.0:
            raise SupportConstructionError(
                "not_observable 的 effective denominator 必须为零"
            )
        if self.status != "not_observable" and self.effective_denominator <= 0.0:
            raise SupportConstructionError(
                f"{self.status} 的 effective denominator 必须为正"
            )

    @property
    def num_vertices(self) -> int:
        return len(self.effective_vertex_numerator)

    def coverage(
        self,
        support_mask: BoolArray,
        *,
        space: Literal["source", "effective"] = "effective",
    ) -> Optional[float]:
        """返回指定视野的 coverage；无观测状态明确返回 ``None``。"""

        support = np.asarray(support_mask)
        if support.dtype != np.bool_ or support.shape != (self.num_vertices,):
            raise SupportConstructionError("support mask 必须为 bool [N_v]")
        if space == "source":
            numerator = self.source_vertex_numerator
            denominator = self.source_denominator
        elif space == "effective":
            numerator = self.effective_vertex_numerator
            denominator = self.effective_denominator
        else:
            raise SupportConstructionError(f"未知 coverage space: {space}")
        if denominator <= 0.0:
            return None
        return float(numerator[support].sum(dtype=np.float64) / denominator)

    def per_instance_coverage(
        self,
        support_mask: BoolArray,
        *,
        space: Literal["source", "effective"] = "effective",
    ) -> tuple[Optional[float], ...]:
        """返回逐共享纹理实例 coverage；零观测实例返回 ``None``。"""

        support = np.asarray(support_mask)
        if support.dtype != np.bool_ or support.shape != (self.num_vertices,):
            raise SupportConstructionError("support mask 必须为 bool [N_v]")
        if space == "source":
            numerators = self.source_instance_vertex_numerator
            denominators = self.source_instance_denominators
        elif space == "effective":
            numerators = self.effective_instance_vertex_numerator
            denominators = self.effective_instance_denominators
        else:
            raise SupportConstructionError(f"未知 coverage space: {space}")
        return tuple(
            (
                float(numerators[index, support].sum(dtype=np.float64) / denominator)
                if denominator > 0.0
                else None
            )
            for index, denominator in enumerate(denominators)
        )


@dataclass(frozen=True)
class StateCoverageResult:
    """一个 state 的 raw/effective coverage 与四状态标签。"""

    state_id: int
    view_name: str
    status: VisibilityEvidenceStatus
    source_coverage: Optional[float]
    effective_coverage: Optional[float]
    source_denominator: float
    effective_denominator: float
    source_per_instance_coverage: tuple[Optional[float], ...]
    effective_per_instance_coverage: tuple[Optional[float], ...]
    source_per_instance_denominators: tuple[float, ...]
    effective_per_instance_denominators: tuple[float, ...]


@dataclass(frozen=True)
class RegionConstructionResult:
    """一个连通区域的确定性增长结果。"""

    region_index: int
    seed_vertex_id: int
    vertex_ids: tuple[int, ...]
    target_mass: float
    actual_mass: float
    last_added_vertex_mass: float
    perimeter: float
    compactness: Optional[float]
    complete: bool
    failure: Optional[str]


@dataclass(frozen=True)
class SeedSelectionDiagnostics:
    """新增 seed 的 coverage 增益和测地分离拒绝统计。"""

    seed_vertex_id: int
    worst_primary_state_id: int
    marginal_coverage_gain: float
    geodesic_distance: float
    nonpositive_gain_rejected_count: int
    separation_rejected_count: int


@dataclass(frozen=True)
class SupportCandidateResult:
    """固定区域数 ``r`` 的候选及全部 Gate 证据。"""

    num_regions: int
    seed_vertex_ids: tuple[int, ...]
    seed_diagnostics: tuple[SeedSelectionDiagnostics, ...]
    regions: tuple[RegionConstructionResult, ...]
    support_mask: BoolArray
    target_mass: float
    actual_mass: float
    area_fraction_actual: float
    primary_coverage: tuple[StateCoverageResult, ...]
    wrist_coverage: tuple[StateCoverageResult, ...]
    primary_valid_min: Optional[float]
    primary_valid_mean: Optional[float]
    primary_valid_median: Optional[float]
    worst_primary_state_id: Optional[int]
    tied_worst_primary_state_ids: tuple[int, ...]
    construction_complete: bool
    primary_gate_pass: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class SupportConstructionResult:
    """依次尝试 r=1..3 后的只读构造结果。"""

    schema_version: str
    total_surface_area: float
    target_area_fraction: float
    max_regions: int
    primary_coverage_min: float
    smoothing_alpha: float
    smoothing_length: float
    candidates: tuple[SupportCandidateResult, ...]
    selected_candidate_index: Optional[int]
    production_support_constructed: bool


def _validate_geometry_inputs(
    vertices: FloatArray,
    faces: IntArray,
    vertex_mass: FloatArray,
    smoothed_density: FloatArray,
) -> None:
    if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise TypeError("vertices 必须为 float64 [N_v,3]")
    if faces.dtype != np.int64 or faces.ndim != 2 or faces.shape[1] != 3:
        raise TypeError("faces 必须为 int64 [N_f,3]")
    num_vertices = len(vertices)
    if num_vertices == 0 or len(faces) == 0:
        raise SupportConstructionError("mesh 不得为空")
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= num_vertices:
        raise SupportConstructionError("faces 顶点索引越界")
    for name, values in (
        ("vertex_mass", vertex_mass),
        ("smoothed_density", smoothed_density),
    ):
        if values.dtype != np.float64 or values.shape != (num_vertices,):
            raise TypeError(f"{name} 必须为 float64 [N_v]")
        if not np.isfinite(values).all():
            raise SupportConstructionError(f"{name} 包含 NaN/Inf")
    if np.any(vertex_mass <= 0.0):
        raise SupportConstructionError("vertex_mass 必须严格为正")


def build_face_adjacency(
    faces: IntArray,
    *,
    num_vertices: int,
) -> tuple[tuple[int, ...], ...]:
    """构造按顶点 ID 排序的原始 OBJ face-edge 邻接表。"""

    if faces.dtype != np.int64 or faces.ndim != 2 or faces.shape[1] != 3:
        raise TypeError("faces 必须为 int64 [N_f,3]")
    if num_vertices <= 0 or faces.min(initial=0) < 0 or faces.max(initial=-1) >= num_vertices:
        raise SupportConstructionError("face adjacency 输入索引越界")
    neighbors: list[set[int]] = [set() for _ in range(num_vertices)]
    for triangle in faces:
        first, second, third = (int(value) for value in triangle)
        for left, right in (
            (first, second),
            (second, third),
            (third, first),
        ):
            if left == right:
                raise SupportConstructionError("mesh 包含退化自环 edge")
            neighbors[left].add(right)
            neighbors[right].add(left)
    return tuple(tuple(sorted(row)) for row in neighbors)


def geodesic_distance_from_support(
    vertices: FloatArray,
    adjacency: tuple[tuple[int, ...], ...],
    support_mask: BoolArray,
) -> FloatArray:
    """在原始 OBJ edge-length 图上运行确定性 multi-source Dijkstra。"""

    num_vertices = len(vertices)
    if vertices.dtype != np.float64 or vertices.shape != (num_vertices, 3):
        raise TypeError("vertices 必须为 float64 [N_v,3]")
    if len(adjacency) != num_vertices:
        raise SupportConstructionError("adjacency 与 vertices 数量不一致")
    if support_mask.dtype != np.bool_ or support_mask.shape != (num_vertices,):
        raise SupportConstructionError("support_mask 必须为 bool [N_v]")
    source_ids = np.flatnonzero(support_mask)
    if len(source_ids) == 0:
        raise SupportConstructionError("测地距离 source support 不得为空")
    distances = np.full(num_vertices, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for source in source_ids:
        vertex_id = int(source)
        distances[vertex_id] = 0.0
        heapq.heappush(queue, (0.0, vertex_id))
    while queue:
        distance, vertex_id = heapq.heappop(queue)
        if distance != distances[vertex_id]:
            continue
        for neighbor in adjacency[vertex_id]:
            edge_length = float(np.linalg.norm(vertices[vertex_id] - vertices[neighbor]))
            candidate = distance + edge_length
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _region_shape_diagnostics(
    region_mask: BoolArray,
    *,
    vertices: FloatArray,
    adjacency: tuple[tuple[int, ...], ...],
    area: float,
) -> tuple[float, Optional[float]]:
    """报告 vertex-cut edge 长度及 ``4*pi*A/P^2``，只作形状诊断。"""

    perimeter = 0.0
    for vertex_id, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            if neighbor <= vertex_id:
                continue
            if bool(region_mask[vertex_id]) != bool(region_mask[neighbor]):
                perimeter += float(np.linalg.norm(vertices[vertex_id] - vertices[neighbor]))
    compactness = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0.0
        else None
    )
    return perimeter, compactness


def grow_equal_mass_regions(
    *,
    seed_vertex_ids: tuple[int, ...],
    target_mass_per_region: float,
    vertices: FloatArray,
    adjacency: tuple[tuple[int, ...], ...],
    vertex_mass: FloatArray,
    smoothed_density: FloatArray,
) -> tuple[tuple[RegionConstructionResult, ...], BoolArray]:
    """按 seed 顺序轮转执行确定性、互不 face-adjacent 的 best-first 增长。"""

    num_vertices = len(vertices)
    if not seed_vertex_ids or len(set(seed_vertex_ids)) != len(seed_vertex_ids):
        raise SupportConstructionError("region seeds 必须非空且互异")
    if not math.isfinite(target_mass_per_region) or target_mass_per_region <= 0.0:
        raise SupportConstructionError("每区 target mass 必须为有限正数")
    if any(seed < 0 or seed >= num_vertices for seed in seed_vertex_ids):
        raise SupportConstructionError("seed 顶点索引越界")
    if len(adjacency) != num_vertices:
        raise SupportConstructionError("adjacency 与 geometry 不一致")

    owner = np.full(num_vertices, -1, dtype=np.int64)
    region_vertices: list[list[int]] = []
    region_mass: list[float] = []
    last_mass: list[float] = []
    frontiers: list[list[tuple[float, int]]] = []
    failures: list[Optional[str]] = [None] * len(seed_vertex_ids)
    for region_index, seed in enumerate(seed_vertex_ids):
        if any(owner[neighbor] >= 0 for neighbor in adjacency[seed]):
            raise SupportConstructionError("不同 region seed 不得 face-adjacent")
        owner[seed] = region_index
        region_vertices.append([seed])
        seed_mass = float(vertex_mass[seed])
        region_mass.append(seed_mass)
        last_mass.append(seed_mass)
        frontier: list[tuple[float, int]] = []
        for neighbor in adjacency[seed]:
            heapq.heappush(frontier, (-float(smoothed_density[neighbor]), neighbor))
        frontiers.append(frontier)

    def pop_legal_frontier(region_index: int) -> Optional[int]:
        frontier = frontiers[region_index]
        while frontier:
            _, candidate = heapq.heappop(frontier)
            if owner[candidate] >= 0:
                continue
            if any(
                owner[neighbor] >= 0 and owner[neighbor] != region_index
                for neighbor in adjacency[candidate]
            ):
                continue
            if not any(owner[neighbor] == region_index for neighbor in adjacency[candidate]):
                continue
            return candidate
        return None

    while True:
        incomplete = [
            index
            for index, mass in enumerate(region_mass)
            if mass < target_mass_per_region and failures[index] is None
        ]
        if not incomplete:
            break
        progress = False
        for region_index in incomplete:
            candidate = pop_legal_frontier(region_index)
            if candidate is None:
                failures[region_index] = "legal_frontier_exhausted"
                continue
            owner[candidate] = region_index
            region_vertices[region_index].append(candidate)
            added_mass = float(vertex_mass[candidate])
            region_mass[region_index] += added_mass
            last_mass[region_index] = added_mass
            for neighbor in adjacency[candidate]:
                if owner[neighbor] < 0:
                    heapq.heappush(
                        frontiers[region_index],
                        (-float(smoothed_density[neighbor]), neighbor),
                    )
            progress = True
        if not progress and any(failure is None for failure in failures):
            raise RuntimeError("region growing 未进展但仍有 active region")

    support_mask = owner >= 0
    results: list[RegionConstructionResult] = []
    for region_index, seed in enumerate(seed_vertex_ids):
        region_mask = owner == region_index
        actual_mass = float(region_mass[region_index])
        perimeter, compactness = _region_shape_diagnostics(
            region_mask,
            vertices=vertices,
            adjacency=adjacency,
            area=actual_mass,
        )
        complete = actual_mass >= target_mass_per_region and failures[region_index] is None
        if complete and actual_mass - target_mass_per_region > last_mass[region_index] + SUPPORT_MARGINAL_GAIN_ATOL:
            raise RuntimeError("region mass overshoot 超过一个边界顶点")
        results.append(
            RegionConstructionResult(
                region_index=region_index,
                seed_vertex_id=seed,
                vertex_ids=tuple(region_vertices[region_index]),
                target_mass=target_mass_per_region,
                actual_mass=actual_mass,
                last_added_vertex_mass=last_mass[region_index],
                perimeter=perimeter,
                compactness=compactness,
                complete=complete,
                failure=failures[region_index],
            )
        )
    return tuple(results), support_mask.astype(np.bool_)


def _coverage_results(
    evidence: tuple[ProjectionCoverageEvidence, ...],
    support_mask: BoolArray,
) -> tuple[StateCoverageResult, ...]:
    return tuple(
        StateCoverageResult(
            state_id=row.state_id,
            view_name=row.view_name,
            status=row.status,
            source_coverage=row.coverage(support_mask, space="source"),
            effective_coverage=row.coverage(support_mask, space="effective"),
            source_denominator=row.source_denominator,
            effective_denominator=row.effective_denominator,
            source_per_instance_coverage=row.per_instance_coverage(
                support_mask,
                space="source",
            ),
            effective_per_instance_coverage=row.per_instance_coverage(
                support_mask,
                space="effective",
            ),
            source_per_instance_denominators=tuple(
                float(value) for value in row.source_instance_denominators
            ),
            effective_per_instance_denominators=tuple(
                float(value) for value in row.effective_instance_denominators
            ),
        )
        for row in evidence
    )


def _primary_statistics(
    rows: tuple[StateCoverageResult, ...],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], tuple[int, ...]]:
    valid = tuple(
        row
        for row in rows
        if row.status == "valid" and row.effective_coverage is not None
    )
    if not valid:
        return None, None, None, None, ()
    values = np.asarray(
        [float(row.effective_coverage) for row in valid],
        dtype=np.float64,
    )
    minimum = float(values.min())
    tied_ids = tuple(
        sorted(
            row.state_id
            for row in valid
            if math.isclose(
                float(row.effective_coverage),
                minimum,
                rel_tol=0.0,
                abs_tol=SUPPORT_MARGINAL_GAIN_ATOL,
            )
        )
    )
    return (
        minimum,
        float(values.mean()),
        float(np.median(values)),
        tied_ids[0],
        tied_ids,
    )


def _ranked_visible_first_seed(
    smoothed_density: FloatArray,
    primary_evidence: tuple[ProjectionCoverageEvidence, ...],
) -> int:
    valid = tuple(row for row in primary_evidence if row.status == "valid")
    if not valid:
        raise SupportConstructionError("至少需要一个 valid Primary state")
    visible = np.zeros(len(smoothed_density), dtype=np.bool_)
    for row in valid:
        visible |= row.effective_vertex_numerator > SUPPORT_MARGINAL_GAIN_ATOL
    candidate_ids = np.flatnonzero(visible)
    if len(candidate_ids) == 0:
        raise SupportConstructionError("没有顶点在 valid Primary 中产生正 coverage")
    order = np.lexsort((candidate_ids, -smoothed_density[candidate_ids]))
    return int(candidate_ids[order[0]])


def _select_additional_seed(
    *,
    support_mask: BoolArray,
    primary_evidence: tuple[ProjectionCoverageEvidence, ...],
    smoothed_density: FloatArray,
    vertices: FloatArray,
    adjacency: tuple[tuple[int, ...], ...],
    minimum_geodesic_distance: float,
) -> SeedSelectionDiagnostics:
    valid = tuple(row for row in primary_evidence if row.status == "valid")
    if not valid:
        raise SupportConstructionError("新增 seed 需要 valid Primary state")
    coverages = tuple(
        (float(row.coverage(support_mask, space="effective")), row.state_id, row)
        for row in valid
    )
    _, worst_state_id, worst = min(coverages, key=lambda item: (item[0], item[1]))
    gains = worst.effective_vertex_numerator / worst.effective_denominator
    positive = gains > SUPPORT_MARGINAL_GAIN_ATOL
    unoccupied = ~support_mask
    distances = geodesic_distance_from_support(vertices, adjacency, support_mask)
    separated = distances + SUPPORT_MARGINAL_GAIN_ATOL >= minimum_geodesic_distance
    candidates = np.flatnonzero(positive & unoccupied & separated)
    if len(candidates) == 0:
        raise SupportConstructionError(
            f"state {worst_state_id} 没有同时满足正 coverage 增益与测地分离的 seed"
        )
    order = np.lexsort((candidates, -smoothed_density[candidates]))
    seed = int(candidates[order[0]])
    return SeedSelectionDiagnostics(
        seed_vertex_id=seed,
        worst_primary_state_id=worst_state_id,
        marginal_coverage_gain=float(gains[seed]),
        geodesic_distance=float(distances[seed]),
        nonpositive_gain_rejected_count=int(np.count_nonzero(unoccupied & ~positive)),
        separation_rejected_count=int(np.count_nonzero(unoccupied & positive & ~separated)),
    )


def construct_akita_support_candidates(
    *,
    vertices: FloatArray,
    faces: IntArray,
    vertex_mass: FloatArray,
    smoothed_density: FloatArray,
    primary_evidence: tuple[ProjectionCoverageEvidence, ...],
    wrist_evidence: tuple[ProjectionCoverageEvidence, ...] = (),
) -> SupportConstructionResult:
    """严格按冻结的 Akita MVP 规则依次构造 ``r=1,2,3`` 候选。

    返回的 ``selected_candidate_index`` 只标记第一个通过 Gate 的候选；
    ``production_support_constructed`` 永远为 ``False``，防止 contract audit
    产物被误当作已经冻结的训练输入。
    """

    _validate_geometry_inputs(vertices, faces, vertex_mass, smoothed_density)
    num_vertices = len(vertices)
    if not primary_evidence:
        raise SupportConstructionError("Primary coverage evidence 不得为空")
    all_evidence = primary_evidence + wrist_evidence
    if any(row.num_vertices != num_vertices for row in all_evidence):
        raise SupportConstructionError("coverage evidence 与 mesh 顶点数不一致")
    if any(row.view_name != "primary" for row in primary_evidence):
        raise SupportConstructionError("primary_evidence 含非 primary view")
    if any(row.view_name != "wrist_source_crop_proxy" for row in wrist_evidence):
        raise SupportConstructionError("wrist evidence view 名称不合法")
    primary_ids = [row.state_id for row in primary_evidence]
    if len(set(primary_ids)) != len(primary_ids):
        raise SupportConstructionError("Primary state ID 不得重复")
    if any(row.status == "invalid_alignment" for row in primary_evidence):
        raise SupportConstructionError("Primary invalid_alignment 必须使构造验收失败")

    total_area = float(vertex_mass.sum(dtype=np.float64))
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise SupportConstructionError("total surface area 必须为有限正数")
    target_mass = AKITA_SUPPORT_AREA_FRACTION * total_area
    smoothing_length = AKITA_SMOOTHING_ALPHA * math.sqrt(total_area)
    adjacency = build_face_adjacency(faces, num_vertices=num_vertices)
    seeds: list[int] = [
        _ranked_visible_first_seed(smoothed_density, primary_evidence)
    ]
    seed_diagnostics: list[SeedSelectionDiagnostics] = []
    candidates: list[SupportCandidateResult] = []
    previous_support: Optional[BoolArray] = None

    for num_regions in range(1, AKITA_MAX_REGIONS + 1):
        if num_regions > 1:
            if previous_support is None:
                raise RuntimeError("缺少上一候选 support")
            diagnostics = _select_additional_seed(
                support_mask=previous_support,
                primary_evidence=primary_evidence,
                smoothed_density=smoothed_density,
                vertices=vertices,
                adjacency=adjacency,
                minimum_geodesic_distance=smoothing_length,
            )
            seeds.append(diagnostics.seed_vertex_id)
            seed_diagnostics.append(diagnostics)
        failures: list[str] = []
        try:
            regions, support_mask = grow_equal_mass_regions(
                seed_vertex_ids=tuple(seeds),
                target_mass_per_region=target_mass / float(num_regions),
                vertices=vertices,
                adjacency=adjacency,
                vertex_mass=vertex_mass,
                smoothed_density=smoothed_density,
            )
        except SupportConstructionError as error:
            raise SupportConstructionError(
                f"r={num_regions} region growing 输入失败: {error}"
            ) from error
        construction_complete = all(region.complete for region in regions)
        if not construction_complete:
            failures.extend(
                f"region_{region.region_index}:{region.failure}"
                for region in regions
                if not region.complete
            )
        primary_rows = _coverage_results(primary_evidence, support_mask)
        wrist_rows = _coverage_results(wrist_evidence, support_mask)
        minimum, mean, median, worst_id, tied_ids = _primary_statistics(primary_rows)
        if minimum is None:
            failures.append("no_valid_primary_state")
        elif minimum + SUPPORT_MARGINAL_GAIN_ATOL < AKITA_PRIMARY_COVERAGE_MIN:
            failures.append("primary_coverage_below_frozen_minimum")
        actual_mass = float(vertex_mass[support_mask].sum(dtype=np.float64))
        primary_gate_pass = (
            construction_complete
            and minimum is not None
            and minimum + SUPPORT_MARGINAL_GAIN_ATOL >= AKITA_PRIMARY_COVERAGE_MIN
        )
        candidate = SupportCandidateResult(
            num_regions=num_regions,
            seed_vertex_ids=tuple(seeds),
            seed_diagnostics=tuple(seed_diagnostics),
            regions=regions,
            support_mask=support_mask,
            target_mass=target_mass,
            actual_mass=actual_mass,
            area_fraction_actual=actual_mass / total_area,
            primary_coverage=primary_rows,
            wrist_coverage=wrist_rows,
            primary_valid_min=minimum,
            primary_valid_mean=mean,
            primary_valid_median=median,
            worst_primary_state_id=worst_id,
            tied_worst_primary_state_ids=tied_ids,
            construction_complete=construction_complete,
            primary_gate_pass=primary_gate_pass,
            failures=tuple(failures),
        )
        candidates.append(candidate)
        previous_support = support_mask
        if primary_gate_pass:
            break

    selected = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.primary_gate_pass
        ),
        None,
    )
    return SupportConstructionResult(
        schema_version=SUPPORT_CONSTRUCTION_SCHEMA_VERSION,
        total_surface_area=total_area,
        target_area_fraction=AKITA_SUPPORT_AREA_FRACTION,
        max_regions=AKITA_MAX_REGIONS,
        primary_coverage_min=AKITA_PRIMARY_COVERAGE_MIN,
        smoothing_alpha=AKITA_SMOOTHING_ALPHA,
        smoothing_length=smoothing_length,
        candidates=tuple(candidates),
        selected_candidate_index=selected,
        production_support_constructed=False,
    )
