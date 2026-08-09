"""Support Construction 的完整 NPZ artifact 与独立重算契约。

一个 audit 保存两层只读证据：

* coverage contribution artifact：逐 ``(state, view, instance, vertex)`` 的
  source/effective 分子与分母；
* candidate artifact：每个已尝试 ``r`` 的 support mask、region owner、seed、
  面积、逐 state coverage 和 Gate 结果。

两层产物都明确声明没有冻结或构造 production Fixed Support。加载器禁用 pickle，
校验固定 keys、SHA-256 和 provenance；候选层可以从 score + contribution 完整
重算，防止只信任 JSON 摘要。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .seed_score_audit import SeedScoreArrays
from .support_construction import (
    ProjectionCoverageEvidence,
    StateCoverageResult,
    SupportCandidateResult,
    SupportConstructionResult,
    construct_akita_support_candidates,
)
from .support_coverage_contribution import CoverageContributionDiagnostics


SUPPORT_AUDIT_SCHEMA_VERSION: Final[str] = "openvla-support-audit-v1"
COVERAGE_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "code_commit",
        "seed_score_artifact_sha256",
        "visibility_manifest_sha256",
        "visibility_metrics_sha256",
        "state_ids",
        "view_names",
        "statuses",
        "visibility_npz_relative_paths",
        "visibility_npz_sha256",
        "source_vertex_numerator",
        "effective_vertex_numerator",
        "source_denominators",
        "effective_denominators",
        "instance_offsets",
        "source_instance_vertex_numerator",
        "effective_instance_vertex_numerator",
        "source_instance_denominators",
        "effective_instance_denominators",
        "production_support_constructed",
    }
)
CANDIDATE_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "code_commit",
        "seed_score_artifact_sha256",
        "visibility_manifest_sha256",
        "visibility_metrics_sha256",
        "coverage_artifact_sha256",
        "total_surface_area",
        "target_area_fraction",
        "max_regions",
        "primary_coverage_min",
        "smoothing_alpha",
        "smoothing_length",
        "candidate_region_counts",
        "candidate_seed_vertex_ids",
        "candidate_support_masks",
        "candidate_region_owner",
        "candidate_target_mass",
        "candidate_actual_mass",
        "candidate_area_fraction_actual",
        "candidate_construction_complete",
        "candidate_primary_gate_pass",
        "region_target_mass",
        "region_actual_mass",
        "region_last_added_vertex_mass",
        "region_perimeter",
        "region_compactness",
        "region_complete",
        "primary_state_ids",
        "primary_statuses",
        "primary_source_coverage",
        "primary_effective_coverage",
        "wrist_state_ids",
        "wrist_statuses",
        "wrist_source_coverage",
        "wrist_effective_coverage",
        "selected_candidate_index",
        "production_support_constructed",
        "fixed_support_frozen",
    }
)


class SupportAuditError(ValueError):
    """Support audit artifact 或 provenance 不满足固定契约。"""


@dataclass(frozen=True)
class SupportAuditProvenance:
    """Support audit 对 score 与 visibility bundle 的哈希绑定。"""

    code_commit: str
    seed_score_artifact_sha256: str
    visibility_manifest_sha256: str
    visibility_metrics_sha256: str
    visibility_npz_relative_paths: tuple[str, ...]
    visibility_npz_sha256: tuple[str, ...]


@dataclass(frozen=True)
class SupportArtifactEvidence:
    """两个禁止覆盖的 NPZ artifact 及其哈希。"""

    schema_version: str
    provenance: SupportAuditProvenance
    coverage_artifact_relative_path: str
    coverage_artifact_sha256: str
    candidate_artifact_relative_path: str
    candidate_artifact_sha256: str
    production_support_constructed: bool
    fixed_support_frozen: bool


@dataclass(frozen=True)
class SupportArtifactDecision:
    """artifact 完整重算的判定。"""

    gate_pass: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class SupportRepeatStabilityReport:
    """两个同规则 support audit 的只读稳定性报告，不定义通过阈值。"""

    compatible: bool
    failures: tuple[str, ...]
    selected_region_count_equal: Optional[bool]
    selected_seed_vertex_ids_equal: Optional[bool]
    selected_support_jaccard: Optional[float]
    selected_support_intersection_count: Optional[int]
    selected_support_union_count: Optional[int]
    selected_vertex_count_first: Optional[int]
    selected_vertex_count_second: Optional[int]
    selected_area_fraction_absolute_difference: Optional[float]
    primary_effective_coverage_max_absolute_difference: Optional[float]
    threshold_registered: bool


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SupportAuditError(f"{name} 必须是64位小写 SHA-256")


def _validate_provenance(provenance: SupportAuditProvenance) -> None:
    if len(provenance.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in provenance.code_commit
    ):
        raise SupportAuditError("code_commit 必须是40位小写 Git SHA")
    for name, value in (
        ("seed_score_artifact_sha256", provenance.seed_score_artifact_sha256),
        ("visibility_manifest_sha256", provenance.visibility_manifest_sha256),
        ("visibility_metrics_sha256", provenance.visibility_metrics_sha256),
    ):
        _validate_sha256(value, name=name)
    if not provenance.visibility_npz_relative_paths or len(
        provenance.visibility_npz_relative_paths
    ) != len(provenance.visibility_npz_sha256):
        raise SupportAuditError("visibility NPZ path/hash 列表为空或长度不一致")
    if len(set(provenance.visibility_npz_relative_paths)) != len(
        provenance.visibility_npz_relative_paths
    ):
        raise SupportAuditError("visibility NPZ relative path 不得重复")
    for value in provenance.visibility_npz_sha256:
        _validate_sha256(value, name="visibility_npz_sha256")


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _coverage_rows_to_arrays(
    rows: Sequence[ProjectionCoverageEvidence],
) -> dict[str, NDArray[Any]]:
    if not rows:
        raise SupportAuditError("coverage rows 不得为空")
    num_vertices = rows[0].num_vertices
    if any(row.num_vertices != num_vertices for row in rows):
        raise SupportAuditError("coverage rows 顶点数不一致")
    instance_offsets = [0]
    for row in rows:
        instance_offsets.append(
            instance_offsets[-1] + len(row.source_instance_denominators)
        )
    return {
        "state_ids": np.asarray([row.state_id for row in rows], dtype=np.int64),
        "view_names": np.asarray([row.view_name for row in rows], dtype=np.str_),
        "statuses": np.asarray([row.status for row in rows], dtype=np.str_),
        "source_vertex_numerator": np.stack(
            [row.source_vertex_numerator for row in rows]
        ),
        "effective_vertex_numerator": np.stack(
            [row.effective_vertex_numerator for row in rows]
        ),
        "source_denominators": np.asarray(
            [row.source_denominator for row in rows], dtype=np.float64
        ),
        "effective_denominators": np.asarray(
            [row.effective_denominator for row in rows], dtype=np.float64
        ),
        "instance_offsets": np.asarray(instance_offsets, dtype=np.int64),
        "source_instance_vertex_numerator": np.concatenate(
            [row.source_instance_vertex_numerator for row in rows], axis=0
        ),
        "effective_instance_vertex_numerator": np.concatenate(
            [row.effective_instance_vertex_numerator for row in rows], axis=0
        ),
        "source_instance_denominators": np.concatenate(
            [row.source_instance_denominators for row in rows]
        ),
        "effective_instance_denominators": np.concatenate(
            [row.effective_instance_denominators for row in rows]
        ),
    }


def write_coverage_contribution_artifact(
    output_path: str | Path,
    *,
    rows: Sequence[ProjectionCoverageEvidence],
    provenance: SupportAuditProvenance,
) -> str:
    """写入完整 contribution artifact，拒绝覆盖并返回 SHA-256。"""

    _validate_provenance(provenance)
    if len(rows) != len(provenance.visibility_npz_relative_paths):
        raise SupportAuditError("coverage rows 与 visibility provenance 数量不一致")
    path = Path(output_path)
    _refuse_existing(path)
    arrays = _coverage_rows_to_arrays(rows)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SUPPORT_AUDIT_SCHEMA_VERSION),
        code_commit=np.asarray(provenance.code_commit),
        seed_score_artifact_sha256=np.asarray(
            provenance.seed_score_artifact_sha256
        ),
        visibility_manifest_sha256=np.asarray(
            provenance.visibility_manifest_sha256
        ),
        visibility_metrics_sha256=np.asarray(
            provenance.visibility_metrics_sha256
        ),
        visibility_npz_relative_paths=np.asarray(
            provenance.visibility_npz_relative_paths, dtype=np.str_
        ),
        visibility_npz_sha256=np.asarray(
            provenance.visibility_npz_sha256, dtype=np.str_
        ),
        production_support_constructed=np.asarray(False, dtype=np.bool_),
        **arrays,
    )
    return file_sha256(path)


def _scalar(archive: Mapping[str, NDArray[Any]], name: str) -> Any:
    value = archive[name]
    if value.ndim != 0:
        raise SupportAuditError(f"{name} 必须是 scalar")
    return value.item()


def load_coverage_contribution_artifact(
    path: str | Path,
) -> tuple[tuple[ProjectionCoverageEvidence, ...], SupportAuditProvenance]:
    """加载完整 contribution，重新触发逐实例/union 数据类不变量。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != COVERAGE_ARTIFACT_KEYS:
            raise SupportAuditError("coverage artifact keys 与冻结 schema 不一致")
        if _scalar(archive, "schema_version") != SUPPORT_AUDIT_SCHEMA_VERSION:
            raise SupportAuditError("coverage artifact schema 不匹配")
        if bool(_scalar(archive, "production_support_constructed")):
            raise SupportAuditError("coverage artifact 不得声称 production support")
        provenance = SupportAuditProvenance(
            code_commit=str(_scalar(archive, "code_commit")),
            seed_score_artifact_sha256=str(
                _scalar(archive, "seed_score_artifact_sha256")
            ),
            visibility_manifest_sha256=str(
                _scalar(archive, "visibility_manifest_sha256")
            ),
            visibility_metrics_sha256=str(
                _scalar(archive, "visibility_metrics_sha256")
            ),
            visibility_npz_relative_paths=tuple(
                str(value) for value in archive["visibility_npz_relative_paths"]
            ),
            visibility_npz_sha256=tuple(
                str(value) for value in archive["visibility_npz_sha256"]
            ),
        )
        _validate_provenance(provenance)
        state_ids = archive["state_ids"]
        view_names = archive["view_names"]
        statuses = archive["statuses"]
        offsets = archive["instance_offsets"]
        row_count = len(state_ids)
        if (
            state_ids.dtype != np.int64
            or view_names.shape != (row_count,)
            or statuses.shape != (row_count,)
            or offsets.dtype != np.int64
            or offsets.shape != (row_count + 1,)
            or int(offsets[0]) != 0
            or np.any(np.diff(offsets) <= 0)
        ):
            raise SupportAuditError("coverage row metadata/instance offsets 非法")
        union_source = archive["source_vertex_numerator"]
        union_effective = archive["effective_vertex_numerator"]
        if union_source.dtype != np.float64 or union_source.ndim != 2 or union_effective.shape != union_source.shape:
            raise SupportAuditError("coverage union matrix 非法")
        all_source_instances = archive["source_instance_vertex_numerator"]
        all_effective_instances = archive["effective_instance_vertex_numerator"]
        all_source_denominators = archive["source_instance_denominators"]
        all_effective_denominators = archive["effective_instance_denominators"]
        if int(offsets[-1]) != len(all_source_instances):
            raise SupportAuditError("coverage instance offsets 末端不匹配")
        rows: list[ProjectionCoverageEvidence] = []
        for index in range(row_count):
            start, end = int(offsets[index]), int(offsets[index + 1])
            rows.append(
                ProjectionCoverageEvidence(
                    state_id=int(state_ids[index]),
                    view_name=str(view_names[index]),
                    status=str(statuses[index]),  # type: ignore[arg-type]
                    source_vertex_numerator=union_source[index].copy(),
                    effective_vertex_numerator=union_effective[index].copy(),
                    source_denominator=float(archive["source_denominators"][index]),
                    effective_denominator=float(archive["effective_denominators"][index]),
                    source_instance_vertex_numerator=all_source_instances[start:end].copy(),
                    effective_instance_vertex_numerator=all_effective_instances[start:end].copy(),
                    source_instance_denominators=all_source_denominators[start:end].copy(),
                    effective_instance_denominators=all_effective_denominators[start:end].copy(),
                )
            )
    return tuple(rows), provenance


def _optional_coverage_matrix(
    candidates: Sequence[SupportCandidateResult],
    *,
    view: str,
    space: str,
) -> tuple[NDArray[np.int64], NDArray[np.str_], NDArray[np.float64]]:
    if not candidates:
        raise SupportAuditError("candidate 列表不得为空")
    rows = candidates[0].primary_coverage if view == "primary" else candidates[0].wrist_coverage
    state_ids = np.asarray([row.state_id for row in rows], dtype=np.int64)
    statuses = np.asarray([row.status for row in rows], dtype=np.str_)
    matrix = np.full((len(candidates), len(rows)), np.nan, dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        current = candidate.primary_coverage if view == "primary" else candidate.wrist_coverage
        if tuple(row.state_id for row in current) != tuple(state_ids.tolist()):
            raise SupportAuditError(f"{view} candidate state IDs 不一致")
        for row_index, row in enumerate(current):
            value = row.source_coverage if space == "source" else row.effective_coverage
            if value is not None:
                matrix[candidate_index, row_index] = value
    return state_ids, statuses, matrix


def _candidate_arrays(result: SupportConstructionResult) -> dict[str, NDArray[Any]]:
    candidates = result.candidates
    if not candidates:
        raise SupportAuditError("Support Construction 没有候选")
    num_vertices = len(candidates[0].support_mask)
    num_candidates = len(candidates)
    seed_ids = np.full((num_candidates, result.max_regions), -1, dtype=np.int64)
    owner = np.full((num_candidates, num_vertices), -1, dtype=np.int8)
    region_shape = (num_candidates, result.max_regions)
    region_target = np.full(region_shape, np.nan, dtype=np.float64)
    region_actual = np.full(region_shape, np.nan, dtype=np.float64)
    region_last = np.full(region_shape, np.nan, dtype=np.float64)
    region_perimeter = np.full(region_shape, np.nan, dtype=np.float64)
    region_compactness = np.full(region_shape, np.nan, dtype=np.float64)
    region_complete = np.zeros(region_shape, dtype=np.bool_)
    for candidate_index, candidate in enumerate(candidates):
        seed_ids[candidate_index, : len(candidate.seed_vertex_ids)] = candidate.seed_vertex_ids
        for region in candidate.regions:
            owner[candidate_index, np.asarray(region.vertex_ids, dtype=np.int64)] = region.region_index
            region_target[candidate_index, region.region_index] = region.target_mass
            region_actual[candidate_index, region.region_index] = region.actual_mass
            region_last[candidate_index, region.region_index] = region.last_added_vertex_mass
            region_perimeter[candidate_index, region.region_index] = region.perimeter
            if region.compactness is not None:
                region_compactness[candidate_index, region.region_index] = region.compactness
            region_complete[candidate_index, region.region_index] = region.complete
        if not np.array_equal(owner[candidate_index] >= 0, candidate.support_mask):
            raise SupportAuditError("region owner union 与 support mask 不一致")
    primary_ids, primary_statuses, primary_source = _optional_coverage_matrix(
        candidates, view="primary", space="source"
    )
    _, _, primary_effective = _optional_coverage_matrix(
        candidates, view="primary", space="effective"
    )
    wrist_ids, wrist_statuses, wrist_source = _optional_coverage_matrix(
        candidates, view="wrist", space="source"
    )
    _, _, wrist_effective = _optional_coverage_matrix(
        candidates, view="wrist", space="effective"
    )
    return {
        "candidate_region_counts": np.asarray([candidate.num_regions for candidate in candidates], dtype=np.int64),
        "candidate_seed_vertex_ids": seed_ids,
        "candidate_support_masks": np.stack([candidate.support_mask for candidate in candidates]),
        "candidate_region_owner": owner,
        "candidate_target_mass": np.asarray([candidate.target_mass for candidate in candidates], dtype=np.float64),
        "candidate_actual_mass": np.asarray([candidate.actual_mass for candidate in candidates], dtype=np.float64),
        "candidate_area_fraction_actual": np.asarray([candidate.area_fraction_actual for candidate in candidates], dtype=np.float64),
        "candidate_construction_complete": np.asarray([candidate.construction_complete for candidate in candidates], dtype=np.bool_),
        "candidate_primary_gate_pass": np.asarray([candidate.primary_gate_pass for candidate in candidates], dtype=np.bool_),
        "region_target_mass": region_target,
        "region_actual_mass": region_actual,
        "region_last_added_vertex_mass": region_last,
        "region_perimeter": region_perimeter,
        "region_compactness": region_compactness,
        "region_complete": region_complete,
        "primary_state_ids": primary_ids,
        "primary_statuses": primary_statuses,
        "primary_source_coverage": primary_source,
        "primary_effective_coverage": primary_effective,
        "wrist_state_ids": wrist_ids,
        "wrist_statuses": wrist_statuses,
        "wrist_source_coverage": wrist_source,
        "wrist_effective_coverage": wrist_effective,
    }


def write_candidate_artifact(
    output_path: str | Path,
    *,
    result: SupportConstructionResult,
    provenance: SupportAuditProvenance,
    coverage_artifact_sha256: str,
) -> str:
    """写入全部候选与 region membership，拒绝覆盖并返回 SHA-256。"""

    _validate_provenance(provenance)
    _validate_sha256(coverage_artifact_sha256, name="coverage_artifact_sha256")
    if result.production_support_constructed:
        raise SupportAuditError("audit result 不得包含 production support")
    path = Path(output_path)
    _refuse_existing(path)
    arrays = _candidate_arrays(result)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SUPPORT_AUDIT_SCHEMA_VERSION),
        code_commit=np.asarray(provenance.code_commit),
        seed_score_artifact_sha256=np.asarray(provenance.seed_score_artifact_sha256),
        visibility_manifest_sha256=np.asarray(provenance.visibility_manifest_sha256),
        visibility_metrics_sha256=np.asarray(provenance.visibility_metrics_sha256),
        coverage_artifact_sha256=np.asarray(coverage_artifact_sha256),
        total_surface_area=np.asarray(result.total_surface_area, dtype=np.float64),
        target_area_fraction=np.asarray(result.target_area_fraction, dtype=np.float64),
        max_regions=np.asarray(result.max_regions, dtype=np.int64),
        primary_coverage_min=np.asarray(result.primary_coverage_min, dtype=np.float64),
        smoothing_alpha=np.asarray(result.smoothing_alpha, dtype=np.float64),
        smoothing_length=np.asarray(result.smoothing_length, dtype=np.float64),
        selected_candidate_index=np.asarray(
            -1 if result.selected_candidate_index is None else result.selected_candidate_index,
            dtype=np.int64,
        ),
        production_support_constructed=np.asarray(False, dtype=np.bool_),
        fixed_support_frozen=np.asarray(False, dtype=np.bool_),
        **arrays,
    )
    return file_sha256(path)


def _compare_candidate_archive(
    archive: Mapping[str, NDArray[Any]],
    expected: SupportConstructionResult,
) -> list[str]:
    failures: list[str] = []
    expected_arrays = _candidate_arrays(expected)
    for name, value in expected_arrays.items():
        actual = archive[name]
        if np.issubdtype(value.dtype, np.floating):
            if not np.allclose(actual, value, rtol=1e-12, atol=1e-12, equal_nan=True):
                failures.append(f"candidate artifact {name} 与重算不一致")
        elif not np.array_equal(actual, value):
            failures.append(f"candidate artifact {name} 与重算不一致")
    expected_selected = -1 if expected.selected_candidate_index is None else expected.selected_candidate_index
    if int(_scalar(archive, "selected_candidate_index")) != expected_selected:
        failures.append("selected_candidate_index 与重算不一致")
    for name, value in (
        ("total_surface_area", expected.total_surface_area),
        ("target_area_fraction", expected.target_area_fraction),
        ("primary_coverage_min", expected.primary_coverage_min),
        ("smoothing_alpha", expected.smoothing_alpha),
        ("smoothing_length", expected.smoothing_length),
    ):
        if not math.isclose(float(_scalar(archive, name)), value, rel_tol=1e-12, abs_tol=1e-12):
            failures.append(f"candidate artifact {name} 与重算不一致")
    return failures


def evaluate_support_artifacts(
    evidence: SupportArtifactEvidence,
    *,
    artifact_root: str | Path,
    score_arrays: SeedScoreArrays,
) -> SupportArtifactDecision:
    """验证哈希/provenance，并从 score + contribution 独立重算全部候选。"""

    failures: list[str] = []
    if evidence.schema_version != SUPPORT_AUDIT_SCHEMA_VERSION:
        failures.append("Support audit schema 不匹配")
    if evidence.production_support_constructed or evidence.fixed_support_frozen:
        failures.append("audit evidence 不得冻结 production Fixed Support")
    root = Path(artifact_root).resolve()
    coverage_path = (root / evidence.coverage_artifact_relative_path).resolve()
    candidate_path = (root / evidence.candidate_artifact_relative_path).resolve()
    for path in (coverage_path, candidate_path):
        if root not in path.parents or not path.is_file():
            failures.append(f"artifact path 无效: {path}")
            return SupportArtifactDecision(False, tuple(failures))
    if file_sha256(coverage_path) != evidence.coverage_artifact_sha256:
        failures.append("coverage artifact SHA-256 不匹配")
    if file_sha256(candidate_path) != evidence.candidate_artifact_sha256:
        failures.append("candidate artifact SHA-256 不匹配")
    if failures:
        return SupportArtifactDecision(False, tuple(failures))
    try:
        rows, loaded_provenance = load_coverage_contribution_artifact(coverage_path)
        if loaded_provenance != evidence.provenance:
            failures.append("coverage artifact provenance 不匹配")
        primary = tuple(row for row in rows if row.view_name == "primary")
        wrist = tuple(row for row in rows if row.view_name == "wrist_source_crop_proxy")
        expected = construct_akita_support_candidates(
            vertices=score_arrays.vertices,
            faces=score_arrays.faces,
            vertex_mass=score_arrays.vertex_mass,
            smoothed_density=score_arrays.smoothed_density,
            primary_evidence=primary,
            wrist_evidence=wrist,
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            if set(archive.files) != CANDIDATE_ARTIFACT_KEYS:
                failures.append("candidate artifact keys 与冻结 schema 不一致")
            elif _scalar(archive, "schema_version") != SUPPORT_AUDIT_SCHEMA_VERSION:
                failures.append("candidate artifact schema 不匹配")
            elif bool(_scalar(archive, "production_support_constructed")) or bool(
                _scalar(archive, "fixed_support_frozen")
            ):
                failures.append("candidate artifact 声称已冻结 production support")
            else:
                scalar_provenance = {
                    "code_commit": evidence.provenance.code_commit,
                    "seed_score_artifact_sha256": evidence.provenance.seed_score_artifact_sha256,
                    "visibility_manifest_sha256": evidence.provenance.visibility_manifest_sha256,
                    "visibility_metrics_sha256": evidence.provenance.visibility_metrics_sha256,
                    "coverage_artifact_sha256": evidence.coverage_artifact_sha256,
                }
                for name, value in scalar_provenance.items():
                    if str(_scalar(archive, name)) != value:
                        failures.append(f"candidate {name} provenance 不匹配")
                failures.extend(_compare_candidate_archive(archive, expected))
    except (OSError, KeyError, TypeError, ValueError, SupportAuditError) as error:
        failures.append(f"Support artifact 无法严格复核: {error}")
    return SupportArtifactDecision(not failures, tuple(failures))


def support_artifact_evidence_from_mapping(
    value: Mapping[str, Any],
) -> SupportArtifactEvidence:
    """从 manifest JSON mapping 恢复严格 evidence dataclass。"""

    provenance_value = value["provenance"]
    provenance = SupportAuditProvenance(
        code_commit=str(provenance_value["code_commit"]),
        seed_score_artifact_sha256=str(provenance_value["seed_score_artifact_sha256"]),
        visibility_manifest_sha256=str(provenance_value["visibility_manifest_sha256"]),
        visibility_metrics_sha256=str(provenance_value["visibility_metrics_sha256"]),
        visibility_npz_relative_paths=tuple(provenance_value["visibility_npz_relative_paths"]),
        visibility_npz_sha256=tuple(provenance_value["visibility_npz_sha256"]),
    )
    return SupportArtifactEvidence(
        schema_version=str(value["schema_version"]),
        provenance=provenance,
        coverage_artifact_relative_path=str(value["coverage_artifact_relative_path"]),
        coverage_artifact_sha256=str(value["coverage_artifact_sha256"]),
        candidate_artifact_relative_path=str(value["candidate_artifact_relative_path"]),
        candidate_artifact_sha256=str(value["candidate_artifact_sha256"]),
        production_support_constructed=bool(value["production_support_constructed"]),
        fixed_support_frozen=bool(value["fixed_support_frozen"]),
    )


def compare_support_repeat_artifacts(
    first_path: str | Path,
    second_path: str | Path,
) -> SupportRepeatStabilityReport:
    """比较两个 candidate NPZ 的选中 support；只报告，不设置门槛。"""

    failures: list[str] = []
    with np.load(Path(first_path), allow_pickle=False) as first, np.load(
        Path(second_path), allow_pickle=False
    ) as second:
        for archive, name in ((first, "first"), (second, "second")):
            if set(archive.files) != CANDIDATE_ARTIFACT_KEYS:
                failures.append(f"{name} candidate keys 不匹配")
            if bool(_scalar(archive, "production_support_constructed")) or bool(
                _scalar(archive, "fixed_support_frozen")
            ):
                failures.append(f"{name} 不是只读 audit artifact")
        for name in (
            "visibility_manifest_sha256",
            "visibility_metrics_sha256",
            "total_surface_area",
            "target_area_fraction",
            "max_regions",
            "primary_coverage_min",
            "smoothing_alpha",
            "smoothing_length",
            "primary_state_ids",
            "primary_statuses",
        ):
            if not np.array_equal(first[name], second[name]):
                failures.append(f"repeat {name} 不一致")
        first_selected = int(_scalar(first, "selected_candidate_index"))
        second_selected = int(_scalar(second, "selected_candidate_index"))
        if first_selected < 0 or second_selected < 0:
            failures.append("repeat 至少一个没有通过 Gate 的 selected candidate")
        if failures:
            return SupportRepeatStabilityReport(
                compatible=False,
                failures=tuple(failures),
                selected_region_count_equal=None,
                selected_seed_vertex_ids_equal=None,
                selected_support_jaccard=None,
                selected_support_intersection_count=None,
                selected_support_union_count=None,
                selected_vertex_count_first=None,
                selected_vertex_count_second=None,
                selected_area_fraction_absolute_difference=None,
                primary_effective_coverage_max_absolute_difference=None,
                threshold_registered=False,
            )
        first_mask = first["candidate_support_masks"][first_selected]
        second_mask = second["candidate_support_masks"][second_selected]
        intersection = int(np.count_nonzero(first_mask & second_mask))
        union = int(np.count_nonzero(first_mask | second_mask))
        first_count = int(np.count_nonzero(first_mask))
        second_count = int(np.count_nonzero(second_mask))
        first_r = int(first["candidate_region_counts"][first_selected])
        second_r = int(second["candidate_region_counts"][second_selected])
        first_seeds = first["candidate_seed_vertex_ids"][first_selected, :first_r]
        second_seeds = second["candidate_seed_vertex_ids"][second_selected, :second_r]
        coverage_difference = np.abs(
            first["primary_effective_coverage"][first_selected]
            - second["primary_effective_coverage"][second_selected]
        )
        return SupportRepeatStabilityReport(
            compatible=True,
            failures=(),
            selected_region_count_equal=first_r == second_r,
            selected_seed_vertex_ids_equal=np.array_equal(first_seeds, second_seeds),
            selected_support_jaccard=intersection / union if union > 0 else 1.0,
            selected_support_intersection_count=intersection,
            selected_support_union_count=union,
            selected_vertex_count_first=first_count,
            selected_vertex_count_second=second_count,
            selected_area_fraction_absolute_difference=abs(
                float(first["candidate_area_fraction_actual"][first_selected])
                - float(second["candidate_area_fraction_actual"][second_selected])
            ),
            primary_effective_coverage_max_absolute_difference=float(
                np.nanmax(coverage_difference)
            ),
            threshold_registered=False,
        )


def candidate_to_json(candidate: SupportCandidateResult) -> dict[str, Any]:
    """生成不含完整 mask 的人类可读候选摘要。"""

    def coverage_row(row: StateCoverageResult) -> dict[str, Any]:
        return {
            "state_id": row.state_id,
            "view_name": row.view_name,
            "status": row.status,
            "source_coverage": row.source_coverage,
            "effective_coverage": row.effective_coverage,
            "source_denominator": row.source_denominator,
            "effective_denominator": row.effective_denominator,
            "source_per_instance_coverage": list(row.source_per_instance_coverage),
            "effective_per_instance_coverage": list(row.effective_per_instance_coverage),
            "source_per_instance_denominators": list(row.source_per_instance_denominators),
            "effective_per_instance_denominators": list(row.effective_per_instance_denominators),
        }

    return {
        "num_regions": candidate.num_regions,
        "seed_vertex_ids": list(candidate.seed_vertex_ids),
        "seed_diagnostics": [diagnostic.__dict__ for diagnostic in candidate.seed_diagnostics],
        "regions": [
            {
                "region_index": region.region_index,
                "seed_vertex_id": region.seed_vertex_id,
                "vertex_count": len(region.vertex_ids),
                "target_mass": region.target_mass,
                "actual_mass": region.actual_mass,
                "last_added_vertex_mass": region.last_added_vertex_mass,
                "perimeter": region.perimeter,
                "compactness": region.compactness,
                "complete": region.complete,
                "failure": region.failure,
            }
            for region in candidate.regions
        ],
        "selected_vertex_count": int(np.count_nonzero(candidate.support_mask)),
        "trainable_rgb_scalar_count": 3 * int(np.count_nonzero(candidate.support_mask)),
        "target_mass": candidate.target_mass,
        "actual_mass": candidate.actual_mass,
        "area_fraction_actual": candidate.area_fraction_actual,
        "primary_valid_min": candidate.primary_valid_min,
        "primary_valid_mean": candidate.primary_valid_mean,
        "primary_valid_median": candidate.primary_valid_median,
        "worst_primary_state_id": candidate.worst_primary_state_id,
        "tied_worst_primary_state_ids": list(candidate.tied_worst_primary_state_ids),
        "construction_complete": candidate.construction_complete,
        "primary_gate_pass": candidate.primary_gate_pass,
        "failures": list(candidate.failures),
        "primary_coverage": [coverage_row(row) for row in candidate.primary_coverage],
        "wrist_coverage": [coverage_row(row) for row in candidate.wrist_coverage],
    }


def coverage_diagnostics_to_json(
    diagnostics: Sequence[CoverageContributionDiagnostics],
) -> list[dict[str, Any]]:
    """把 diagnostics 转为 JSON-safe mapping。"""

    return [diagnostic.__dict__ for diagnostic in diagnostics]
