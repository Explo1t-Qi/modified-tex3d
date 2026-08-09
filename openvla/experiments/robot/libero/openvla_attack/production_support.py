"""不可变 Production Fixed Vertex Support artifact 契约。

本模块只把已经通过 Support Construction 与 repeat 复核的 canonical candidate
冻结为训练可消费的顶点集合，不重新选点、不计算 ``rho_nat``、不校准
``lambda_spec``，也不允许正式训练。紧凑参数坐标固定为升序几何顶点 ID；任何
candidate、score、visibility、mesh 或 renderer mapping 哈希变化都会使验收失败。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from .seed_score_audit import (
    SeedScoreArrays,
    file_sha256,
    load_seed_score_artifact_arrays,
)


PRODUCTION_SUPPORT_SCHEMA_VERSION: Final[str] = (
    "openvla-production-fixed-support-v1"
)
AKITA_NATURALNESS_K_NONCONSTANT: Final[int] = 128
SOURCE_CANDIDATE_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "selected_candidate_index",
        "candidate_primary_gate_pass",
        "candidate_support_masks",
        "candidate_region_counts",
        "candidate_seed_vertex_ids",
        "primary_state_ids",
        "primary_statuses",
        "primary_source_coverage",
        "primary_effective_coverage",
        "wrist_state_ids",
        "wrist_statuses",
        "wrist_source_coverage",
        "wrist_effective_coverage",
        "total_surface_area",
        "target_area_fraction",
        "candidate_target_mass",
        "candidate_actual_mass",
        "candidate_area_fraction_actual",
        "primary_coverage_min",
        "production_support_constructed",
        "fixed_support_frozen",
    }
)
PRODUCTION_SUPPORT_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "code_commit",
        "object_name",
        "source_support_manifest_sha256",
        "source_candidate_artifact_sha256",
        "source_coverage_artifact_sha256",
        "seed_score_manifest_sha256",
        "seed_score_artifact_sha256",
        "visibility_manifest_sha256",
        "visibility_metrics_sha256",
        "mesh_file_sha256",
        "mesh_array_sha256",
        "renderer_faces_sha256",
        "render_to_geometry_sha256",
        "selected_candidate_index",
        "num_geometry_vertices",
        "support_mask",
        "support_vertex_indices",
        "support_mask_sha256",
        "compact_coordinate_order",
        "num_regions",
        "seed_vertex_ids",
        "total_surface_area",
        "target_area_fraction",
        "target_mass",
        "actual_mass",
        "actual_area_fraction",
        "primary_coverage_min_threshold",
        "primary_state_ids",
        "primary_statuses",
        "primary_source_coverage",
        "primary_effective_coverage",
        "wrist_state_ids",
        "wrist_statuses",
        "wrist_source_coverage",
        "wrist_effective_coverage",
        "naturalness_k_nonconstant",
        "production_support_constructed",
        "fixed_support_frozen",
        "rho_nat_calibrated",
        "lambda_spec_calibrated",
        "formal_training_allowed",
    }
)


class ProductionSupportError(ValueError):
    """Production Support 输入、artifact 或训练边界不满足冻结契约。"""


@dataclass(frozen=True)
class ProductionSupportProvenance:
    """Production Support 对全部上游权威证据的 SHA-256 绑定。"""

    code_commit: str
    object_name: str
    source_support_manifest_sha256: str
    source_candidate_artifact_sha256: str
    source_coverage_artifact_sha256: str
    seed_score_manifest_sha256: str
    seed_score_artifact_sha256: str
    visibility_manifest_sha256: str
    visibility_metrics_sha256: str
    mesh_file_sha256: str
    mesh_array_sha256: str
    renderer_faces_sha256: str
    render_to_geometry_sha256: str


@dataclass(frozen=True)
class FrozenProductionSupport:
    """训练可消费但尚未通过谱校准的不可变 Fixed Support。"""

    schema_version: str
    provenance: ProductionSupportProvenance
    selected_candidate_index: int
    num_geometry_vertices: int
    support_mask: NDArray[np.bool_]
    support_vertex_indices: NDArray[np.int64]
    support_mask_sha256: str
    compact_coordinate_order: str
    num_regions: int
    seed_vertex_ids: tuple[int, ...]
    total_surface_area: float
    target_area_fraction: float
    target_mass: float
    actual_mass: float
    actual_area_fraction: float
    primary_coverage_min_threshold: float
    primary_state_ids: tuple[int, ...]
    primary_statuses: tuple[str, ...]
    primary_source_coverage: tuple[float, ...]
    primary_effective_coverage: tuple[float, ...]
    wrist_state_ids: tuple[int, ...]
    wrist_statuses: tuple[str, ...]
    wrist_source_coverage: tuple[float, ...]
    wrist_effective_coverage: tuple[float, ...]
    naturalness_k_nonconstant: int
    production_support_constructed: bool
    fixed_support_frozen: bool
    rho_nat_calibrated: bool
    lambda_spec_calibrated: bool
    formal_training_allowed: bool


@dataclass(frozen=True)
class ProductionSupportEvidence:
    """一个 Production Support NPZ 与冻结 manifest 证据。"""

    schema_version: str
    artifact_relative_path: str
    artifact_sha256: str
    support_mask_sha256: str
    num_geometry_vertices: int
    num_support_vertices: int
    trainable_rgb_scalar_count: int
    production_support_constructed: bool
    fixed_support_frozen: bool
    rho_nat_calibrated: bool
    lambda_spec_calibrated: bool
    formal_training_allowed: bool


@dataclass(frozen=True)
class ProductionSupportDecision:
    """Production Support 与全部上游 artifact 的独立验收结果。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_provenance(value: ProductionSupportProvenance) -> None:
    if not _is_hex(value.code_commit, 40):
        raise ProductionSupportError("code_commit 必须是40位小写 Git SHA")
    if not value.object_name:
        raise ProductionSupportError("object_name 不得为空")
    for name in (
        "source_support_manifest_sha256",
        "source_candidate_artifact_sha256",
        "source_coverage_artifact_sha256",
        "seed_score_manifest_sha256",
        "seed_score_artifact_sha256",
        "visibility_manifest_sha256",
        "visibility_metrics_sha256",
        "mesh_file_sha256",
        "mesh_array_sha256",
        "renderer_faces_sha256",
        "render_to_geometry_sha256",
    ):
        if not _is_hex(str(getattr(value, name)), 64):
            raise ProductionSupportError(f"{name} 必须是64位小写 SHA-256")


def array_sha256(array: NDArray[Any]) -> str:
    """按 dtype、shape 与连续 bytes 计算稳定数组 SHA-256。"""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _scalar(archive: Mapping[str, NDArray[Any]], name: str) -> Any:
    value = archive[name]
    if value.ndim != 0:
        raise ProductionSupportError(f"{name} 必须是 scalar")
    return value.item()


def _validate_frozen_support(value: FrozenProductionSupport) -> None:
    _validate_provenance(value.provenance)
    if value.schema_version != PRODUCTION_SUPPORT_SCHEMA_VERSION:
        raise ProductionSupportError("Production Support schema 不匹配")
    if value.selected_candidate_index < 0:
        raise ProductionSupportError("selected candidate index 必须非负")
    if value.num_geometry_vertices <= 0:
        raise ProductionSupportError("num_geometry_vertices 必须为正")
    if value.support_mask.dtype != np.bool_ or value.support_mask.shape != (
        value.num_geometry_vertices,
    ):
        raise ProductionSupportError("support_mask 必须为 bool [N_v]")
    if (
        value.support_vertex_indices.dtype != np.int64
        or value.support_vertex_indices.ndim != 1
        or len(value.support_vertex_indices) == 0
    ):
        raise ProductionSupportError(
            "support_vertex_indices 必须为非空 int64 [N_support]"
        )
    expected_indices = np.flatnonzero(value.support_mask).astype(np.int64)
    if not np.array_equal(value.support_vertex_indices, expected_indices):
        raise ProductionSupportError(
            "紧凑坐标必须是 support mask 的升序几何顶点 ID"
        )
    if value.compact_coordinate_order != "ascending_geometry_vertex_id":
        raise ProductionSupportError("未知 compact coordinate order")
    if array_sha256(value.support_mask) != value.support_mask_sha256:
        raise ProductionSupportError("support mask semantic SHA-256 不匹配")
    if value.num_regions <= 0 or len(value.seed_vertex_ids) != value.num_regions:
        raise ProductionSupportError("region count 与 seed 数量不一致")
    if len(set(value.seed_vertex_ids)) != len(value.seed_vertex_ids):
        raise ProductionSupportError("seed vertex IDs 不得重复")
    for name, number in (
        ("total_surface_area", value.total_surface_area),
        ("target_area_fraction", value.target_area_fraction),
        ("target_mass", value.target_mass),
        ("actual_mass", value.actual_mass),
        ("actual_area_fraction", value.actual_area_fraction),
        ("primary_coverage_min_threshold", value.primary_coverage_min_threshold),
    ):
        if not math.isfinite(number) or number <= 0.0:
            raise ProductionSupportError(f"{name} 必须为有限正数")
    if not math.isclose(
        value.actual_mass / value.total_surface_area,
        value.actual_area_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ProductionSupportError("actual mass/fraction 不一致")
    if not math.isclose(
        value.target_mass / value.total_surface_area,
        value.target_area_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ProductionSupportError("target mass/fraction 不一致")
    if not value.primary_state_ids or any(
        status != "valid" for status in value.primary_statuses
    ):
        raise ProductionSupportError("生产 Support 要求全部 Primary states valid")
    primary_length = len(value.primary_state_ids)
    if not (
        len(value.primary_statuses)
        == len(value.primary_source_coverage)
        == len(value.primary_effective_coverage)
        == primary_length
    ):
        raise ProductionSupportError("Primary coverage arrays 长度不一致")
    if min(value.primary_effective_coverage) < value.primary_coverage_min_threshold:
        raise ProductionSupportError("冻结 Support 未通过 Primary coverage Gate")
    wrist_length = len(value.wrist_state_ids)
    if not (
        len(value.wrist_statuses)
        == len(value.wrist_source_coverage)
        == len(value.wrist_effective_coverage)
        == wrist_length
    ):
        raise ProductionSupportError("wrist coverage arrays 长度不一致")
    if value.naturalness_k_nonconstant != AKITA_NATURALNESS_K_NONCONSTANT:
        raise ProductionSupportError("Akita K_nat 必须固定为128个非恒定模态")
    if not value.production_support_constructed or not value.fixed_support_frozen:
        raise ProductionSupportError("artifact 未声明冻结 Production Support")
    if (
        value.rho_nat_calibrated
        or value.lambda_spec_calibrated
        or value.formal_training_allowed
    ):
        raise ProductionSupportError(
            "Support 冻结阶段不得提前声称谱校准或正式训练已放行"
        )


def write_production_support_artifact(
    path: str | Path,
    support: FrozenProductionSupport,
) -> str:
    """拒绝覆盖地写入冻结 NPZ，并返回文件 SHA-256。"""

    _validate_frozen_support(support)
    output_path = Path(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = support.provenance
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(support.schema_version),
        code_commit=np.asarray(provenance.code_commit),
        object_name=np.asarray(provenance.object_name),
        source_support_manifest_sha256=np.asarray(
            provenance.source_support_manifest_sha256
        ),
        source_candidate_artifact_sha256=np.asarray(
            provenance.source_candidate_artifact_sha256
        ),
        source_coverage_artifact_sha256=np.asarray(
            provenance.source_coverage_artifact_sha256
        ),
        seed_score_manifest_sha256=np.asarray(
            provenance.seed_score_manifest_sha256
        ),
        seed_score_artifact_sha256=np.asarray(
            provenance.seed_score_artifact_sha256
        ),
        visibility_manifest_sha256=np.asarray(
            provenance.visibility_manifest_sha256
        ),
        visibility_metrics_sha256=np.asarray(
            provenance.visibility_metrics_sha256
        ),
        mesh_file_sha256=np.asarray(provenance.mesh_file_sha256),
        mesh_array_sha256=np.asarray(provenance.mesh_array_sha256),
        renderer_faces_sha256=np.asarray(provenance.renderer_faces_sha256),
        render_to_geometry_sha256=np.asarray(
            provenance.render_to_geometry_sha256
        ),
        selected_candidate_index=np.asarray(
            support.selected_candidate_index, dtype=np.int64
        ),
        num_geometry_vertices=np.asarray(
            support.num_geometry_vertices, dtype=np.int64
        ),
        support_mask=support.support_mask,
        support_vertex_indices=support.support_vertex_indices,
        support_mask_sha256=np.asarray(support.support_mask_sha256),
        compact_coordinate_order=np.asarray(support.compact_coordinate_order),
        num_regions=np.asarray(support.num_regions, dtype=np.int64),
        seed_vertex_ids=np.asarray(support.seed_vertex_ids, dtype=np.int64),
        total_surface_area=np.asarray(support.total_surface_area, dtype=np.float64),
        target_area_fraction=np.asarray(
            support.target_area_fraction, dtype=np.float64
        ),
        target_mass=np.asarray(support.target_mass, dtype=np.float64),
        actual_mass=np.asarray(support.actual_mass, dtype=np.float64),
        actual_area_fraction=np.asarray(
            support.actual_area_fraction, dtype=np.float64
        ),
        primary_coverage_min_threshold=np.asarray(
            support.primary_coverage_min_threshold, dtype=np.float64
        ),
        primary_state_ids=np.asarray(support.primary_state_ids, dtype=np.int64),
        primary_statuses=np.asarray(support.primary_statuses, dtype=np.str_),
        primary_source_coverage=np.asarray(
            support.primary_source_coverage, dtype=np.float64
        ),
        primary_effective_coverage=np.asarray(
            support.primary_effective_coverage, dtype=np.float64
        ),
        wrist_state_ids=np.asarray(support.wrist_state_ids, dtype=np.int64),
        wrist_statuses=np.asarray(support.wrist_statuses, dtype=np.str_),
        wrist_source_coverage=np.asarray(
            support.wrist_source_coverage, dtype=np.float64
        ),
        wrist_effective_coverage=np.asarray(
            support.wrist_effective_coverage, dtype=np.float64
        ),
        naturalness_k_nonconstant=np.asarray(
            support.naturalness_k_nonconstant, dtype=np.int64
        ),
        production_support_constructed=np.asarray(True, dtype=np.bool_),
        fixed_support_frozen=np.asarray(True, dtype=np.bool_),
        rho_nat_calibrated=np.asarray(False, dtype=np.bool_),
        lambda_spec_calibrated=np.asarray(False, dtype=np.bool_),
        formal_training_allowed=np.asarray(False, dtype=np.bool_),
    )
    return file_sha256(output_path)


def load_production_support_artifact(
    path: str | Path,
    *,
    expected_mesh_file_sha256: Optional[str] = None,
    expected_mesh_array_sha256: Optional[str] = None,
    expected_render_to_geometry_sha256: Optional[str] = None,
) -> FrozenProductionSupport:
    """禁用 pickle 加载冻结 Support，并可绑定当前 renderer geometry。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != PRODUCTION_SUPPORT_ARTIFACT_KEYS:
            raise ProductionSupportError(
                "Production Support NPZ keys 与冻结 schema 不一致"
            )
        provenance = ProductionSupportProvenance(
            code_commit=str(_scalar(archive, "code_commit")),
            object_name=str(_scalar(archive, "object_name")),
            source_support_manifest_sha256=str(
                _scalar(archive, "source_support_manifest_sha256")
            ),
            source_candidate_artifact_sha256=str(
                _scalar(archive, "source_candidate_artifact_sha256")
            ),
            source_coverage_artifact_sha256=str(
                _scalar(archive, "source_coverage_artifact_sha256")
            ),
            seed_score_manifest_sha256=str(
                _scalar(archive, "seed_score_manifest_sha256")
            ),
            seed_score_artifact_sha256=str(
                _scalar(archive, "seed_score_artifact_sha256")
            ),
            visibility_manifest_sha256=str(
                _scalar(archive, "visibility_manifest_sha256")
            ),
            visibility_metrics_sha256=str(
                _scalar(archive, "visibility_metrics_sha256")
            ),
            mesh_file_sha256=str(_scalar(archive, "mesh_file_sha256")),
            mesh_array_sha256=str(_scalar(archive, "mesh_array_sha256")),
            renderer_faces_sha256=str(
                _scalar(archive, "renderer_faces_sha256")
            ),
            render_to_geometry_sha256=str(
                _scalar(archive, "render_to_geometry_sha256")
            ),
        )
        support = FrozenProductionSupport(
            schema_version=str(_scalar(archive, "schema_version")),
            provenance=provenance,
            selected_candidate_index=int(
                _scalar(archive, "selected_candidate_index")
            ),
            num_geometry_vertices=int(
                _scalar(archive, "num_geometry_vertices")
            ),
            support_mask=archive["support_mask"].copy(),
            support_vertex_indices=archive["support_vertex_indices"].copy(),
            support_mask_sha256=str(_scalar(archive, "support_mask_sha256")),
            compact_coordinate_order=str(
                _scalar(archive, "compact_coordinate_order")
            ),
            num_regions=int(_scalar(archive, "num_regions")),
            seed_vertex_ids=tuple(
                int(value) for value in archive["seed_vertex_ids"]
            ),
            total_surface_area=float(_scalar(archive, "total_surface_area")),
            target_area_fraction=float(
                _scalar(archive, "target_area_fraction")
            ),
            target_mass=float(_scalar(archive, "target_mass")),
            actual_mass=float(_scalar(archive, "actual_mass")),
            actual_area_fraction=float(
                _scalar(archive, "actual_area_fraction")
            ),
            primary_coverage_min_threshold=float(
                _scalar(archive, "primary_coverage_min_threshold")
            ),
            primary_state_ids=tuple(
                int(value) for value in archive["primary_state_ids"]
            ),
            primary_statuses=tuple(
                str(value) for value in archive["primary_statuses"]
            ),
            primary_source_coverage=tuple(
                float(value) for value in archive["primary_source_coverage"]
            ),
            primary_effective_coverage=tuple(
                float(value)
                for value in archive["primary_effective_coverage"]
            ),
            wrist_state_ids=tuple(
                int(value) for value in archive["wrist_state_ids"]
            ),
            wrist_statuses=tuple(
                str(value) for value in archive["wrist_statuses"]
            ),
            wrist_source_coverage=tuple(
                float(value) for value in archive["wrist_source_coverage"]
            ),
            wrist_effective_coverage=tuple(
                float(value) for value in archive["wrist_effective_coverage"]
            ),
            naturalness_k_nonconstant=int(
                _scalar(archive, "naturalness_k_nonconstant")
            ),
            production_support_constructed=bool(
                _scalar(archive, "production_support_constructed")
            ),
            fixed_support_frozen=bool(
                _scalar(archive, "fixed_support_frozen")
            ),
            rho_nat_calibrated=bool(_scalar(archive, "rho_nat_calibrated")),
            lambda_spec_calibrated=bool(
                _scalar(archive, "lambda_spec_calibrated")
            ),
            formal_training_allowed=bool(
                _scalar(archive, "formal_training_allowed")
            ),
        )
    _validate_frozen_support(support)
    for expected, actual, name in (
        (
            expected_mesh_file_sha256,
            support.provenance.mesh_file_sha256,
            "mesh file",
        ),
        (
            expected_mesh_array_sha256,
            support.provenance.mesh_array_sha256,
            "mesh array",
        ),
        (
            expected_render_to_geometry_sha256,
            support.provenance.render_to_geometry_sha256,
            "render_to_geometry",
        ),
    ):
        if expected is not None and expected != actual:
            raise ProductionSupportError(f"Production Support {name} SHA-256 不匹配")
    return support


def build_frozen_support_from_accepted_inputs(
    *,
    code_commit: str,
    object_name: str,
    support_manifest_path: str | Path,
    seed_score_manifest_path: str | Path,
    visibility_manifest_path: str | Path,
) -> tuple[FrozenProductionSupport, Path, Path]:
    """验证三个权威 manifest，并从 canonical selected candidate 精确冻结。"""

    support_manifest_file = Path(support_manifest_path)
    seed_manifest_file = Path(seed_score_manifest_path)
    visibility_manifest_file = Path(visibility_manifest_path)
    support_manifest = json.loads(
        support_manifest_file.read_text(encoding="utf-8")
    )
    seed_manifest = json.loads(seed_manifest_file.read_text(encoding="utf-8"))
    visibility_manifest = json.loads(
        visibility_manifest_file.read_text(encoding="utf-8")
    )
    if not support_manifest["decision"]["gate_pass"]:
        raise ProductionSupportError("source Support audit 未通过 artifact Gate")
    if support_manifest["fixed_support_frozen"] or support_manifest[
        "production_support_constructed"
    ]:
        raise ProductionSupportError("source Support audit 必须是未冻结 candidate")
    selected = support_manifest["selected_candidate_index"]
    if selected is None:
        raise ProductionSupportError("source Support audit 没有 selected candidate")
    selected_index = int(selected)
    candidate_summary = support_manifest["candidates"][selected_index]
    if not candidate_summary["primary_gate_pass"] or not candidate_summary[
        "construction_complete"
    ]:
        raise ProductionSupportError("selected candidate 未通过构造与 Primary Gate")

    support_root = support_manifest_file.parent
    candidate_path = support_root / support_manifest["evidence"][
        "candidate_artifact_relative_path"
    ]
    coverage_path = support_root / support_manifest["evidence"][
        "coverage_artifact_relative_path"
    ]
    for path, expected_sha in (
        (
            candidate_path,
            support_manifest["evidence"]["candidate_artifact_sha256"],
        ),
        (
            coverage_path,
            support_manifest["evidence"]["coverage_artifact_sha256"],
        ),
    ):
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ProductionSupportError(f"上游 artifact hash 不匹配: {path}")

    seed_root = seed_manifest_file.parent
    score_path = seed_root / seed_manifest["evidence"]["artifact_relative_path"]
    if not seed_manifest["decision"]["gate_pass"] or file_sha256(score_path) != (
        seed_manifest["evidence"]["artifact_sha256"]
    ):
        raise ProductionSupportError("Seed Score artifact 未通过或哈希不匹配")
    if file_sha256(score_path) != support_manifest["evidence"]["provenance"][
        "seed_score_artifact_sha256"
    ]:
        raise ProductionSupportError("Support audit 与 Seed Score artifact 绑定不一致")
    score_arrays: SeedScoreArrays = load_seed_score_artifact_arrays(score_path)

    if file_sha256(visibility_manifest_file) != support_manifest["evidence"][
        "provenance"
    ]["visibility_manifest_sha256"]:
        raise ProductionSupportError("Support audit 与 Visibility manifest 绑定不一致")
    visibility_metrics_path = (
        visibility_manifest_file.parent / "visibility_alignment_metrics.jsonl"
    )
    if file_sha256(visibility_metrics_path) != visibility_manifest[
        "metrics_jsonl_sha256"
    ]:
        raise ProductionSupportError("Visibility metrics SHA-256 不匹配")
    if object_name != visibility_manifest["metadata"]["object_name"]:
        raise ProductionSupportError("object_name 与 Visibility manifest 不一致")

    with np.load(candidate_path, allow_pickle=False) as candidate:
        missing_keys = SOURCE_CANDIDATE_REQUIRED_KEYS - set(candidate.files)
        if missing_keys:
            raise ProductionSupportError(
                f"source candidate NPZ 缺少冻结所需 keys: {sorted(missing_keys)}"
            )
        if bool(candidate["production_support_constructed"].item()) or bool(
            candidate["fixed_support_frozen"].item()
        ):
            raise ProductionSupportError("source candidate 必须仍是未冻结 audit")
        if selected_index != int(candidate["selected_candidate_index"].item()):
            raise ProductionSupportError("manifest/candidate selected index 不一致")
        if not bool(candidate["candidate_primary_gate_pass"][selected_index]):
            raise ProductionSupportError("source candidate NPZ 未通过 Primary Gate")
        support_mask = candidate["candidate_support_masks"][selected_index].copy()
        num_regions = int(candidate["candidate_region_counts"][selected_index])
        seeds = tuple(
            int(value)
            for value in candidate["candidate_seed_vertex_ids"][
                selected_index, :num_regions
            ]
        )
        primary_state_ids = tuple(
            int(value) for value in candidate["primary_state_ids"]
        )
        primary_statuses = tuple(
            str(value) for value in candidate["primary_statuses"]
        )
        primary_source = tuple(
            float(value)
            for value in candidate["primary_source_coverage"][selected_index]
        )
        primary_effective = tuple(
            float(value)
            for value in candidate["primary_effective_coverage"][selected_index]
        )
        wrist_state_ids = tuple(
            int(value) for value in candidate["wrist_state_ids"]
        )
        wrist_statuses = tuple(
            str(value) for value in candidate["wrist_statuses"]
        )
        wrist_source = tuple(
            float(value)
            for value in candidate["wrist_source_coverage"][selected_index]
        )
        wrist_effective = tuple(
            float(value)
            for value in candidate["wrist_effective_coverage"][selected_index]
        )
        total_surface_area = float(candidate["total_surface_area"].item())
        target_fraction = float(candidate["target_area_fraction"].item())
        target_mass = float(candidate["candidate_target_mass"][selected_index])
        actual_mass = float(candidate["candidate_actual_mass"][selected_index])
        actual_fraction = float(
            candidate["candidate_area_fraction_actual"][selected_index]
        )
        coverage_threshold = float(candidate["primary_coverage_min"].item())

    num_vertices = len(score_arrays.vertices)
    if support_mask.shape != (num_vertices,):
        raise ProductionSupportError("candidate support mask 与 score geometry 不一致")
    if support_manifest["input"]["mesh_file_sha256"] != seed_manifest[
        "evidence"
    ]["provenance"]["mesh_file_sha256"]:
        raise ProductionSupportError("上游 mesh file provenance 不一致")
    provenance = ProductionSupportProvenance(
        code_commit=code_commit,
        object_name=object_name,
        source_support_manifest_sha256=file_sha256(support_manifest_file),
        source_candidate_artifact_sha256=file_sha256(candidate_path),
        source_coverage_artifact_sha256=file_sha256(coverage_path),
        seed_score_manifest_sha256=file_sha256(seed_manifest_file),
        seed_score_artifact_sha256=file_sha256(score_path),
        visibility_manifest_sha256=file_sha256(visibility_manifest_file),
        visibility_metrics_sha256=file_sha256(visibility_metrics_path),
        mesh_file_sha256=str(
            seed_manifest["evidence"]["provenance"]["mesh_file_sha256"]
        ),
        mesh_array_sha256=str(
            seed_manifest["evidence"]["provenance"]["mesh_array_sha256"]
        ),
        renderer_faces_sha256=str(
            visibility_manifest["metadata"]["renderer_topology_sha256"][
                "faces"
            ]
        ),
        render_to_geometry_sha256=str(
            visibility_manifest["metadata"]["renderer_topology_sha256"][
                "render_to_geometry"
            ]
        ),
    )
    frozen = FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=provenance,
        selected_candidate_index=selected_index,
        num_geometry_vertices=num_vertices,
        support_mask=support_mask.astype(np.bool_),
        support_vertex_indices=np.flatnonzero(support_mask).astype(np.int64),
        support_mask_sha256=array_sha256(support_mask.astype(np.bool_)),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=num_regions,
        seed_vertex_ids=seeds,
        total_surface_area=total_surface_area,
        target_area_fraction=target_fraction,
        target_mass=target_mass,
        actual_mass=actual_mass,
        actual_area_fraction=actual_fraction,
        primary_coverage_min_threshold=coverage_threshold,
        primary_state_ids=primary_state_ids,
        primary_statuses=primary_statuses,
        primary_source_coverage=primary_source,
        primary_effective_coverage=primary_effective,
        wrist_state_ids=wrist_state_ids,
        wrist_statuses=wrist_statuses,
        wrist_source_coverage=wrist_source,
        wrist_effective_coverage=wrist_effective,
        naturalness_k_nonconstant=AKITA_NATURALNESS_K_NONCONSTANT,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    _validate_frozen_support(frozen)
    return frozen, candidate_path, score_path


def evaluate_production_support_artifact(
    evidence: ProductionSupportEvidence,
    *,
    artifact_root: str | Path,
    expected: FrozenProductionSupport,
) -> ProductionSupportDecision:
    """验证文件/hash并逐字段比较由上游重建的预期冻结对象。"""

    failures: list[str] = []
    if evidence.schema_version != PRODUCTION_SUPPORT_SCHEMA_VERSION:
        failures.append("Production Support evidence schema 不匹配")
    root = Path(artifact_root).resolve()
    path = (root / evidence.artifact_relative_path).resolve()
    if root not in path.parents or not path.is_file():
        return ProductionSupportDecision(False, ("Production Support path 非法",))
    if file_sha256(path) != evidence.artifact_sha256:
        return ProductionSupportDecision(
            False, ("Production Support artifact SHA-256 不匹配",)
        )
    try:
        actual = load_production_support_artifact(path)
        scalar_names = (
            "schema_version",
            "provenance",
            "selected_candidate_index",
            "num_geometry_vertices",
            "support_mask_sha256",
            "compact_coordinate_order",
            "num_regions",
            "seed_vertex_ids",
            "total_surface_area",
            "target_area_fraction",
            "target_mass",
            "actual_mass",
            "actual_area_fraction",
            "primary_coverage_min_threshold",
            "primary_state_ids",
            "primary_statuses",
            "primary_source_coverage",
            "primary_effective_coverage",
            "wrist_state_ids",
            "wrist_statuses",
            "wrist_source_coverage",
            "wrist_effective_coverage",
            "naturalness_k_nonconstant",
            "production_support_constructed",
            "fixed_support_frozen",
            "rho_nat_calibrated",
            "lambda_spec_calibrated",
            "formal_training_allowed",
        )
        for name in scalar_names:
            if getattr(actual, name) != getattr(expected, name):
                failures.append(f"Production Support {name} 与上游重建不一致")
        if not np.array_equal(actual.support_mask, expected.support_mask):
            failures.append("Production Support mask 与上游 candidate 不一致")
        if not np.array_equal(
            actual.support_vertex_indices,
            expected.support_vertex_indices,
        ):
            failures.append("Production Support compact indices 不一致")
    except (OSError, KeyError, TypeError, ValueError, ProductionSupportError) as error:
        failures.append(f"Production Support artifact 无法严格复核: {error}")
    expected_flags = (
        evidence.production_support_constructed,
        evidence.fixed_support_frozen,
        evidence.rho_nat_calibrated,
        evidence.lambda_spec_calibrated,
        evidence.formal_training_allowed,
    )
    if expected_flags != (True, True, False, False, False):
        failures.append("Production Support evidence 阶段 flags 非法")
    if evidence.support_mask_sha256 != expected.support_mask_sha256:
        failures.append("Production Support evidence semantic hash 不匹配")
    if evidence.num_geometry_vertices != expected.num_geometry_vertices:
        failures.append("Production Support evidence geometry vertex count 不匹配")
    if evidence.num_support_vertices != len(expected.support_vertex_indices):
        failures.append("Production Support evidence support vertex count 不匹配")
    if evidence.trainable_rgb_scalar_count != 3 * len(
        expected.support_vertex_indices
    ):
        failures.append("Production Support evidence RGB scalar count 不匹配")
    return ProductionSupportDecision(not failures, tuple(failures))
