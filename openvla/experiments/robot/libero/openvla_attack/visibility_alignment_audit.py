"""states 0–9 Visibility/Alignment audit 的权威行 schema 与结构判定。

本模块不创建 LIBERO、MuJoCo 或 renderer。真实 runner 负责采集并构造每个
``(state_id, view_name)`` 的一行；这里严格检查行集合、静止事务、Primary
alignment 与 ``A_obs`` 候选边界带。precision/IoU overlay 异常和 recall 是否与
0.95 清晰分离仍需人工审阅，因此结构通过不自动冻结候选门槛。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Optional, TypeAlias

from .visibility_evidence import (
    InstanceVisibilityEvidence,
    SoftAlignmentMetrics,
    VisibilityEvidenceStatus,
    VisibilityThresholdCandidates,
)
from .visibility_view_evidence import VisibilityObservationStages


VISIBILITY_ALIGNMENT_SCHEMA_VERSION: Final[str] = (
    "openvla-visibility-alignment-v1"
)
A_OBS_BOUNDARY_LOW: Final[float] = 0.8e-3
A_OBS_BOUNDARY_HIGH: Final[float] = 1.2e-3
VisibilityAuditViewName: TypeAlias = Literal[
    "primary",
    "wrist_source_crop_proxy",
]


@dataclass(frozen=True)
class VisibilityAuditTransactionEvidence:
    """一行关联的同 state 静止事务证据。"""

    verified: bool
    before_fingerprint_sha256: str
    after_fingerprint_sha256: str
    maximum_position_delta: float
    maximum_quaternion_delta: float


@dataclass(frozen=True)
class VisibilityAlignmentAuditRow:
    """权威 JSONL 中一个 ``(state, view)`` 的完整数值索引。"""

    schema_version: str
    code_commit: str
    state_id: int
    view_name: VisibilityAuditViewName
    status: VisibilityEvidenceStatus
    thresholds: VisibilityThresholdCandidates
    observations: VisibilityObservationStages
    union_alignment: Optional[SoftAlignmentMetrics]
    instances: tuple[InstanceVisibilityEvidence, ...]
    instance_body_ids: tuple[int, ...]
    instance_body_names: tuple[str, ...]
    transaction: VisibilityAuditTransactionEvidence
    initial_state_sha256: str
    arrays_npz_sha256: str
    oriented_segmentation_sha256: str
    renderer_faces_sha256: str
    render_to_geometry_sha256: str
    mesh_sha256: str
    backend: Mapping[str, Optional[str]]
    artifact_sha256: Mapping[str, str]


@dataclass(frozen=True)
class VisibilityAlignmentAuditSummary:
    """结构性验收结果；不代表候选门槛已经人工冻结。"""

    structural_pass: bool
    failures: tuple[str, ...]
    row_count: int
    status_counts_by_view: Mapping[str, Mapping[str, int]]
    primary_invalid_state_ids: tuple[int, ...]
    wrist_invalid_state_ids: tuple[int, ...]
    a_obs_boundary_cases: tuple[tuple[int, str], ...]
    primary_valid_state_ids: tuple[int, ...]
    threshold_freeze_requires_manual_review: bool = True


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def summarize_visibility_alignment_rows(
    rows: list[VisibilityAlignmentAuditRow],
    *,
    expected_state_ids: tuple[int, ...],
) -> VisibilityAlignmentAuditSummary:
    """检查完整二维行集合与预注册的结构性失败条件。"""

    failures: list[str] = []
    if not expected_state_ids or len(set(expected_state_ids)) != len(
        expected_state_ids
    ):
        raise ValueError("expected_state_ids 必须非空且唯一")
    expected_keys = {
        (state_id, view_name)
        for state_id in expected_state_ids
        for view_name in ("primary", "wrist_source_crop_proxy")
    }
    actual_keys = [(row.state_id, row.view_name) for row in rows]
    if len(actual_keys) != len(set(actual_keys)):
        failures.append("duplicate_state_view_rows")
    missing_keys = sorted(expected_keys - set(actual_keys))
    extra_keys = sorted(set(actual_keys) - expected_keys)
    if missing_keys:
        failures.append(f"missing_rows={missing_keys}")
    if extra_keys:
        failures.append(f"extra_rows={extra_keys}")

    code_commits = {row.code_commit for row in rows}
    if len(code_commits) != 1 or any(
        not _is_sha256(commit) for commit in code_commits
    ):
        failures.append("invalid_or_mixed_code_commit")
    if any(
        row.schema_version != VISIBILITY_ALIGNMENT_SCHEMA_VERSION
        for row in rows
    ):
        failures.append("schema_version_mismatch")
    if any(not row.transaction.verified for row in rows):
        failures.append("static_transaction_failed")
    if any(
        not _is_sha256(hash_value)
        for row in rows
        for hash_value in (
            row.transaction.before_fingerprint_sha256,
            row.transaction.after_fingerprint_sha256,
            row.initial_state_sha256,
            row.arrays_npz_sha256,
            row.oriented_segmentation_sha256,
            row.renderer_faces_sha256,
            row.render_to_geometry_sha256,
            row.mesh_sha256,
        )
    ):
        failures.append("invalid_required_sha256")
    if any(
        len(row.instance_body_ids) != len(row.instances)
        or len(row.instance_body_names) != len(row.instances)
        for row in rows
    ):
        failures.append("instance_metadata_count_mismatch")
    if len({row.thresholds for row in rows}) != 1:
        failures.append("mixed_visibility_threshold_candidates")
    if any(
        not row.artifact_sha256
        or any(
            not _is_sha256(hash_value)
            for hash_value in row.artifact_sha256.values()
        )
        for row in rows
    ):
        failures.append("invalid_artifact_sha256")

    primary_invalid = tuple(
        sorted(
            row.state_id
            for row in rows
            if row.view_name == "primary"
            and row.status == "invalid_alignment"
        )
    )
    wrist_invalid = tuple(
        sorted(
            row.state_id
            for row in rows
            if row.view_name == "wrist_source_crop_proxy"
            and row.status == "invalid_alignment"
        )
    )
    if primary_invalid:
        failures.append(f"primary_invalid_alignment={list(primary_invalid)}")

    boundary_cases = tuple(
        sorted(
            (
                row.state_id,
                row.view_name,
            )
            for row in rows
            if A_OBS_BOUNDARY_LOW
            <= row.observations.effective.observation_area
            <= A_OBS_BOUNDARY_HIGH
        )
    )
    if boundary_cases:
        failures.append(f"a_obs_boundary_cases={list(boundary_cases)}")
    primary_valid = tuple(
        sorted(
            row.state_id
            for row in rows
            if row.view_name == "primary" and row.status == "valid"
        )
    )
    if not primary_valid:
        failures.append("no_valid_primary_state")

    status_counts: dict[str, dict[str, int]] = {
        "primary": {},
        "wrist_source_crop_proxy": {},
    }
    for row in rows:
        view_counts = status_counts[row.view_name]
        view_counts[row.status] = view_counts.get(row.status, 0) + 1
    return VisibilityAlignmentAuditSummary(
        structural_pass=not failures,
        failures=tuple(failures),
        row_count=len(rows),
        status_counts_by_view=status_counts,
        primary_invalid_state_ids=primary_invalid,
        wrist_invalid_state_ids=wrist_invalid,
        a_obs_boundary_cases=boundary_cases,
        primary_valid_state_ids=primary_valid,
    )


def write_visibility_alignment_jsonl(
    rows: list[VisibilityAlignmentAuditRow],
    *,
    output_path: Path,
) -> str:
    """按 state/view 确定顺序写权威 JSONL，并返回文件 SHA-256。"""

    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 visibility JSONL: {output_path}")
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.state_id,
            0 if row.view_name == "primary" else 1,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(
                json.dumps(
                    asdict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def write_visibility_alignment_manifest(
    *,
    summary: VisibilityAlignmentAuditSummary,
    metadata: Mapping[str, Any],
    metrics_jsonl_sha256: str,
    output_path: Path,
) -> str:
    """写只索引权威 JSONL 的 manifest；不会据此自动冻结门槛。"""

    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 visibility manifest: {output_path}")
    payload = {
        "schema_version": VISIBILITY_ALIGNMENT_SCHEMA_VERSION,
        "metrics_jsonl_sha256": metrics_jsonl_sha256,
        "summary": asdict(summary),
        "metadata": dict(metadata),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(output_path.read_bytes()).hexdigest()
