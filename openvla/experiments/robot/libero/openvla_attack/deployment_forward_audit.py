"""Gate 1D 完整 deployment-path forward equivalence 的纯判定层。

真实 LIBERO/OpenVLA runner 只负责为每个原始 state 采集两条路径的 RGB、
processor tensor、action token 与连续 action。本模块不加载模型或环境，只做
严格 shape/dtype 校验、逐 state 零误差判定、完整性汇总和权威 JSONL 保存。

Gate 1D 是等价性 Gate，不允许近似容差：Policy Pre-Crop Canvas、Effective
View、checkpoint fused pixel values、全部 action token 和最终连续 action 任一
不一致都使该 state 失败。浮点 MAE 只用于定位，不能替代 L∞=0 的硬条件。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray


DEPLOYMENT_FORWARD_SCHEMA_VERSION = "openvla-gate-1d-v1"


class DeploymentForwardAuditError(RuntimeError):
    """采集证据 shape、dtype 或数值不满足 Gate 1D schema。"""


@dataclass(frozen=True)
class DeploymentForwardEvidence:
    """一个原始 LIBERO state 的 rollout/candidate 完整前向证据。"""

    state_id: int
    source_rgb: NDArray[np.uint8]
    exact_pre_crop_rgb: NDArray[np.uint8]
    candidate_pre_crop_rgb: NDArray[np.uint8]
    rollout_effective_rgb: NDArray[np.uint8]
    candidate_effective_rgb: NDArray[np.uint8]
    rollout_pixel_values: NDArray[np.floating]
    candidate_pixel_values: NDArray[np.floating]
    rollout_action_token_ids: NDArray[np.integer]
    candidate_action_token_ids: NDArray[np.integer]
    rollout_action: NDArray[np.floating]
    candidate_action: NDArray[np.floating]


class DeploymentForwardRow(TypedDict):
    """权威 JSONL 中一个 state 的固定 schema。"""

    schema_version: str
    code_commit: str
    state_id: int
    source_rgb_shape: list[int]
    pre_crop_rgb_shape: list[int]
    effective_view_rgb_shape: list[int]
    processor_pixel_shape: list[int]
    action_dim: int
    source_rgb_sha256: str
    exact_pre_crop_rgb_sha256: str
    candidate_pre_crop_rgb_sha256: str
    rollout_effective_rgb_sha256: str
    candidate_effective_rgb_sha256: str
    pre_crop_mae: float
    pre_crop_linf: float
    effective_view_mae: float
    effective_view_linf: float
    processor_pixel_mae: float
    processor_pixel_linf: float
    rollout_action_token_ids: list[int]
    candidate_action_token_ids: list[int]
    action_token_hamming: int
    first_action_token_difference: int | None
    rollout_action: list[float]
    candidate_action: list[float]
    action_mae: float
    action_linf: float
    gate_pass: bool


class DeploymentForwardSummary(TypedDict):
    """全 state 完整性与最坏误差汇总。"""

    schema_version: str
    expected_state_ids: list[int]
    observed_state_ids: list[int]
    missing_state_ids: list[int]
    unexpected_state_ids: list[int]
    duplicate_state_ids: list[int]
    code_commits: list[str]
    row_count: int
    passed_state_count: int
    total_action_token_count: int
    matching_action_token_count: int
    worst_pre_crop_linf: float
    worst_effective_view_linf: float
    worst_processor_pixel_linf: float
    worst_action_linf: float
    gate_pass: bool


def _validate_rgb(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise DeploymentForwardAuditError(f"{name} 必须是 numpy array")
    if value.dtype != np.uint8:
        raise DeploymentForwardAuditError(
            f"{name} 必须为 uint8，收到 {value.dtype}"
        )
    if value.ndim != 3 or value.shape[2] != 3:
        raise DeploymentForwardAuditError(
            f"{name} 必须为 HWC RGB，收到 {value.shape}"
        )


def _validate_float_pair(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    if not isinstance(reference, np.ndarray) or not isinstance(
        candidate,
        np.ndarray,
    ):
        raise DeploymentForwardAuditError(f"{name} 必须是 numpy array")
    if reference.shape != candidate.shape:
        raise DeploymentForwardAuditError(
            f"{name} shape 不一致：{reference.shape} != {candidate.shape}"
        )
    if not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(
        candidate.dtype,
        np.floating,
    ):
        raise DeploymentForwardAuditError(f"{name} 必须为浮点数组")
    if not np.all(np.isfinite(reference)) or not np.all(
        np.isfinite(candidate)
    ):
        raise DeploymentForwardAuditError(f"{name} 包含 NaN/Inf")


def _error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise DeploymentForwardAuditError(
            "等价性数组 shape 不一致："
            f"{reference.shape} != {candidate.shape}"
        )
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    if difference.size == 0:
        raise DeploymentForwardAuditError("等价性数组不得为空")
    return float(np.mean(np.abs(difference))), float(np.max(np.abs(difference)))


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def evaluate_deployment_forward_evidence(
    evidence: DeploymentForwardEvidence,
    *,
    code_commit: str,
) -> DeploymentForwardRow:
    """校验并判定一个 state；硬 Gate 只接受所有关键数组逐值一致。"""

    if evidence.state_id < 0:
        raise DeploymentForwardAuditError("state_id 必须为非负整数")
    if not code_commit.strip():
        raise DeploymentForwardAuditError("code_commit 不得为空")
    for name, value in (
        ("source_rgb", evidence.source_rgb),
        ("exact_pre_crop_rgb", evidence.exact_pre_crop_rgb),
        ("candidate_pre_crop_rgb", evidence.candidate_pre_crop_rgb),
        ("rollout_effective_rgb", evidence.rollout_effective_rgb),
        ("candidate_effective_rgb", evidence.candidate_effective_rgb),
    ):
        _validate_rgb(name, value)
    if evidence.exact_pre_crop_rgb.shape != (
        evidence.candidate_pre_crop_rgb.shape
    ):
        raise DeploymentForwardAuditError("pre-crop RGB shape 不一致")
    if evidence.rollout_effective_rgb.shape != (
        evidence.candidate_effective_rgb.shape
    ):
        raise DeploymentForwardAuditError("effective-view RGB shape 不一致")
    _validate_float_pair(
        "processor pixel values",
        evidence.rollout_pixel_values,
        evidence.candidate_pixel_values,
    )
    _validate_float_pair(
        "continuous action",
        evidence.rollout_action,
        evidence.candidate_action,
    )
    rollout_tokens = np.asarray(evidence.rollout_action_token_ids)
    candidate_tokens = np.asarray(evidence.candidate_action_token_ids)
    if rollout_tokens.ndim != 1 or rollout_tokens.size == 0:
        raise DeploymentForwardAuditError("rollout action tokens 必须为非空一维数组")
    if rollout_tokens.shape != candidate_tokens.shape:
        raise DeploymentForwardAuditError("candidate action token shape 不一致")
    if not np.issubdtype(rollout_tokens.dtype, np.integer) or not np.issubdtype(
        candidate_tokens.dtype,
        np.integer,
    ):
        raise DeploymentForwardAuditError("action token IDs 必须为整数")
    if evidence.rollout_action.shape != (rollout_tokens.size,):
        raise DeploymentForwardAuditError(
            "连续 action shape 必须等于 action token 数"
        )

    pre_crop_mae, pre_crop_linf = _error_metrics(
        evidence.exact_pre_crop_rgb,
        evidence.candidate_pre_crop_rgb,
    )
    effective_mae, effective_linf = _error_metrics(
        evidence.rollout_effective_rgb,
        evidence.candidate_effective_rgb,
    )
    processor_mae, processor_linf = _error_metrics(
        evidence.rollout_pixel_values,
        evidence.candidate_pixel_values,
    )
    action_mae, action_linf = _error_metrics(
        evidence.rollout_action,
        evidence.candidate_action,
    )
    token_differences = np.flatnonzero(
        rollout_tokens != candidate_tokens
    )
    action_token_hamming = int(token_differences.size)
    first_token_difference = (
        int(token_differences[0]) if token_differences.size > 0 else None
    )
    numeric_values = (
        pre_crop_mae,
        pre_crop_linf,
        effective_mae,
        effective_linf,
        processor_mae,
        processor_linf,
        action_mae,
        action_linf,
    )
    gate_pass = bool(
        all(math.isfinite(value) and value == 0.0 for value in numeric_values)
        and action_token_hamming == 0
    )
    return {
        "schema_version": DEPLOYMENT_FORWARD_SCHEMA_VERSION,
        "code_commit": code_commit,
        "state_id": evidence.state_id,
        "source_rgb_shape": list(evidence.source_rgb.shape),
        "pre_crop_rgb_shape": list(evidence.exact_pre_crop_rgb.shape),
        "effective_view_rgb_shape": list(
            evidence.rollout_effective_rgb.shape
        ),
        "processor_pixel_shape": list(evidence.rollout_pixel_values.shape),
        "action_dim": int(rollout_tokens.size),
        "source_rgb_sha256": _array_sha256(evidence.source_rgb),
        "exact_pre_crop_rgb_sha256": _array_sha256(
            evidence.exact_pre_crop_rgb
        ),
        "candidate_pre_crop_rgb_sha256": _array_sha256(
            evidence.candidate_pre_crop_rgb
        ),
        "rollout_effective_rgb_sha256": _array_sha256(
            evidence.rollout_effective_rgb
        ),
        "candidate_effective_rgb_sha256": _array_sha256(
            evidence.candidate_effective_rgb
        ),
        "pre_crop_mae": pre_crop_mae,
        "pre_crop_linf": pre_crop_linf,
        "effective_view_mae": effective_mae,
        "effective_view_linf": effective_linf,
        "processor_pixel_mae": processor_mae,
        "processor_pixel_linf": processor_linf,
        "rollout_action_token_ids": rollout_tokens.astype(np.int64).tolist(),
        "candidate_action_token_ids": candidate_tokens.astype(
            np.int64
        ).tolist(),
        "action_token_hamming": action_token_hamming,
        "first_action_token_difference": first_token_difference,
        "rollout_action": evidence.rollout_action.astype(np.float64).tolist(),
        "candidate_action": evidence.candidate_action.astype(
            np.float64
        ).tolist(),
        "action_mae": action_mae,
        "action_linf": action_linf,
        "gate_pass": gate_pass,
    }


def summarize_deployment_forward_rows(
    rows: Sequence[DeploymentForwardRow],
    *,
    expected_state_ids: Sequence[int],
) -> DeploymentForwardSummary:
    """汇总完整性与最坏误差；缺行、重复、额外 state 均失败。"""

    expected = [int(state_id) for state_id in expected_state_ids]
    if len(expected) != len(set(expected)) or any(
        state_id < 0 for state_id in expected
    ):
        raise DeploymentForwardAuditError(
            "expected_state_ids 必须非负且不能重复"
        )
    observed = [int(row["state_id"]) for row in rows]
    counts = Counter(observed)
    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    duplicate = sorted(
        state_id for state_id, count in counts.items() if count > 1
    )
    commits = sorted({str(row["code_commit"]) for row in rows})
    schemas = {str(row["schema_version"]) for row in rows}
    rows_by_state: dict[int, list[DeploymentForwardRow]] = {}
    for row in rows:
        rows_by_state.setdefault(int(row["state_id"]), []).append(row)
    passed_state_count = sum(
        len(rows_by_state.get(state_id, ())) == 1
        and rows_by_state[state_id][0]["gate_pass"]
        for state_id in expected
    )
    total_action_tokens = sum(int(row["action_dim"]) for row in rows)
    matching_action_tokens = sum(
        int(row["action_dim"]) - int(row["action_token_hamming"])
        for row in rows
    )

    def maximum(field_name: str) -> float:
        return max((float(row[field_name]) for row in rows), default=0.0)

    gate_pass = bool(
        len(rows) == len(expected)
        and not missing
        and not unexpected
        and not duplicate
        and len(commits) == 1
        and schemas == {DEPLOYMENT_FORWARD_SCHEMA_VERSION}
        and passed_state_count == len(expected)
    )
    return {
        "schema_version": DEPLOYMENT_FORWARD_SCHEMA_VERSION,
        "expected_state_ids": expected,
        "observed_state_ids": observed,
        "missing_state_ids": missing,
        "unexpected_state_ids": unexpected,
        "duplicate_state_ids": duplicate,
        "code_commits": commits,
        "row_count": len(rows),
        "passed_state_count": passed_state_count,
        "total_action_token_count": total_action_tokens,
        "matching_action_token_count": matching_action_tokens,
        "worst_pre_crop_linf": maximum("pre_crop_linf"),
        "worst_effective_view_linf": maximum("effective_view_linf"),
        "worst_processor_pixel_linf": maximum("processor_pixel_linf"),
        "worst_action_linf": maximum("action_linf"),
        "gate_pass": gate_pass,
    }


def write_deployment_forward_jsonl(
    rows: Sequence[DeploymentForwardRow],
    *,
    output_path: Path,
) -> str:
    """按采集顺序保存权威 JSONL，并返回带换行 payload 的 SHA-256。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
