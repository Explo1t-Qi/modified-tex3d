"""Untargeted Clean-Action Objective GPU Audit 的证据契约。

本模块只定义纯 CPU、可独立重算的 JSONL schema 与 Gate 判定，不导入
LIBERO、OpenVLA 模型或 CUDA。真实 runner 必须在零 Surface Delta 的全几何
顶点 RGB 参数空间中，对 states 0--9 分别运行固定 clean sequence 的
teacher-forced forward/backward，并把逐 token margin 与五级梯度摘要写入这里。

负 margin 只因本 Gate 的参考点严格为零扰动才表示输入或 causal alignment
失败；正式攻击训练中的负 margin 是合法的 hinge 非激活状态。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .deployment_backward_audit import GradientEvidence
from .objective import ACTION_TOKEN_START, NUM_ACTION_BINS


ACTION_OBJECTIVE_SCHEMA_VERSION: Final[str] = (
    "openvla-action-objective-audit-v1"
)
ACTION_OBJECTIVE_NAME: Final[str] = (
    "untargeted_clean_action_margin_hinge"
)
EXPECTED_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10))
NUMERIC_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class ActionObjectiveAuditEvidence:
    """一个 source OpenVLA state 的零扰动 objective/backward 证据。"""

    schema_version: str
    code_commit: str
    state_id: int
    initial_state_sha256: str
    objective: str
    parameterization: str
    zero_surface_delta_linf: float
    teacher_forced_clean_prefix: bool
    clean_action_token_ids: list[int]
    clean_classes: list[int]
    margins: list[float]
    hinge_values: list[float]
    action_loss: float
    parameter_shape: tuple[int, ...]
    source_rgb_gradient: GradientEvidence
    pre_crop_gradient: GradientEvidence
    effective_view_gradient: GradientEvidence
    render_surface_delta_gradient: GradientEvidence
    dense_geometry_gradient: GradientEvidence
    dense_geometry_gradient_sha256: str


@dataclass(frozen=True)
class ActionObjectiveAuditDecision:
    """单 state Gate 判定及精确 tie/负 margin 位置。"""

    gate_pass: bool
    failures: tuple[str, ...]
    tie_indices: tuple[int, ...]
    negative_margin_indices: tuple[int, ...]


@dataclass(frozen=True)
class ActionObjectiveAuditSummary:
    """states 0--9 完整集合的 Gate 汇总。"""

    gate_pass: bool
    failures: tuple[str, ...]
    state_ids: tuple[int, ...]
    valid_state_count: int
    action_token_count: int
    tie_locations: tuple[tuple[int, int], ...]
    negative_margin_locations: tuple[tuple[int, int], ...]
    margin_mean: float
    margin_min: float
    margin_max: float
    action_loss_mean: float
    dense_gradient_l2_min: float
    dense_gradient_l2_max: float


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_commit_sha(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def evaluate_action_objective_evidence(
    evidence: ActionObjectiveAuditEvidence,
) -> ActionObjectiveAuditDecision:
    """重算单 state Gate，不相信 runner 预先给出的 PASS 文本。"""

    failures: list[str] = []
    if evidence.schema_version != ACTION_OBJECTIVE_SCHEMA_VERSION:
        failures.append("schema_version 不匹配")
    if not _is_commit_sha(evidence.code_commit):
        failures.append("code_commit 不是40位小写十六进制 SHA")
    if not _is_sha256(evidence.initial_state_sha256):
        failures.append("initial_state_sha256 非法")
    if evidence.objective != ACTION_OBJECTIVE_NAME:
        failures.append("objective 不是冻结的 clean-action margin hinge")
    if evidence.parameterization != "geometry_vertex_dense":
        failures.append("Objective Audit 必须使用 dense geometry vertex 参数化")
    if not (
        np.isfinite(evidence.zero_surface_delta_linf)
        and 0.0 <= evidence.zero_surface_delta_linf <= NUMERIC_TOLERANCE
    ):
        failures.append("参考点不是零 Surface Delta")
    if not evidence.teacher_forced_clean_prefix:
        failures.append("未使用固定 teacher-forced clean prefix")

    action_count = len(evidence.clean_action_token_ids)
    vector_lengths = {
        action_count,
        len(evidence.clean_classes),
        len(evidence.margins),
        len(evidence.hinge_values),
    }
    if action_count == 0:
        failures.append("Action token 证据为空")
    if len(vector_lengths) != 1:
        failures.append("逐 token 证据长度不一致")

    margins = np.asarray(evidence.margins, dtype=np.float64)
    hinge_values = np.asarray(evidence.hinge_values, dtype=np.float64)
    tie_indices: tuple[int, ...] = ()
    negative_indices: tuple[int, ...] = ()
    if margins.size > 0:
        if not np.all(np.isfinite(margins)):
            failures.append("Action margins 包含 NaN/Inf")
        else:
            tie_indices = tuple(
                int(index) for index in np.flatnonzero(margins == 0.0)
            )
            negative_indices = tuple(
                int(index) for index in np.flatnonzero(margins < 0.0)
            )
            if negative_indices:
                failures.append("零 Surface Delta 出现负 clean-action margin")
    if hinge_values.size > 0 and not np.all(np.isfinite(hinge_values)):
        failures.append("Action hinge values 包含 NaN/Inf")
    if (
        margins.shape == hinge_values.shape
        and margins.size > 0
        and np.all(np.isfinite(margins))
        and np.all(np.isfinite(hinge_values))
        and not np.allclose(
            hinge_values,
            np.maximum(margins, 0.0),
            rtol=0.0,
            atol=NUMERIC_TOLERANCE,
        )
    ):
        failures.append("hinge_values 与 relu(margins) 不一致")
    if not np.isfinite(evidence.action_loss):
        failures.append("Action loss 不是有限值")
    elif hinge_values.size > 0 and np.all(np.isfinite(hinge_values)) and not (
        abs(float(hinge_values.mean()) - evidence.action_loss)
        <= NUMERIC_TOLERANCE
    ):
        failures.append("Action loss 不等于逐 token hinge 均值")

    if action_count > 0 and len(vector_lengths) == 1:
        classes = np.asarray(evidence.clean_classes, dtype=np.int64)
        token_ids = np.asarray(
            evidence.clean_action_token_ids,
            dtype=np.int64,
        )
        if np.any(classes < 0) or np.any(classes >= NUM_ACTION_BINS):
            failures.append("clean_classes 存在越界类别")
        elif not np.array_equal(token_ids, classes + ACTION_TOKEN_START):
            failures.append("clean token IDs 与 action classes 不一致")

    if (
        len(evidence.parameter_shape) != 2
        or evidence.parameter_shape[0] <= 0
        or evidence.parameter_shape[1] != 3
    ):
        failures.append("Dense Geometry 参数 shape 不是非空 [N_v,3]")

    gradients = {
        "Policy Source": evidence.source_rgb_gradient,
        "Pre-Crop": evidence.pre_crop_gradient,
        "Effective View": evidence.effective_view_gradient,
        "Render Surface Delta": evidence.render_surface_delta_gradient,
        "Dense Geometry": evidence.dense_geometry_gradient,
    }
    for stage_name, gradient in gradients.items():
        if not gradient.is_finite_nonzero():
            failures.append(f"{stage_name} 梯度不是有限非零")
    if evidence.dense_geometry_gradient.shape != evidence.parameter_shape:
        failures.append("Dense Geometry 梯度 shape 与参数不一致")
    if not _is_sha256(evidence.dense_geometry_gradient_sha256):
        failures.append("Dense Geometry 梯度 SHA-256 非法")

    return ActionObjectiveAuditDecision(
        gate_pass=not failures,
        failures=tuple(failures),
        tie_indices=tie_indices,
        negative_margin_indices=negative_indices,
    )


def summarize_action_objective_evidence(
    evidence_rows: Sequence[ActionObjectiveAuditEvidence],
) -> ActionObjectiveAuditSummary:
    """严格要求恰好覆盖 states 0--9，并汇总逐 state 决策。"""

    rows = list(evidence_rows)
    state_ids = tuple(row.state_id for row in rows)
    failures: list[str] = []
    if state_ids != EXPECTED_STATE_IDS:
        failures.append("state IDs 必须精确等于 0-9")

    decisions = [evaluate_action_objective_evidence(row) for row in rows]
    for row, decision in zip(rows, decisions):
        failures.extend(
            f"state {row.state_id}: {failure}"
            for failure in decision.failures
        )

    tie_locations = tuple(
        (row.state_id, token_index)
        for row, decision in zip(rows, decisions)
        for token_index in decision.tie_indices
    )
    negative_locations = tuple(
        (row.state_id, token_index)
        for row, decision in zip(rows, decisions)
        for token_index in decision.negative_margin_indices
    )
    all_margins = np.asarray(
        [margin for row in rows for margin in row.margins],
        dtype=np.float64,
    )
    action_losses = np.asarray(
        [row.action_loss for row in rows],
        dtype=np.float64,
    )
    gradient_norms = np.asarray(
        [row.dense_geometry_gradient.l2_norm for row in rows],
        dtype=np.float64,
    )

    def finite_stat(values: np.ndarray, kind: str) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        if kind == "mean":
            return float(finite.mean())
        if kind == "min":
            return float(finite.min())
        return float(finite.max())

    return ActionObjectiveAuditSummary(
        gate_pass=not failures,
        failures=tuple(failures),
        state_ids=state_ids,
        valid_state_count=sum(decision.gate_pass for decision in decisions),
        action_token_count=int(all_margins.size),
        tie_locations=tie_locations,
        negative_margin_locations=negative_locations,
        margin_mean=finite_stat(all_margins, "mean"),
        margin_min=finite_stat(all_margins, "min"),
        margin_max=finite_stat(all_margins, "max"),
        action_loss_mean=finite_stat(action_losses, "mean"),
        dense_gradient_l2_min=finite_stat(gradient_norms, "min"),
        dense_gradient_l2_max=finite_stat(gradient_norms, "max"),
    )


def write_action_objective_evidence(
    path: str | Path,
    evidence_rows: Sequence[ActionObjectiveAuditEvidence],
) -> Path:
    """写入逐 state JSONL；每行同时保存可独立重算的 decision。"""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("x", encoding="utf-8") as handle:
        for evidence in evidence_rows:
            payload = {
                "evidence": asdict(evidence),
                "decision": asdict(
                    evaluate_action_objective_evidence(evidence)
                ),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return resolved_path


def _gradient_from_mapping(value: Mapping[str, Any]) -> GradientEvidence:
    return GradientEvidence(
        shape=tuple(int(item) for item in value["shape"]),
        numel=int(value["numel"]),
        finite_count=int(value["finite_count"]),
        nonzero_count=int(value["nonzero_count"]),
        l2_norm=float(value["l2_norm"]),
        linf_norm=float(value["linf_norm"]),
    )


def load_action_objective_evidence(
    path: str | Path,
) -> list[ActionObjectiveAuditEvidence]:
    """严格读取 runner JSONL，缺字段或字段类型错误时直接失败。"""

    rows: list[ActionObjectiveAuditEvidence] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        value: Mapping[str, Any] = payload["evidence"]
        resolved = dict(value)
        for field_name in (
            "source_rgb_gradient",
            "pre_crop_gradient",
            "effective_view_gradient",
            "render_surface_delta_gradient",
            "dense_geometry_gradient",
        ):
            resolved[field_name] = _gradient_from_mapping(value[field_name])
        resolved["parameter_shape"] = tuple(value["parameter_shape"])
        rows.append(ActionObjectiveAuditEvidence(**resolved))
    return rows
