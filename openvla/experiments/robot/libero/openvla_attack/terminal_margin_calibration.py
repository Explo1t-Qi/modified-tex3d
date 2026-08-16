r"""Action-only κ intervention 前的纯 CPU margin/drift evidence 报告。

本模块复用已经通过 Gate 6g 的 Action-only states 0--9 C/A/B artifact。
clean class ``y`` 固定来自同一state的Clean cached generation；A（training
Renderer Delta Composition）与B（MuJoCo Active Texture）使用同一个clean
teacher-forced prefix，两者模型调用只改变Effective RGB。逐token计算：

.. math::

   m_A=z_A[y]-\max_{j\ne y}z_A[j],\qquad
   m_B=z_B[y]-\max_{j\ne y}z_B[j],\qquad
   \Delta m=m_B-m_A.

``m_A <= 0``精确定义为training路径已经越过或到达clean-token决策边界，
``Delta m > 0``表示deployment把clean token向重新占优方向推回。本模块只报告
完整分布和身份绑定，不输出feasibility pass/fail或推荐κ；κ的统计总体、规则和
数值必须在读取报告后另行讨论并预注册。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .terminal_deployment_response_audit import (
    TerminalDeploymentResponseEvidence,
    evaluate_terminal_response_bundle,
    load_terminal_response_npz,
)
from .terminal_endpoint_action_audit import compute_action_hinge
from .terminal_endpoint_action_evidence import array_sha256, file_sha256


Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]
MARGIN_CALIBRATION_SCHEMA_VERSION: Final[str] = (
    "openvla-action-only-margin-drift-report-v1"
)
ACTION_ONLY_VARIANT: Final[str] = "action_only_control"


class TerminalMarginCalibrationError(ValueError):
    """Gate 6g输入、margin inventory或输出违反κ校准证据合同。"""


@dataclass(frozen=True)
class MarginDriftCase:
    """一个state的共享clean target与A/B teacher logits。"""

    state_id: int
    state_fingerprint: str
    response_artifact_sha256: str
    # int64 [action_dim]，来自Clean cached generation的action-vocabulary class。
    clean_classes: Int64Array
    teacher_input_ids_sha256: str
    training_effective_rgb_sha256: str
    deployment_effective_rgb_sha256: str
    # float32 [action_dim,num_action_classes]，共享clean prefix下的A/B logits。
    training_teacher_logits: Float32Array
    deployment_teacher_logits: Float32Array


@dataclass(frozen=True)
class MarginDriftRow:
    """一个state/action token的training margin与deployment drift。"""

    state_id: int
    action_index: int
    state_fingerprint: str
    response_artifact_sha256: str
    teacher_input_ids_sha256: str
    training_effective_rgb_sha256: str
    deployment_effective_rgb_sha256: str
    clean_class: int
    training_margin: float
    deployment_margin: float
    margin_drift: float
    crossed_in_training: bool
    positive_deployment_drift: bool
    deployment_reversal: bool


@dataclass(frozen=True)
class MarginDriftCounts:
    """精确零边界下的主要token inventory。"""

    token_count: int
    uncrossed_token_count: int
    crossed_token_count: int
    positive_drift_token_count: int
    crossed_positive_drift_token_count: int
    deployment_reversal_token_count: int


@dataclass(frozen=True)
class DistributionSummary:
    """不设阈值的便读统计；空子集全部统计量为None。"""

    name: str
    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    q10: float | None
    q25: float | None
    q75: float | None
    q90: float | None
    median_absolute_deviation: float | None


@dataclass(frozen=True)
class MarginDriftDecision:
    """内存case的inventory验收与全部逐token派生结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    rows: tuple[MarginDriftRow, ...]
    counts: MarginDriftCounts
    distributions: tuple[DistributionSummary, ...]


@dataclass(frozen=True)
class MarginCalibrationSource:
    """报告绑定的Gate 6g正式manifest身份。"""

    manifest_sha256: str
    code_commit: str
    variant: str


@dataclass(frozen=True)
class MarginCalibrationReport:
    """可JSON序列化的κ feasibility/calibration只读报告。"""

    schema_version: str
    audit_valid: bool
    failures: tuple[str, ...]
    source: MarginCalibrationSource | None
    comparison_paths: tuple[str, str]
    clean_target_source: str
    teacher_prefix_relation: str
    only_model_input_change: str
    margin_arithmetic: str
    boundary_rule: str
    kappa_selection_performed: bool
    feasibility_boolean_gate: bool
    rows: tuple[MarginDriftRow, ...]
    counts: MarginDriftCounts
    distributions: tuple[DistributionSummary, ...]


def _empty_counts() -> MarginDriftCounts:
    return MarginDriftCounts(0, 0, 0, 0, 0, 0)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _distribution(name: str, values: Sequence[float]) -> DistributionSummary:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return DistributionSummary(
            name=name,
            count=0,
            mean=None,
            median=None,
            minimum=None,
            maximum=None,
            q10=None,
            q25=None,
            q75=None,
            q90=None,
            median_absolute_deviation=None,
        )
    median = float(np.median(array))
    quantiles = np.quantile(
        array,
        np.asarray([0.10, 0.25, 0.75, 0.90], dtype=np.float64),
        method="linear",
    )
    return DistributionSummary(
        name=name,
        count=int(array.size),
        mean=float(array.mean()),
        median=median,
        minimum=float(array.min()),
        maximum=float(array.max()),
        q10=float(quantiles[0]),
        q25=float(quantiles[1]),
        q75=float(quantiles[2]),
        q90=float(quantiles[3]),
        median_absolute_deviation=float(np.median(np.abs(array - median))),
    )


def _invalid_decision(failures: Sequence[str]) -> MarginDriftDecision:
    return MarginDriftDecision(
        audit_valid=False,
        failures=tuple(failures),
        rows=(),
        counts=_empty_counts(),
        distributions=(),
    )


def _validate_case(
    case: MarginDriftCase,
    *,
    expected_action_dim: int,
) -> tuple[str, ...]:
    failures: list[str] = []
    if type(case.state_id) is not int or case.state_id not in range(10):
        failures.append("state_id必须为0--9")
    for name in (
        "state_fingerprint",
        "response_artifact_sha256",
        "teacher_input_ids_sha256",
        "training_effective_rgb_sha256",
        "deployment_effective_rgb_sha256",
    ):
        if not _is_sha256(getattr(case, name)):
            failures.append(f"{name}必须为SHA-256")
    classes = np.asarray(case.clean_classes)
    training = np.asarray(case.training_teacher_logits)
    deployment = np.asarray(case.deployment_teacher_logits)
    if (
        classes.dtype != np.int64
        or classes.shape != (expected_action_dim,)
        or training.dtype != np.float32
        or training.ndim != 2
        or training.shape[0] != expected_action_dim
        or training.shape[1] < 2
        or deployment.dtype != np.float32
        or deployment.shape != training.shape
        or not bool(np.isfinite(training).all())
        or not bool(np.isfinite(deployment).all())
    ):
        failures.append(
            "clean classes/logits必须为int64 [A]及同shape finite float32 [A,C]"
        )
    elif bool(np.any(classes < 0)) or bool(np.any(classes >= training.shape[1])):
        failures.append("clean class越界")
    return tuple(failures)


def analyze_margin_drift_cases(
    cases: Sequence[MarginDriftCase],
    *,
    expected_state_ids: Sequence[int] = tuple(range(10)),
    expected_action_dim: int = 7,
) -> MarginDriftDecision:
    """复核完整state inventory并从A/B raw teacher logits计算margin分布。"""

    state_ids = tuple(expected_state_ids)
    if (
        not state_ids
        or len(set(state_ids)) != len(state_ids)
        or any(type(state_id) is not int or state_id not in range(10) for state_id in state_ids)
        or type(expected_action_dim) is not int
        or expected_action_dim <= 0
    ):
        return _invalid_decision(("expected states/action_dim无效",))
    failures: list[str] = []
    indexed: dict[int, MarginDriftCase] = {}
    for case in cases:
        failures.extend(
            f"state{case.state_id}: {failure}"
            for failure in _validate_case(
                case, expected_action_dim=expected_action_dim
            )
        )
        if case.state_id in indexed:
            failures.append(f"state inventory重复: {case.state_id}")
        indexed[case.state_id] = case
    if len(cases) != len(state_ids) or set(indexed) != set(state_ids):
        failures.append(
            f"state inventory必须恰好包含{len(state_ids)}个唯一case"
        )
    if failures:
        return _invalid_decision(failures)

    rows: list[MarginDriftRow] = []
    for state_id in state_ids:
        case = indexed[state_id]
        training_margins = compute_action_hinge(
            case.training_teacher_logits,
            case.clean_classes,
        ).margins
        deployment_margins = compute_action_hinge(
            case.deployment_teacher_logits,
            case.clean_classes,
        ).margins
        for action_index in range(expected_action_dim):
            margin_a = float(training_margins[action_index])
            margin_b = float(deployment_margins[action_index])
            # 保持Action objective的float32 margin尺度；统计聚合再提升float64。
            drift = float(np.float32(np.float32(margin_b) - np.float32(margin_a)))
            crossed = margin_a <= 0.0
            positive_drift = drift > 0.0
            rows.append(
                MarginDriftRow(
                    state_id=state_id,
                    action_index=action_index,
                    state_fingerprint=case.state_fingerprint,
                    response_artifact_sha256=case.response_artifact_sha256,
                    teacher_input_ids_sha256=case.teacher_input_ids_sha256,
                    training_effective_rgb_sha256=(
                        case.training_effective_rgb_sha256
                    ),
                    deployment_effective_rgb_sha256=(
                        case.deployment_effective_rgb_sha256
                    ),
                    clean_class=int(case.clean_classes[action_index]),
                    training_margin=margin_a,
                    deployment_margin=margin_b,
                    margin_drift=drift,
                    crossed_in_training=crossed,
                    positive_deployment_drift=positive_drift,
                    deployment_reversal=(crossed and margin_b > 0.0),
                )
            )
    crossed_rows = tuple(row for row in rows if row.crossed_in_training)
    uncrossed_rows = tuple(row for row in rows if not row.crossed_in_training)
    positive_rows = tuple(row for row in rows if row.positive_deployment_drift)
    crossed_positive_rows = tuple(
        row
        for row in crossed_rows
        if row.positive_deployment_drift
    )
    counts = MarginDriftCounts(
        token_count=len(rows),
        uncrossed_token_count=len(uncrossed_rows),
        crossed_token_count=len(crossed_rows),
        positive_drift_token_count=len(positive_rows),
        crossed_positive_drift_token_count=len(crossed_positive_rows),
        deployment_reversal_token_count=sum(
            row.deployment_reversal for row in rows
        ),
    )
    distributions = (
        _distribution(
            "training_margin_all",
            tuple(row.training_margin for row in rows),
        ),
        _distribution(
            "deployment_margin_all",
            tuple(row.deployment_margin for row in rows),
        ),
        _distribution(
            "margin_drift_all",
            tuple(row.margin_drift for row in rows),
        ),
        _distribution(
            "training_margin_uncrossed",
            tuple(row.training_margin for row in uncrossed_rows),
        ),
        _distribution(
            "training_margin_crossed",
            tuple(row.training_margin for row in crossed_rows),
        ),
        _distribution(
            "achieved_negative_buffer_crossed",
            tuple(-row.training_margin for row in crossed_rows),
        ),
        _distribution(
            "drift_crossed",
            tuple(row.margin_drift for row in crossed_rows),
        ),
        _distribution(
            "positive_drift_all",
            tuple(row.margin_drift for row in positive_rows),
        ),
        _distribution(
            "positive_drift_crossed",
            tuple(row.margin_drift for row in crossed_positive_rows),
        ),
    )
    return MarginDriftDecision(
        audit_valid=True,
        failures=(),
        rows=tuple(rows),
        counts=counts,
        distributions=distributions,
    )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerminalMarginCalibrationError(
            f"manifest无法读取: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TerminalMarginCalibrationError("manifest根必须为object")
    return value


def _safe_relative_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TerminalMarginCalibrationError("NPZ路径必须为非空相对路径")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise TerminalMarginCalibrationError("NPZ路径必须为安全相对路径")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TerminalMarginCalibrationError("NPZ路径逃逸bundle root") from error
    return resolved


def _invalid_report(failures: Sequence[str]) -> MarginCalibrationReport:
    return MarginCalibrationReport(
        schema_version=MARGIN_CALIBRATION_SCHEMA_VERSION,
        audit_valid=False,
        failures=tuple(failures),
        source=None,
        comparison_paths=("training_A", "deployment_B"),
        clean_target_source="shared_clean_cached_generation_class",
        teacher_prefix_relation="same_clean_teacher_forced_prefix",
        only_model_input_change="effective_rgb_A_vs_B",
        margin_arithmetic="float32_action_margin_and_drift_float64_statistics",
        boundary_rule="crossed_iff_m_A_le_0_exact",
        kappa_selection_performed=False,
        feasibility_boolean_gate=False,
        rows=(),
        counts=_empty_counts(),
        distributions=(),
    )


def analyze_terminal_margin_calibration_bundle(
    manifest_path: str | Path,
) -> MarginCalibrationReport:
    """从正式Gate 6g bundle提取Action-only 70-token margin/drift报告。"""

    path = Path(manifest_path).resolve()
    bundle_decision = evaluate_terminal_response_bundle(path)
    if not bundle_decision.audit_valid:
        return _invalid_report(
            tuple(
                f"Gate 6g bundle无效: {failure}"
                for failure in bundle_decision.failures
            )
        )
    try:
        manifest = _load_json_object(path)
        authority = manifest.get("response_authority")
        if not isinstance(authority, dict) or authority.get(
            "training_proxy"
        ) != "clean_prefix_teacher_forward":
            raise TerminalMarginCalibrationError(
                "Gate 6g未绑定clean-prefix teacher training proxy"
            )
        fingerprints = manifest.get("state_fingerprints")
        if (
            not isinstance(fingerprints, list)
            or len(fingerprints) != 10
            or not all(_is_sha256(value) for value in fingerprints)
        ):
            raise TerminalMarginCalibrationError(
                "state_fingerprints必须为states 0--9完整SHA inventory"
            )
        raw_records = manifest.get("cases")
        if not isinstance(raw_records, list):
            raise TerminalMarginCalibrationError("cases必须为list")
        records: dict[int, Mapping[str, object]] = {}
        for record in raw_records:
            if not isinstance(record, dict):
                raise TerminalMarginCalibrationError("case record必须为object")
            if record.get("variant") != ACTION_ONLY_VARIANT:
                continue
            state_id = record.get("state_id")
            if type(state_id) is not int or state_id in records:
                raise TerminalMarginCalibrationError(
                    "Action-only state identity无效或重复"
                )
            records[state_id] = record
        if set(records) != set(range(10)):
            raise TerminalMarginCalibrationError(
                "Action-only必须恰好包含states 0--9"
            )

        cases: list[MarginDriftCase] = []
        for state_id in range(10):
            record = records[state_id]
            artifact = _safe_relative_path(
                path.parent, record.get("npz_relative_path")
            )
            digest = record.get("npz_sha256")
            if not _is_sha256(digest) or file_sha256(artifact) != digest:
                raise TerminalMarginCalibrationError(
                    f"state{state_id} response artifact SHA漂移"
                )
            evidence: TerminalDeploymentResponseEvidence = (
                load_terminal_response_npz(artifact)
            )
            if (
                evidence.variant != ACTION_ONLY_VARIANT
                or evidence.state_id != state_id
                or record.get("initial_state_sha256") != fingerprints[state_id]
            ):
                raise TerminalMarginCalibrationError(
                    f"state{state_id} response/state fingerprint身份漂移"
                )
            clean_classes = np.asarray(
                evidence.generated_classes[0], dtype=np.int64
            )
            cases.append(
                MarginDriftCase(
                    state_id=state_id,
                    state_fingerprint=str(fingerprints[state_id]),
                    response_artifact_sha256=digest,
                    clean_classes=clean_classes,
                    teacher_input_ids_sha256=array_sha256(
                        evidence.teacher_input_ids
                    ),
                    training_effective_rgb_sha256=array_sha256(
                        evidence.effective_rgb[1]
                    ),
                    deployment_effective_rgb_sha256=array_sha256(
                        evidence.effective_rgb[2]
                    ),
                    training_teacher_logits=np.asarray(
                        evidence.teacher_logits[1], dtype=np.float32
                    ),
                    deployment_teacher_logits=np.asarray(
                        evidence.teacher_logits[2], dtype=np.float32
                    ),
                )
            )
        decision = analyze_margin_drift_cases(cases)
        source = MarginCalibrationSource(
            manifest_sha256=file_sha256(path),
            code_commit=str(manifest.get("code_commit")),
            variant=ACTION_ONLY_VARIANT,
        )
        return MarginCalibrationReport(
            schema_version=MARGIN_CALIBRATION_SCHEMA_VERSION,
            audit_valid=decision.audit_valid,
            failures=decision.failures,
            source=source,
            comparison_paths=("training_A", "deployment_B"),
            clean_target_source="shared_clean_cached_generation_class",
            teacher_prefix_relation="same_clean_teacher_forced_prefix",
            only_model_input_change="effective_rgb_A_vs_B",
            margin_arithmetic=(
                "float32_action_margin_and_drift_float64_statistics"
            ),
            boundary_rule="crossed_iff_m_A_le_0_exact",
            kappa_selection_performed=False,
            feasibility_boolean_gate=False,
            rows=decision.rows,
            counts=decision.counts,
            distributions=decision.distributions,
        )
    except (KeyError, OSError, ValueError) as error:
        return _invalid_report((str(error),))


def margin_calibration_report_dict(
    report: MarginCalibrationReport,
) -> dict[str, object]:
    """转换为稳定JSON object。"""

    value = asdict(report)
    assert isinstance(value, dict)
    return value


def write_margin_calibration_report(
    report: MarginCalibrationReport,
    *,
    output_path: str | Path,
) -> str:
    """不覆盖既有文件，原子写入只读报告并返回SHA-256。"""

    path = Path(output_path)
    if path.suffix != ".json":
        raise TerminalMarginCalibrationError("output_path必须为.json")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f"{path.stem}_candidate.json")
    if candidate.exists():
        raise FileExistsError(candidate)
    try:
        candidate.write_text(
            json.dumps(
                margin_calibration_report_dict(report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)
    return file_sha256(path)
