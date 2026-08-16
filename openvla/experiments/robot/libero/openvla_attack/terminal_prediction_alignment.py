r"""Gate 6g--6j 诊断收尾所用的逐 state 一阶预测/精确响应分析。

本模块只读取已经通过 Gate 6i/6j CPU 合同的 raw artifact，不加载模型、
CUDA、LIBERO、renderer 或训练器。对每个 endpoint、state 和非平凡更新 arm，
它使用该 state 自己保存的 dense Action gradient ``g[e,s]`` 计算：

.. math::

   P_{e,s,a}=-\langle g_{e,s}, \Delta_{e,a}\rangle,
   \qquad
   I_{e,s,a}=L^0_{e,s}-L^a_{e,s}.

``P`` 的乘法和归约显式提升到 float64；正/零/负使用精确比较，不引入隐藏
阈值。baseline 只是 ``I`` 和 generation 变化的共同参考，不作为平凡零步加入
contingency。输出只包含描述性证据，不产生 ``bpda_bad``、共同瓶颈或机制归因。
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

from .terminal_endpoint_action_audit import (
    ENDPOINT_NAMES,
    EndpointGradientEvidence,
    EndpointResponseEvidence,
    load_endpoint_gradient_npz,
)
from .terminal_endpoint_action_evidence import (
    EndpointStepEvidence,
    array_sha256,
    evaluate_terminal_endpoint_bundle,
    file_sha256,
    load_endpoint_step_npz,
)
from .terminal_projection_counterfactual import (
    MatchedBoxEndpointEvidence,
    evaluate_terminal_projection_bundle,
    load_matched_box_endpoint_npz,
    load_projection_response_npz,
)


Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]
PREDICTION_ALIGNMENT_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-prediction-alignment-report-v1"
)
ANALYZED_ARMS: Final[tuple[str, ...]] = (
    "radial_support",
    "matched_box_support",
)
SIGN_NAMES: Final[tuple[str, ...]] = ("positive", "zero", "negative")


class TerminalPredictionAlignmentError(ValueError):
    """P/I 输入、bundle 绑定或输出路径违反只读分析合同。"""


@dataclass(frozen=True)
class PredictionAlignmentCase:
    """一个 ``endpoint × state × arm`` 的全部权威输入及其SHA绑定。"""

    endpoint: str
    state_id: int
    arm: str
    state_fingerprint: str
    gradient_artifact_sha256: str
    baseline_response_artifact_sha256: str
    arm_response_artifact_sha256: str
    step_artifact_sha256: str
    step_array_sha256: str
    # float32 [num_geometry_vertices,3]，逐state source Action gradient。
    dense_surface_gradient: Float32Array
    # float32 [num_geometry_vertices,3]，从同一endpoint出发的实际可执行步。
    executed_step: Float32Array
    baseline_action_loss: float
    arm_action_loss: float
    baseline_generated_token_ids: Int64Array
    arm_generated_token_ids: Int64Array


@dataclass(frozen=True)
class PredictionAlignmentRow:
    """从一个case复算的P/I和离散generation描述量。"""

    endpoint: str
    state_id: int
    arm: str
    state_fingerprint: str
    gradient_artifact_sha256: str
    baseline_response_artifact_sha256: str
    arm_response_artifact_sha256: str
    step_artifact_sha256: str
    step_array_sha256: str
    baseline_action_loss: float
    arm_action_loss: float
    predicted_decrease: float
    exact_improvement: float
    prediction_sign: str
    improvement_sign: str
    generation_token_change_count: int
    first_generation_token_change_index: int | None


@dataclass(frozen=True)
class SignContingencyCell:
    """P/I精确符号3×3 contingency中的一个格子。"""

    prediction_sign: str
    improvement_sign: str
    count: int


@dataclass(frozen=True)
class PredictionAlignmentSummary:
    """一个endpoint和一个arm的描述性聚合；禁止跨endpoint先平均。"""

    endpoint: str
    arm: str
    case_count: int
    predicted_decrease_mean: float
    predicted_decrease_median: float
    exact_improvement_mean: float
    exact_improvement_median: float
    generation_changed_state_count: int
    generation_changed_token_count: int
    sign_contingency: tuple[SignContingencyCell, ...]


@dataclass(frozen=True)
class PredictionAlignmentDecision:
    """内存case的inventory/数值验收和P/I结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    rows: tuple[PredictionAlignmentRow, ...]
    summaries: tuple[PredictionAlignmentSummary, ...]


@dataclass(frozen=True)
class PredictionAlignmentSource:
    """正式报告绑定的Gate 6i/6j manifest身份。"""

    projection_manifest_sha256: str
    projection_code_commit: str
    endpoint_manifest_sha256: str
    endpoint_code_commit: str


@dataclass(frozen=True)
class PredictionAlignmentReport:
    """可JSON序列化的最终只读报告。"""

    schema_version: str
    audit_valid: bool
    failures: tuple[str, ...]
    source: PredictionAlignmentSource | None
    analyzed_arms: tuple[str, ...]
    baseline_role: str
    prediction_reduction: str
    sign_rule: str
    scientific_boolean_gate: bool
    rows: tuple[PredictionAlignmentRow, ...]
    summaries: tuple[PredictionAlignmentSummary, ...]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _mean(values: Sequence[float]) -> float:
    return float(np.asarray(tuple(values), dtype=np.float64).mean())


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(tuple(values), dtype=np.float64)))


def _validate_case(case: PredictionAlignmentCase) -> tuple[str, ...]:
    failures: list[str] = []
    if case.endpoint not in ENDPOINT_NAMES:
        failures.append("endpoint无效")
    if type(case.state_id) is not int or case.state_id not in range(10):
        failures.append("state_id必须为0--9")
    if case.arm not in ANALYZED_ARMS:
        failures.append("arm必须为radial_support或matched_box_support")
    for name in (
        "state_fingerprint",
        "gradient_artifact_sha256",
        "baseline_response_artifact_sha256",
        "arm_response_artifact_sha256",
        "step_artifact_sha256",
        "step_array_sha256",
    ):
        if not _is_sha256(getattr(case, name)):
            failures.append(f"{name}必须为SHA-256")
    gradient = np.asarray(case.dense_surface_gradient)
    step = np.asarray(case.executed_step)
    if (
        gradient.dtype != np.float32
        or gradient.ndim != 2
        or gradient.shape[0] <= 0
        or gradient.shape[1] != 3
        or not bool(np.isfinite(gradient).all())
    ):
        failures.append("dense_surface_gradient必须为finite float32 [N,3]")
    if (
        step.dtype != np.float32
        or step.shape != gradient.shape
        or not bool(np.isfinite(step).all())
    ):
        failures.append("executed_step必须为同shape finite float32 [N,3]")
    elif case.step_array_sha256 != array_sha256(step):
        failures.append("step_array_sha256不能从executed_step复算")
    for name in ("baseline_action_loss", "arm_action_loss"):
        value = getattr(case, name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{name}必须为finite scalar")
    baseline_tokens = np.asarray(case.baseline_generated_token_ids)
    arm_tokens = np.asarray(case.arm_generated_token_ids)
    if (
        baseline_tokens.dtype != np.int64
        or baseline_tokens.ndim != 1
        or baseline_tokens.size <= 0
        or arm_tokens.dtype != np.int64
        or arm_tokens.shape != baseline_tokens.shape
    ):
        failures.append("generation token必须为同shape非空int64 [action_dim]")
    return tuple(failures)


def _invalid_decision(failures: Sequence[str]) -> PredictionAlignmentDecision:
    return PredictionAlignmentDecision(
        audit_valid=False,
        failures=tuple(failures),
        rows=(),
        summaries=(),
    )


def analyze_prediction_alignment_cases(
    cases: Sequence[PredictionAlignmentCase],
    *,
    expected_state_ids: Sequence[int] = tuple(range(10)),
) -> PredictionAlignmentDecision:
    """严格验收40个非平凡case并逐state复算P/I。

    ``baseline`` 不作为零步case加入输入；每个arm case都显式携带其baseline
    loss、generation和artifact SHA。相同endpoint/state的两个arm必须绑定同一
    gradient、fingerprint和baseline，相同endpoint/arm跨state必须绑定同一步。
    """

    state_ids = tuple(expected_state_ids)
    if (
        not state_ids
        or len(set(state_ids)) != len(state_ids)
        or any(type(state_id) is not int or state_id not in range(10) for state_id in state_ids)
    ):
        return _invalid_decision(("expected_state_ids必须为0--9非空唯一子集",))
    failures: list[str] = []
    indexed: dict[tuple[str, int, str], PredictionAlignmentCase] = {}
    for case in cases:
        case_failures = _validate_case(case)
        failures.extend(
            f"{case.endpoint}/state{case.state_id}/{case.arm}: {failure}"
            for failure in case_failures
        )
        key = (case.endpoint, case.state_id, case.arm)
        if key in indexed:
            failures.append(f"inventory重复key: {key}")
        indexed[key] = case
    expected_keys = {
        (endpoint, state_id, arm)
        for endpoint in ENDPOINT_NAMES
        for state_id in state_ids
        for arm in ANALYZED_ARMS
    }
    if len(cases) != len(expected_keys) or set(indexed) != expected_keys:
        failures.append(
            f"inventory必须恰好包含{len(expected_keys)}个唯一非平凡case"
        )
    if failures:
        return _invalid_decision(failures)

    for endpoint in ENDPOINT_NAMES:
        for state_id in state_ids:
            radial = indexed[(endpoint, state_id, "radial_support")]
            matched = indexed[(endpoint, state_id, "matched_box_support")]
            if (
                radial.state_fingerprint != matched.state_fingerprint
                or radial.gradient_artifact_sha256
                != matched.gradient_artifact_sha256
                or radial.baseline_response_artifact_sha256
                != matched.baseline_response_artifact_sha256
                or radial.baseline_action_loss != matched.baseline_action_loss
                or not np.array_equal(
                    radial.dense_surface_gradient,
                    matched.dense_surface_gradient,
                )
                or not np.array_equal(
                    radial.baseline_generated_token_ids,
                    matched.baseline_generated_token_ids,
                )
            ):
                failures.append(
                    f"{endpoint}/state{state_id}两个arm的state/gradient/baseline绑定漂移"
                )
        for arm in ANALYZED_ARMS:
            reference = indexed[(endpoint, state_ids[0], arm)]
            for state_id in state_ids[1:]:
                candidate = indexed[(endpoint, state_id, arm)]
                if (
                    candidate.step_artifact_sha256
                    != reference.step_artifact_sha256
                    or candidate.step_array_sha256 != reference.step_array_sha256
                    or not np.array_equal(
                        candidate.executed_step,
                        reference.executed_step,
                    )
                ):
                    failures.append(f"{endpoint}/{arm}跨state step绑定漂移")
    if failures:
        return _invalid_decision(failures)

    rows: list[PredictionAlignmentRow] = []
    for endpoint in ENDPOINT_NAMES:
        for arm in ANALYZED_ARMS:
            for state_id in state_ids:
                case = indexed[(endpoint, state_id, arm)]
                gradient64 = np.asarray(
                    case.dense_surface_gradient, dtype=np.float64
                )
                step64 = np.asarray(case.executed_step, dtype=np.float64)
                predicted = -float(
                    np.sum(gradient64 * step64, dtype=np.float64)
                )
                improvement = float(
                    np.float64(case.baseline_action_loss)
                    - np.float64(case.arm_action_loss)
                )
                changed = np.flatnonzero(
                    np.asarray(case.baseline_generated_token_ids)
                    != np.asarray(case.arm_generated_token_ids)
                )
                rows.append(
                    PredictionAlignmentRow(
                        endpoint=endpoint,
                        state_id=state_id,
                        arm=arm,
                        state_fingerprint=case.state_fingerprint,
                        gradient_artifact_sha256=case.gradient_artifact_sha256,
                        baseline_response_artifact_sha256=(
                            case.baseline_response_artifact_sha256
                        ),
                        arm_response_artifact_sha256=(
                            case.arm_response_artifact_sha256
                        ),
                        step_artifact_sha256=case.step_artifact_sha256,
                        step_array_sha256=case.step_array_sha256,
                        baseline_action_loss=float(case.baseline_action_loss),
                        arm_action_loss=float(case.arm_action_loss),
                        predicted_decrease=predicted,
                        exact_improvement=improvement,
                        prediction_sign=_sign(predicted),
                        improvement_sign=_sign(improvement),
                        generation_token_change_count=int(changed.size),
                        first_generation_token_change_index=(
                            None if changed.size == 0 else int(changed[0])
                        ),
                    )
                )

    summaries: list[PredictionAlignmentSummary] = []
    for endpoint in ENDPOINT_NAMES:
        for arm in ANALYZED_ARMS:
            group = tuple(
                row
                for row in rows
                if row.endpoint == endpoint and row.arm == arm
            )
            contingency = tuple(
                SignContingencyCell(
                    prediction_sign=prediction_sign,
                    improvement_sign=improvement_sign,
                    count=sum(
                        row.prediction_sign == prediction_sign
                        and row.improvement_sign == improvement_sign
                        for row in group
                    ),
                )
                for prediction_sign in SIGN_NAMES
                for improvement_sign in SIGN_NAMES
            )
            summaries.append(
                PredictionAlignmentSummary(
                    endpoint=endpoint,
                    arm=arm,
                    case_count=len(group),
                    predicted_decrease_mean=_mean(
                        tuple(row.predicted_decrease for row in group)
                    ),
                    predicted_decrease_median=_median(
                        tuple(row.predicted_decrease for row in group)
                    ),
                    exact_improvement_mean=_mean(
                        tuple(row.exact_improvement for row in group)
                    ),
                    exact_improvement_median=_median(
                        tuple(row.exact_improvement for row in group)
                    ),
                    generation_changed_state_count=sum(
                        row.generation_token_change_count > 0 for row in group
                    ),
                    generation_changed_token_count=sum(
                        row.generation_token_change_count for row in group
                    ),
                    sign_contingency=contingency,
                )
            )
    return PredictionAlignmentDecision(
        audit_valid=True,
        failures=(),
        rows=tuple(rows),
        summaries=tuple(summaries),
    )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerminalPredictionAlignmentError(
            f"manifest无法读取: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TerminalPredictionAlignmentError("manifest根必须为object")
    return value


def _safe_relative_path(root: Path, relative: object, *, name: str) -> Path:
    if not isinstance(relative, str):
        raise TerminalPredictionAlignmentError(f"{name}必须为相对路径")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise TerminalPredictionAlignmentError(f"{name}必须为安全相对路径")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TerminalPredictionAlignmentError(f"{name}逃逸bundle root") from error
    return resolved


def _record_map(
    records: object,
    *,
    key_names: Sequence[str],
    name: str,
) -> dict[tuple[object, ...], Mapping[str, object]]:
    if not isinstance(records, list):
        raise TerminalPredictionAlignmentError(f"{name}必须为list")
    result: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TerminalPredictionAlignmentError(f"{name} record必须为object")
        try:
            key = tuple(record[field] for field in key_names)
        except KeyError as error:
            raise TerminalPredictionAlignmentError(
                f"{name} record缺少身份字段"
            ) from error
        if key in result:
            raise TerminalPredictionAlignmentError(f"{name}重复key: {key}")
        result[key] = record
    return result


def _bound_artifact(
    root: Path,
    record: Mapping[str, object],
    *,
    name: str,
) -> tuple[Path, str]:
    path = _safe_relative_path(root, record.get("npz_relative_path"), name=name)
    digest = record.get("npz_sha256")
    if not _is_sha256(digest) or file_sha256(path) != digest:
        raise TerminalPredictionAlignmentError(f"{name} artifact SHA漂移")
    return path, digest


def _invalid_report(failures: Sequence[str]) -> PredictionAlignmentReport:
    return PredictionAlignmentReport(
        schema_version=PREDICTION_ALIGNMENT_SCHEMA_VERSION,
        audit_valid=False,
        failures=tuple(failures),
        source=None,
        analyzed_arms=ANALYZED_ARMS,
        baseline_role="reference_only_not_in_contingency",
        prediction_reduction="float64_negative_inner_product",
        sign_rule="exact_positive_zero_negative_no_tolerance",
        scientific_boolean_gate=False,
        rows=(),
        summaries=(),
    )


def analyze_terminal_prediction_alignment_bundle(
    manifest_path: str | Path,
) -> PredictionAlignmentReport:
    """从正式Gate 6j自包含bundle产生最终40行P/I报告。"""

    projection_path = Path(manifest_path).resolve()
    projection_decision = evaluate_terminal_projection_bundle(projection_path)
    if not projection_decision.audit_valid:
        return _invalid_report(
            tuple(
                f"Gate 6j bundle无效: {failure}"
                for failure in projection_decision.failures
            )
        )
    try:
        projection = _load_json_object(projection_path)
        parent_record = projection.get("parent")
        if not isinstance(parent_record, dict):
            raise TerminalPredictionAlignmentError("Gate 6j parent必须为object")
        parent_path = _safe_relative_path(
            projection_path.parent,
            parent_record.get("manifest_relative_path"),
            name="Gate 6i parent manifest",
        )
        parent_sha = parent_record.get("manifest_sha256")
        if not _is_sha256(parent_sha) or file_sha256(parent_path) != parent_sha:
            raise TerminalPredictionAlignmentError("Gate 6i parent manifest SHA漂移")
        copied_dense_metrics = _safe_relative_path(
            projection_path.parent,
            parent_record.get("source_dense_seed_metrics_relative_path"),
            name="Gate 6i Dense Seed metrics",
        )
        copied_support = _safe_relative_path(
            projection_path.parent,
            parent_record.get("production_support_relative_path"),
            name="Gate 6i Production Support",
        )
        if file_sha256(copied_dense_metrics) != parent_record.get(
            "source_dense_seed_metrics_sha256"
        ):
            raise TerminalPredictionAlignmentError(
                "Gate 6i Dense Seed metrics SHA漂移"
            )
        if file_sha256(copied_support) != parent_record.get(
            "production_support_sha256"
        ):
            raise TerminalPredictionAlignmentError(
                "Gate 6i Production Support SHA漂移"
            )
        endpoint_decision = evaluate_terminal_endpoint_bundle(
            parent_path,
            source_dense_seed_metrics_path=copied_dense_metrics,
            production_support_path=copied_support,
        )
        if not endpoint_decision.audit_valid:
            raise TerminalPredictionAlignmentError(
                "Gate 6i parent bundle无效: "
                + "; ".join(endpoint_decision.failures)
            )
        parent = _load_json_object(parent_path)
        gradient_records = _record_map(
            parent.get("gradient_cases"),
            key_names=("endpoint", "state_id"),
            name="gradient_cases",
        )
        parent_step_records = _record_map(
            parent.get("endpoint_steps"),
            key_names=("endpoint",),
            name="parent endpoint_steps",
        )
        child_step_records = _record_map(
            projection.get("endpoint_steps"),
            key_names=("endpoint",),
            name="child endpoint_steps",
        )
        response_records = _record_map(
            projection.get("response_records"),
            key_names=("endpoint", "state_id", "arm"),
            name="projection response_records",
        )

        gradients: dict[tuple[str, int], tuple[EndpointGradientEvidence, str]] = {}
        for key, record in gradient_records.items():
            artifact, digest = _bound_artifact(
                parent_path.parent, record, name="gradient"
            )
            evidence = load_endpoint_gradient_npz(artifact)
            gradients[(str(key[0]), int(key[1]))] = (evidence, digest)
        parent_steps: dict[str, tuple[EndpointStepEvidence, str]] = {}
        child_steps: dict[str, tuple[MatchedBoxEndpointEvidence, str]] = {}
        for endpoint in ENDPOINT_NAMES:
            parent_artifact, parent_digest = _bound_artifact(
                parent_path.parent,
                parent_step_records[(endpoint,)],
                name="parent endpoint step",
            )
            child_artifact, child_digest = _bound_artifact(
                projection_path.parent,
                child_step_records[(endpoint,)],
                name="child matched step",
            )
            parent_steps[endpoint] = (
                load_endpoint_step_npz(parent_artifact),
                parent_digest,
            )
            child_steps[endpoint] = (
                load_matched_box_endpoint_npz(child_artifact),
                child_digest,
            )

        responses: dict[
            tuple[str, int, str], tuple[EndpointResponseEvidence, str]
        ] = {}
        for key, record in response_records.items():
            artifact, digest = _bound_artifact(
                projection_path.parent, record, name="projection response"
            )
            responses[(str(key[0]), int(key[1]), str(key[2]))] = (
                load_projection_response_npz(artifact),
                digest,
            )

        cases: list[PredictionAlignmentCase] = []
        for endpoint in ENDPOINT_NAMES:
            parent_step, parent_step_sha = parent_steps[endpoint]
            matched_step, matched_step_sha = child_steps[endpoint]
            steps = {
                "radial_support": (
                    parent_step.support_step.executed_step,
                    parent_step_sha,
                ),
                "matched_box_support": (
                    matched_step.matched_step,
                    matched_step_sha,
                ),
            }
            for state_id in range(10):
                gradient, gradient_sha = gradients[(endpoint, state_id)]
                baseline, baseline_sha = responses[
                    (endpoint, state_id, "baseline")
                ]
                for arm in ANALYZED_ARMS:
                    response, response_sha = responses[(endpoint, state_id, arm)]
                    step, step_artifact_sha = steps[arm]
                    cases.append(
                        PredictionAlignmentCase(
                            endpoint=endpoint,
                            state_id=state_id,
                            arm=arm,
                            state_fingerprint=gradient.state_fingerprint,
                            gradient_artifact_sha256=gradient_sha,
                            baseline_response_artifact_sha256=baseline_sha,
                            arm_response_artifact_sha256=response_sha,
                            step_artifact_sha256=step_artifact_sha,
                            step_array_sha256=array_sha256(step),
                            dense_surface_gradient=(
                                gradient.dense_surface_gradient
                            ),
                            executed_step=step,
                            baseline_action_loss=baseline.action_loss,
                            arm_action_loss=response.action_loss,
                            baseline_generated_token_ids=(
                                baseline.generated_token_ids
                            ),
                            arm_generated_token_ids=response.generated_token_ids,
                        )
                    )
        decision = analyze_prediction_alignment_cases(cases)
        source = PredictionAlignmentSource(
            projection_manifest_sha256=file_sha256(projection_path),
            projection_code_commit=str(projection.get("code_commit")),
            endpoint_manifest_sha256=parent_sha,
            endpoint_code_commit=str(parent.get("code_commit")),
        )
        return PredictionAlignmentReport(
            schema_version=PREDICTION_ALIGNMENT_SCHEMA_VERSION,
            audit_valid=decision.audit_valid,
            failures=decision.failures,
            source=source,
            analyzed_arms=ANALYZED_ARMS,
            baseline_role="reference_only_not_in_contingency",
            prediction_reduction="float64_negative_inner_product",
            sign_rule="exact_positive_zero_negative_no_tolerance",
            scientific_boolean_gate=False,
            rows=decision.rows,
            summaries=decision.summaries,
        )
    except (KeyError, OSError, ValueError) as error:
        return _invalid_report((str(error),))


def prediction_alignment_report_dict(
    report: PredictionAlignmentReport,
) -> dict[str, object]:
    """转换为稳定JSON对象；tuple由dataclass递归转为JSON list。"""

    value = asdict(report)
    assert isinstance(value, dict)
    return value


def write_prediction_alignment_report(
    report: PredictionAlignmentReport,
    *,
    output_path: str | Path,
) -> str:
    """不覆盖既有文件，原子写入报告并返回文件SHA-256。"""

    path = Path(output_path)
    if path.suffix != ".json":
        raise TerminalPredictionAlignmentError("output_path必须为.json")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f"{path.stem}_candidate.json")
    if candidate.exists():
        raise FileExistsError(candidate)
    try:
        candidate.write_text(
            json.dumps(
                prediction_alignment_report_dict(report),
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
