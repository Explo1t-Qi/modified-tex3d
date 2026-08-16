r"""Gate 6j Radial-vs-Matched-Box 的纯 Surface-space 反事实。

本模块只消费 Gate 6i 已保存的 Support 单步，不加载 OpenVLA、renderer、
LIBERO 或训练器。它把同一个 ``realized_endpoint`` 与同一个已归一化下降步
逐坐标投影到 Surface-:math:`L_\infty` box，再沿 endpoint--box 线段缩放，
使最终实际 :math:`L_\infty` 步长与 parent radial step 一致。

这里比较的是两种可行化算子；即使实际 :math:`L_\infty` 相同，两步的 L2、
坐标分布和方向仍可不同，因此不能称为纯方向消融。
"""

from __future__ import annotations

import math
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .terminal_endpoint_action_audit import (
    EndpointResponseEvidence,
    SurfaceCounterfactualStep,
    load_endpoint_response_npz,
    validate_endpoint_response_evidence,
)
from .terminal_endpoint_action_evidence import (
    EndpointStepEvidence,
    array_sha256,
    evaluate_terminal_endpoint_bundle,
    json_sha256,
    load_endpoint_step_npz,
)


Float32Array: TypeAlias = NDArray[np.float32]
MATCHED_BOX_STEP_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-projection-matched-box-step-v1"
)
PROJECTION_RESPONSE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-projection-response-v1"
)
PROJECTION_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-projection-counterfactual-bundle-v1"
)
PROJECTION_RESPONSE_ARMS: Final[tuple[str, ...]] = (
    "baseline",
    "radial_support",
    "matched_box_support",
)
DERIVED_FLOAT_TOLERANCE: Final[float] = 1e-12


class TerminalProjectionCounterfactualError(ValueError):
    """Matched-box 输入或派生结果违反冻结几何合同。"""


@dataclass(frozen=True)
class MatchedBoxStepStats:
    """从 parent radial 与 matched-box raw step 派生的核心标量。"""

    matched_scale: float
    radial_actual_linf: float
    box_actual_linf: float
    matched_actual_linf: float
    radial_l2: float
    box_l2: float
    matched_l2: float
    matched_max_abs_delta: float
    radial_predicted_decrease: float
    box_predicted_decrease: float
    matched_predicted_decrease: float
    radial_descent_alignment_cosine: float
    box_descent_alignment_cosine: float
    matched_descent_alignment_cosine: float


@dataclass(frozen=True)
class MatchedBoxCounterfactual:
    """一个 endpoint 的 box point 与 Linf-matched 可行反事实。"""

    box_step: Float32Array
    box_surface_delta: Float32Array
    matched_step: Float32Array
    matched_surface_delta: Float32Array
    stats: MatchedBoxStepStats


@dataclass(frozen=True)
class MatchedBoxEndpointEvidence:
    """一个 endpoint 的 parent-bound matched-box raw artifact。"""

    endpoint: str
    parent_bundle_sha256: str
    parent_step_npz_sha256: str
    support_mask_sha256: str
    realized_endpoint_sha256: str
    aggregate_gradient_sha256: str
    radial_surface_delta_sha256: str
    box_surface_delta_sha256: str
    matched_surface_delta_sha256: str
    epsilon: float
    box_step: Float32Array
    box_surface_delta: Float32Array
    matched_step: Float32Array
    matched_surface_delta: Float32Array
    matched_scale: float
    radial_actual_linf: float
    box_actual_linf: float
    matched_actual_linf: float
    radial_l2: float
    box_l2: float
    matched_l2: float
    matched_max_abs_delta: float
    radial_predicted_decrease: float
    box_predicted_decrease: float
    matched_predicted_decrease: float
    radial_descent_alignment_cosine: float
    box_descent_alignment_cosine: float
    matched_descent_alignment_cosine: float


@dataclass(frozen=True)
class MatchedBoxEndpointDecision:
    """Matched-box endpoint artifact 的独立 parent-bound 验收结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    endpoint: str


@dataclass(frozen=True)
class ProjectionEndpointMetrics:
    """一个 endpoint 的三臂 Action hinge 描述量，不是科学 pass/fail。"""

    endpoint: str
    baseline_mean_action_loss: float
    radial_mean_action_loss: float
    matched_mean_action_loss: float
    radial_minus_matched: float
    baseline_minus_matched: float
    per_state_radial_minus_matched: tuple[float, ...]
    per_state_baseline_minus_matched: tuple[float, ...]
    radial_minus_matched_median: float
    baseline_minus_matched_median: float
    radial_minus_matched_positive_count: int
    radial_minus_matched_zero_count: int
    radial_minus_matched_negative_count: int
    baseline_minus_matched_positive_count: int
    baseline_minus_matched_zero_count: int
    baseline_minus_matched_negative_count: int


@dataclass(frozen=True)
class ProjectionResponseDecision:
    """三臂inventory、parent replay与逐endpoint功能量的工程验收。"""

    audit_valid: bool
    failures: tuple[str, ...]
    response_record_count: int
    endpoint_metrics: tuple[ProjectionEndpointMetrics, ...]


@dataclass(frozen=True)
class TerminalProjectionBundleDecision:
    """Gate 6j自包含child bundle的工程验收结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    response_record_count: int
    endpoint_step_count: int
    response_decision: ProjectionResponseDecision | None
    endpoint_step_decisions: tuple[MatchedBoxEndpointDecision, ...]


def _geometry(value: np.ndarray, *, name: str) -> Float32Array:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] != 3
        or not bool(np.isfinite(array).all())
    ):
        raise TerminalProjectionCounterfactualError(
            f"{name}必须为finite float32 [num_geometry_vertices,3]"
        )
    return array


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    if value.shape != ():
        raise TerminalProjectionCounterfactualError(
            f"{name}必须为scalar"
        )
    return value.item()


def build_matched_box_counterfactual(
    parent_support_step: SurfaceCounterfactualStep,
    *,
    epsilon: float,
) -> MatchedBoxCounterfactual:
    r"""从 Gate 6i Support step 构造相同 actual-Linf 的 box 反事实。

    Args:
        parent_support_step: Gate 6i 保存并通过 CPU 复核的 Support arm。全部
            Surface 数组均为 float32 ``[num_geometry_vertices, 3]``。
        epsilon: Surface-:math:`L_\infty` 预算。

    Returns:
        box raw step/point、matched step/point 和核心几何派生量。
    """

    if not isinstance(epsilon, (int, float)) or not math.isfinite(
        float(epsilon)
    ) or float(epsilon) <= 0.0:
        raise TerminalProjectionCounterfactualError(
            "epsilon必须为finite正数"
        )
    endpoint = _geometry(
        parent_support_step.realized_endpoint,
        name="realized_endpoint",
    )
    gradient = _geometry(
        parent_support_step.masked_gradient,
        name="masked_gradient",
    )
    normalized = _geometry(
        parent_support_step.unconstrained_normalized_step,
        name="unconstrained_normalized_step",
    )
    radial = _geometry(
        parent_support_step.executed_step,
        name="radial_executed_step",
    )
    if not (
        endpoint.shape == gradient.shape == normalized.shape == radial.shape
    ):
        raise TerminalProjectionCounterfactualError(
            "parent Support Surface数组shape不一致"
        )

    epsilon32 = np.float32(epsilon)
    candidate = np.ascontiguousarray(endpoint + normalized, dtype=np.float32)
    box_surface = np.ascontiguousarray(
        np.clip(candidate, -epsilon32, epsilon32),
        dtype=np.float32,
    )
    box_step = np.ascontiguousarray(box_surface - endpoint, dtype=np.float32)
    box_linf = float(np.max(np.abs(box_step)))
    radial_linf = float(np.max(np.abs(radial)))
    if not math.isfinite(box_linf) or box_linf <= 0.0:
        raise TerminalProjectionCounterfactualError(
            "box step必须为finite nonzero"
        )
    if not math.isfinite(radial_linf) or radial_linf <= 0.0:
        raise TerminalProjectionCounterfactualError(
            "parent radial actual Linf必须为finite正数"
        )
    matched_scale = float(np.float32(radial_linf / box_linf))
    if not math.isfinite(matched_scale) or not 0.0 < matched_scale <= 1.0:
        raise TerminalProjectionCounterfactualError(
            "matched scale必须满足0 < s <= 1"
        )
    matched_step = np.ascontiguousarray(
        box_step * np.float32(matched_scale),
        dtype=np.float32,
    )
    matched_surface = np.ascontiguousarray(
        endpoint + matched_step,
        dtype=np.float32,
    )
    matched_linf = float(np.max(np.abs(matched_step)))
    matched_max = float(np.max(np.abs(matched_surface)))
    if matched_max > float(epsilon32):
        raise TerminalProjectionCounterfactualError(
            "matched endpoint超出Surface Linf预算"
        )
    flattened_gradient = gradient.reshape(-1).astype(np.float64)
    gradient_l2 = float(np.linalg.norm(flattened_gradient))
    if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
        raise TerminalProjectionCounterfactualError(
            "parent masked gradient必须为finite nonzero"
        )

    def step_metrics(step: Float32Array) -> tuple[float, float, float]:
        flattened = step.reshape(-1).astype(np.float64)
        l2 = float(np.linalg.norm(flattened))
        if not math.isfinite(l2) or l2 <= 0.0:
            raise TerminalProjectionCounterfactualError(
                "radial/box/matched step必须为finite nonzero"
            )
        predicted = -float(np.dot(flattened_gradient, flattened))
        cosine = predicted / (gradient_l2 * l2)
        return l2, predicted, float(np.clip(cosine, -1.0, 1.0))

    radial_l2, radial_predicted, radial_cosine = step_metrics(radial)
    box_l2, box_predicted, box_cosine = step_metrics(box_step)
    matched_l2, matched_predicted, matched_cosine = step_metrics(matched_step)
    return MatchedBoxCounterfactual(
        box_step=box_step,
        box_surface_delta=box_surface,
        matched_step=matched_step,
        matched_surface_delta=matched_surface,
        stats=MatchedBoxStepStats(
            matched_scale=matched_scale,
            radial_actual_linf=radial_linf,
            box_actual_linf=box_linf,
            matched_actual_linf=matched_linf,
            radial_l2=radial_l2,
            box_l2=box_l2,
            matched_l2=matched_l2,
            matched_max_abs_delta=matched_max,
            radial_predicted_decrease=radial_predicted,
            box_predicted_decrease=box_predicted,
            matched_predicted_decrease=matched_predicted,
            radial_descent_alignment_cosine=radial_cosine,
            box_descent_alignment_cosine=box_cosine,
            matched_descent_alignment_cosine=matched_cosine,
        ),
    )


def build_matched_box_endpoint_evidence(
    *,
    parent_bundle_sha256: str,
    parent_step_npz_sha256: str,
    parent_step: EndpointStepEvidence,
) -> MatchedBoxEndpointEvidence:
    """从一个已复核 Gate 6i endpoint step 构造 child raw evidence。"""

    if not _is_sha256(parent_bundle_sha256) or not _is_sha256(
        parent_step_npz_sha256
    ):
        raise TerminalProjectionCounterfactualError(
            "parent bundle/step SHA无效"
        )
    result = build_matched_box_counterfactual(
        parent_step.support_step,
        epsilon=parent_step.epsilon,
    )
    stats = result.stats
    return MatchedBoxEndpointEvidence(
        endpoint=parent_step.endpoint,
        parent_bundle_sha256=parent_bundle_sha256,
        parent_step_npz_sha256=parent_step_npz_sha256,
        support_mask_sha256=parent_step.support_mask_sha256,
        realized_endpoint_sha256=parent_step.realized_endpoint_sha256,
        aggregate_gradient_sha256=array_sha256(
            parent_step.support_step.masked_gradient
        ),
        radial_surface_delta_sha256=(
            parent_step.support_surface_delta_sha256
        ),
        box_surface_delta_sha256=array_sha256(result.box_surface_delta),
        matched_surface_delta_sha256=array_sha256(
            result.matched_surface_delta
        ),
        epsilon=parent_step.epsilon,
        box_step=result.box_step,
        box_surface_delta=result.box_surface_delta,
        matched_step=result.matched_step,
        matched_surface_delta=result.matched_surface_delta,
        **{
            field.name: getattr(stats, field.name)
            for field in fields(MatchedBoxStepStats)
        },
    )


_RAW_ARRAY_FIELDS: Final[tuple[str, ...]] = (
    "box_step",
    "box_surface_delta",
    "matched_step",
    "matched_surface_delta",
)
_DERIVED_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(MatchedBoxStepStats)
)
_SHA_FIELDS: Final[tuple[str, ...]] = (
    "parent_bundle_sha256",
    "parent_step_npz_sha256",
    "support_mask_sha256",
    "realized_endpoint_sha256",
    "aggregate_gradient_sha256",
    "radial_surface_delta_sha256",
    "box_surface_delta_sha256",
    "matched_surface_delta_sha256",
)
_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "endpoint",
        "epsilon",
        *_SHA_FIELDS,
        *_RAW_ARRAY_FIELDS,
        *_DERIVED_FIELDS,
    }
)


def _validate_endpoint_evidence(
    evidence: MatchedBoxEndpointEvidence,
) -> None:
    if evidence.endpoint not in ("action_spectral", "action_only_control"):
        raise TerminalProjectionCounterfactualError("endpoint无效")
    for name in _SHA_FIELDS:
        if not _is_sha256(getattr(evidence, name)):
            raise TerminalProjectionCounterfactualError(f"{name}无效")
    arrays = {
        name: _geometry(np.asarray(getattr(evidence, name)), name=name)
        for name in _RAW_ARRAY_FIELDS
    }
    if len({array.shape for array in arrays.values()}) != 1:
        raise TerminalProjectionCounterfactualError(
            "matched-box全部Surface数组shape必须一致"
        )
    if not math.isfinite(float(evidence.epsilon)) or evidence.epsilon <= 0.0:
        raise TerminalProjectionCounterfactualError("epsilon必须为finite正数")
    if not 0.0 < float(evidence.matched_scale) <= 1.0:
        raise TerminalProjectionCounterfactualError(
            "matched scale必须满足0 < s <= 1"
        )
    for name in _DERIVED_FIELDS:
        value = getattr(evidence, name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TerminalProjectionCounterfactualError(
                f"{name}必须为finite scalar"
            )
    if float(evidence.box_actual_linf) <= 0.0:
        raise TerminalProjectionCounterfactualError(
            "box step必须为finite nonzero"
        )
    if float(evidence.matched_max_abs_delta) > float(
        np.float32(evidence.epsilon)
    ):
        raise TerminalProjectionCounterfactualError(
            "matched endpoint超出Surface Linf预算"
        )
    if array_sha256(arrays["box_surface_delta"]) != (
        evidence.box_surface_delta_sha256
    ):
        raise TerminalProjectionCounterfactualError(
            "box Surface SHA不能从raw array复算"
        )
    if array_sha256(arrays["matched_surface_delta"]) != (
        evidence.matched_surface_delta_sha256
    ):
        raise TerminalProjectionCounterfactualError(
            "matched Surface SHA不能从raw array复算"
        )


def write_matched_box_endpoint_npz(
    evidence: MatchedBoxEndpointEvidence,
    *,
    output_path: str | Path,
) -> str:
    """拒绝覆盖地保存一个 endpoint 的 matched-box child artifact。"""

    _validate_endpoint_evidence(evidence)
    path = Path(output_path)
    if path.suffix != ".npz":
        raise TerminalProjectionCounterfactualError(
            "matched-box output_path必须为.npz"
        )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(MATCHED_BOX_STEP_SCHEMA_VERSION),
        "endpoint": np.asarray(evidence.endpoint),
        "epsilon": np.asarray(evidence.epsilon, dtype=np.float64),
    }
    payload.update(
        {name: np.asarray(getattr(evidence, name)) for name in _SHA_FIELDS}
    )
    payload.update(
        {name: getattr(evidence, name) for name in _RAW_ARRAY_FIELDS}
    )
    payload.update(
        {
            name: np.asarray(getattr(evidence, name), dtype=np.float64)
            for name in _DERIVED_FIELDS
        }
    )
    np.savez_compressed(path, **payload)
    return _file_sha256(path)


def load_matched_box_endpoint_npz(
    path: str | Path,
) -> MatchedBoxEndpointEvidence:
    """禁用 pickle 加载 child endpoint artifact 并执行结构校验。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != _ARTIFACT_KEYS:
            raise TerminalProjectionCounterfactualError(
                "matched-box NPZ keys与冻结schema不一致"
            )
        if str(_scalar(archive, "schema_version")) != (
            MATCHED_BOX_STEP_SCHEMA_VERSION
        ):
            raise TerminalProjectionCounterfactualError(
                "matched-box schema_version不匹配"
            )
        evidence = MatchedBoxEndpointEvidence(
            endpoint=str(_scalar(archive, "endpoint")),
            epsilon=float(_scalar(archive, "epsilon")),
            **{
                name: str(_scalar(archive, name)) for name in _SHA_FIELDS
            },
            **{name: archive[name].copy() for name in _RAW_ARRAY_FIELDS},
            **{
                name: float(_scalar(archive, name))
                for name in _DERIVED_FIELDS
            },
        )
    _validate_endpoint_evidence(evidence)
    return evidence


def evaluate_matched_box_endpoint_evidence(
    evidence: MatchedBoxEndpointEvidence,
    *,
    parent_bundle_sha256: str,
    parent_step_npz_sha256: str,
    parent_step: EndpointStepEvidence,
) -> MatchedBoxEndpointDecision:
    """从 parent raw step 独立重建 child，并逐项复核全部字段。"""

    failures: list[str] = []
    try:
        _validate_endpoint_evidence(evidence)
        expected = build_matched_box_endpoint_evidence(
            parent_bundle_sha256=parent_bundle_sha256,
            parent_step_npz_sha256=parent_step_npz_sha256,
            parent_step=parent_step,
        )
    except (TypeError, ValueError) as error:
        return MatchedBoxEndpointDecision(
            audit_valid=False,
            failures=(str(error),),
            endpoint=evidence.endpoint,
        )
    for field in fields(MatchedBoxEndpointEvidence):
        name = field.name
        observed = getattr(evidence, name)
        wanted = getattr(expected, name)
        if name in _RAW_ARRAY_FIELDS:
            if not np.array_equal(observed, wanted):
                failures.append(f"{name}不能从parent raw step复算")
        elif name in _DERIVED_FIELDS or name == "epsilon":
            if not math.isclose(
                float(observed),
                float(wanted),
                rel_tol=0.0,
                abs_tol=DERIVED_FLOAT_TOLERANCE,
            ):
                failures.append(f"{name}不能从parent raw step复算")
        elif observed != wanted:
            failures.append(f"{name}未严格绑定parent")
    return MatchedBoxEndpointDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        endpoint=evidence.endpoint,
    )


_PROJECTION_RESPONSE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "endpoint",
        "state_id",
        "arm",
        "state_fingerprint",
        "surface_delta_sha256",
        "clean_action_token_ids",
        "clean_classes",
        "action_loss",
        "margins",
        "hinge_values",
        "generated_token_ids",
        "generated_classes",
        "generation_logits",
        "teacher_logits",
        "decoded_action",
        "effective_view_rgb",
        "processor_bf16_bits",
    }
)


def _validate_projection_response(
    evidence: EndpointResponseEvidence,
) -> None:
    try:
        validate_endpoint_response_evidence(
            evidence,
            allowed_arms=PROJECTION_RESPONSE_ARMS,
        )
    except ValueError as error:
        raise TerminalProjectionCounterfactualError(str(error)) from error


def write_projection_response_npz(
    evidence: EndpointResponseEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存Gate 6j三臂之一的无pickle OpenVLA raw response。"""

    _validate_projection_response(evidence)
    path = Path(output_path)
    if path.suffix != ".npz":
        raise TerminalProjectionCounterfactualError(
            "projection response output_path必须为.npz"
        )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(PROJECTION_RESPONSE_SCHEMA_VERSION),
        endpoint=np.asarray(evidence.endpoint),
        state_id=np.asarray(evidence.state_id, dtype=np.int64),
        arm=np.asarray(evidence.arm),
        state_fingerprint=np.asarray(evidence.state_fingerprint),
        surface_delta_sha256=np.asarray(evidence.surface_delta_sha256),
        clean_action_token_ids=evidence.clean_action_token_ids,
        clean_classes=evidence.clean_classes,
        action_loss=np.asarray(evidence.action_loss, dtype=np.float32),
        margins=evidence.margins,
        hinge_values=evidence.hinge_values,
        generated_token_ids=evidence.generated_token_ids,
        generated_classes=evidence.generated_classes,
        generation_logits=evidence.generation_logits,
        teacher_logits=evidence.teacher_logits,
        decoded_action=evidence.decoded_action,
        effective_view_rgb=evidence.effective_view_rgb,
        processor_bf16_bits=evidence.processor_bf16_bits,
    )
    return _file_sha256(path)


def load_projection_response_npz(
    path: str | Path,
) -> EndpointResponseEvidence:
    """无pickle加载三臂response并从raw teacher logits复核objective。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != _PROJECTION_RESPONSE_KEYS:
            raise TerminalProjectionCounterfactualError(
                "projection response NPZ keys与冻结schema不一致"
            )
        if str(_scalar(archive, "schema_version")) != (
            PROJECTION_RESPONSE_SCHEMA_VERSION
        ):
            raise TerminalProjectionCounterfactualError(
                "projection response schema_version不匹配"
            )
        evidence = EndpointResponseEvidence(
            endpoint=str(_scalar(archive, "endpoint")),
            state_id=int(_scalar(archive, "state_id")),
            arm=str(_scalar(archive, "arm")),
            state_fingerprint=str(_scalar(archive, "state_fingerprint")),
            surface_delta_sha256=str(
                _scalar(archive, "surface_delta_sha256")
            ),
            clean_action_token_ids=archive["clean_action_token_ids"].copy(),
            clean_classes=archive["clean_classes"].copy(),
            action_loss=float(_scalar(archive, "action_loss")),
            margins=archive["margins"].copy(),
            hinge_values=archive["hinge_values"].copy(),
            generated_token_ids=archive["generated_token_ids"].copy(),
            generated_classes=archive["generated_classes"].copy(),
            generation_logits=archive["generation_logits"].copy(),
            teacher_logits=archive["teacher_logits"].copy(),
            decoded_action=archive["decoded_action"].copy(),
            effective_view_rgb=archive["effective_view_rgb"].copy(),
            processor_bf16_bits=archive["processor_bf16_bits"].copy(),
        )
    _validate_projection_response(evidence)
    return evidence


def _raw_response_equal(
    observed: EndpointResponseEvidence,
    expected: EndpointResponseEvidence,
) -> bool:
    """除arm名称映射外，对parent replay raw字段执行严格比较。"""

    for field in fields(EndpointResponseEvidence):
        if field.name == "arm":
            continue
        left = getattr(observed, field.name)
        right = getattr(expected, field.name)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            if (
                left_array.dtype != right_array.dtype
                or left_array.shape != right_array.shape
                or not np.array_equal(left_array, right_array)
            ):
                return False
        elif field.name == "action_loss":
            if np.float32(left).tobytes() != np.float32(right).tobytes():
                return False
        elif left != right:
            return False
    return True


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _sign_counts(values: Sequence[float]) -> tuple[int, int, int]:
    return (
        sum(value > 0.0 for value in values),
        sum(value == 0.0 for value in values),
        sum(value < 0.0 for value in values),
    )


def evaluate_projection_response_evidence(
    response_records: Sequence[EndpointResponseEvidence],
    *,
    parent_responses: Sequence[EndpointResponseEvidence],
    matched_steps_by_endpoint: Mapping[str, MatchedBoxEndpointEvidence],
) -> ProjectionResponseDecision:
    """严格复核60条新response及40条parent baseline/Support replay。

    本函数不输出 ``projection_harm`` 等科学布尔值。两个 endpoint 始终分别
    计算 ``D = L_radial - L_matched`` 与
    ``I_M = L_baseline - L_matched``，禁止先跨 endpoint 混合。
    """

    failures: list[str] = []
    child: dict[tuple[str, int, str], EndpointResponseEvidence] = {}
    for record in response_records:
        try:
            _validate_projection_response(record)
        except ValueError as error:
            failures.append(f"child response无效: {error}")
            continue
        key = (record.endpoint, record.state_id, record.arm)
        if key in child:
            failures.append(f"child response重复key: {key}")
        child[key] = record
    expected_child_keys = {
        (endpoint, state_id, arm)
        for endpoint in ("action_spectral", "action_only_control")
        for state_id in range(10)
        for arm in PROJECTION_RESPONSE_ARMS
    }
    if len(response_records) != 60 or set(child) != expected_child_keys:
        failures.append("必须恰好包含60个唯一三臂response")

    parent: dict[tuple[str, int, str], EndpointResponseEvidence] = {}
    for record in parent_responses:
        try:
            validate_endpoint_response_evidence(
                record,
                allowed_arms=("baseline", "support"),
            )
        except ValueError as error:
            failures.append(f"parent response无效: {error}")
            continue
        key = (record.endpoint, record.state_id, record.arm)
        if key in parent:
            failures.append(f"parent response重复key: {key}")
        parent[key] = record
    expected_parent_keys = {
        (endpoint, state_id, arm)
        for endpoint in ("action_spectral", "action_only_control")
        for state_id in range(10)
        for arm in ("baseline", "support")
    }
    if len(parent_responses) != 40 or set(parent) != expected_parent_keys:
        failures.append("parent必须恰好提供40个唯一baseline/Support response")
    if set(matched_steps_by_endpoint) != {
        "action_spectral",
        "action_only_control",
    }:
        failures.append("matched step必须逐endpoint各有一个")
    if failures:
        return ProjectionResponseDecision(
            audit_valid=False,
            failures=tuple(failures),
            response_record_count=len(response_records),
            endpoint_metrics=(),
        )

    for endpoint in ("action_spectral", "action_only_control"):
        matched_step = matched_steps_by_endpoint[endpoint]
        for state_id in range(10):
            baseline = child[(endpoint, state_id, "baseline")]
            radial = child[(endpoint, state_id, "radial_support")]
            matched = child[(endpoint, state_id, "matched_box_support")]
            parent_baseline = parent[(endpoint, state_id, "baseline")]
            parent_radial = parent[(endpoint, state_id, "support")]
            if not _raw_response_equal(baseline, parent_baseline):
                failures.append(
                    f"parent replay baseline漂移: {endpoint}/state{state_id}"
                )
            if not _raw_response_equal(radial, parent_radial):
                failures.append(
                    f"parent replay radial漂移: {endpoint}/state{state_id}"
                )
            if baseline.surface_delta_sha256 != (
                matched_step.realized_endpoint_sha256
            ):
                failures.append(
                    f"{endpoint}/state{state_id} baseline未绑定endpoint"
                )
            if radial.surface_delta_sha256 != (
                matched_step.radial_surface_delta_sha256
            ):
                failures.append(
                    f"{endpoint}/state{state_id} radial未绑定parent step"
                )
            if matched.surface_delta_sha256 != (
                matched_step.matched_surface_delta_sha256
            ):
                failures.append(
                    f"{endpoint}/state{state_id} matched未绑定child step"
                )
            if (
                matched.state_fingerprint != baseline.state_fingerprint
                or not np.array_equal(
                    matched.clean_action_token_ids,
                    baseline.clean_action_token_ids,
                )
                or not np.array_equal(
                    matched.clean_classes,
                    baseline.clean_classes,
                )
            ):
                failures.append(
                    f"{endpoint}/state{state_id} matched clean绑定漂移"
                )
    if failures:
        return ProjectionResponseDecision(
            audit_valid=False,
            failures=tuple(failures),
            response_record_count=len(response_records),
            endpoint_metrics=(),
        )

    metrics: list[ProjectionEndpointMetrics] = []
    for endpoint in ("action_spectral", "action_only_control"):
        losses = {
            arm: tuple(
                float(child[(endpoint, state_id, arm)].action_loss)
                for state_id in range(10)
            )
            for arm in PROJECTION_RESPONSE_ARMS
        }
        per_state_d = tuple(
            radial - matched
            for radial, matched in zip(
                losses["radial_support"],
                losses["matched_box_support"],
                strict=True,
            )
        )
        per_state_i = tuple(
            baseline - matched
            for baseline, matched in zip(
                losses["baseline"],
                losses["matched_box_support"],
                strict=True,
            )
        )
        d_counts = _sign_counts(per_state_d)
        i_counts = _sign_counts(per_state_i)
        baseline_mean = _mean(losses["baseline"])
        radial_mean = _mean(losses["radial_support"])
        matched_mean = _mean(losses["matched_box_support"])
        metrics.append(
            ProjectionEndpointMetrics(
                endpoint=endpoint,
                baseline_mean_action_loss=baseline_mean,
                radial_mean_action_loss=radial_mean,
                matched_mean_action_loss=matched_mean,
                radial_minus_matched=radial_mean - matched_mean,
                baseline_minus_matched=baseline_mean - matched_mean,
                per_state_radial_minus_matched=per_state_d,
                per_state_baseline_minus_matched=per_state_i,
                radial_minus_matched_median=_median(per_state_d),
                baseline_minus_matched_median=_median(per_state_i),
                radial_minus_matched_positive_count=d_counts[0],
                radial_minus_matched_zero_count=d_counts[1],
                radial_minus_matched_negative_count=d_counts[2],
                baseline_minus_matched_positive_count=i_counts[0],
                baseline_minus_matched_zero_count=i_counts[1],
                baseline_minus_matched_negative_count=i_counts[2],
            )
        )
    return ProjectionResponseDecision(
        audit_valid=True,
        failures=(),
        response_record_count=len(response_records),
        endpoint_metrics=tuple(metrics),
    )


def frozen_projection_configuration() -> dict[str, object]:
    """返回Gate 6j不可由CLI改写的科学与工程配置。"""

    return {
        "gate": "terminal_projection_counterfactual",
        "parent_gate": "terminal_endpoint_action_audit",
        "endpoint_names": ["action_spectral", "action_only_control"],
        "state_ids": list(range(10)),
        "response_arms": list(PROJECTION_RESPONSE_ARMS),
        "objective": "untargeted_clean_action_margin_hinge",
        "teacher_forced_clean_prefix": True,
        "gradient_recomputed": False,
        "training_or_rollout_run": False,
        "matched_actual_linf_to_parent_radial": True,
        "matched_scale_interval": "0 < s <= 1",
        "endpoint_metrics_kept_separate": True,
        "scientific_boolean_gate": False,
        "raw_parent_replay": "strict",
        "derived_float_tolerance": DERIVED_FLOAT_TOLERANCE,
    }


def _safe_relative_path(root: Path, relative: object, *, name: str) -> Path:
    if not isinstance(relative, str):
        raise TerminalProjectionCounterfactualError(f"{name}必须为相对路径")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise TerminalProjectionCounterfactualError(f"{name}必须为安全相对路径")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TerminalProjectionCounterfactualError(
            f"{name}逃逸bundle root"
        ) from error
    return resolved


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerminalProjectionCounterfactualError(
            f"manifest无法读取: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TerminalProjectionCounterfactualError("manifest根必须为object")
    return value


def _projection_derived(
    response_decision: ProjectionResponseDecision,
    endpoint_steps: Sequence[MatchedBoxEndpointEvidence],
) -> dict[str, object]:
    derived = {
        "endpoint_metrics": [
            asdict(metric) for metric in response_decision.endpoint_metrics
        ],
        "step_metrics": [
            {
                "endpoint": evidence.endpoint,
                **{
                    name: getattr(evidence, name)
                    for name in _DERIVED_FIELDS
                },
            }
            for evidence in endpoint_steps
        ],
        "response_record_count": response_decision.response_record_count,
        "endpoint_step_count": len(endpoint_steps),
    }
    # dataclass中的tuple经JSON保存后会成为list；在比较前统一为JSON原生结构。
    normalized = json.loads(json.dumps(derived, sort_keys=True))
    assert isinstance(normalized, dict)
    return normalized


def _derived_equal(observed: object, expected: object) -> bool:
    """递归比较derived JSON；只对浮点归约使用冻结的1e-12容差。"""

    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and all(
                _derived_equal(observed[name], expected[name])
                for name in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _derived_equal(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isclose(
                float(observed),
                expected,
                rel_tol=0.0,
                abs_tol=DERIVED_FLOAT_TOLERANCE,
            )
        )
    return type(observed) is type(expected) and observed == expected


def _invalid_bundle(
    failures: Sequence[str],
    *,
    response_count: int,
    step_decisions: Sequence[MatchedBoxEndpointDecision] = (),
) -> TerminalProjectionBundleDecision:
    return TerminalProjectionBundleDecision(
        audit_valid=False,
        failures=tuple(failures),
        response_record_count=response_count,
        endpoint_step_count=len(step_decisions),
        response_decision=None,
        endpoint_step_decisions=tuple(step_decisions),
    )


def evaluate_terminal_projection_bundle(
    manifest_path: str | Path,
    *,
    require_derived: bool = True,
) -> TerminalProjectionBundleDecision:
    """只读复核parent Gate 6i、child steps、严格replay和三臂响应。"""

    path = Path(manifest_path).resolve()
    root = path.parent
    try:
        manifest = _load_json_object(path)
    except ValueError as error:
        return _invalid_bundle((str(error),), response_count=0)
    failures: list[str] = []
    if manifest.get("schema_version") != PROJECTION_BUNDLE_SCHEMA_VERSION:
        failures.append("projection bundle schema_version不匹配")
    if manifest.get("status") != "complete":
        failures.append("projection bundle status必须为complete")
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or not all(
        character in "0123456789abcdef" for character in code_commit
    ):
        failures.append("code_commit必须为40位小写Git SHA")
    configuration = manifest.get("configuration")
    if configuration != frozen_projection_configuration() or (
        manifest.get("config_sha256") != json_sha256(configuration)
        if isinstance(configuration, dict)
        else True
    ):
        failures.append("projection frozen configuration漂移")
    provenance = manifest.get("provenance")
    expected_provenance = {
        "gradient_recomputed": False,
        "training_or_rollout_run": False,
        "feature_gradient": False,
        "wrist_gradient": False,
        "oft_gradient": False,
    }
    if provenance != expected_provenance:
        failures.append("projection provenance违反source-only静态审计合同")

    parent_record = manifest.get("parent")
    parent_manifest: Path | None = None
    parent_json: dict[str, object] | None = None
    parent_sha = ""
    if not isinstance(parent_record, dict):
        failures.append("parent record必须为object")
    else:
        try:
            parent_manifest = _safe_relative_path(
                root,
                parent_record.get("manifest_relative_path"),
                name="parent manifest",
            )
            parent_sha = _file_sha256(parent_manifest)
            if parent_record.get("manifest_sha256") != parent_sha:
                failures.append("parent manifest SHA漂移")
            copied_dense_metrics = _safe_relative_path(
                root,
                parent_record.get(
                    "source_dense_seed_metrics_relative_path"
                ),
                name="parent Dense Seed metrics",
            )
            copied_support = _safe_relative_path(
                root,
                parent_record.get("production_support_relative_path"),
                name="parent Production Support",
            )
            if _file_sha256(copied_dense_metrics) != parent_record.get(
                "source_dense_seed_metrics_sha256"
            ):
                failures.append("parent Dense Seed metrics SHA漂移")
            if _file_sha256(copied_support) != parent_record.get(
                "production_support_sha256"
            ):
                failures.append("parent Production Support SHA漂移")
            parent_decision = evaluate_terminal_endpoint_bundle(
                parent_manifest,
                source_dense_seed_metrics_path=copied_dense_metrics,
                production_support_path=copied_support,
            )
            if not parent_decision.audit_valid:
                failures.append("parent Gate 6i bundle复核失败")
            parent_json = _load_json_object(parent_manifest)
        except (OSError, ValueError) as error:
            failures.append(f"parent bundle无效: {error}")

    raw_step_records = manifest.get("endpoint_steps")
    step_records = raw_step_records if isinstance(raw_step_records, list) else []
    if not isinstance(raw_step_records, list):
        failures.append("endpoint_steps必须为list")
    matched_steps: dict[str, MatchedBoxEndpointEvidence] = {}
    step_decisions: list[MatchedBoxEndpointDecision] = []
    parent_steps: dict[str, tuple[EndpointStepEvidence, str]] = {}
    if parent_manifest is not None and parent_json is not None:
        parent_raw_steps = parent_json.get("endpoint_steps")
        if isinstance(parent_raw_steps, list):
            for record in parent_raw_steps:
                if not isinstance(record, dict):
                    continue
                try:
                    endpoint = str(record["endpoint"])
                    artifact = _safe_relative_path(
                        parent_manifest.parent,
                        record["npz_relative_path"],
                        name="parent endpoint step",
                    )
                    digest = _file_sha256(artifact)
                    if digest != record.get("npz_sha256"):
                        failures.append(f"{endpoint} parent step SHA漂移")
                        continue
                    parent_steps[endpoint] = (
                        load_endpoint_step_npz(artifact),
                        digest,
                    )
                except (KeyError, OSError, ValueError) as error:
                    failures.append(f"parent endpoint step无效: {error}")
    seen_step_endpoints: set[str] = set()
    for record in step_records:
        if not isinstance(record, dict):
            failures.append("child endpoint step record必须为object")
            continue
        try:
            endpoint = str(record["endpoint"])
            if endpoint in seen_step_endpoints:
                raise TerminalProjectionCounterfactualError(
                    "child endpoint step重复"
                )
            seen_step_endpoints.add(endpoint)
            artifact = _safe_relative_path(
                root,
                record["npz_relative_path"],
                name="child endpoint step",
            )
            digest = _file_sha256(artifact)
            if digest != record.get("npz_sha256"):
                raise TerminalProjectionCounterfactualError(
                    "child endpoint step SHA漂移"
                )
            evidence = load_matched_box_endpoint_npz(artifact)
            parent_step, expected_parent_digest = parent_steps[endpoint]
            if record.get("parent_step_npz_sha256") != expected_parent_digest:
                raise TerminalProjectionCounterfactualError(
                    "child record未绑定parent step SHA"
                )
            decision = evaluate_matched_box_endpoint_evidence(
                evidence,
                parent_bundle_sha256=parent_sha,
                parent_step_npz_sha256=expected_parent_digest,
                parent_step=parent_step,
            )
            step_decisions.append(decision)
            if not decision.audit_valid:
                failures.extend(decision.failures)
            matched_steps[endpoint] = evidence
        except (KeyError, OSError, ValueError) as error:
            failures.append(f"child endpoint step无效: {error}")
    if set(matched_steps) != {"action_spectral", "action_only_control"}:
        failures.append("child必须恰好包含两个endpoint step")

    raw_response_records = manifest.get("response_records")
    response_records = (
        raw_response_records if isinstance(raw_response_records, list) else []
    )
    if not isinstance(raw_response_records, list):
        failures.append("response_records必须为list")
    responses: list[EndpointResponseEvidence] = []
    for record in response_records:
        if not isinstance(record, dict):
            failures.append("child response record必须为object")
            continue
        try:
            artifact = _safe_relative_path(
                root,
                record["npz_relative_path"],
                name="child response",
            )
            if _file_sha256(artifact) != record.get("npz_sha256"):
                raise TerminalProjectionCounterfactualError(
                    "child response SHA漂移"
                )
            evidence = load_projection_response_npz(artifact)
            if (
                evidence.endpoint != record.get("endpoint")
                or evidence.state_id != record.get("state_id")
                or evidence.arm != record.get("arm")
            ):
                raise TerminalProjectionCounterfactualError(
                    "child response record身份漂移"
                )
            responses.append(evidence)
        except (KeyError, OSError, ValueError) as error:
            failures.append(f"child response无效: {error}")

    parent_responses: list[EndpointResponseEvidence] = []
    if parent_manifest is not None and parent_json is not None:
        parent_raw_responses = parent_json.get("response_records")
        if isinstance(parent_raw_responses, list):
            for record in parent_raw_responses:
                if not isinstance(record, dict) or record.get("arm") not in (
                    "baseline",
                    "support",
                ):
                    continue
                try:
                    artifact = _safe_relative_path(
                        parent_manifest.parent,
                        record["npz_relative_path"],
                        name="parent response",
                    )
                    if _file_sha256(artifact) != record.get("npz_sha256"):
                        raise TerminalProjectionCounterfactualError(
                            "parent response SHA漂移"
                        )
                    parent_responses.append(load_endpoint_response_npz(artifact))
                except (KeyError, OSError, ValueError) as error:
                    failures.append(f"parent response无效: {error}")
    if failures:
        return _invalid_bundle(
            failures,
            response_count=len(response_records),
            step_decisions=step_decisions,
        )
    response_decision = evaluate_projection_response_evidence(
        responses,
        parent_responses=parent_responses,
        matched_steps_by_endpoint=matched_steps,
    )
    if not response_decision.audit_valid:
        failures.extend(response_decision.failures)
    ordered_steps = tuple(
        matched_steps[endpoint]
        for endpoint in ("action_spectral", "action_only_control")
    )
    expected_derived = _projection_derived(response_decision, ordered_steps)
    if require_derived and not _derived_equal(
        manifest.get("derived"), expected_derived
    ):
        failures.append("projection derived不能从raw evidence复算")
    return TerminalProjectionBundleDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        response_record_count=len(response_records),
        endpoint_step_count=len(step_records),
        response_decision=(response_decision if not failures else None),
        endpoint_step_decisions=tuple(step_decisions),
    )


def publish_terminal_projection_bundle(
    payload: Mapping[str, object],
    *,
    output_path: str | Path,
) -> str:
    """先两次CPU复核，再原子发布Gate 6j成功manifest。"""

    path = Path(output_path)
    if path.suffix != ".json":
        raise TerminalProjectionCounterfactualError(
            "projection manifest output_path必须为.json"
        )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(f"{path.stem}_raw_candidate.json")
    candidate_path = path.with_name(f"{path.stem}_candidate.json")
    if raw_path.exists() or candidate_path.exists():
        raise FileExistsError(raw_path if raw_path.exists() else candidate_path)
    try:
        raw_payload = dict(payload)
        raw_payload["status"] = "complete"
        raw_path.write_text(
            json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raw_decision = evaluate_terminal_projection_bundle(
            raw_path,
            require_derived=False,
        )
        if not raw_decision.audit_valid or raw_decision.response_decision is None:
            raise TerminalProjectionCounterfactualError(
                "projection raw candidate复核失败: "
                + "; ".join(raw_decision.failures)
            )
        endpoint_steps: list[MatchedBoxEndpointEvidence] = []
        raw_manifest = _load_json_object(raw_path)
        for record in raw_manifest["endpoint_steps"]:  # type: ignore[index]
            endpoint_steps.append(
                load_matched_box_endpoint_npz(
                    _safe_relative_path(
                        path.parent,
                        record["npz_relative_path"],
                        name="child endpoint step",
                    )
                )
            )
        ordered = tuple(
            next(row for row in endpoint_steps if row.endpoint == endpoint)
            for endpoint in ("action_spectral", "action_only_control")
        )
        final_payload = dict(payload)
        final_payload["status"] = "complete"
        final_payload["derived"] = _projection_derived(
            raw_decision.response_decision,
            ordered,
        )
        candidate_path.write_text(
            json.dumps(
                final_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        final_decision = evaluate_terminal_projection_bundle(candidate_path)
        if not final_decision.audit_valid:
            raise TerminalProjectionCounterfactualError(
                "projection candidate复核失败: "
                + "; ".join(final_decision.failures)
            )
        os.replace(candidate_path, path)
    finally:
        raw_path.unlink(missing_ok=True)
        candidate_path.unlink(missing_ok=True)
    return _file_sha256(path)
