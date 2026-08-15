"""Gate 6i Terminal Endpoint Action-Gradient/Response 的纯 CPU 合同核心。

本模块不加载 OpenVLA、CUDA、LIBERO 或 renderer。GPU runner 未来只允许
向这里定义的无 pickle artifact 写入 authoritative raw evidence；CPU
evaluator 只从这些数组复算 retention、Action hinge、单步几何和
20/60 inventory，绝不在 CPU 上重跑神经网络。

更新顺序严格冻结为：``realize endpoint -> aggregate -> mask ->
normalize -> add -> global projection -> post-projection step cap``。Dense
和 Support 分别做 Surface-Linf normalization，不人为匹配 L2 norm。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray


Float32Array: TypeAlias = NDArray[np.float32]
EndpointName: TypeAlias = Literal["action_spectral", "action_only_control"]
ResponseArm: TypeAlias = Literal["baseline", "dense", "support"]

ENDPOINT_GRADIENT_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-endpoint-gradient-v1"
)
ENDPOINT_RESPONSE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-endpoint-response-v1"
)
ENDPOINT_NAMES: Final[tuple[str, ...]] = (
    "action_spectral",
    "action_only_control",
)
RESPONSE_ARMS: Final[tuple[str, ...]] = ("baseline", "dense", "support")
EXPECTED_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10))
ACTION_TOKEN_START: Final[int] = 31_744
ACTION_TOKEN_END: Final[int] = 32_000
NUM_ACTION_BINS: Final[int] = ACTION_TOKEN_END - ACTION_TOKEN_START


class TerminalEndpointActionAuditError(ValueError):
    """Gate 6i raw evidence 或派生量违反冻结合同。"""


@dataclass(frozen=True)
class ActionHingeEvidence:
    """CPU 从 teacher action logits 复算的 Action objective。

    ``margins`` 和 ``hinge_values`` 是 float32 ``[action_dim]``；
    ``action_loss`` 是它们中 hinge 的 float32 算术平均。
    """

    action_loss: float
    margins: Float32Array
    hinge_values: Float32Array


@dataclass(frozen=True)
class EndpointGradientEvidence:
    """GPU 产生的一个 ``(endpoint,state)`` dense Surface 梯度。"""

    endpoint: str
    state_id: int
    state_fingerprint: str
    realized_endpoint_sha256: str
    # float32 [num_geometry_vertices,3]，已汇总全部共享纹理实例
    # 并从 renderer vertex 严格 scatter-add 回 OBJ geometry vertex。
    dense_surface_gradient: Float32Array


@dataclass(frozen=True)
class EndpointResponseEvidence:
    """GPU 产生的一个 ``(endpoint,state,arm)`` 静态响应。"""

    endpoint: str
    state_id: int
    arm: str
    state_fingerprint: str
    surface_delta_sha256: str
    clean_action_token_ids: NDArray[np.int64]
    clean_classes: NDArray[np.int64]
    action_loss: float
    margins: Float32Array
    hinge_values: Float32Array
    generated_token_ids: NDArray[np.int64]
    generated_classes: NDArray[np.int64]
    generation_logits: Float32Array
    teacher_logits: Float32Array
    decoded_action: NDArray[np.floating]
    # 真实forward的最终224x224 Effective View与processor BF16 bit pattern。
    effective_view_rgb: NDArray[np.uint8]
    processor_bf16_bits: NDArray[np.uint16]


@dataclass(frozen=True)
class SurfaceCounterfactualStepStats:
    """CPU 复算的 surface-space 更新标量。"""

    endpoint_projection_scale: float
    direction_surface_max: float
    parameter_scale: float
    projection_scale: float
    step_cap_scale: float
    actual_surface_step: float
    max_abs_delta: float


@dataclass(frozen=True)
class SurfaceCounterfactualStep:
    """Dense 或 Support arm 的完整可复算更新数组。

    所有数组都是 float32 ``[num_geometry_vertices,3]``。
    ``projection_residual = executed_step - unconstrained_normalized_step``。
    """

    realized_endpoint: Float32Array
    masked_gradient: Float32Array
    unconstrained_normalized_step: Float32Array
    projected_step: Float32Array
    executed_step: Float32Array
    projection_residual: Float32Array
    stats: SurfaceCounterfactualStepStats


@dataclass(frozen=True)
class EndpointInventoryDecision:
    """20 gradient / 60 response 唯一键工程验收结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    gradient_case_count: int
    response_record_count: int


@dataclass(frozen=True)
class EndpointDerivedMetrics:
    """一个endpoint的retention与单步Action派生量，不是科学pass/fail。"""

    endpoint: str
    zero_retentions: tuple[float, ...]
    terminal_retentions: tuple[float, ...]
    retention_differences: tuple[float, ...]
    zero_retention_mean: float
    zero_retention_median: float
    terminal_retention_mean: float
    terminal_retention_median: float
    negative_retention_difference_count: int
    zero_aggregate_retention: float
    terminal_aggregate_retention: float
    baseline_mean_action_loss: float
    dense_mean_action_loss: float
    support_mean_action_loss: float
    dense_improvement: float
    support_improvement: float
    dense_advantage: float
    per_state_dense_advantages: tuple[float, ...]
    dense_better_state_count: int
    support_better_state_count: int
    exact_equal_state_count: int


@dataclass(frozen=True)
class EndpointEvidenceDecision:
    """CPU从20 gradient/60 response raw evidence复算的描述性结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    inventory: EndpointInventoryDecision
    endpoint_metrics: tuple[EndpointDerivedMetrics, ...]
    per_state_endpoint_gradient_cosines: tuple[float, ...]
    aggregate_endpoint_gradient_cosine: float | None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_endpoint_state(endpoint: object, state_id: object) -> None:
    if endpoint not in ENDPOINT_NAMES:
        raise TerminalEndpointActionAuditError("endpoint无效")
    if type(state_id) is not int or state_id not in EXPECTED_STATE_IDS:
        raise TerminalEndpointActionAuditError("state_id必须为0--9")


def _float32_geometry(value: np.ndarray, *, name: str) -> Float32Array:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[1] != 3
        or array.shape[0] <= 0
        or not bool(np.isfinite(array).all())
    ):
        raise TerminalEndpointActionAuditError(
            f"{name}必须为finite float32 [num_geometry_vertices,3]"
        )
    return array


def aggregate_state_gradients(gradients: np.ndarray) -> Float32Array:
    """按正式 trainer 语义对state轴作float32算术平均。"""

    array = np.asarray(gradients)
    if (
        array.dtype != np.float32
        or array.ndim != 3
        or array.shape[0] <= 0
        or array.shape[2] != 3
        or not bool(np.isfinite(array).all())
    ):
        raise TerminalEndpointActionAuditError(
            "gradients必须为finite float32 [num_states,num_vertices,3]"
        )
    # NumPy对float32输入默认用float32 accumulator，与
    # torch.stack(...).mean(dim=0)的正式语义一致。
    return np.ascontiguousarray(array.mean(axis=0, dtype=np.float32))


def compute_gradient_retention(
    dense_gradient: np.ndarray,
    support_mask: np.ndarray,
) -> float:
    """计算冻结的普通Euclidean gradient-energy retention。"""

    gradient = _float32_geometry(dense_gradient, name="dense_gradient")
    mask = np.asarray(support_mask)
    if mask.dtype != np.bool_ or mask.shape != (gradient.shape[0],):
        raise TerminalEndpointActionAuditError(
            "support_mask必须为bool [num_geometry_vertices]"
        )
    # 比例用float64归约，但不改变authoritative float32梯度数组。
    full_energy = float(
        np.sum(gradient.astype(np.float64) ** 2, dtype=np.float64)
    )
    if not math.isfinite(full_energy) or full_energy <= 0.0:
        raise TerminalEndpointActionAuditError(
            "dense_gradient必须为finite nonzero"
        )
    support = gradient[mask].astype(np.float64)
    support_energy = float(np.sum(support**2, dtype=np.float64))
    return support_energy / full_energy


def compute_action_hinge(
    teacher_logits: np.ndarray,
    clean_classes: np.ndarray,
) -> ActionHingeEvidence:
    """CPU从已抽取的action-subvocabulary logits复算margin hinge。"""

    logits = np.asarray(teacher_logits)
    classes = np.asarray(clean_classes)
    if (
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[0] <= 0
        or logits.shape[1] < 2
        or not bool(np.isfinite(logits).all())
        or classes.dtype != np.int64
        or classes.shape != (logits.shape[0],)
        or bool(np.any(classes < 0))
        or bool(np.any(classes >= logits.shape[1]))
    ):
        raise TerminalEndpointActionAuditError(
            "teacher logits/clean classes shape、dtype或数值无效"
        )
    rows = np.arange(classes.size)
    clean_logits = logits[rows, classes]
    other_logits = logits.copy()
    other_logits[rows, classes] = np.float32(-np.inf)
    best_other = np.max(other_logits, axis=1)
    margins = np.ascontiguousarray(
        (clean_logits - best_other).astype(np.float32)
    )
    hinges = np.ascontiguousarray(np.maximum(margins, np.float32(0.0)))
    loss = float(np.mean(hinges, dtype=np.float32))
    return ActionHingeEvidence(loss, margins, hinges)


def _project_surface(
    surface_delta: Float32Array,
    epsilon: float,
) -> tuple[Float32Array, float]:
    maximum = np.max(np.abs(surface_delta))
    tiny = np.float32(np.finfo(np.float32).tiny)
    scale = np.minimum(
        np.float32(epsilon) / np.maximum(maximum, tiny),
        np.float32(1.0),
    ).astype(np.float32)
    return (
        np.ascontiguousarray(surface_delta * scale, dtype=np.float32),
        float(scale),
    )


def execute_surface_counterfactual_step(
    endpoint: np.ndarray,
    dense_gradient: np.ndarray,
    *,
    epsilon: float,
    surface_step: float,
    support_mask: np.ndarray | None = None,
) -> SurfaceCounterfactualStep:
    """按冻结顺序执行一次Dense或Support surface-space反事实更新。"""

    raw_endpoint = _float32_geometry(endpoint, name="endpoint")
    gradient = _float32_geometry(dense_gradient, name="dense_gradient")
    if gradient.shape != raw_endpoint.shape:
        raise TerminalEndpointActionAuditError("endpoint/gradient shape不一致")
    if (
        not math.isfinite(float(epsilon))
        or not math.isfinite(float(surface_step))
        or epsilon <= 0.0
        or surface_step <= 0.0
        or surface_step > epsilon
    ):
        raise TerminalEndpointActionAuditError(
            "epsilon/surface_step必须为有效有限正数"
        )
    # 与正式trainer一致，先把已有endpoint的functional projection固化。
    realized, endpoint_projection_scale = _project_surface(
        raw_endpoint,
        float(epsilon),
    )
    masked = gradient.copy()
    if support_mask is not None:
        mask = np.asarray(support_mask)
        if mask.dtype != np.bool_ or mask.shape != (gradient.shape[0],):
            raise TerminalEndpointActionAuditError(
                "support_mask必须为bool [num_geometry_vertices]"
            )
        masked[~mask] = np.float32(0.0)
    if not bool(np.any(masked != 0.0)):
        raise TerminalEndpointActionAuditError(
            "counterfactual direction必须为finite nonzero"
        )

    direction = np.negative(masked, dtype=np.float32)
    direction_max = np.max(np.abs(direction))
    if float(direction_max) <= np.finfo(np.float32).tiny:
        raise TerminalEndpointActionAuditError(
            "counterfactual direction数值上为零"
        )
    parameter_scale = (
        np.float32(surface_step) / direction_max
    ).astype(np.float32)
    normalized_step = np.ascontiguousarray(
        direction * parameter_scale,
        dtype=np.float32,
    )
    candidate = np.ascontiguousarray(realized + normalized_step)
    projected, projection_scale = _project_surface(candidate, float(epsilon))
    projected_step = np.ascontiguousarray(projected - realized)
    projected_max = np.max(np.abs(projected_step))
    step_cap_scale = np.float32(1.0)
    if float(projected_max) > float(surface_step):
        step_cap_scale = (
            np.float32(surface_step) / projected_max
        ).astype(np.float32)
    executed_step = np.ascontiguousarray(
        projected_step * step_cap_scale,
        dtype=np.float32,
    )
    after = np.ascontiguousarray(realized + executed_step)
    actual_step = float(np.max(np.abs(executed_step)))
    residual = np.ascontiguousarray(executed_step - normalized_step)
    return SurfaceCounterfactualStep(
        realized_endpoint=realized,
        masked_gradient=np.ascontiguousarray(masked),
        unconstrained_normalized_step=normalized_step,
        projected_step=projected_step,
        executed_step=executed_step,
        projection_residual=residual,
        stats=SurfaceCounterfactualStepStats(
            endpoint_projection_scale=endpoint_projection_scale,
            direction_surface_max=float(direction_max),
            parameter_scale=float(parameter_scale),
            projection_scale=projection_scale,
            step_cap_scale=float(step_cap_scale),
            actual_surface_step=actual_step,
            max_abs_delta=float(np.max(np.abs(after))),
        ),
    )


def evaluate_endpoint_inventory(
    gradient_keys: Sequence[tuple[str, int]],
    response_keys: Sequence[tuple[str, int, str]],
) -> EndpointInventoryDecision:
    """验证2 endpoint x 10 state x 3 arm的完整唯一inventory。"""

    failures: list[str] = []
    expected_gradients = {
        (endpoint, state_id)
        for endpoint in ENDPOINT_NAMES
        for state_id in EXPECTED_STATE_IDS
    }
    expected_responses = {
        (endpoint, state_id, arm)
        for endpoint in ENDPOINT_NAMES
        for state_id in EXPECTED_STATE_IDS
        for arm in RESPONSE_ARMS
    }
    observed_gradients = list(gradient_keys)
    observed_responses = list(response_keys)
    if (
        len(observed_gradients) != 20
        or len(set(observed_gradients)) != 20
        or set(observed_gradients) != expected_gradients
    ):
        failures.append("inventory必须包含20个唯一gradient case")
    if (
        len(observed_responses) != 60
        or len(set(observed_responses)) != 60
        or set(observed_responses) != expected_responses
    ):
        failures.append("inventory必须包含60个唯一response record")
    return EndpointInventoryDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        gradient_case_count=len(observed_gradients),
        response_record_count=len(observed_responses),
    )


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """用float64归约复算两个已保存float32数组的普通Euclidean cosine。"""

    left = np.asarray(first, dtype=np.float32).reshape(-1).astype(np.float64)
    right = np.asarray(second, dtype=np.float32).reshape(-1).astype(np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 0.0
        or right_norm <= 0.0
    ):
        raise TerminalEndpointActionAuditError(
            "gradient cosine要求两个finite nonzero数组"
        )
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    # 浮点归约可能在完全同向时略微越过[-1,1]。
    return float(np.clip(cosine, -1.0, 1.0))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _invalid_evidence_decision(
    inventory: EndpointInventoryDecision,
    failures: Sequence[str],
) -> EndpointEvidenceDecision:
    return EndpointEvidenceDecision(
        audit_valid=False,
        failures=tuple(failures),
        inventory=inventory,
        endpoint_metrics=(),
        per_state_endpoint_gradient_cosines=(),
        aggregate_endpoint_gradient_cosine=None,
    )


def evaluate_endpoint_evidence(
    gradient_cases: Sequence[EndpointGradientEvidence],
    response_records: Sequence[EndpointResponseEvidence],
    *,
    zero_gradients_by_state: Mapping[int, np.ndarray],
    support_mask: np.ndarray,
) -> EndpointEvidenceDecision:
    """复核完整20/60 evidence并计算Gate 6i冻结的描述性派生量。

    本函数不运行OpenVLA。``gradient_cases``与``response_records``是GPU
    authoritative raw evidence；``zero_gradients_by_state``是Production
    Support绑定的零点Dense artifact中十个float32几何梯度。任一inventory、
    provenance、shape/dtype或跨记录绑定失败，都会使整组evidence invalid，且
    不返回可能被误读为有效结果的部分指标。
    """

    gradient_keys = [
        (case.endpoint, case.state_id) for case in gradient_cases
    ]
    response_keys = [
        (record.endpoint, record.state_id, record.arm)
        for record in response_records
    ]
    inventory = evaluate_endpoint_inventory(gradient_keys, response_keys)
    failures: list[str] = list(inventory.failures)
    if not inventory.audit_valid:
        return _invalid_evidence_decision(inventory, failures)

    gradients: dict[tuple[str, int], EndpointGradientEvidence] = {}
    for case in gradient_cases:
        try:
            _validate_gradient_evidence(case)
        except (TypeError, ValueError) as error:
            failures.append(
                f"gradient ({case.endpoint},{case.state_id})无效: {error}"
            )
        else:
            gradients[(case.endpoint, case.state_id)] = case

    responses: dict[tuple[str, int, str], EndpointResponseEvidence] = {}
    for record in response_records:
        try:
            _validate_response_evidence(record)
        except (TypeError, ValueError) as error:
            failures.append(
                "response "
                f"({record.endpoint},{record.state_id},{record.arm})无效: "
                f"{error}"
            )
        else:
            responses[(record.endpoint, record.state_id, record.arm)] = record
    if failures:
        return _invalid_evidence_decision(inventory, failures)

    zero_state_ids = set(zero_gradients_by_state)
    if (
        any(type(state_id) is not int for state_id in zero_gradients_by_state)
        or zero_state_ids != set(EXPECTED_STATE_IDS)
    ):
        failures.append("zero gradient必须恰好绑定states 0--9")
        return _invalid_evidence_decision(inventory, failures)

    first_gradient = gradients[(ENDPOINT_NAMES[0], EXPECTED_STATE_IDS[0])]
    geometry_shape = first_gradient.dense_surface_gradient.shape
    mask = np.asarray(support_mask)
    if mask.dtype != np.bool_ or mask.shape != (geometry_shape[0],):
        failures.append("support mask shape/dtype与geometry gradient不一致")

    zero_gradients: dict[int, Float32Array] = {}
    for state_id in EXPECTED_STATE_IDS:
        try:
            zero_gradient = _float32_geometry(
                np.asarray(zero_gradients_by_state[state_id]),
                name=f"zero_gradient[{state_id}]",
            )
            if (
                zero_gradient.shape != geometry_shape
                or not bool(np.any(zero_gradient != 0.0))
            ):
                raise TerminalEndpointActionAuditError(
                    "shape不一致或数值上为零"
                )
        except (TypeError, ValueError) as error:
            failures.append(f"zero gradient state {state_id}无效: {error}")
        else:
            zero_gradients[state_id] = zero_gradient

    for endpoint in ENDPOINT_NAMES:
        endpoint_cases = [
            gradients[(endpoint, state_id)]
            for state_id in EXPECTED_STATE_IDS
        ]
        shapes = {
            case.dense_surface_gradient.shape for case in endpoint_cases
        }
        if shapes != {geometry_shape}:
            failures.append(f"{endpoint} gradient geometry shape不唯一")
        endpoint_hashes = {
            case.realized_endpoint_sha256 for case in endpoint_cases
        }
        if len(endpoint_hashes) != 1:
            failures.append(f"{endpoint} realized endpoint SHA不唯一")
        for arm in RESPONSE_ARMS:
            surface_hashes = {
                responses[(endpoint, state_id, arm)].surface_delta_sha256
                for state_id in EXPECTED_STATE_IDS
            }
            if len(surface_hashes) != 1:
                failures.append(f"{endpoint}/{arm} surface delta SHA不唯一")

    canonical_fingerprints: list[str] = []
    for state_id in EXPECTED_STATE_IDS:
        fingerprints = {
            gradients[(endpoint, state_id)].state_fingerprint
            for endpoint in ENDPOINT_NAMES
        }
        fingerprints.update(
            responses[(endpoint, state_id, arm)].state_fingerprint
            for endpoint in ENDPOINT_NAMES
            for arm in RESPONSE_ARMS
        )
        if len(fingerprints) != 1:
            failures.append(f"state {state_id} fingerprint跨记录不一致")
        else:
            canonical_fingerprints.append(next(iter(fingerprints)))
        clean_tokens = {
            responses[(endpoint, state_id, arm)]
            .clean_action_token_ids.tobytes()
            for endpoint in ENDPOINT_NAMES
            for arm in RESPONSE_ARMS
        }
        clean_classes = {
            responses[(endpoint, state_id, arm)].clean_classes.tobytes()
            for endpoint in ENDPOINT_NAMES
            for arm in RESPONSE_ARMS
        }
        if len(clean_tokens) != 1 or len(clean_classes) != 1:
            failures.append(f"state {state_id} clean Action目标跨记录不一致")
    if len(set(canonical_fingerprints)) != len(EXPECTED_STATE_IDS):
        failures.append("states 0--9 fingerprint必须彼此唯一")
    if failures:
        return _invalid_evidence_decision(inventory, failures)

    assert mask.dtype == np.bool_  # 已由上面的工程验收收窄。
    zero_stack = np.stack(
        [zero_gradients[state_id] for state_id in EXPECTED_STATE_IDS],
        axis=0,
    ).astype(np.float32, copy=False)
    zero_retentions = tuple(
        compute_gradient_retention(zero_gradients[state_id], mask)
        for state_id in EXPECTED_STATE_IDS
    )
    zero_aggregate = aggregate_state_gradients(zero_stack)
    try:
        zero_aggregate_retention = compute_gradient_retention(
            zero_aggregate,
            mask,
        )
    except ValueError as error:
        failures.append(f"zero aggregate gradient无效: {error}")
        return _invalid_evidence_decision(inventory, failures)

    metrics: list[EndpointDerivedMetrics] = []
    aggregate_gradients: dict[str, Float32Array] = {}
    for endpoint in ENDPOINT_NAMES:
        terminal_stack = np.stack(
            [
                gradients[(endpoint, state_id)].dense_surface_gradient
                for state_id in EXPECTED_STATE_IDS
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        terminal_retentions = tuple(
            compute_gradient_retention(terminal_stack[index], mask)
            for index, _ in enumerate(EXPECTED_STATE_IDS)
        )
        retention_differences = tuple(
            terminal - zero
            for terminal, zero in zip(
                terminal_retentions,
                zero_retentions,
                strict=True,
            )
        )
        aggregate = aggregate_state_gradients(terminal_stack)
        try:
            terminal_aggregate_retention = compute_gradient_retention(
                aggregate,
                mask,
            )
        except ValueError as error:
            failures.append(f"{endpoint} aggregate gradient无效: {error}")
            continue
        aggregate_gradients[endpoint] = aggregate

        losses: dict[str, tuple[float, ...]] = {
            arm: tuple(
                float(responses[(endpoint, state_id, arm)].action_loss)
                for state_id in EXPECTED_STATE_IDS
            )
            for arm in RESPONSE_ARMS
        }
        baseline_mean = _mean(losses["baseline"])
        dense_mean = _mean(losses["dense"])
        support_mean = _mean(losses["support"])
        per_state_advantages = tuple(
            support_loss - dense_loss
            for support_loss, dense_loss in zip(
                losses["support"],
                losses["dense"],
                strict=True,
            )
        )
        exact_equal_count = sum(
            np.float32(support_loss).tobytes()
            == np.float32(dense_loss).tobytes()
            for support_loss, dense_loss in zip(
                losses["support"],
                losses["dense"],
                strict=True,
            )
        )
        metrics.append(
            EndpointDerivedMetrics(
                endpoint=endpoint,
                zero_retentions=zero_retentions,
                terminal_retentions=terminal_retentions,
                retention_differences=retention_differences,
                zero_retention_mean=_mean(zero_retentions),
                zero_retention_median=_median(zero_retentions),
                terminal_retention_mean=_mean(terminal_retentions),
                terminal_retention_median=_median(terminal_retentions),
                negative_retention_difference_count=sum(
                    difference < 0.0
                    for difference in retention_differences
                ),
                zero_aggregate_retention=zero_aggregate_retention,
                terminal_aggregate_retention=terminal_aggregate_retention,
                baseline_mean_action_loss=baseline_mean,
                dense_mean_action_loss=dense_mean,
                support_mean_action_loss=support_mean,
                dense_improvement=baseline_mean - dense_mean,
                support_improvement=baseline_mean - support_mean,
                dense_advantage=support_mean - dense_mean,
                per_state_dense_advantages=per_state_advantages,
                dense_better_state_count=sum(
                    advantage > 0.0 for advantage in per_state_advantages
                ),
                support_better_state_count=sum(
                    advantage < 0.0 for advantage in per_state_advantages
                ),
                exact_equal_state_count=exact_equal_count,
            )
        )
    if failures:
        return _invalid_evidence_decision(inventory, failures)

    try:
        per_state_cosines = tuple(
            _cosine_similarity(
                gradients[(ENDPOINT_NAMES[0], state_id)].dense_surface_gradient,
                gradients[(ENDPOINT_NAMES[1], state_id)].dense_surface_gradient,
            )
            for state_id in EXPECTED_STATE_IDS
        )
        aggregate_cosine = _cosine_similarity(
            aggregate_gradients[ENDPOINT_NAMES[0]],
            aggregate_gradients[ENDPOINT_NAMES[1]],
        )
    except ValueError as error:
        failures.append(f"跨endpoint gradient cosine无效: {error}")
        return _invalid_evidence_decision(inventory, failures)
    return EndpointEvidenceDecision(
        audit_valid=True,
        failures=(),
        inventory=inventory,
        endpoint_metrics=tuple(metrics),
        per_state_endpoint_gradient_cosines=per_state_cosines,
        aggregate_endpoint_gradient_cosine=aggregate_cosine,
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
        raise TerminalEndpointActionAuditError(f"NPZ字段{name}必须为scalar")
    return value.item()


def _validate_gradient_evidence(evidence: EndpointGradientEvidence) -> None:
    _validate_endpoint_state(evidence.endpoint, evidence.state_id)
    if not _is_sha256(evidence.state_fingerprint) or not _is_sha256(
        evidence.realized_endpoint_sha256
    ):
        raise TerminalEndpointActionAuditError("gradient provenance SHA无效")
    raw_gradient = np.asarray(evidence.dense_surface_gradient)
    if (
        raw_gradient.dtype != np.float32
        or raw_gradient.ndim != 2
        or raw_gradient.shape[0] <= 0
        or raw_gradient.shape[1] != 3
        or not bool(np.isfinite(raw_gradient).all())
        or not bool(np.any(raw_gradient != 0.0))
    ):
        raise TerminalEndpointActionAuditError(
            "dense_surface_gradient必须为finite nonzero"
        )


def write_endpoint_gradient_npz(
    evidence: EndpointGradientEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存一个无pickle gradient case并返回文件SHA-256。"""

    _validate_gradient_evidence(evidence)
    path = Path(output_path)
    if path.suffix != ".npz":
        raise TerminalEndpointActionAuditError(
            "gradient output_path必须使用.npz后缀"
        )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(ENDPOINT_GRADIENT_SCHEMA_VERSION),
        endpoint=np.asarray(evidence.endpoint),
        state_id=np.asarray(evidence.state_id, dtype=np.int64),
        state_fingerprint=np.asarray(evidence.state_fingerprint),
        realized_endpoint_sha256=np.asarray(
            evidence.realized_endpoint_sha256
        ),
        dense_surface_gradient=evidence.dense_surface_gradient,
    )
    return _file_sha256(path)


def load_endpoint_gradient_npz(path: str | Path) -> EndpointGradientEvidence:
    """无pickle加载gradient case并重新执行schema验证。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(_scalar(archive, "schema_version")) != (
            ENDPOINT_GRADIENT_SCHEMA_VERSION
        ):
            raise TerminalEndpointActionAuditError("gradient NPZ schema不匹配")
        evidence = EndpointGradientEvidence(
            endpoint=str(_scalar(archive, "endpoint")),
            state_id=int(_scalar(archive, "state_id")),
            state_fingerprint=str(_scalar(archive, "state_fingerprint")),
            realized_endpoint_sha256=str(
                _scalar(archive, "realized_endpoint_sha256")
            ),
            dense_surface_gradient=archive["dense_surface_gradient"].copy(),
        )
    _validate_gradient_evidence(evidence)
    return evidence


def _validate_response_evidence(evidence: EndpointResponseEvidence) -> None:
    _validate_endpoint_state(evidence.endpoint, evidence.state_id)
    if evidence.arm not in RESPONSE_ARMS:
        raise TerminalEndpointActionAuditError("response arm无效")
    if not _is_sha256(evidence.state_fingerprint) or not _is_sha256(
        evidence.surface_delta_sha256
    ):
        raise TerminalEndpointActionAuditError("response provenance SHA无效")
    clean_tokens = np.asarray(evidence.clean_action_token_ids)
    clean_classes = np.asarray(evidence.clean_classes)
    generated_tokens = np.asarray(evidence.generated_token_ids)
    generated_classes = np.asarray(evidence.generated_classes)
    generation = np.asarray(evidence.generation_logits)
    teacher = np.asarray(evidence.teacher_logits)
    action_dim = clean_classes.size
    if (
        action_dim <= 0
        or clean_tokens.dtype != np.int64
        or clean_classes.dtype != np.int64
        or clean_tokens.shape != (action_dim,)
        or generated_tokens.dtype != np.int64
        or generated_classes.dtype != np.int64
        or generated_tokens.shape != (action_dim,)
        or generated_classes.shape != (action_dim,)
        or generation.dtype != np.float32
        or teacher.dtype != np.float32
        or generation.ndim != 2
        or generation.shape != teacher.shape
        or generation.shape[0] != action_dim
        or generation.shape[1] != NUM_ACTION_BINS
        or not bool(np.isfinite(generation).all())
        or not bool(np.isfinite(teacher).all())
    ):
        raise TerminalEndpointActionAuditError(
            "response token/logits shape、dtype或数值无效"
        )
    margins = np.asarray(evidence.margins)
    hinges = np.asarray(evidence.hinge_values)
    if (
        margins.dtype != np.float32
        or hinges.dtype != np.float32
        or margins.shape != (action_dim,)
        or hinges.shape != (action_dim,)
        or not math.isfinite(float(evidence.action_loss))
    ):
        raise TerminalEndpointActionAuditError(
            "response margin/hinge/loss shape、dtype或数值无效"
        )
    if not np.array_equal(clean_tokens, clean_classes + ACTION_TOKEN_START):
        raise TerminalEndpointActionAuditError("clean token/class映射无效")
    if not np.array_equal(
        generated_tokens,
        generated_classes + ACTION_TOKEN_START,
    ):
        raise TerminalEndpointActionAuditError("generated token/class映射无效")
    if bool(
        np.any(generated_classes < 0)
        or np.any(generated_classes >= generation.shape[1])
    ):
        raise TerminalEndpointActionAuditError("generated class越界")
    selected = generation[np.arange(action_dim), generated_classes]
    if not np.array_equal(selected, np.max(generation, axis=1)):
        raise TerminalEndpointActionAuditError(
            "generated class不属于generation logits精确argmax"
        )
    objective = compute_action_hinge(teacher, clean_classes)
    if not np.array_equal(np.asarray(evidence.margins), objective.margins):
        raise TerminalEndpointActionAuditError("response margin不能从logits复算")
    if not np.array_equal(
        np.asarray(evidence.hinge_values),
        objective.hinge_values,
    ):
        raise TerminalEndpointActionAuditError("response hinge不能从logits复算")
    if np.float32(evidence.action_loss).tobytes() != np.float32(
        objective.action_loss
    ).tobytes():
        raise TerminalEndpointActionAuditError("response action loss不能从logits复算")
    decoded = np.asarray(evidence.decoded_action)
    if (
        decoded.ndim != 1
        or decoded.size != action_dim
        or not np.issubdtype(decoded.dtype, np.floating)
        or not bool(np.isfinite(decoded).all())
    ):
        raise TerminalEndpointActionAuditError("decoded_action无效")
    rgb = np.asarray(evidence.effective_view_rgb)
    bits = np.asarray(evidence.processor_bf16_bits)
    if (
        rgb.dtype != np.uint8
        or rgb.shape != (224, 224, 3)
        or bits.dtype != np.uint16
        or bits.shape != (1, 6, 224, 224)
    ):
        raise TerminalEndpointActionAuditError(
            "effective RGB/processor BF16 bits shape或dtype无效"
        )


def write_endpoint_response_npz(
    evidence: EndpointResponseEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存一个无pickle response record并返回文件SHA-256。"""

    _validate_response_evidence(evidence)
    path = Path(output_path)
    if path.suffix != ".npz":
        raise TerminalEndpointActionAuditError(
            "response output_path必须使用.npz后缀"
        )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(ENDPOINT_RESPONSE_SCHEMA_VERSION),
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


def load_endpoint_response_npz(path: str | Path) -> EndpointResponseEvidence:
    """无pickle加载response record并独立复算Action objective。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(_scalar(archive, "schema_version")) != (
            ENDPOINT_RESPONSE_SCHEMA_VERSION
        ):
            raise TerminalEndpointActionAuditError("response NPZ schema不匹配")
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
    _validate_response_evidence(evidence)
    return evidence
