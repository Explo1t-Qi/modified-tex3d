r"""Fixed Support 参数量与 source Action 梯度信号的纯 CPU 只读审计。

本模块只消费已经通过 Gate 6i 的 ``(endpoint,state)`` raw dense Surface
gradient、Production Support 与 endpoint step artifact，不加载 OpenVLA、CUDA、
LIBERO、renderer 或训练器，也不改变训练行为。它区分三类容易混淆的量：

1. 完整 dense 向量的总范数；
2. 只在实际可训练 Support 坐标上计算的 per-coordinate mean-absolute/RMS；
3. 按正式 ``mask -> surface-normalize -> project -> cap`` 规则得到的更新。

Support 统计的分母只包含 ``3 * N_support`` 个可训练 RGB 坐标；不会把 Support
外补零坐标放进 mean/RMS，从而避免把参数数量的机械变化误写成每参数信号变弱。
报告是描述性 evidence，不产生机制 Gate、ASR 因果判定或新训练许可。
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

from .production_support import load_production_support_artifact
from .terminal_endpoint_action_audit import (
    ENDPOINT_NAMES,
    EXPECTED_STATE_IDS,
    EndpointGradientEvidence,
    aggregate_state_gradients,
    compute_gradient_retention,
    execute_surface_counterfactual_step,
    load_endpoint_gradient_npz,
)
from .terminal_endpoint_action_evidence import (
    evaluate_terminal_endpoint_bundle,
    file_sha256,
    load_endpoint_step_npz,
)


Float32Array: TypeAlias = NDArray[np.float32]
GRADIENT_SIGNAL_SCHEMA_VERSION: Final[str] = (
    "openvla-action-gradient-signal-report-v1"
)


class ActionGradientSignalAuditError(ValueError):
    """输入bundle、梯度inventory或输出路径违反只读审计合同。"""


@dataclass(frozen=True)
class GradientSignalCase:
    """一个固定endpoint和state上的raw dense Surface gradient。"""

    endpoint: str
    state_id: int
    state_fingerprint: str
    gradient_artifact_sha256: str
    # float32 [num_geometry_vertices,3]，mask/normalize/project之前的梯度。
    dense_surface_gradient: Float32Array
    # float32 [num_geometry_vertices,3]，该endpoint的已实现Surface Delta。
    realized_endpoint: Float32Array


@dataclass(frozen=True)
class VectorSignalStats:
    """一个非空向量子集的普通Euclidean及per-coordinate统计。"""

    vertex_count: int
    coordinate_count: int
    l2_norm: float
    linf_norm: float
    mean_absolute: float
    rms: float


@dataclass(frozen=True)
class UpdateNorms:
    """一个Surface更新的总L2和逐坐标L∞。"""

    l2_norm: float
    linf_norm: float


@dataclass(frozen=True)
class GradientSignalMetrics:
    """同一起点、同一raw gradient下Dense/Support的配对信号。"""

    dense: VectorSignalStats
    support: VectorSignalStats
    outside_support: VectorSignalStats
    gradient_energy_retention: float
    dense_raw_update: UpdateNorms
    support_raw_update: UpdateNorms
    dense_executed_update: UpdateNorms
    support_executed_update: UpdateNorms


@dataclass(frozen=True)
class GradientSignalRow(GradientSignalMetrics):
    """一个endpoint/state的描述量及身份绑定。"""

    endpoint: str
    state_id: int
    state_fingerprint: str
    gradient_artifact_sha256: str


@dataclass(frozen=True)
class MetricDistribution:
    """一个逐state标量的mean/median及范围。"""

    name: str
    mean: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class GradientSignalAggregate(GradientSignalMetrics):
    """正式float32十state算术平均梯度上的配对描述量。"""


@dataclass(frozen=True)
class EndpointGradientSignalSummary:
    """一个endpoint的逐state分布与aggregate-gradient结果。"""

    endpoint: str
    state_count: int
    per_state_distributions: tuple[MetricDistribution, ...]
    aggregate: GradientSignalAggregate


@dataclass(frozen=True)
class GradientSignalDecision:
    """内存case的工程验收与描述性结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    rows: tuple[GradientSignalRow, ...]
    summaries: tuple[EndpointGradientSignalSummary, ...]


@dataclass(frozen=True)
class GradientSignalSource:
    """报告绑定的Gate 6i与Production Support身份。"""

    endpoint_manifest_sha256: str
    endpoint_code_commit: str
    production_support_sha256: str
    support_mask_sha256: str


@dataclass(frozen=True)
class GradientSignalReport:
    """可JSON序列化的Action Gradient Signal只读报告。"""

    schema_version: str
    audit_valid: bool
    failures: tuple[str, ...]
    source: GradientSignalSource | None
    objective: str
    gradient_stage: str
    support_statistic_denominator: str
    gradient_aggregation: str
    update_rule: str
    training_or_rollout_run: bool
    scientific_boolean_gate: bool
    rows: tuple[GradientSignalRow, ...]
    summaries: tuple[EndpointGradientSignalSummary, ...]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _geometry(value: np.ndarray, *, name: str) -> Float32Array:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] != 3
        or not bool(np.isfinite(array).all())
    ):
        raise ActionGradientSignalAuditError(
            f"{name}必须为finite float32 [num_geometry_vertices,3]"
        )
    return array


def _vector_stats(value: np.ndarray) -> VectorSignalStats:
    array = _geometry(value, name="vector")
    flat = array.reshape(-1).astype(np.float64)
    absolute = np.abs(flat)
    return VectorSignalStats(
        vertex_count=int(array.shape[0]),
        coordinate_count=int(flat.size),
        l2_norm=float(np.linalg.norm(flat)),
        linf_norm=float(absolute.max()),
        mean_absolute=float(absolute.mean()),
        rms=float(np.sqrt(np.mean(flat**2, dtype=np.float64))),
    )


def _update_norms(value: np.ndarray) -> UpdateNorms:
    array = _geometry(value, name="update")
    flat = array.reshape(-1).astype(np.float64)
    return UpdateNorms(
        l2_norm=float(np.linalg.norm(flat)),
        linf_norm=float(np.max(np.abs(flat))),
    )


def _metrics(
    gradient: Float32Array,
    realized_endpoint: Float32Array,
    support_mask: NDArray[np.bool_],
    *,
    epsilon: float,
    surface_step: float,
) -> GradientSignalMetrics:
    dense_step = execute_surface_counterfactual_step(
        realized_endpoint,
        gradient,
        epsilon=epsilon,
        surface_step=surface_step,
    )
    support_step = execute_surface_counterfactual_step(
        realized_endpoint,
        gradient,
        epsilon=epsilon,
        surface_step=surface_step,
        support_mask=support_mask,
    )
    return GradientSignalMetrics(
        dense=_vector_stats(gradient),
        # 只统计真实可训练坐标，不把Support外的补零坐标纳入分母。
        support=_vector_stats(gradient[support_mask]),
        outside_support=_vector_stats(gradient[~support_mask]),
        gradient_energy_retention=compute_gradient_retention(
            gradient, support_mask
        ),
        dense_raw_update=_update_norms(
            dense_step.unconstrained_normalized_step
        ),
        support_raw_update=_update_norms(
            support_step.unconstrained_normalized_step
        ),
        dense_executed_update=_update_norms(dense_step.executed_step),
        support_executed_update=_update_norms(support_step.executed_step),
    )


def _metric_values(row: GradientSignalRow) -> Mapping[str, float]:
    return {
        "dense_l2_norm": row.dense.l2_norm,
        "dense_linf_norm": row.dense.linf_norm,
        "dense_mean_absolute": row.dense.mean_absolute,
        "dense_rms": row.dense.rms,
        "support_l2_norm": row.support.l2_norm,
        "support_linf_norm": row.support.linf_norm,
        "support_mean_absolute": row.support.mean_absolute,
        "support_rms": row.support.rms,
        "outside_support_rms": row.outside_support.rms,
        "gradient_energy_retention": row.gradient_energy_retention,
        "dense_raw_update_l2_norm": row.dense_raw_update.l2_norm,
        "dense_raw_update_linf_norm": row.dense_raw_update.linf_norm,
        "support_raw_update_l2_norm": row.support_raw_update.l2_norm,
        "support_raw_update_linf_norm": row.support_raw_update.linf_norm,
        "dense_executed_update_l2_norm": row.dense_executed_update.l2_norm,
        "dense_executed_update_linf_norm": row.dense_executed_update.linf_norm,
        "support_executed_update_l2_norm": (
            row.support_executed_update.l2_norm
        ),
        "support_executed_update_linf_norm": (
            row.support_executed_update.linf_norm
        ),
    }


def _distribution(name: str, values: Sequence[float]) -> MetricDistribution:
    array = np.asarray(tuple(values), dtype=np.float64)
    return MetricDistribution(
        name=name,
        mean=float(array.mean()),
        median=float(np.median(array)),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def _invalid_decision(failures: Sequence[str]) -> GradientSignalDecision:
    return GradientSignalDecision(False, tuple(failures), (), ())


def analyze_gradient_signal_cases(
    cases: Sequence[GradientSignalCase],
    *,
    support_mask: np.ndarray,
    epsilon: float,
    surface_step: float,
    expected_state_ids: Sequence[int] = EXPECTED_STATE_IDS,
    expected_endpoints: Sequence[str] = ENDPOINT_NAMES,
) -> GradientSignalDecision:
    """从raw梯度复算逐state与aggregate的信号和正式单步统计。"""

    states = tuple(expected_state_ids)
    endpoints = tuple(expected_endpoints)
    if (
        not states
        or len(set(states)) != len(states)
        or any(
            type(state) is not int or state not in EXPECTED_STATE_IDS
            for state in states
        )
        or not endpoints
        or len(set(endpoints)) != len(endpoints)
        or any(endpoint not in ENDPOINT_NAMES for endpoint in endpoints)
    ):
        return _invalid_decision(("expected endpoint/state inventory无效",))
    if (
        not math.isfinite(float(epsilon))
        or not math.isfinite(float(surface_step))
        or epsilon <= 0.0
        or surface_step <= 0.0
        or surface_step > epsilon
    ):
        return _invalid_decision(("epsilon/surface_step无效",))

    mask = np.asarray(support_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        return _invalid_decision(("support_mask必须为bool [N_v]",))
    if not bool(mask.any()) or bool(mask.all()):
        return _invalid_decision(("Support与outside-Support都必须非空",))

    failures: list[str] = []
    indexed: dict[tuple[str, int], GradientSignalCase] = {}
    num_vertices: int | None = None
    endpoint_realized: dict[str, Float32Array] = {}
    for case in cases:
        key = (case.endpoint, case.state_id)
        if key in indexed:
            failures.append(f"inventory重复key: {key}")
        indexed[key] = case
        if case.endpoint not in endpoints or case.state_id not in states:
            failures.append(f"非预期case: {key}")
        if not _is_sha256(case.state_fingerprint):
            failures.append(f"{key}: state_fingerprint必须为SHA-256")
        if not _is_sha256(case.gradient_artifact_sha256):
            failures.append(f"{key}: gradient artifact必须为SHA-256")
        try:
            gradient = _geometry(
                case.dense_surface_gradient, name="dense_surface_gradient"
            )
            endpoint_value = _geometry(
                case.realized_endpoint, name="realized_endpoint"
            )
            if gradient.shape != endpoint_value.shape:
                failures.append(f"{key}: gradient/endpoint shape不一致")
            if num_vertices is None:
                num_vertices = int(gradient.shape[0])
            elif gradient.shape[0] != num_vertices:
                failures.append(f"{key}: num_geometry_vertices漂移")
            previous = endpoint_realized.get(case.endpoint)
            if previous is None:
                endpoint_realized[case.endpoint] = endpoint_value
            elif not np.array_equal(previous, endpoint_value):
                failures.append(f"{key}: 同endpoint realized Surface漂移")
        except ActionGradientSignalAuditError as error:
            failures.append(f"{key}: {error}")

    expected_keys = {
        (endpoint, state) for endpoint in endpoints for state in states
    }
    if len(cases) != len(expected_keys) or set(indexed) != expected_keys:
        failures.append(
            f"inventory必须恰好包含{len(expected_keys)}个唯一case"
        )
    if num_vertices is not None and mask.shape != (num_vertices,):
        failures.append("support_mask shape与geometry不一致")
    if failures:
        return _invalid_decision(failures)

    rows: list[GradientSignalRow] = []
    summaries: list[EndpointGradientSignalSummary] = []
    for endpoint in endpoints:
        endpoint_rows: list[GradientSignalRow] = []
        gradients: list[Float32Array] = []
        for state_id in states:
            case = indexed[(endpoint, state_id)]
            gradient = np.asarray(case.dense_surface_gradient, dtype=np.float32)
            gradients.append(gradient)
            metrics = _metrics(
                gradient,
                np.asarray(case.realized_endpoint, dtype=np.float32),
                mask,
                epsilon=float(epsilon),
                surface_step=float(surface_step),
            )
            row = GradientSignalRow(
                **metrics.__dict__,
                endpoint=endpoint,
                state_id=state_id,
                state_fingerprint=case.state_fingerprint,
                gradient_artifact_sha256=case.gradient_artifact_sha256,
            )
            rows.append(row)
            endpoint_rows.append(row)

        aggregate_gradient = aggregate_state_gradients(
            np.stack(gradients, axis=0)
        )
        aggregate_metrics = _metrics(
            aggregate_gradient,
            endpoint_realized[endpoint],
            mask,
            epsilon=float(epsilon),
            surface_step=float(surface_step),
        )
        metric_names = tuple(_metric_values(endpoint_rows[0]))
        distributions = tuple(
            _distribution(
                name,
                tuple(_metric_values(row)[name] for row in endpoint_rows),
            )
            for name in metric_names
        )
        summaries.append(
            EndpointGradientSignalSummary(
                endpoint=endpoint,
                state_count=len(states),
                per_state_distributions=distributions,
                aggregate=GradientSignalAggregate(**aggregate_metrics.__dict__),
            )
        )
    return GradientSignalDecision(True, (), tuple(rows), tuple(summaries))


def _invalid_report(failures: Sequence[str]) -> GradientSignalReport:
    return GradientSignalReport(
        schema_version=GRADIENT_SIGNAL_SCHEMA_VERSION,
        audit_valid=False,
        failures=tuple(failures),
        source=None,
        objective="untargeted_clean_action_margin_hinge",
        gradient_stage="raw_dense_surface_before_mask_normalization_projection",
        support_statistic_denominator="trainable_support_rgb_coordinates_only",
        gradient_aggregation="float32_arithmetic_mean_states_0_9",
        update_rule="mask_then_surface_normalize_project_and_cap",
        training_or_rollout_run=False,
        scientific_boolean_gate=False,
        rows=(),
        summaries=(),
    )


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActionGradientSignalAuditError("manifest必须为JSON object")
    return value


def _safe_relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ActionGradientSignalAuditError("artifact relative path无效")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ActionGradientSignalAuditError(
            "artifact path逃逸manifest目录"
        ) from error
    return candidate


def analyze_action_gradient_signal_bundle(
    manifest_path: str | Path,
    *,
    production_support_path: str | Path,
    source_dense_seed_metrics_path: str | Path,
) -> GradientSignalReport:
    """复核Gate 6i后，从其raw evidence生成参数量/梯度信号报告。"""

    path = Path(manifest_path).resolve()
    support_path = Path(production_support_path).resolve()
    dense_seed_path = Path(source_dense_seed_metrics_path).resolve()
    bundle = evaluate_terminal_endpoint_bundle(
        path,
        production_support_path=support_path,
        source_dense_seed_metrics_path=dense_seed_path,
    )
    if not bundle.audit_valid:
        return _invalid_report(
            tuple(f"Gate 6i bundle无效: {value}" for value in bundle.failures)
        )
    try:
        manifest = _load_json_object(path)
        provenance = manifest.get("provenance")
        configuration = manifest.get("configuration")
        if not isinstance(provenance, dict) or not isinstance(
            configuration, dict
        ):
            raise ActionGradientSignalAuditError(
                "Gate 6i provenance/configuration无效"
            )
        support_digest = file_sha256(support_path)
        if support_digest != provenance.get("production_support_sha256"):
            raise ActionGradientSignalAuditError(
                "Production Support SHA与Gate 6i绑定不一致"
            )
        support = load_production_support_artifact(support_path)
        raw_steps = manifest.get("endpoint_steps")
        if not isinstance(raw_steps, list):
            raise ActionGradientSignalAuditError("endpoint_steps必须为list")
        realized: dict[str, Float32Array] = {}
        for record in raw_steps:
            if not isinstance(record, dict):
                raise ActionGradientSignalAuditError(
                    "endpoint step record必须为object"
                )
            artifact = _safe_relative_path(
                path.parent, record.get("npz_relative_path")
            )
            digest = record.get("npz_sha256")
            if not _is_sha256(digest) or file_sha256(artifact) != digest:
                raise ActionGradientSignalAuditError(
                    "endpoint step artifact SHA漂移"
                )
            evidence = load_endpoint_step_npz(artifact)
            if evidence.support_mask_sha256 != support.support_mask_sha256:
                raise ActionGradientSignalAuditError(
                    "endpoint step Support mask SHA漂移"
                )
            if not np.array_equal(
                evidence.dense_step.realized_endpoint,
                evidence.support_step.realized_endpoint,
            ):
                raise ActionGradientSignalAuditError(
                    "Dense/Support realized endpoint不一致"
                )
            realized[evidence.endpoint] = (
                evidence.dense_step.realized_endpoint.copy()
            )
        if set(realized) != set(ENDPOINT_NAMES):
            raise ActionGradientSignalAuditError(
                "endpoint step inventory必须包含两个正式endpoint"
            )

        raw_gradients = manifest.get("gradient_cases")
        if not isinstance(raw_gradients, list):
            raise ActionGradientSignalAuditError("gradient_cases必须为list")
        cases: list[GradientSignalCase] = []
        for record in raw_gradients:
            if not isinstance(record, dict):
                raise ActionGradientSignalAuditError(
                    "gradient record必须为object"
                )
            artifact = _safe_relative_path(
                path.parent, record.get("npz_relative_path")
            )
            digest = record.get("npz_sha256")
            if not _is_sha256(digest) or file_sha256(artifact) != digest:
                raise ActionGradientSignalAuditError(
                    "gradient artifact SHA漂移"
                )
            evidence: EndpointGradientEvidence = load_endpoint_gradient_npz(
                artifact
            )
            if evidence.endpoint not in realized:
                raise ActionGradientSignalAuditError("gradient endpoint无step")
            cases.append(
                GradientSignalCase(
                    endpoint=evidence.endpoint,
                    state_id=evidence.state_id,
                    state_fingerprint=evidence.state_fingerprint,
                    gradient_artifact_sha256=str(digest),
                    dense_surface_gradient=evidence.dense_surface_gradient,
                    realized_endpoint=realized[evidence.endpoint],
                )
            )
        decision = analyze_gradient_signal_cases(
            cases,
            support_mask=support.support_mask,
            epsilon=float(configuration.get("epsilon")),
            surface_step=float(configuration.get("surface_step")),
        )
        source = GradientSignalSource(
            endpoint_manifest_sha256=file_sha256(path),
            endpoint_code_commit=str(manifest.get("code_commit")),
            production_support_sha256=support_digest,
            support_mask_sha256=support.support_mask_sha256,
        )
        return GradientSignalReport(
            schema_version=GRADIENT_SIGNAL_SCHEMA_VERSION,
            audit_valid=decision.audit_valid,
            failures=decision.failures,
            source=source,
            objective=str(configuration.get("objective")),
            gradient_stage=(
                "raw_dense_surface_before_mask_normalization_projection"
            ),
            support_statistic_denominator=(
                "trainable_support_rgb_coordinates_only"
            ),
            gradient_aggregation="float32_arithmetic_mean_states_0_9",
            update_rule="mask_then_surface_normalize_project_and_cap",
            training_or_rollout_run=False,
            scientific_boolean_gate=False,
            rows=decision.rows,
            summaries=decision.summaries,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return _invalid_report((str(error),))


def gradient_signal_report_dict(
    report: GradientSignalReport,
) -> dict[str, object]:
    """转换为稳定JSON object。"""

    value = asdict(report)
    assert isinstance(value, dict)
    return value


def write_gradient_signal_report(
    report: GradientSignalReport,
    *,
    output_path: str | Path,
) -> str:
    """不覆盖既有文件，原子写入只读报告并返回SHA-256。"""

    path = Path(output_path)
    if path.suffix != ".json":
        raise ActionGradientSignalAuditError("output_path必须为.json")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f"{path.stem}_candidate.json")
    if candidate.exists():
        raise FileExistsError(candidate)
    try:
        candidate.write_text(
            json.dumps(
                gradient_signal_report_dict(report),
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
