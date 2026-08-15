"""Gate 6i endpoint 单步反事实的无 pickle artifact 合同。

本模块只处理两个正式终态上已经产生的 Surface 数组，不加载 OpenVLA、
renderer、LIBERO 或训练代码。GPU runner 必须复用正式
``surface_normalized_step_`` 产生 authoritative raw step；这里的 NumPy 实现
用于在 WSL 独立复算更新顺序、数组关系、SHA 与描述性范数。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from .action_objective_audit import ACTION_OBJECTIVE_NAME
from .dense_seed_audit import (
    load_dense_seed_gradient_evidence,
    summarize_dense_seed_gradient_evidence,
)
from .production_support import load_production_support_artifact
from .terminal_endpoint_action_audit import (
    ENDPOINT_NAMES,
    EXPECTED_STATE_IDS,
    RESPONSE_ARMS,
    EndpointEvidenceDecision,
    EndpointGradientEvidence,
    EndpointResponseEvidence,
    Float32Array,
    SurfaceCounterfactualStep,
    SurfaceCounterfactualStepStats,
    TerminalEndpointActionAuditError,
    aggregate_state_gradients,
    evaluate_endpoint_evidence,
    execute_surface_counterfactual_step,
    load_endpoint_gradient_npz,
    load_endpoint_response_npz,
)
from .terminal_gain_counterfactual_audit import (
    processor_bf16_bits_from_effective_rgb,
)


ENDPOINT_STEP_SCHEMA_VERSION: Final[str] = "openvla-terminal-endpoint-step-v1"
ENDPOINT_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-endpoint-action-audit-bundle-v1"
)
FORMAL_SURFACE_EPSILON: Final[float] = 128.0 / 255.0
FORMAL_SURFACE_STEP: Final[float] = 2.0 / 255.0
STEP_ARMS: Final[tuple[str, ...]] = ("dense", "support")
NUMERIC_TOLERANCE: Final[float] = 1e-6
_STEP_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "realized_endpoint",
    "masked_gradient",
    "unconstrained_normalized_step",
    "projected_step",
    "executed_step",
    "projection_residual",
)
_STEP_STAT_NAMES: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(SurfaceCounterfactualStepStats)
)
ENDPOINT_STEP_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "endpoint",
        "epsilon",
        "surface_step",
        "support_mask_sha256",
        "realized_endpoint_sha256",
        "dense_surface_delta_sha256",
        "support_surface_delta_sha256",
    }
    | {
        f"{arm}_{name}"
        for arm in STEP_ARMS
        for name in (*_STEP_ARRAY_NAMES, *_STEP_STAT_NAMES)
    }
)


class TerminalEndpointActionEvidenceError(ValueError):
    """Gate 6i endpoint step artifact 违反冻结合同。"""


@dataclass(frozen=True)
class EndpointStepEvidence:
    """一个终态的共同起点及 Dense/Support authoritative raw step。"""

    endpoint: str
    epsilon: float
    surface_step: float
    support_mask_sha256: str
    realized_endpoint_sha256: str
    dense_surface_delta_sha256: str
    support_surface_delta_sha256: str
    dense_step: SurfaceCounterfactualStep
    support_step: SurfaceCounterfactualStep


@dataclass(frozen=True)
class EndpointArmStepMetrics:
    """从 raw step 数组复算的描述性几何量，不是科学 pass/fail。"""

    arm: str
    normalized_l2: float
    normalized_linf: float
    projected_l2: float
    projected_linf: float
    executed_l2: float
    executed_linf: float
    projection_residual_l2: float
    projection_residual_linf: float
    gradient_step_inner_product: float
    descent_alignment_cosine: float


@dataclass(frozen=True)
class EndpointStepDecision:
    """一个 endpoint step artifact 的独立 CPU 验收结果。"""

    audit_valid: bool
    failures: tuple[str, ...]
    endpoint: str
    arm_metrics: tuple[EndpointArmStepMetrics, ...]


def array_sha256(array: np.ndarray) -> str:
    """按 contiguous dtype、shape 和 bytes 计算稳定数组 SHA-256。"""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _geometry_array(value: np.ndarray, *, name: str) -> Float32Array:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] != 3
        or not bool(np.isfinite(array).all())
    ):
        raise TerminalEndpointActionEvidenceError(
            f"{name}必须为finite float32 [num_geometry_vertices,3]"
        )
    return array


def _validate_scalar_configuration(
    endpoint: object,
    epsilon: object,
    surface_step: object,
) -> None:
    if endpoint not in ENDPOINT_NAMES:
        raise TerminalEndpointActionEvidenceError("endpoint无效")
    if (
        not isinstance(epsilon, (int, float))
        or not isinstance(surface_step, (int, float))
        or not math.isfinite(float(epsilon))
        or not math.isfinite(float(surface_step))
        or float(epsilon) <= 0.0
        or float(surface_step) <= 0.0
        or float(surface_step) > float(epsilon)
    ):
        raise TerminalEndpointActionEvidenceError(
            "epsilon/surface_step必须为finite正数且step不大于预算"
        )


def _validate_step_structure(
    step: SurfaceCounterfactualStep,
    *,
    name: str,
    expected_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    arrays = {
        field_name: _geometry_array(
            np.asarray(getattr(step, field_name)),
            name=f"{name}.{field_name}",
        )
        for field_name in _STEP_ARRAY_NAMES
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise TerminalEndpointActionEvidenceError(
            f"{name}全部Surface数组shape必须一致"
        )
    shape = next(iter(shapes))
    if expected_shape is not None and shape != expected_shape:
        raise TerminalEndpointActionEvidenceError(
            f"{name}与另一个arm的geometry shape不一致"
        )
    for stat_name in _STEP_STAT_NAMES:
        value = getattr(step.stats, stat_name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TerminalEndpointActionEvidenceError(
                f"{name}.stats.{stat_name}必须为finite scalar"
            )
    return shape


def _validate_evidence_structure(evidence: EndpointStepEvidence) -> None:
    _validate_scalar_configuration(
        evidence.endpoint,
        evidence.epsilon,
        evidence.surface_step,
    )
    for name in (
        "support_mask_sha256",
        "realized_endpoint_sha256",
        "dense_surface_delta_sha256",
        "support_surface_delta_sha256",
    ):
        if not _is_sha256(getattr(evidence, name)):
            raise TerminalEndpointActionEvidenceError(f"{name}无效")
    shape = _validate_step_structure(
        evidence.dense_step,
        name="dense_step",
        expected_shape=None,
    )
    _validate_step_structure(
        evidence.support_step,
        name="support_step",
        expected_shape=shape,
    )


def build_endpoint_step_evidence(
    *,
    endpoint: str,
    endpoint_surface_delta: np.ndarray,
    aggregate_gradient: np.ndarray,
    support_mask: np.ndarray,
    epsilon: float,
    surface_step: float,
) -> EndpointStepEvidence:
    """用冻结的 NumPy 规则构造可用于测试/CPU复算的 endpoint step。"""

    mask = np.asarray(support_mask)
    if mask.dtype != np.bool_:
        raise TerminalEndpointActionEvidenceError("support_mask必须为bool")
    dense_step = execute_surface_counterfactual_step(
        endpoint_surface_delta,
        aggregate_gradient,
        epsilon=epsilon,
        surface_step=surface_step,
    )
    support_step = execute_surface_counterfactual_step(
        endpoint_surface_delta,
        aggregate_gradient,
        support_mask=mask,
        epsilon=epsilon,
        surface_step=surface_step,
    )
    realized = dense_step.realized_endpoint
    dense_after = np.ascontiguousarray(
        realized + dense_step.executed_step,
        dtype=np.float32,
    )
    support_after = np.ascontiguousarray(
        realized + support_step.executed_step,
        dtype=np.float32,
    )
    evidence = EndpointStepEvidence(
        endpoint=endpoint,
        epsilon=float(epsilon),
        surface_step=float(surface_step),
        support_mask_sha256=array_sha256(mask),
        realized_endpoint_sha256=array_sha256(realized),
        dense_surface_delta_sha256=array_sha256(dense_after),
        support_surface_delta_sha256=array_sha256(support_after),
        dense_step=dense_step,
        support_step=support_step,
    )
    _validate_evidence_structure(evidence)
    return evidence


def _array_matches(observed: np.ndarray, expected: np.ndarray) -> bool:
    return bool(
        np.allclose(
            observed,
            expected,
            rtol=0.0,
            atol=NUMERIC_TOLERANCE,
        )
    )


def _step_metrics(
    arm: str,
    step: SurfaceCounterfactualStep,
) -> EndpointArmStepMetrics:
    def norms(value: np.ndarray) -> tuple[float, float]:
        flattened = np.asarray(value, dtype=np.float32).reshape(-1)
        as_float64 = flattened.astype(np.float64)
        return (
            float(np.linalg.norm(as_float64)),
            float(np.max(np.abs(as_float64))),
        )

    normalized_l2, normalized_linf = norms(
        step.unconstrained_normalized_step
    )
    projected_l2, projected_linf = norms(step.projected_step)
    executed_l2, executed_linf = norms(step.executed_step)
    residual_l2, residual_linf = norms(step.projection_residual)
    gradient = step.masked_gradient.reshape(-1).astype(np.float64)
    executed = step.executed_step.reshape(-1).astype(np.float64)
    inner_product = float(np.dot(gradient, executed))
    denominator = float(np.linalg.norm(gradient) * np.linalg.norm(executed))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise TerminalEndpointActionEvidenceError(
            f"{arm} gradient/executed step无法定义cosine"
        )
    descent_cosine = float(np.clip(-inner_product / denominator, -1.0, 1.0))
    return EndpointArmStepMetrics(
        arm=arm,
        normalized_l2=normalized_l2,
        normalized_linf=normalized_linf,
        projected_l2=projected_l2,
        projected_linf=projected_linf,
        executed_l2=executed_l2,
        executed_linf=executed_linf,
        projection_residual_l2=residual_l2,
        projection_residual_linf=residual_linf,
        gradient_step_inner_product=inner_product,
        descent_alignment_cosine=descent_cosine,
    )


def _compare_step(
    observed: SurfaceCounterfactualStep,
    expected: SurfaceCounterfactualStep,
    *,
    arm: str,
    failures: list[str],
) -> None:
    for name in _STEP_ARRAY_NAMES:
        if not _array_matches(
            np.asarray(getattr(observed, name)),
            np.asarray(getattr(expected, name)),
        ):
            failures.append(f"{arm} {name}不能由冻结更新顺序复算")
    for name in _STEP_STAT_NAMES:
        if not math.isclose(
            float(getattr(observed.stats, name)),
            float(getattr(expected.stats, name)),
            rel_tol=0.0,
            abs_tol=NUMERIC_TOLERANCE,
        ):
            failures.append(f"{arm} SurfaceStepStats.{name}不能复算")


def evaluate_endpoint_step_evidence(
    evidence: EndpointStepEvidence,
    *,
    aggregate_gradient: np.ndarray,
    support_mask: np.ndarray,
) -> EndpointStepDecision:
    """从 raw arrays 独立复算一个终态的 Dense/Support step。"""

    failures: list[str] = []
    try:
        _validate_evidence_structure(evidence)
        gradient = _geometry_array(
            np.asarray(aggregate_gradient),
            name="aggregate_gradient",
        )
        mask = np.asarray(support_mask)
        if mask.dtype != np.bool_ or mask.shape != (gradient.shape[0],):
            raise TerminalEndpointActionEvidenceError(
                "support_mask必须为bool [num_geometry_vertices]"
            )
    except (TypeError, ValueError) as error:
        return EndpointStepDecision(
            False,
            (f"endpoint step结构无效: {error}",),
            str(evidence.endpoint),
            (),
        )
    if evidence.support_mask_sha256 != array_sha256(mask):
        failures.append("support mask SHA不匹配")

    dense_realized = evidence.dense_step.realized_endpoint
    support_realized = evidence.support_step.realized_endpoint
    if not np.array_equal(dense_realized, support_realized):
        failures.append("Dense/Support没有从逐值相同realized endpoint开始")
    if evidence.realized_endpoint_sha256 != array_sha256(dense_realized):
        failures.append("realized endpoint SHA不能由raw array复算")

    try:
        expected_dense = execute_surface_counterfactual_step(
            dense_realized,
            gradient,
            epsilon=evidence.epsilon,
            surface_step=evidence.surface_step,
        )
        expected_support = execute_surface_counterfactual_step(
            support_realized,
            gradient,
            support_mask=mask,
            epsilon=evidence.epsilon,
            surface_step=evidence.surface_step,
        )
    except TerminalEndpointActionAuditError as error:
        failures.append(f"冻结更新无法复算: {error}")
        return EndpointStepDecision(False, tuple(failures), evidence.endpoint, ())
    _compare_step(
        evidence.dense_step,
        expected_dense,
        arm="dense",
        failures=failures,
    )
    _compare_step(
        evidence.support_step,
        expected_support,
        arm="support",
        failures=failures,
    )
    dense_after = np.ascontiguousarray(
        dense_realized + evidence.dense_step.executed_step,
        dtype=np.float32,
    )
    support_after = np.ascontiguousarray(
        support_realized + evidence.support_step.executed_step,
        dtype=np.float32,
    )
    if evidence.dense_surface_delta_sha256 != array_sha256(dense_after):
        failures.append("dense Surface Delta SHA不能由raw step复算")
    if evidence.support_surface_delta_sha256 != array_sha256(support_after):
        failures.append("support Surface Delta SHA不能由raw step复算")

    metrics: tuple[EndpointArmStepMetrics, ...] = ()
    if not failures:
        try:
            metrics = (
                _step_metrics("dense", evidence.dense_step),
                _step_metrics("support", evidence.support_step),
            )
        except ValueError as error:
            failures.append(str(error))
    return EndpointStepDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        endpoint=evidence.endpoint,
        arm_metrics=() if failures else metrics,
    )


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    if value.shape != ():
        raise TerminalEndpointActionEvidenceError(f"NPZ字段{name}必须为scalar")
    return value.item()


def _step_payload(
    prefix: str,
    step: SurfaceCounterfactualStep,
) -> dict[str, np.ndarray]:
    payload = {
        f"{prefix}_{name}": np.asarray(getattr(step, name))
        for name in _STEP_ARRAY_NAMES
    }
    payload.update(
        {
            f"{prefix}_{name}": np.asarray(
                getattr(step.stats, name),
                dtype=np.float64,
            )
            for name in _STEP_STAT_NAMES
        }
    )
    return payload


def write_endpoint_step_npz(
    evidence: EndpointStepEvidence,
    *,
    output_path: str | Path,
) -> str:
    """拒绝覆盖地保存一个 endpoint 的完整 raw step artifact。"""

    _validate_evidence_structure(evidence)
    path = Path(output_path)
    if path.suffix != ".npz":
        raise TerminalEndpointActionEvidenceError("step output_path必须为.npz")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ENDPOINT_STEP_SCHEMA_VERSION),
        "endpoint": np.asarray(evidence.endpoint),
        "epsilon": np.asarray(evidence.epsilon, dtype=np.float64),
        "surface_step": np.asarray(evidence.surface_step, dtype=np.float64),
        "support_mask_sha256": np.asarray(evidence.support_mask_sha256),
        "realized_endpoint_sha256": np.asarray(
            evidence.realized_endpoint_sha256
        ),
        "dense_surface_delta_sha256": np.asarray(
            evidence.dense_surface_delta_sha256
        ),
        "support_surface_delta_sha256": np.asarray(
            evidence.support_surface_delta_sha256
        ),
    }
    payload.update(_step_payload("dense", evidence.dense_step))
    payload.update(_step_payload("support", evidence.support_step))
    np.savez_compressed(path, **payload)
    return file_sha256(path)


def _load_step(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
) -> SurfaceCounterfactualStep:
    arrays = {
        name: archive[f"{prefix}_{name}"].copy()
        for name in _STEP_ARRAY_NAMES
    }
    stats = SurfaceCounterfactualStepStats(
        **{
            name: float(_scalar(archive, f"{prefix}_{name}"))
            for name in _STEP_STAT_NAMES
        }
    )
    return SurfaceCounterfactualStep(stats=stats, **arrays)


def load_endpoint_step_npz(path: str | Path) -> EndpointStepEvidence:
    """禁用 pickle 加载 endpoint step，并重新执行完整结构校验。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != ENDPOINT_STEP_ARTIFACT_KEYS:
            raise TerminalEndpointActionEvidenceError(
                "endpoint step NPZ keys与冻结schema不一致"
            )
        if str(_scalar(archive, "schema_version")) != (
            ENDPOINT_STEP_SCHEMA_VERSION
        ):
            raise TerminalEndpointActionEvidenceError(
                "endpoint step schema_version不匹配"
            )
        evidence = EndpointStepEvidence(
            endpoint=str(_scalar(archive, "endpoint")),
            epsilon=float(_scalar(archive, "epsilon")),
            surface_step=float(_scalar(archive, "surface_step")),
            support_mask_sha256=str(
                _scalar(archive, "support_mask_sha256")
            ),
            realized_endpoint_sha256=str(
                _scalar(archive, "realized_endpoint_sha256")
            ),
            dense_surface_delta_sha256=str(
                _scalar(archive, "dense_surface_delta_sha256")
            ),
            support_surface_delta_sha256=str(
                _scalar(archive, "support_surface_delta_sha256")
            ),
            dense_step=_load_step(archive, "dense"),
            support_step=_load_step(archive, "support"),
        )
    _validate_evidence_structure(evidence)
    return evidence


@dataclass(frozen=True)
class TerminalEndpointBundleDecision:
    """Gate 6i完整bundle的工程验收；不包含科学结论字段。"""

    audit_valid: bool
    failures: tuple[str, ...]
    gradient_case_count: int
    response_record_count: int
    endpoint_step_count: int
    evidence_decision: EndpointEvidenceDecision | None
    step_decisions: tuple[EndpointStepDecision, ...]


def frozen_bundle_configuration() -> dict[str, Any]:
    """返回Gate 6i不得由runner改写的正式配置。"""

    return {
        "objective": ACTION_OBJECTIVE_NAME,
        "epsilon": FORMAL_SURFACE_EPSILON,
        "surface_step": FORMAL_SURFACE_STEP,
        "expected_endpoints": list(ENDPOINT_NAMES),
        "expected_state_ids": list(EXPECTED_STATE_IDS),
        "expected_response_arms": list(RESPONSE_ARMS),
        "gradient_parameterization": "geometry_vertex_dense",
        "gradient_aggregation": "float32_arithmetic_mean_states_0_9",
        "retention": "ordinary_euclidean_gradient_energy",
    }


def json_sha256(value: Mapping[str, Any]) -> str:
    """对JSON object作稳定排序序列化后计算SHA-256。"""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TerminalEndpointActionEvidenceError("artifact relative path缺失")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TerminalEndpointActionEvidenceError("artifact relative path越界")
    return root / relative


def _external_path(
    manifest_value: object,
    override: str | Path | None,
    *,
    manifest_root: Path,
) -> Path:
    raw = override if override is not None else manifest_value
    if not isinstance(raw, (str, Path)):
        raise TerminalEndpointActionEvidenceError("source artifact路径缺失")
    candidate = Path(raw)
    if not candidate.is_absolute() and override is None:
        candidate = manifest_root / candidate
    return candidate.resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TerminalEndpointActionEvidenceError("manifest必须为JSON object")
    return payload


def _load_zero_gradient_inputs(
    metrics_path: Path,
    support_path: Path,
) -> tuple[
    dict[int, Float32Array],
    NDArray[np.bool_],
    dict[int, str],
    dict[int, tuple[NDArray[np.int64], NDArray[np.int64]]],
]:
    """复核Dense Seed/Production Support并提取Gate 6i零点输入。"""

    rows = load_dense_seed_gradient_evidence(metrics_path)
    summary = summarize_dense_seed_gradient_evidence(
        rows,
        artifact_root=metrics_path.parent,
    )
    if not summary.gate_pass:
        raise TerminalEndpointActionEvidenceError(
            "source Dense Seed evidence复核失败: "
            + "; ".join(summary.failures)
        )
    support = load_production_support_artifact(support_path)
    if not support.production_support_constructed or not support.fixed_support_frozen:
        raise TerminalEndpointActionEvidenceError(
            "Production Support未处于constructed/frozen状态"
        )
    if support.num_geometry_vertices != summary.num_geometry_vertices:
        raise TerminalEndpointActionEvidenceError(
            "Production Support与Dense Seed geometry数量不一致"
        )
    first_metadata = rows[0].metadata
    if (
        support.provenance.mesh_file_sha256 != first_metadata.mesh_sha256
        or support.provenance.render_to_geometry_sha256
        != first_metadata.render_to_geometry_sha256
    ):
        raise TerminalEndpointActionEvidenceError(
            "Production Support未绑定相同mesh/render-to-geometry"
        )

    gradients: dict[int, Float32Array] = {}
    fingerprints: dict[int, str] = {}
    clean_targets: dict[
        int,
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ] = {}
    for row in rows:
        state_id = row.objective_evidence.state_id
        artifact = _safe_relative_path(
            metrics_path.parent,
            row.artifact_relative_path,
        )
        with np.load(artifact, allow_pickle=False) as archive:
            gradient = archive["dense_geometry_gradient"].copy()
        gradients[state_id] = _geometry_array(
            gradient,
            name=f"zero_gradient[{state_id}]",
        )
        fingerprints[state_id] = row.objective_evidence.initial_state_sha256
        clean_targets[state_id] = (
            np.asarray(
                row.objective_evidence.clean_action_token_ids,
                dtype=np.int64,
            ),
            np.asarray(
                row.objective_evidence.clean_classes,
                dtype=np.int64,
            ),
        )
    return gradients, support.support_mask.copy(), fingerprints, clean_targets


def _load_indexed_artifact(
    root: Path,
    record: Mapping[str, Any],
) -> Path:
    path = _safe_relative_path(root, record.get("npz_relative_path"))
    if file_sha256(path) != record.get("npz_sha256"):
        raise TerminalEndpointActionEvidenceError("artifact file SHA不匹配")
    return path


def _decision_record(value: object) -> Any:
    """把dataclass派生结果正规化为可逐值比较的JSON值。"""

    return json.loads(json.dumps(asdict(value), sort_keys=True))


def terminal_endpoint_bundle_decision_record(
    decision: TerminalEndpointBundleDecision,
) -> dict[str, Any]:
    """生成成功manifest中的derived字段，明确排除科学归因。"""

    if not decision.audit_valid or decision.evidence_decision is None:
        raise TerminalEndpointActionEvidenceError(
            "无效bundle不能生成derived成功记录"
        )
    return {
        "evidence": _decision_record(decision.evidence_decision),
        "endpoint_steps": [
            _decision_record(step) for step in decision.step_decisions
        ],
    }


def evaluate_terminal_endpoint_bundle(
    manifest_path: str | Path,
    *,
    source_dense_seed_metrics_path: str | Path | None = None,
    production_support_path: str | Path | None = None,
    require_derived: bool = True,
) -> TerminalEndpointBundleDecision:
    """独立复核Gate 6i完整20/60/2 bundle，不运行OpenVLA。"""

    failures: list[str] = []
    gradients: list[EndpointGradientEvidence] = []
    responses: list[EndpointResponseEvidence] = []
    steps: list[EndpointStepEvidence] = []
    evidence_decision: EndpointEvidenceDecision | None = None
    step_decisions: tuple[EndpointStepDecision, ...] = ()
    try:
        path = Path(manifest_path).resolve()
        root = path.parent
        manifest = _load_json_object(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return TerminalEndpointBundleDecision(
            False,
            (f"Gate 6i manifest无法加载: {error}",),
            0,
            0,
            0,
            None,
            (),
        )

    if manifest.get("schema_version") != ENDPOINT_BUNDLE_SCHEMA_VERSION:
        failures.append("bundle schema_version不匹配")
    if manifest.get("status") != "complete":
        failures.append("bundle status必须为complete")
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or not all(
        character in "0123456789abcdef" for character in code_commit
    ):
        failures.append("code_commit必须为40位小写Git SHA")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        failures.append("configuration必须为object")
    elif configuration != frozen_bundle_configuration():
        failures.append("configuration偏离Gate 6i冻结值")
    elif manifest.get("config_sha256") != json_sha256(configuration):
        failures.append("config_sha256不能由configuration复算")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("provenance必须为object")
        provenance = {}
    restrictions = {
        "gradient_modalities": ["source_openvla_primary_action"],
        "teacher_forced_clean_prefix": True,
        "shared_texture_instances_aggregated": True,
        "render_to_geometry_mapping": "strict_face_corner_scatter_add",
        "feature_gradient": False,
        "wrist_gradient": False,
        "oft_gradient": False,
        "training_or_rollout_run": False,
        "legacy_optimizer_modules": [],
    }
    for name, expected in restrictions.items():
        if provenance.get(name) != expected:
            failures.append(f"provenance.{name}偏离冻结合同")
    processor_specification = provenance.get("processor_specification")
    if not isinstance(processor_specification, dict):
        failures.append("processor_specification必须为object")
        processor_specification = {}
    elif provenance.get("processor_specification_sha256") != json_sha256(
        processor_specification
    ):
        failures.append("processor_specification SHA不能复算")

    try:
        dense_metrics = _external_path(
            provenance.get("source_dense_seed_metrics_path"),
            source_dense_seed_metrics_path,
            manifest_root=root,
        )
        support_artifact = _external_path(
            provenance.get("production_support_path"),
            production_support_path,
            manifest_root=root,
        )
        if file_sha256(dense_metrics) != provenance.get(
            "source_dense_seed_metrics_sha256"
        ):
            raise TerminalEndpointActionEvidenceError(
                "source Dense Seed metrics SHA不匹配"
            )
        if file_sha256(support_artifact) != provenance.get(
            "production_support_sha256"
        ):
            raise TerminalEndpointActionEvidenceError(
                "Production Support artifact SHA不匹配"
            )
        (
            zero_gradients,
            support_mask,
            source_fingerprints,
            source_clean_targets,
        ) = _load_zero_gradient_inputs(dense_metrics, support_artifact)
    except (OSError, ValueError, TypeError, KeyError) as error:
        failures.append(f"零点/Support来源无法复核: {error}")
        zero_gradients = {}
        support_mask = np.asarray([], dtype=np.bool_)
        source_fingerprints = {}
        source_clean_targets = {}

    endpoint_sources = provenance.get("endpoint_sources")
    observed_endpoint_sources: list[str] = []
    if not isinstance(endpoint_sources, list):
        failures.append("endpoint_sources必须为list")
    else:
        for source in endpoint_sources:
            if not isinstance(source, dict):
                failures.append("endpoint source必须为object")
                continue
            endpoint = source.get("endpoint")
            observed_endpoint_sources.append(str(endpoint))
            for prefix in ("formal_manifest", "compact_parameter"):
                try:
                    # 两个文件都很小，runner必须逐字节复制到bundle内，保证
                    # rsync到WSL后无需依赖服务器绝对路径。
                    source_path = _safe_relative_path(
                        root,
                        source.get(f"{prefix}_relative_path"),
                    )
                    if file_sha256(source_path) != source.get(f"{prefix}_sha256"):
                        failures.append(f"{endpoint} {prefix} SHA不匹配")
                except (OSError, ValueError, TypeError) as error:
                    failures.append(f"{endpoint} {prefix}无法复核: {error}")
        if (
            len(observed_endpoint_sources) != len(ENDPOINT_NAMES)
            or set(observed_endpoint_sources) != set(ENDPOINT_NAMES)
            or len(set(observed_endpoint_sources)) != len(ENDPOINT_NAMES)
        ):
            failures.append("endpoint_sources必须恰好绑定两个正式终态")

    raw_gradient_records = manifest.get("gradient_cases")
    if not isinstance(raw_gradient_records, list):
        failures.append("gradient_cases必须为list")
        raw_gradient_records = []
    for record in raw_gradient_records:
        if not isinstance(record, dict):
            failures.append("gradient case record必须为object")
            continue
        key = (record.get("endpoint"), record.get("state_id"))
        try:
            artifact = _load_indexed_artifact(root, record)
            evidence = load_endpoint_gradient_npz(artifact)
        except (OSError, ValueError, TypeError, KeyError) as error:
            failures.append(f"gradient {key}无法加载: {error}")
            continue
        if key != (evidence.endpoint, evidence.state_id):
            failures.append(f"gradient {key} NPZ身份漂移")
        if record.get("state_fingerprint") != evidence.state_fingerprint:
            failures.append(f"gradient {key} fingerprint索引漂移")
        if source_fingerprints.get(evidence.state_id) != (
            evidence.state_fingerprint
        ):
            failures.append(f"gradient {key}未绑定零点state fingerprint")
        gradients.append(evidence)

    raw_response_records = manifest.get("response_records")
    if not isinstance(raw_response_records, list):
        failures.append("response_records必须为list")
        raw_response_records = []
    for record in raw_response_records:
        if not isinstance(record, dict):
            failures.append("response record必须为object")
            continue
        key = (
            record.get("endpoint"),
            record.get("state_id"),
            record.get("arm"),
        )
        try:
            artifact = _load_indexed_artifact(root, record)
            evidence = load_endpoint_response_npz(artifact)
        except (OSError, ValueError, TypeError, KeyError) as error:
            failures.append(f"response {key}无法加载: {error}")
            continue
        if key != (evidence.endpoint, evidence.state_id, evidence.arm):
            failures.append(f"response {key} NPZ身份漂移")
        expected_target = source_clean_targets.get(evidence.state_id)
        if expected_target is None or not (
            np.array_equal(evidence.clean_action_token_ids, expected_target[0])
            and np.array_equal(evidence.clean_classes, expected_target[1])
        ):
            failures.append(f"response {key}未绑定零点clean Action target")
        try:
            expected_bits = processor_bf16_bits_from_effective_rgb(
                evidence.effective_view_rgb,
                processor_specification,
            )
        except (TypeError, ValueError) as error:
            failures.append(f"response {key} processor bits无法复算: {error}")
        else:
            if not np.array_equal(
                evidence.processor_bf16_bits,
                expected_bits,
            ):
                failures.append(f"response {key} processor BF16 bits不一致")
        responses.append(evidence)

    raw_step_records = manifest.get("endpoint_steps")
    if not isinstance(raw_step_records, list):
        failures.append("endpoint_steps必须为list")
        raw_step_records = []
    for record in raw_step_records:
        if not isinstance(record, dict):
            failures.append("endpoint step record必须为object")
            continue
        endpoint = record.get("endpoint")
        try:
            artifact = _load_indexed_artifact(root, record)
            evidence = load_endpoint_step_npz(artifact)
        except (OSError, ValueError, TypeError, KeyError) as error:
            failures.append(f"step {endpoint}无法加载: {error}")
            continue
        if endpoint != evidence.endpoint:
            failures.append(f"step {endpoint} NPZ身份漂移")
        steps.append(evidence)

    if not failures:
        evidence_decision = evaluate_endpoint_evidence(
            gradients,
            responses,
            zero_gradients_by_state=zero_gradients,
            support_mask=support_mask,
        )
        failures.extend(evidence_decision.failures)
    if not failures and evidence_decision is not None:
        gradient_map = {
            (case.endpoint, case.state_id): case for case in gradients
        }
        step_map = {step.endpoint: step for step in steps}
        if len(steps) != 2 or set(step_map) != set(ENDPOINT_NAMES):
            failures.append("endpoint step inventory必须恰好为两个终态")
        else:
            collected_step_decisions: list[EndpointStepDecision] = []
            response_map = {
                (response.endpoint, response.state_id, response.arm): response
                for response in responses
            }
            for endpoint in ENDPOINT_NAMES:
                aggregate = aggregate_state_gradients(
                    np.stack(
                        [
                            gradient_map[(endpoint, state_id)]
                            .dense_surface_gradient
                            for state_id in EXPECTED_STATE_IDS
                        ],
                        axis=0,
                    ).astype(np.float32, copy=False)
                )
                step = step_map[endpoint]
                decision = evaluate_endpoint_step_evidence(
                    step,
                    aggregate_gradient=aggregate,
                    support_mask=support_mask,
                )
                collected_step_decisions.append(decision)
                failures.extend(decision.failures)
                if step.epsilon != FORMAL_SURFACE_EPSILON:
                    failures.append(f"{endpoint} epsilon偏离正式预算")
                if step.surface_step != FORMAL_SURFACE_STEP:
                    failures.append(f"{endpoint} surface_step偏离正式步长")
                if any(
                    case.realized_endpoint_sha256
                    != step.realized_endpoint_sha256
                    for case in gradients
                    if case.endpoint == endpoint
                ):
                    failures.append(f"{endpoint} gradient未绑定step realized endpoint")
                arm_hashes = {
                    "baseline": step.realized_endpoint_sha256,
                    "dense": step.dense_surface_delta_sha256,
                    "support": step.support_surface_delta_sha256,
                }
                for state_id in EXPECTED_STATE_IDS:
                    for arm in RESPONSE_ARMS:
                        response = response_map[(endpoint, state_id, arm)]
                        if response.surface_delta_sha256 != arm_hashes[arm]:
                            failures.append(
                                f"{endpoint}/state{state_id}/{arm} response未绑定step"
                            )
            step_decisions = tuple(collected_step_decisions)

    provisional = TerminalEndpointBundleDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        gradient_case_count=len(raw_gradient_records),
        response_record_count=len(raw_response_records),
        endpoint_step_count=len(raw_step_records),
        evidence_decision=(None if failures else evidence_decision),
        step_decisions=(() if failures else step_decisions),
    )
    if not failures and require_derived:
        expected_derived = terminal_endpoint_bundle_decision_record(provisional)
        if manifest.get("derived") != expected_derived:
            failures.append("manifest derived不能由raw evidence独立复算")
    if failures:
        return TerminalEndpointBundleDecision(
            audit_valid=False,
            failures=tuple(failures),
            gradient_case_count=len(raw_gradient_records),
            response_record_count=len(raw_response_records),
            endpoint_step_count=len(raw_step_records),
            evidence_decision=None,
            step_decisions=(),
        )
    return provisional


def write_json_atomically(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    """拒绝覆盖地原子写JSON，并返回文件SHA-256。"""

    path = Path(output_path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_sha256(path)


def publish_terminal_endpoint_bundle(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    """先复核raw candidate，再最后原子发布含derived的成功manifest。"""

    path = Path(output_path)
    raw_candidate = path.with_name(path.stem + "_raw_candidate.json")
    validated_candidate = path.with_name(path.stem + "_candidate.json")
    if path.exists() or raw_candidate.exists() or validated_candidate.exists():
        raise FileExistsError(path)
    candidate_payload = dict(payload)
    candidate_payload["status"] = "complete"
    write_json_atomically(candidate_payload, output_path=raw_candidate)
    decision = evaluate_terminal_endpoint_bundle(
        raw_candidate,
        require_derived=False,
    )
    if not decision.audit_valid:
        raw_candidate.unlink()
        raise TerminalEndpointActionEvidenceError(
            "Gate 6i raw candidate复核失败: " + "; ".join(decision.failures)
        )
    candidate_payload["derived"] = terminal_endpoint_bundle_decision_record(
        decision
    )
    write_json_atomically(candidate_payload, output_path=validated_candidate)
    validated = evaluate_terminal_endpoint_bundle(validated_candidate)
    if not validated.audit_valid:
        raw_candidate.unlink()
        validated_candidate.unlink()
        raise TerminalEndpointActionEvidenceError(
            "Gate 6i validated candidate复核失败: "
            + "; ".join(validated.failures)
        )
    raw_candidate.unlink()
    os.replace(validated_candidate, path)
    return file_sha256(path)
