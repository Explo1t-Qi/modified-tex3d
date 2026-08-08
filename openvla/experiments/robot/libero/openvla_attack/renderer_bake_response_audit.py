"""Gate 2R renderer-to-bake response 的纯数值审计层。

真实 runner 负责在同一个 LIBERO 初始 state 上采集 renderer compositor 与
bake→MuJoCo 两条路径。本模块只处理 effective-view 浮点响应、MuJoCo 可见权重、
Action margin 诊断、证据完整性和确定性持久化；它不导入 LIBERO、OpenVLA、
nvdiffrast 或 CUDA，因此服务器生成的证据可以在 WSL 上独立复算。

响应数组统一使用 ``float32/float64 [height,width,3]``，alpha 使用
``float32/float64 [height,width]``。所有加权归约提升到 float64。Gate 只要求
两条路径响应充分且加权 cosine 严格大于零；Action margin、relative L2 与符号
一致率均只作诊断，不参与第一版硬判定。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


RENDERER_BAKE_RESPONSE_SCHEMA_VERSION: Final[str] = (
    "openvla-renderer-bake-response-v1"
)
PROBE_RMS_MIN_CANDIDATE: Final[float] = 1e-6
METRIC_EPSILON: Final[float] = 1e-12
PROBE_CHANNELS: Final[tuple[str, ...]] = ("r", "g", "b")
PROBE_SURFACE_DELTA: Final[float] = 2.0 / 255.0
REQUIRED_VISUALIZATION_ROLES: Final[tuple[str, ...]] = (
    "alpha",
    "d_sur",
    "d_bake",
    "difference",
    "weighted_scatter",
)

ProbeChannel: TypeAlias = Literal["r", "g", "b"]
ResponseStatus: TypeAlias = Literal[
    "valid",
    "insufficient_probe_response",
    "not_observable",
    "insufficient_observation",
    "invalid_alignment",
    "invalid_evidence",
]
VisibilityResponseStatus: TypeAlias = Literal[
    "valid",
    "not_observable",
    "insufficient_observation",
    "invalid_alignment",
]
FloatArray: TypeAlias = NDArray[np.floating[Any]]


class RendererBakeResponseAuditError(RuntimeError):
    """Gate 2R 输入或证据 schema 无法被可靠解释。"""


@dataclass(frozen=True)
class ResponsePathStatistics:
    """一个 RGB 响应路径的 alpha 加权逐通道统计。"""

    weighted_mean: tuple[float, float, float]
    weighted_rms: tuple[float, float, float]
    visible_minimum: tuple[float, float, float]
    visible_maximum: tuple[float, float, float]
    positive_weight_fraction: tuple[float, float, float]
    negative_weight_fraction: tuple[float, float, float]
    zero_weight_fraction: tuple[float, float, float]


@dataclass(frozen=True)
class WeightedResponseMetrics:
    """一对 surrogate/bake 响应的可复算指标。"""

    alpha_sum: float
    alpha_positive_pixel_count: int
    surrogate_rms: float | None
    bake_rms: float | None
    cosine: float | None
    relative_l2: float | None
    sign_consistency: float | None
    sign_consistent_weight: float
    sign_denominator_weight: float
    sign_denominator_component_count: int
    surrogate_channels: ResponsePathStatistics
    bake_channels: ResponsePathStatistics
    difference_channels: ResponsePathStatistics
    response_sufficient: bool


@dataclass(frozen=True)
class RendererBakeResponseEvidence:
    """一个 ``(state,probe)`` 的未量化 effective-view 数组证据。

    ``alpha`` shape 为 ``[H,W]``；``d_sur`` 与 ``d_bake`` shape 均为
    ``[H,W,3]``。三组 Action margin shape 均为 ``[action_dim]``，并且使用
    同一 clean action sequence 及 teacher-forced prefix。
    """

    state_id: int
    probe_channel: ProbeChannel
    alpha: FloatArray
    d_sur: FloatArray
    d_bake: FloatArray
    clean_action_margins: FloatArray
    surrogate_action_margins: FloatArray
    bake_action_margins: FloatArray
    visibility_status: VisibilityResponseStatus = "valid"


@dataclass(frozen=True)
class RendererBakeResponseProvenance:
    """一次真实 bake、状态绑定和产物 inventory 的来源证据。"""

    code_commit: str
    config_sha256: str
    evidence_sha256: str
    initial_state_sha256: str
    clean_sim_state_sha256: str
    bake_sim_state_sha256: str
    asset_xml_path: str
    real_texture_path: str
    transaction_verified: bool
    xml_sha256_before: str
    xml_sha256_after_restore: str
    clean_texture_sha256: str
    baked_texture_sha256: str
    active_texture_sha256: str
    restored_texture_sha256: str
    arrays_npz_path: str
    arrays_npz_sha256: str
    artifact_sha256: Mapping[str, str]


class RendererBakeResponseRow(TypedDict):
    """权威 ``response_metrics.jsonl`` 的固定单行 schema。"""

    schema_version: str
    code_commit: str
    state_id: int
    probe_channel: str
    probe_surface_delta: float
    probe_rms_min_candidate: float
    metric_epsilon: float
    status: str
    visibility_status: str
    alpha_shape: list[int]
    alpha_dtype: str
    d_sur_shape: list[int]
    d_sur_dtype: str
    d_bake_shape: list[int]
    d_bake_dtype: str
    alpha_sum: float
    alpha_positive_pixel_count: int
    surrogate_rms: float
    bake_rms: float
    cos_alpha: float | None
    rel_l2_alpha: float | None
    sign_consistency: float | None
    sign_consistent_weight: float
    sign_denominator_weight: float
    sign_denominator_component_count: int
    per_channel_statistics: dict[str, Any]
    action_dim: int
    clean_action_margins: list[float]
    surrogate_action_margins: list[float]
    bake_action_margins: list[float]
    config_sha256: str
    evidence_sha256: str
    initial_state_sha256: str
    clean_sim_state_sha256: str
    bake_sim_state_sha256: str
    asset_xml_path: str
    real_texture_path: str
    transaction_verified: bool
    xml_sha256_before: str
    xml_sha256_after_restore: str
    clean_texture_sha256: str
    baked_texture_sha256: str
    active_texture_sha256: str
    restored_texture_sha256: str
    arrays_npz_path: str
    arrays_npz_sha256: str
    artifact_sha256: dict[str, str]
    failures: list[str]
    gate_pass: bool


class ProbeResponseSummary(TypedDict):
    """单一颜色 probe 的分布摘要与最坏 state ID。"""

    row_count: int
    status_counts: dict[str, int]
    minimum_cosine: float | None
    minimum_cosine_state_ids: list[int]
    median_cosine: float | None
    minimum_surrogate_rms: float | None
    minimum_surrogate_rms_state_ids: list[int]
    minimum_bake_rms: float | None
    minimum_bake_rms_state_ids: list[int]
    maximum_relative_l2: float | None
    maximum_relative_l2_state_ids: list[int]
    median_relative_l2: float | None
    minimum_sign_consistency: float | None
    minimum_sign_consistency_state_ids: list[int]
    median_sign_consistency: float | None
    gate_pass: bool


class RendererBakeResponseSummary(TypedDict):
    """完整 ``states × RGB probes`` 的结构与 Gate 汇总。"""

    schema_version: str
    expected_state_ids: list[int]
    expected_probe_channels: list[str]
    row_count: int
    missing_keys: list[list[int | str]]
    unexpected_keys: list[list[int | str]]
    duplicate_keys: list[list[int | str]]
    code_commits: list[str]
    config_sha256_values: list[str]
    structural_failures: list[str]
    structural_pass: bool
    status_counts: dict[str, int]
    per_probe: dict[str, ProbeResponseSummary]
    passed_row_count: int
    gate_pass: bool


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_response_arrays(
    alpha: np.ndarray,
    d_sur: np.ndarray,
    d_bake: np.ndarray,
    *,
    require_positive_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """验证并返回 float64 view，不改变原始未量化数组。"""

    for name, value in (("alpha", alpha), ("D_sur", d_sur), ("D_bake", d_bake)):
        if not isinstance(value, np.ndarray):
            raise RendererBakeResponseAuditError(f"{name} 必须是 numpy array")
        if not np.issubdtype(value.dtype, np.floating):
            raise RendererBakeResponseAuditError(
                f"{name} 必须是浮点数组，收到 {value.dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise RendererBakeResponseAuditError(f"{name} 包含 NaN/Inf")
    if alpha.ndim != 2 or alpha.size == 0:
        raise RendererBakeResponseAuditError(
            f"alpha 必须为非空 [H,W]，收到 {alpha.shape}"
        )
    expected_rgb_shape = (*alpha.shape, 3)
    if d_sur.shape != expected_rgb_shape or d_bake.shape != expected_rgb_shape:
        raise RendererBakeResponseAuditError(
            "D_sur/D_bake 必须为与 alpha 对齐的 [H,W,3]："
            f"{d_sur.shape}, {d_bake.shape}, alpha={alpha.shape}"
        )
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise RendererBakeResponseAuditError("alpha 必须位于 [0,1]")
    if require_positive_alpha and not bool(np.any(alpha > 0.0)):
        raise RendererBakeResponseAuditError("alpha 没有正权重像素")
    return (
        alpha.astype(np.float64, copy=False),
        d_sur.astype(np.float64, copy=False),
        d_bake.astype(np.float64, copy=False),
    )


def _path_statistics(
    response: np.ndarray,
    alpha: np.ndarray,
) -> ResponsePathStatistics:
    """计算 RGB 每通道的 alpha 加权幅值与符号占比。"""

    alpha_sum = float(np.sum(alpha, dtype=np.float64))
    visible = alpha > 0.0
    means: list[float] = []
    rms_values: list[float] = []
    minimums: list[float] = []
    maximums: list[float] = []
    positive: list[float] = []
    negative: list[float] = []
    zero: list[float] = []
    for channel_index in range(3):
        values = response[..., channel_index]
        means.append(float(np.sum(alpha * values) / alpha_sum))
        rms_values.append(
            float(np.sqrt(np.sum(alpha * values * values) / alpha_sum))
        )
        visible_values = values[visible]
        minimums.append(float(np.min(visible_values)))
        maximums.append(float(np.max(visible_values)))
        positive.append(float(np.sum(alpha[values > 0.0]) / alpha_sum))
        negative.append(float(np.sum(alpha[values < 0.0]) / alpha_sum))
        zero.append(float(np.sum(alpha[values == 0.0]) / alpha_sum))
    return ResponsePathStatistics(
        weighted_mean=tuple(means),  # type: ignore[arg-type]
        weighted_rms=tuple(rms_values),  # type: ignore[arg-type]
        visible_minimum=tuple(minimums),  # type: ignore[arg-type]
        visible_maximum=tuple(maximums),  # type: ignore[arg-type]
        positive_weight_fraction=tuple(positive),  # type: ignore[arg-type]
        negative_weight_fraction=tuple(negative),  # type: ignore[arg-type]
        zero_weight_fraction=tuple(zero),  # type: ignore[arg-type]
    )


def compute_weighted_response_metrics(
    alpha: np.ndarray,
    d_sur: np.ndarray,
    d_bake: np.ndarray,
    *,
    response_minimum: float = PROBE_RMS_MIN_CANDIDATE,
    epsilon: float = METRIC_EPSILON,
) -> WeightedResponseMetrics:
    """按冻结公式计算 alpha 加权 RMS、cosine、relative L2 和符号一致率。

    符号一致率的分母只包含 ``D_sur`` 或 ``D_bake`` 至少一个严格非零的可见
    通道分量；两者同时为零的分量不提供方向证据，一条为零而另一条非零则进入
    分母但不进入分子。分母同时保存加权值和未加权分量数，避免解释歧义。
    """

    if not math.isfinite(response_minimum) or response_minimum <= 0.0:
        raise RendererBakeResponseAuditError("response_minimum 必须为有限正数")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise RendererBakeResponseAuditError("epsilon 必须为有限正数")
    alpha64, d_sur64, d_bake64 = _validate_response_arrays(
        alpha,
        d_sur,
        d_bake,
    )
    alpha_sum = float(np.sum(alpha64, dtype=np.float64))
    channel_weights = alpha64[..., None]
    surrogate_squared = float(np.sum(channel_weights * d_sur64 * d_sur64))
    bake_squared = float(np.sum(channel_weights * d_bake64 * d_bake64))
    denominator = 3.0 * alpha_sum + epsilon
    surrogate_rms = float(np.sqrt(surrogate_squared / denominator))
    bake_rms = float(np.sqrt(bake_squared / denominator))
    response_sufficient = bool(
        math.isfinite(surrogate_rms)
        and math.isfinite(bake_rms)
        and surrogate_rms >= response_minimum
        and bake_rms >= response_minimum
    )

    cosine: float | None = None
    relative_l2: float | None = None
    if response_sufficient:
        inner_product = float(np.sum(channel_weights * d_sur64 * d_bake64))
        cosine_denominator = (
            math.sqrt(surrogate_squared) * math.sqrt(bake_squared) + epsilon
        )
        cosine = float(inner_product / cosine_denominator)
        difference_squared = float(
            np.sum(channel_weights * (d_sur64 - d_bake64) ** 2)
        )
        relative_l2 = float(
            math.sqrt(difference_squared) / (math.sqrt(bake_squared) + epsilon)
        )

    visible_components = np.broadcast_to(
        (alpha64 > 0.0)[..., None],
        d_sur64.shape,
    )
    evidence_components = (
        ((d_sur64 != 0.0) | (d_bake64 != 0.0)) & visible_components
    )
    same_sign_components = (d_sur64 * d_bake64 > 0.0) & evidence_components
    broadcast_weights = np.broadcast_to(channel_weights, d_sur64.shape)
    sign_denominator_weight = float(
        np.sum(broadcast_weights[evidence_components], dtype=np.float64)
    )
    sign_consistent_weight = float(
        np.sum(broadcast_weights[same_sign_components], dtype=np.float64)
    )
    sign_consistency = (
        float(sign_consistent_weight / sign_denominator_weight)
        if sign_denominator_weight > 0.0
        else None
    )
    return WeightedResponseMetrics(
        alpha_sum=alpha_sum,
        alpha_positive_pixel_count=int(np.count_nonzero(alpha64 > 0.0)),
        surrogate_rms=surrogate_rms,
        bake_rms=bake_rms,
        cosine=cosine,
        relative_l2=relative_l2,
        sign_consistency=sign_consistency,
        sign_consistent_weight=sign_consistent_weight,
        sign_denominator_weight=sign_denominator_weight,
        sign_denominator_component_count=int(np.count_nonzero(evidence_components)),
        surrogate_channels=_path_statistics(d_sur64, alpha64),
        bake_channels=_path_statistics(d_bake64, alpha64),
        difference_channels=_path_statistics(d_sur64 - d_bake64, alpha64),
        response_sufficient=response_sufficient,
    )


def compute_untargeted_clean_action_margins(
    action_logits: np.ndarray,
    clean_classes: np.ndarray,
) -> NDArray[np.float64]:
    """计算冻结 objective 的 ``z[y] - max(z[j!=y])`` 逐 token margin。

    Args:
        action_logits: 浮点 ``[action_dim,num_action_classes]`` teacher-forced
            logits；通常第二维为 256。
        clean_classes: 整数 ``[action_dim]``，来自同一 clean action sequence。
    """

    logits = np.asarray(action_logits)
    classes = np.asarray(clean_classes)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise RendererBakeResponseAuditError(
            "action_logits 必须为非空 [A,C>=2]"
        )
    if not np.issubdtype(logits.dtype, np.floating) or not np.all(
        np.isfinite(logits)
    ):
        raise RendererBakeResponseAuditError(
            "action_logits 必须为有限浮点数组"
        )
    if classes.shape != (logits.shape[0],) or not np.issubdtype(
        classes.dtype,
        np.integer,
    ):
        raise RendererBakeResponseAuditError(
            "clean_classes 必须为与 logits 第一维相同的一维整数数组"
        )
    if np.any(classes < 0) or np.any(classes >= logits.shape[1]):
        raise RendererBakeResponseAuditError("clean_classes 存在越界类别")
    logits64 = logits.astype(np.float64, copy=False)
    row_indices = np.arange(logits.shape[0])
    clean_values = logits64[row_indices, classes]
    other_values = logits64.copy()
    other_values[row_indices, classes] = -np.inf
    return clean_values - np.max(other_values, axis=1)


def _validate_margins(evidence: RendererBakeResponseEvidence) -> int:
    action_dim = int(np.asarray(evidence.clean_action_margins).size)
    if action_dim <= 0:
        raise RendererBakeResponseAuditError("Action margin 不得为空")
    for name, values in (
        ("clean", evidence.clean_action_margins),
        ("surrogate", evidence.surrogate_action_margins),
        ("bake", evidence.bake_action_margins),
    ):
        array = np.asarray(values)
        if array.shape != (action_dim,):
            raise RendererBakeResponseAuditError(
                f"{name} Action margin shape 不一致：{array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating) or not np.all(
            np.isfinite(array)
        ):
            raise RendererBakeResponseAuditError(
                f"{name} Action margin 必须为有限浮点数组"
            )
    return action_dim


def evaluate_renderer_bake_response_evidence(
    evidence: RendererBakeResponseEvidence,
    provenance: RendererBakeResponseProvenance,
) -> RendererBakeResponseRow:
    """生成一个权威行，并按第一版必要条件计算行级 Gate。"""

    if evidence.state_id < 0:
        raise RendererBakeResponseAuditError("state_id 必须为非负整数")
    if evidence.probe_channel not in PROBE_CHANNELS:
        raise RendererBakeResponseAuditError("probe_channel 必须为 r/g/b")
    if evidence.visibility_status not in (
        "valid",
        "not_observable",
        "insufficient_observation",
        "invalid_alignment",
    ):
        raise RendererBakeResponseAuditError("未知 visibility_status")
    action_dim = _validate_margins(evidence)
    metrics: WeightedResponseMetrics | None
    if evidence.visibility_status == "valid":
        metrics = compute_weighted_response_metrics(
            evidence.alpha,
            evidence.d_sur,
            evidence.d_bake,
        )
    else:
        _validate_response_arrays(
            evidence.alpha,
            evidence.d_sur,
            evidence.d_bake,
            require_positive_alpha=False,
        )
        metrics = None

    failures: list[str] = []
    evidence_failures: list[str] = []
    if evidence.visibility_status != "valid":
        failures.append(f"visibility_status={evidence.visibility_status}")
    elif metrics is None:
        raise AssertionError("valid visibility 必须生成 response metrics")
    elif not metrics.response_sufficient:
        failures.append("insufficient_probe_response")
    if metrics is not None and metrics.response_sufficient and (
        metrics.cosine is None
        or not math.isfinite(metrics.cosine)
        or metrics.cosine <= 0.0
    ):
        failures.append("cos_alpha_not_positive")
    if not _is_git_commit(provenance.code_commit):
        evidence_failures.append("invalid_code_commit")
    required_hashes = {
        "config_sha256": provenance.config_sha256,
        "evidence_sha256": provenance.evidence_sha256,
        "initial_state_sha256": provenance.initial_state_sha256,
        "clean_sim_state_sha256": provenance.clean_sim_state_sha256,
        "bake_sim_state_sha256": provenance.bake_sim_state_sha256,
        "xml_sha256_before": provenance.xml_sha256_before,
        "xml_sha256_after_restore": provenance.xml_sha256_after_restore,
        "clean_texture_sha256": provenance.clean_texture_sha256,
        "baked_texture_sha256": provenance.baked_texture_sha256,
        "active_texture_sha256": provenance.active_texture_sha256,
        "restored_texture_sha256": provenance.restored_texture_sha256,
        "arrays_npz_sha256": provenance.arrays_npz_sha256,
    }
    invalid_hash_names = sorted(
        name for name, value in required_hashes.items() if not _is_sha256(value)
    )
    if invalid_hash_names:
        evidence_failures.append(
            f"invalid_required_sha256={invalid_hash_names}"
        )
    if not provenance.asset_xml_path or not provenance.real_texture_path:
        evidence_failures.append("missing_asset_paths")
    if (
        _is_sha256(provenance.clean_sim_state_sha256)
        and _is_sha256(provenance.bake_sim_state_sha256)
        and provenance.clean_sim_state_sha256
        != provenance.bake_sim_state_sha256
    ):
        evidence_failures.append("clean_bake_sim_state_mismatch")
    if not provenance.arrays_npz_path:
        evidence_failures.append("missing_arrays_npz_path")
    if not provenance.transaction_verified:
        evidence_failures.append("runtime_asset_transaction_failed")
    if (
        _is_sha256(provenance.xml_sha256_before)
        and _is_sha256(provenance.xml_sha256_after_restore)
        and provenance.xml_sha256_before != provenance.xml_sha256_after_restore
    ):
        evidence_failures.append("xml_not_restored")
    if (
        _is_sha256(provenance.clean_texture_sha256)
        and _is_sha256(provenance.restored_texture_sha256)
        and provenance.clean_texture_sha256
        != provenance.restored_texture_sha256
    ):
        evidence_failures.append("texture_not_restored")
    if (
        _is_sha256(provenance.baked_texture_sha256)
        and _is_sha256(provenance.active_texture_sha256)
        and provenance.baked_texture_sha256
        != provenance.active_texture_sha256
    ):
        evidence_failures.append("active_texture_not_baked_probe")
    if not provenance.artifact_sha256 or any(
        not path or not _is_sha256(fingerprint)
        for path, fingerprint in provenance.artifact_sha256.items()
    ):
        evidence_failures.append("invalid_artifact_sha256_inventory")
    elif provenance.artifact_sha256.get(provenance.arrays_npz_path) != (
        provenance.arrays_npz_sha256
    ):
        evidence_failures.append("artifact_inventory_missing_arrays_npz")
    if provenance.artifact_sha256:
        artifact_names = tuple(
            Path(path).name for path in provenance.artifact_sha256
        )
        missing_visualizations = [
            role
            for role in REQUIRED_VISUALIZATION_ROLES
            if not any(
                name.endswith(f"_{role}.png") for name in artifact_names
            )
        ]
        if missing_visualizations:
            evidence_failures.append(
                f"missing_visualizations={missing_visualizations}"
            )

    failures.extend(evidence_failures)

    if evidence.visibility_status != "valid":
        status: ResponseStatus = evidence.visibility_status
    elif metrics is None:
        raise AssertionError("valid visibility 必须生成 response metrics")
    elif not metrics.response_sufficient:
        status = "insufficient_probe_response"
    elif evidence_failures:
        status = "invalid_evidence"
    else:
        status = "valid"
    statistics = (
        {
            "surrogate": asdict(metrics.surrogate_channels),
            "bake": asdict(metrics.bake_channels),
            "difference": asdict(metrics.difference_channels),
        }
        if metrics is not None
        else {}
    )
    return RendererBakeResponseRow(
        schema_version=RENDERER_BAKE_RESPONSE_SCHEMA_VERSION,
        code_commit=provenance.code_commit,
        state_id=int(evidence.state_id),
        probe_channel=evidence.probe_channel,
        probe_surface_delta=PROBE_SURFACE_DELTA,
        probe_rms_min_candidate=PROBE_RMS_MIN_CANDIDATE,
        metric_epsilon=METRIC_EPSILON,
        status=status,
        visibility_status=evidence.visibility_status,
        alpha_shape=list(evidence.alpha.shape),
        alpha_dtype=str(evidence.alpha.dtype),
        d_sur_shape=list(evidence.d_sur.shape),
        d_sur_dtype=str(evidence.d_sur.dtype),
        d_bake_shape=list(evidence.d_bake.shape),
        d_bake_dtype=str(evidence.d_bake.dtype),
        alpha_sum=(
            metrics.alpha_sum
            if metrics is not None
            else float(np.sum(evidence.alpha, dtype=np.float64))
        ),
        alpha_positive_pixel_count=(
            metrics.alpha_positive_pixel_count
            if metrics is not None
            else int(np.count_nonzero(evidence.alpha > 0.0))
        ),
        surrogate_rms=(metrics.surrogate_rms if metrics is not None else None),
        bake_rms=(metrics.bake_rms if metrics is not None else None),
        cos_alpha=(metrics.cosine if metrics is not None else None),
        rel_l2_alpha=(metrics.relative_l2 if metrics is not None else None),
        sign_consistency=(
            metrics.sign_consistency if metrics is not None else None
        ),
        sign_consistent_weight=(
            metrics.sign_consistent_weight if metrics is not None else 0.0
        ),
        sign_denominator_weight=(
            metrics.sign_denominator_weight if metrics is not None else 0.0
        ),
        sign_denominator_component_count=(
            metrics.sign_denominator_component_count
            if metrics is not None
            else 0
        ),
        per_channel_statistics=statistics,
        action_dim=action_dim,
        clean_action_margins=[
            float(value) for value in evidence.clean_action_margins
        ],
        surrogate_action_margins=[
            float(value) for value in evidence.surrogate_action_margins
        ],
        bake_action_margins=[
            float(value) for value in evidence.bake_action_margins
        ],
        config_sha256=provenance.config_sha256,
        evidence_sha256=provenance.evidence_sha256,
        initial_state_sha256=provenance.initial_state_sha256,
        clean_sim_state_sha256=provenance.clean_sim_state_sha256,
        bake_sim_state_sha256=provenance.bake_sim_state_sha256,
        asset_xml_path=provenance.asset_xml_path,
        real_texture_path=provenance.real_texture_path,
        transaction_verified=provenance.transaction_verified,
        xml_sha256_before=provenance.xml_sha256_before,
        xml_sha256_after_restore=provenance.xml_sha256_after_restore,
        clean_texture_sha256=provenance.clean_texture_sha256,
        baked_texture_sha256=provenance.baked_texture_sha256,
        active_texture_sha256=provenance.active_texture_sha256,
        restored_texture_sha256=provenance.restored_texture_sha256,
        arrays_npz_path=provenance.arrays_npz_path,
        arrays_npz_sha256=provenance.arrays_npz_sha256,
        artifact_sha256=dict(sorted(provenance.artifact_sha256.items())),
        failures=failures,
        gate_pass=not failures,
    )


def _extreme_with_state_ids(
    rows: Sequence[RendererBakeResponseRow],
    field: str,
    *,
    maximum: bool = False,
) -> tuple[float | None, list[int]]:
    values = [
        (int(row["state_id"]), float(value))
        for row in rows
        if (value := row[field]) is not None and math.isfinite(float(value))
    ]
    if not values:
        return None, []
    extreme = (max if maximum else min)(value for _, value in values)
    return extreme, sorted(
        state_id for state_id, value in values if value == extreme
    )


def _median(rows: Sequence[RendererBakeResponseRow], field: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row[field]) is not None and math.isfinite(float(value))
    ]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _summarize_probe(
    rows: Sequence[RendererBakeResponseRow],
) -> ProbeResponseSummary:
    minimum_cosine, minimum_cosine_states = _extreme_with_state_ids(
        rows, "cos_alpha"
    )
    minimum_surrogate, minimum_surrogate_states = _extreme_with_state_ids(
        rows, "surrogate_rms"
    )
    minimum_bake, minimum_bake_states = _extreme_with_state_ids(
        rows, "bake_rms"
    )
    maximum_relative, maximum_relative_states = _extreme_with_state_ids(
        rows, "rel_l2_alpha", maximum=True
    )
    minimum_sign, minimum_sign_states = _extreme_with_state_ids(
        rows, "sign_consistency"
    )
    status_counts = Counter(str(row["status"]) for row in rows)
    return ProbeResponseSummary(
        row_count=len(rows),
        status_counts=dict(sorted(status_counts.items())),
        minimum_cosine=minimum_cosine,
        minimum_cosine_state_ids=minimum_cosine_states,
        median_cosine=_median(rows, "cos_alpha"),
        minimum_surrogate_rms=minimum_surrogate,
        minimum_surrogate_rms_state_ids=minimum_surrogate_states,
        minimum_bake_rms=minimum_bake,
        minimum_bake_rms_state_ids=minimum_bake_states,
        maximum_relative_l2=maximum_relative,
        maximum_relative_l2_state_ids=maximum_relative_states,
        median_relative_l2=_median(rows, "rel_l2_alpha"),
        minimum_sign_consistency=minimum_sign,
        minimum_sign_consistency_state_ids=minimum_sign_states,
        median_sign_consistency=_median(rows, "sign_consistency"),
        gate_pass=bool(rows) and all(bool(row["gate_pass"]) for row in rows),
    )


def summarize_renderer_bake_response_rows(
    rows: Sequence[RendererBakeResponseRow],
    *,
    expected_state_ids: Sequence[int],
) -> RendererBakeResponseSummary:
    """严格检查 10×3 行集合、单一 commit/config 与逐行 Gate。"""

    expected_states = tuple(int(state_id) for state_id in expected_state_ids)
    if not expected_states or len(set(expected_states)) != len(expected_states):
        raise RendererBakeResponseAuditError(
            "expected_state_ids 必须为非空唯一集合"
        )
    expected_keys = {
        (state_id, channel)
        for state_id in expected_states
        for channel in PROBE_CHANNELS
    }
    actual_keys = [
        (int(row["state_id"]), str(row["probe_channel"])) for row in rows
    ]
    actual_key_set = set(actual_keys)
    key_counts = Counter(actual_keys)
    missing = sorted(expected_keys - actual_key_set)
    unexpected = sorted(actual_key_set - expected_keys)
    duplicates = sorted(key for key, count in key_counts.items() if count > 1)
    code_commits = sorted({str(row["code_commit"]) for row in rows})
    config_hashes = sorted({str(row["config_sha256"]) for row in rows})
    schemas = {str(row["schema_version"]) for row in rows}
    structural_failures: list[str] = []
    if len(rows) != len(expected_keys):
        structural_failures.append("row_count_mismatch")
    if missing:
        structural_failures.append("missing_state_probe_rows")
    if unexpected:
        structural_failures.append("unexpected_state_probe_rows")
    if duplicates:
        structural_failures.append("duplicate_state_probe_rows")
    if len(code_commits) != 1 or any(
        not _is_git_commit(commit) for commit in code_commits
    ):
        structural_failures.append("invalid_or_mixed_code_commit")
    if len(config_hashes) != 1 or any(
        not _is_sha256(value) for value in config_hashes
    ):
        structural_failures.append("invalid_or_mixed_config_sha256")
    if schemas != {RENDERER_BAKE_RESPONSE_SCHEMA_VERSION}:
        structural_failures.append("schema_version_mismatch")
    if any(
        float(row["probe_surface_delta"]) != PROBE_SURFACE_DELTA
        or float(row["probe_rms_min_candidate"])
        != PROBE_RMS_MIN_CANDIDATE
        or float(row["metric_epsilon"]) != METRIC_EPSILON
        for row in rows
    ):
        structural_failures.append("mixed_or_invalid_numeric_contract")

    channel_order = {channel: index for index, channel in enumerate(PROBE_CHANNELS)}
    per_probe = {
        channel: _summarize_probe(
            sorted(
                (
                    row
                    for row in rows
                    if row["probe_channel"] == channel
                ),
                key=lambda row: int(row["state_id"]),
            )
        )
        for channel in sorted(PROBE_CHANNELS, key=channel_order.__getitem__)
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    structural_pass = not structural_failures
    return RendererBakeResponseSummary(
        schema_version=RENDERER_BAKE_RESPONSE_SCHEMA_VERSION,
        expected_state_ids=sorted(expected_states),
        expected_probe_channels=list(PROBE_CHANNELS),
        row_count=len(rows),
        missing_keys=[[state_id, channel] for state_id, channel in missing],
        unexpected_keys=[
            [state_id, channel] for state_id, channel in unexpected
        ],
        duplicate_keys=[
            [state_id, channel] for state_id, channel in duplicates
        ],
        code_commits=code_commits,
        config_sha256_values=config_hashes,
        structural_failures=structural_failures,
        structural_pass=structural_pass,
        status_counts=dict(sorted(status_counts.items())),
        per_probe=per_probe,
        passed_row_count=sum(bool(row["gate_pass"]) for row in rows),
        gate_pass=structural_pass and all(
            bool(row["gate_pass"]) for row in rows
        ),
    )


def write_renderer_bake_response_npz(
    evidence: RendererBakeResponseEvidence,
    *,
    output_path: Path,
) -> str:
    """保存无 pickle 的未量化数组和显式 shape/dtype 元数据。"""

    _validate_response_arrays(
        evidence.alpha,
        evidence.d_sur,
        evidence.d_bake,
        require_positive_alpha=(evidence.visibility_status == "valid"),
    )
    _validate_margins(evidence)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Gate 2R NPZ: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        alpha=evidence.alpha,
        d_sur=evidence.d_sur,
        d_bake=evidence.d_bake,
        clean_action_margins=evidence.clean_action_margins,
        surrogate_action_margins=evidence.surrogate_action_margins,
        bake_action_margins=evidence.bake_action_margins,
        visibility_status=np.asarray(evidence.visibility_status),
        alpha_shape=np.asarray(evidence.alpha.shape, dtype=np.int64),
        d_sur_shape=np.asarray(evidence.d_sur.shape, dtype=np.int64),
        d_bake_shape=np.asarray(evidence.d_bake.shape, dtype=np.int64),
        alpha_dtype=np.asarray(str(evidence.alpha.dtype)),
        d_sur_dtype=np.asarray(str(evidence.d_sur.dtype)),
        d_bake_dtype=np.asarray(str(evidence.d_bake.dtype)),
    )
    return _file_sha256(output_path)


def _ordered_rows(
    rows: Sequence[RendererBakeResponseRow],
) -> list[RendererBakeResponseRow]:
    channel_order = {channel: index for index, channel in enumerate(PROBE_CHANNELS)}
    return sorted(
        rows,
        key=lambda row: (
            int(row["state_id"]),
            channel_order.get(str(row["probe_channel"]), len(PROBE_CHANNELS)),
        ),
    )


def write_renderer_bake_response_jsonl(
    rows: Sequence[RendererBakeResponseRow],
    *,
    output_path: Path,
) -> str:
    """按原始 state ID、R/G/B 顺序写唯一权威 JSONL。"""

    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Gate 2R JSONL: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in _ordered_rows(rows):
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
    return _file_sha256(output_path)


def write_renderer_bake_response_csv(
    rows: Sequence[RendererBakeResponseRow],
    *,
    output_path: Path,
) -> str:
    """从内存中的权威行确定性派生便读 CSV；不保存嵌套数组。"""

    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Gate 2R CSV: {output_path}")
    fieldnames = (
        "state_id",
        "probe_channel",
        "status",
        "surrogate_rms",
        "bake_rms",
        "cos_alpha",
        "rel_l2_alpha",
        "sign_consistency",
        "sign_denominator_weight",
        "action_dim",
        "gate_pass",
        "arrays_npz_sha256",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in _ordered_rows(rows):
            writer.writerow({field: row[field] for field in fieldnames})
    return _file_sha256(output_path)


def write_renderer_bake_response_manifest(
    *,
    summary: RendererBakeResponseSummary,
    metadata: Mapping[str, Any],
    metrics_jsonl_sha256: str,
    derived_csv_sha256: str,
    output_path: Path,
) -> str:
    """写入绑定权威 JSONL、派生 CSV 和 Gate summary 的 manifest。"""

    if not _is_sha256(metrics_jsonl_sha256) or not _is_sha256(
        derived_csv_sha256
    ):
        raise RendererBakeResponseAuditError("JSONL/CSV fingerprint 必须为 SHA-256")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Gate 2R manifest: {output_path}")
    payload = {
        "schema_version": RENDERER_BAKE_RESPONSE_SCHEMA_VERSION,
        "metrics_jsonl_sha256": metrics_jsonl_sha256,
        "derived_csv_sha256": derived_csv_sha256,
        "summary": summary,
        "metadata": dict(metadata),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _file_sha256(output_path)
