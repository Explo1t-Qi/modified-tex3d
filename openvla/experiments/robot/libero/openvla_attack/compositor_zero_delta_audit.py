"""零 Surface Delta Compositor Gate 的纯判定与持久化层。

真实 runner 负责从同一个静止 LIBERO state 采集 clean 与 compositor 两条路径；
本模块只验证 schema、逐阶段严格恒等、renderer 梯度分支和 state 集合完整性。
它不导入 LIBERO、OpenVLA、nvdiffrast 或 CUDA，因此服务器产生的 JSONL 可以在
WSL 上独立复算。

Gate 是严格的工程等价性检查，不设置浮点近似容差：raw Policy Source、Policy
Pre-Crop Canvas、center-crop Effective View、checkpoint fused pixel values、
action token 和连续 action 任一差异都会失败。零 Surface Delta 时 renderer
前向响应也必须严格为零；同时 clean renderer 必须断图，而 adversarial renderer
在有效投影像素上对 Surface 参数的梯度必须有限非零。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from .deployment_backward_audit import GradientEvidence


COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION: Final[str] = (
    "openvla-compositor-zero-delta-v1"
)


class ZeroDeltaCompositorAuditError(RuntimeError):
    """输入证据不满足零 delta audit schema。"""


@dataclass(frozen=True)
class ZeroDeltaCompositorEvidence:
    """一个原始 LIBERO state 的 clean/compositor 完整证据。

    RGB 数组均为 ``uint8 [H,W,3]``；pixel values 为浮点
    ``[1,6,H_model,W_model]``；token/action 均为 ``[action_dim]``。梯度统计针对
    renderer 的 trainable Surface 参数，不是 Action loss 梯度。
    """

    state_id: int
    clean_source_rgb: NDArray[np.uint8]
    composited_source_rgb: NDArray[np.uint8]
    clean_pre_crop_rgb: NDArray[np.uint8]
    composited_pre_crop_rgb: NDArray[np.uint8]
    clean_effective_rgb: NDArray[np.uint8]
    composited_effective_rgb: NDArray[np.uint8]
    clean_pixel_values: NDArray[np.floating]
    composited_pixel_values: NDArray[np.floating]
    clean_action_token_ids: NDArray[np.integer]
    composited_action_token_ids: NDArray[np.integer]
    clean_action: NDArray[np.floating]
    composited_action: NDArray[np.floating]
    renderer_delta_linf: float
    per_instance_delta_linf: tuple[float, ...]
    total_delta_linf: float
    saturated_pixel_fraction: float
    saturated_channel_fraction: float
    clean_renderer_requires_grad: bool
    valid_renderer_pixel_count: int
    instance_visible_pixel_counts: tuple[int, ...]
    surface_parameter_gradient: GradientEvidence
    transaction_verified: bool
    transaction_before_sha256: str
    transaction_after_sha256: str
    arrays_npz_sha256: str
    artifact_sha256: Mapping[str, str]


class ZeroDeltaCompositorRow(TypedDict):
    """权威 JSONL 中一个 state 的固定 schema。"""

    schema_version: str
    code_commit: str
    state_id: int
    source_rgb_shape: list[int]
    pre_crop_rgb_shape: list[int]
    effective_view_rgb_shape: list[int]
    processor_pixel_shape: list[int]
    action_dim: int
    clean_source_rgb_sha256: str
    composited_source_rgb_sha256: str
    clean_pre_crop_rgb_sha256: str
    composited_pre_crop_rgb_sha256: str
    clean_effective_rgb_sha256: str
    composited_effective_rgb_sha256: str
    source_rgb_mae: float
    source_rgb_linf: float
    pre_crop_rgb_mae: float
    pre_crop_rgb_linf: float
    effective_view_mae: float
    effective_view_linf: float
    processor_pixel_mae: float
    processor_pixel_linf: float
    clean_action_token_ids: list[int]
    composited_action_token_ids: list[int]
    action_token_hamming: int
    first_action_token_difference: int | None
    clean_action: list[float]
    composited_action: list[float]
    action_mae: float
    action_linf: float
    renderer_delta_linf: float
    per_instance_delta_linf: list[float]
    total_delta_linf: float
    saturated_pixel_fraction: float
    saturated_channel_fraction: float
    clean_renderer_requires_grad: bool
    valid_renderer_pixel_count: int
    instance_visible_pixel_counts: list[int]
    surface_parameter_gradient: dict[str, Any]
    transaction_verified: bool
    transaction_before_sha256: str
    transaction_after_sha256: str
    arrays_npz_sha256: str
    artifact_sha256: dict[str, str]
    failures: list[str]
    gate_pass: bool


class ZeroDeltaCompositorSummary(TypedDict):
    """states 集合完整性和最坏等价误差汇总。"""

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
    minimum_valid_renderer_pixel_count: int
    worst_source_rgb_linf: float
    worst_pre_crop_rgb_linf: float
    worst_effective_view_linf: float
    worst_processor_pixel_linf: float
    worst_action_linf: float
    structural_pass: bool
    gate_pass: bool


def _validate_rgb(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise ZeroDeltaCompositorAuditError(f"{name} 必须是 numpy array")
    if value.dtype != np.uint8:
        raise ZeroDeltaCompositorAuditError(
            f"{name} 必须为 uint8，收到 {value.dtype}"
        )
    if value.ndim != 3 or value.shape[2] != 3:
        raise ZeroDeltaCompositorAuditError(
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
        raise ZeroDeltaCompositorAuditError(f"{name} 必须是 numpy array")
    if reference.shape != candidate.shape:
        raise ZeroDeltaCompositorAuditError(
            f"{name} shape 不一致：{reference.shape} != {candidate.shape}"
        )
    if not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(
        candidate.dtype,
        np.floating,
    ):
        raise ZeroDeltaCompositorAuditError(f"{name} 必须为浮点数组")
    if not np.all(np.isfinite(reference)) or not np.all(
        np.isfinite(candidate)
    ):
        raise ZeroDeltaCompositorAuditError(f"{name} 包含 NaN/Inf")


def _error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ZeroDeltaCompositorAuditError(
            f"等价性数组 shape 不一致：{reference.shape} != {candidate.shape}"
        )
    if reference.size == 0:
        raise ZeroDeltaCompositorAuditError("等价性数组不得为空")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return (
        float(np.mean(np.abs(difference))),
        float(np.max(np.abs(difference))),
    )


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ZeroDeltaCompositorAuditError(
            "code_commit 必须是40位小写十六进制 Git SHA"
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def evaluate_zero_delta_compositor_evidence(
    evidence: ZeroDeltaCompositorEvidence,
    *,
    code_commit: str,
) -> ZeroDeltaCompositorRow:
    """判定一个 state；全部前向路径只接受逐值恒等。"""

    _validate_commit(code_commit)
    if evidence.state_id < 0:
        raise ZeroDeltaCompositorAuditError("state_id 必须为非负整数")
    rgb_pairs = (
        (
            "Policy Source",
            evidence.clean_source_rgb,
            evidence.composited_source_rgb,
        ),
        (
            "Policy Pre-Crop Canvas",
            evidence.clean_pre_crop_rgb,
            evidence.composited_pre_crop_rgb,
        ),
        (
            "Effective View",
            evidence.clean_effective_rgb,
            evidence.composited_effective_rgb,
        ),
    )
    for stage_name, clean, composited in rgb_pairs:
        _validate_rgb(f"clean {stage_name}", clean)
        _validate_rgb(f"composited {stage_name}", composited)
        if clean.shape != composited.shape:
            raise ZeroDeltaCompositorAuditError(
                f"{stage_name} RGB shape 不一致"
            )
    _validate_float_pair(
        "processor pixel values",
        evidence.clean_pixel_values,
        evidence.composited_pixel_values,
    )
    _validate_float_pair(
        "continuous action",
        evidence.clean_action,
        evidence.composited_action,
    )
    clean_tokens = np.asarray(evidence.clean_action_token_ids)
    composited_tokens = np.asarray(evidence.composited_action_token_ids)
    if clean_tokens.ndim != 1 or clean_tokens.size == 0:
        raise ZeroDeltaCompositorAuditError(
            "clean action token IDs 必须为非空一维数组"
        )
    if clean_tokens.shape != composited_tokens.shape:
        raise ZeroDeltaCompositorAuditError("action token shape 不一致")
    if not np.issubdtype(clean_tokens.dtype, np.integer) or not np.issubdtype(
        composited_tokens.dtype,
        np.integer,
    ):
        raise ZeroDeltaCompositorAuditError("action token IDs 必须为整数")
    if evidence.clean_action.shape != (clean_tokens.size,):
        raise ZeroDeltaCompositorAuditError(
            "连续 action shape 必须等于 action token 数"
        )

    source_mae, source_linf = _error_metrics(
        evidence.clean_source_rgb,
        evidence.composited_source_rgb,
    )
    pre_crop_mae, pre_crop_linf = _error_metrics(
        evidence.clean_pre_crop_rgb,
        evidence.composited_pre_crop_rgb,
    )
    effective_mae, effective_linf = _error_metrics(
        evidence.clean_effective_rgb,
        evidence.composited_effective_rgb,
    )
    pixel_mae, pixel_linf = _error_metrics(
        evidence.clean_pixel_values,
        evidence.composited_pixel_values,
    )
    action_mae, action_linf = _error_metrics(
        evidence.clean_action,
        evidence.composited_action,
    )
    token_differences = np.flatnonzero(clean_tokens != composited_tokens)
    token_hamming = int(token_differences.size)
    first_token_difference = (
        int(token_differences[0]) if token_differences.size > 0 else None
    )

    failures: list[str] = []
    for stage_name, linf in (
        ("Policy Source", source_linf),
        ("Policy Pre-Crop Canvas", pre_crop_linf),
        ("Effective View", effective_linf),
        ("checkpoint fused pixel values", pixel_linf),
        ("连续 action", action_linf),
    ):
        if not math.isfinite(linf) or linf != 0.0:
            failures.append(f"{stage_name} 不是逐值恒等")
    if token_hamming != 0:
        failures.append("action token sequence 不是逐值恒等")
    for name, value in (
        ("renderer F_adv-F_clean L∞", evidence.renderer_delta_linf),
        ("compositor total delta L∞", evidence.total_delta_linf),
        ("clamp 饱和像素比例", evidence.saturated_pixel_fraction),
        ("clamp 饱和通道比例", evidence.saturated_channel_fraction),
    ):
        if not math.isfinite(value) or value != 0.0:
            failures.append(f"零 Surface Delta 时{name}不为零")
    if not evidence.per_instance_delta_linf:
        failures.append("缺少逐实例 renderer delta 统计")
    elif any(
        not math.isfinite(value) or value != 0.0
        for value in evidence.per_instance_delta_linf
    ):
        failures.append("零 Surface Delta 时逐实例 renderer delta L∞不为零")
    if evidence.clean_renderer_requires_grad:
        failures.append("clean renderer 分支仍连接 autograd")
    if evidence.valid_renderer_pixel_count <= 0:
        failures.append("没有 valid renderer pixel")
    if (
        len(evidence.instance_visible_pixel_counts)
        != len(evidence.per_instance_delta_linf)
        or any(count < 0 for count in evidence.instance_visible_pixel_counts)
        or sum(evidence.instance_visible_pixel_counts) <= 0
    ):
        failures.append("逐实例 valid renderer pixel 统计无效")
    if not evidence.surface_parameter_gradient.is_finite_nonzero():
        failures.append(
            "adversarial renderer 对 Surface 参数的梯度不是有限非零"
        )
    if not evidence.transaction_verified:
        failures.append("static transaction 未通过")
    if not _is_sha256(evidence.transaction_before_sha256) or not _is_sha256(
        evidence.transaction_after_sha256
    ):
        failures.append("static transaction fingerprint 不是 SHA-256")
    elif (
        evidence.transaction_before_sha256
        != evidence.transaction_after_sha256
    ):
        failures.append("static transaction 前后 fingerprint 不一致")
    if not _is_sha256(evidence.arrays_npz_sha256):
        failures.append("arrays NPZ fingerprint 不是 SHA-256")
    if not evidence.artifact_sha256:
        failures.append("缺少 artifact SHA-256 inventory")
    elif any(
        not path or not _is_sha256(fingerprint)
        for path, fingerprint in evidence.artifact_sha256.items()
    ):
        failures.append("artifact inventory 包含无效路径或 SHA-256")
    elif evidence.arrays_npz_sha256 not in evidence.artifact_sha256.values():
        failures.append("artifact inventory 未绑定 arrays NPZ SHA-256")

    return ZeroDeltaCompositorRow(
        schema_version=COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION,
        code_commit=code_commit,
        state_id=evidence.state_id,
        source_rgb_shape=list(evidence.clean_source_rgb.shape),
        pre_crop_rgb_shape=list(evidence.clean_pre_crop_rgb.shape),
        effective_view_rgb_shape=list(evidence.clean_effective_rgb.shape),
        processor_pixel_shape=list(evidence.clean_pixel_values.shape),
        action_dim=int(clean_tokens.size),
        clean_source_rgb_sha256=_array_sha256(evidence.clean_source_rgb),
        composited_source_rgb_sha256=_array_sha256(
            evidence.composited_source_rgb
        ),
        clean_pre_crop_rgb_sha256=_array_sha256(
            evidence.clean_pre_crop_rgb
        ),
        composited_pre_crop_rgb_sha256=_array_sha256(
            evidence.composited_pre_crop_rgb
        ),
        clean_effective_rgb_sha256=_array_sha256(
            evidence.clean_effective_rgb
        ),
        composited_effective_rgb_sha256=_array_sha256(
            evidence.composited_effective_rgb
        ),
        source_rgb_mae=source_mae,
        source_rgb_linf=source_linf,
        pre_crop_rgb_mae=pre_crop_mae,
        pre_crop_rgb_linf=pre_crop_linf,
        effective_view_mae=effective_mae,
        effective_view_linf=effective_linf,
        processor_pixel_mae=pixel_mae,
        processor_pixel_linf=pixel_linf,
        clean_action_token_ids=[int(value) for value in clean_tokens],
        composited_action_token_ids=[
            int(value) for value in composited_tokens
        ],
        action_token_hamming=token_hamming,
        first_action_token_difference=first_token_difference,
        clean_action=[float(value) for value in evidence.clean_action],
        composited_action=[
            float(value) for value in evidence.composited_action
        ],
        action_mae=action_mae,
        action_linf=action_linf,
        renderer_delta_linf=float(evidence.renderer_delta_linf),
        per_instance_delta_linf=[
            float(value) for value in evidence.per_instance_delta_linf
        ],
        total_delta_linf=float(evidence.total_delta_linf),
        saturated_pixel_fraction=float(evidence.saturated_pixel_fraction),
        saturated_channel_fraction=float(
            evidence.saturated_channel_fraction
        ),
        clean_renderer_requires_grad=bool(
            evidence.clean_renderer_requires_grad
        ),
        valid_renderer_pixel_count=int(evidence.valid_renderer_pixel_count),
        instance_visible_pixel_counts=[
            int(value) for value in evidence.instance_visible_pixel_counts
        ],
        surface_parameter_gradient=asdict(
            evidence.surface_parameter_gradient
        ),
        transaction_verified=bool(evidence.transaction_verified),
        transaction_before_sha256=evidence.transaction_before_sha256,
        transaction_after_sha256=evidence.transaction_after_sha256,
        arrays_npz_sha256=evidence.arrays_npz_sha256,
        artifact_sha256=dict(sorted(evidence.artifact_sha256.items())),
        failures=failures,
        gate_pass=not failures,
    )


def summarize_zero_delta_compositor_rows(
    rows: Sequence[ZeroDeltaCompositorRow],
    *,
    expected_state_ids: Sequence[int],
) -> ZeroDeltaCompositorSummary:
    """汇总 state 完整性；重复、缺失、额外或混合 commit 均失败。"""

    expected = tuple(int(state_id) for state_id in expected_state_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ZeroDeltaCompositorAuditError(
            "expected_state_ids 必须为非空唯一集合"
        )
    observed = [int(row["state_id"]) for row in rows]
    counts = Counter(observed)
    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    duplicates = sorted(
        state_id for state_id, count in counts.items() if count > 1
    )
    code_commits = sorted({str(row["code_commit"]) for row in rows})
    schemas = {str(row["schema_version"]) for row in rows}
    structural_pass = bool(
        len(rows) == len(expected)
        and not missing
        and not unexpected
        and not duplicates
        and len(code_commits) == 1
        and schemas == {COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION}
    )
    total_tokens = sum(int(row["action_dim"]) for row in rows)
    matching_tokens = sum(
        int(row["action_dim"]) - int(row["action_token_hamming"])
        for row in rows
    )

    def _worst(field: str) -> float:
        return max((float(row[field]) for row in rows), default=float("nan"))

    minimum_valid_pixels = min(
        (int(row["valid_renderer_pixel_count"]) for row in rows),
        default=0,
    )
    return ZeroDeltaCompositorSummary(
        schema_version=COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION,
        expected_state_ids=sorted(expected),
        observed_state_ids=sorted(observed),
        missing_state_ids=missing,
        unexpected_state_ids=unexpected,
        duplicate_state_ids=duplicates,
        code_commits=code_commits,
        row_count=len(rows),
        passed_state_count=sum(bool(row["gate_pass"]) for row in rows),
        total_action_token_count=total_tokens,
        matching_action_token_count=matching_tokens,
        minimum_valid_renderer_pixel_count=minimum_valid_pixels,
        worst_source_rgb_linf=_worst("source_rgb_linf"),
        worst_pre_crop_rgb_linf=_worst("pre_crop_rgb_linf"),
        worst_effective_view_linf=_worst("effective_view_linf"),
        worst_processor_pixel_linf=_worst("processor_pixel_linf"),
        worst_action_linf=_worst("action_linf"),
        structural_pass=structural_pass,
        gate_pass=bool(
            structural_pass and all(bool(row["gate_pass"]) for row in rows)
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zero_delta_compositor_jsonl(
    rows: Sequence[ZeroDeltaCompositorRow],
    *,
    output_path: str | Path,
) -> str:
    """按原始 state ID 确定性写入权威 JSONL，并返回文件 SHA-256。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["state_id"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return _sha256_file(path)


def write_zero_delta_compositor_manifest(
    *,
    summary: ZeroDeltaCompositorSummary,
    metadata: Mapping[str, Any],
    metrics_jsonl_sha256: str,
    output_path: str | Path,
) -> str:
    """写入绑定 JSONL hash 的 manifest，并返回 manifest SHA-256。"""

    if len(metrics_jsonl_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in metrics_jsonl_sha256
    ):
        raise ZeroDeltaCompositorAuditError(
            "metrics_jsonl_sha256 必须是64位小写十六进制 SHA-256"
        )
    payload = {
        "schema_version": COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION,
        "summary": summary,
        "metadata": dict(metadata),
        "metrics_jsonl_sha256": metrics_jsonl_sha256,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256_file(path)
