"""Gate 2E deployment backward/update/bake 的机器可读证据契约。

本模块只定义数值统计、严格判定和 JSON 持久化，不导入 LIBERO、模型或 CUDA。
真实 runner 负责采集一条权威记录；WSL 可以独立重算 ``gate_pass``，避免只相信
服务器 stdout 中的一句 PASS。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION: Final[str] = "openvla-gate-2e-v1"
EXPECTED_SPECTRAL_PARAMETER_SHAPE: Final[tuple[int, int]] = (256, 3)
MAX_SURFACE_STEP: Final[float] = 2.0 / 255.0
MAX_SURFACE_DELTA: Final[float] = 128.0 / 255.0
NUMERIC_TOLERANCE: Final[float] = 1e-7


@dataclass(frozen=True)
class GradientEvidence:
    """一个 autograd stage 的有限/非零统计。"""

    shape: tuple[int, ...]
    numel: int
    finite_count: int
    nonzero_count: int
    l2_norm: float
    linf_norm: float

    @classmethod
    def from_tensor(cls, gradient: torch.Tensor) -> "GradientEvidence":
        """从已 detach 的任意 shape 梯度构造可 JSON 化统计。"""

        flattened = gradient.detach().to(torch.float64).reshape(-1).cpu()
        finite_mask = torch.isfinite(flattened)
        finite_values = flattened[finite_mask]
        if finite_values.numel() == 0:
            l2_norm = float("nan")
            linf_norm = float("nan")
        else:
            l2_norm = float(torch.linalg.vector_norm(finite_values).item())
            linf_norm = float(finite_values.abs().amax().item())
        return cls(
            shape=tuple(int(size) for size in gradient.shape),
            numel=int(flattened.numel()),
            finite_count=int(finite_mask.sum().item()),
            nonzero_count=int((finite_values != 0).sum().item()),
            l2_norm=l2_norm,
            linf_norm=linf_norm,
        )

    def is_finite_nonzero(self) -> bool:
        return (
            self.numel > 0
            and self.finite_count == self.numel
            and self.nonzero_count > 0
            and np.isfinite(self.l2_norm)
            and np.isfinite(self.linf_norm)
            and self.l2_norm > 0.0
            and self.linf_norm > 0.0
        )


@dataclass(frozen=True)
class DeploymentBackwardEvidence:
    """一次单 state、单 update Gate 2E 的完整权威记录。"""

    schema_version: str
    code_commit: str
    state_id: int
    objective: str
    action_loss: float
    parameter_shape: tuple[int, ...]
    parameter_changed_count: int
    parameter_linf_change: float
    source_rgb_gradient: GradientEvidence
    pre_crop_gradient: GradientEvidence
    effective_view_gradient: GradientEvidence
    surface_delta_gradient: GradientEvidence
    parameter_gradient: GradientEvidence
    actual_surface_step: float
    max_surface_delta: float
    baked_texture_path: str
    baked_texture_sha256: str
    active_texture_sha256: str
    rollout_completed: bool
    rollout_success: bool
    xml_sha256_before: str
    xml_sha256_after_restore: str
    real_texture_sha256_before: str
    real_texture_sha256_after_restore: str
    backup_paths_removed: bool
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class DeploymentBackwardDecision:
    """Gate 2E 判定及逐项失败原因。"""

    gate_pass: bool
    failures: tuple[str, ...]


def evaluate_deployment_backward_evidence(
    evidence: DeploymentBackwardEvidence,
) -> DeploymentBackwardDecision:
    """按冻结门槛重算 Gate 2E；不接受缺字段或近似人工解释。"""

    failures: list[str] = []
    if evidence.schema_version != SCHEMA_VERSION:
        failures.append("schema_version 不匹配")
    if (
        len(evidence.code_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in evidence.code_commit
        )
    ):
        failures.append("code_commit 不是40位小写十六进制 SHA")
    if evidence.parameter_shape != EXPECTED_SPECTRAL_PARAMETER_SHAPE:
        failures.append("谱系数 shape 不是 [256,3]")
    if not np.isfinite(evidence.action_loss):
        failures.append("Action loss 不是有限值")
    expected_parameter_count = int(np.prod(evidence.parameter_shape))
    if not 0 < evidence.parameter_changed_count <= expected_parameter_count:
        failures.append("谱系数没有真实更新")
    if not (
        np.isfinite(evidence.parameter_linf_change)
        and evidence.parameter_linf_change > 0.0
    ):
        failures.append("谱系数 L∞ change 不是有限正数")

    gradients = {
        "Policy Source": evidence.source_rgb_gradient,
        "Pre-Crop": evidence.pre_crop_gradient,
        "Effective View": evidence.effective_view_gradient,
        "Surface Delta": evidence.surface_delta_gradient,
        "Parameter": evidence.parameter_gradient,
    }
    for stage_name, gradient in gradients.items():
        if not gradient.is_finite_nonzero():
            failures.append(f"{stage_name} 梯度不是有限非零")

    if not (
        np.isfinite(evidence.actual_surface_step)
        and evidence.actual_surface_step > 0.0
        and evidence.actual_surface_step
        <= MAX_SURFACE_STEP + NUMERIC_TOLERANCE
    ):
        failures.append("Actual Surface Step 超限或非正")
    if not (
        np.isfinite(evidence.max_surface_delta)
        and evidence.max_surface_delta >= 0.0
        and evidence.max_surface_delta
        <= MAX_SURFACE_DELTA + NUMERIC_TOLERANCE
    ):
        failures.append("Max Surface Delta 超过 128/255")
    if not evidence.baked_texture_sha256:
        failures.append("缺少 bake PNG hash")
    if evidence.active_texture_sha256 != evidence.baked_texture_sha256:
        failures.append("Active Texture 与 bake PNG 不一致")
    if not evidence.rollout_completed:
        failures.append("rollout 未正常完成")
    if evidence.xml_sha256_after_restore != evidence.xml_sha256_before:
        failures.append("XML 未恢复到事务前内容")
    if (
        evidence.real_texture_sha256_after_restore
        != evidence.real_texture_sha256_before
    ):
        failures.append("真实纹理未恢复到事务前内容")
    if not evidence.backup_paths_removed:
        failures.append("Runtime Asset backup 未删除")
    return DeploymentBackwardDecision(
        gate_pass=not failures,
        failures=tuple(failures),
    )


def write_deployment_backward_evidence(
    path: str | Path,
    evidence: DeploymentBackwardEvidence,
) -> Path:
    """写入单文件 JSON，并附上可由 WSL 独立重算的 Gate decision。"""

    decision = evaluate_deployment_backward_evidence(evidence)
    payload = {
        "evidence": asdict(evidence),
        "decision": asdict(decision),
    }
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def load_deployment_backward_evidence(
    path: str | Path,
) -> DeploymentBackwardEvidence:
    """以严格字段访问读取 JSON；缺少字段时直接失败。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value: Mapping[str, Any] = payload["evidence"]
    gradient_fields: Sequence[str] = (
        "source_rgb_gradient",
        "pre_crop_gradient",
        "effective_view_gradient",
        "surface_delta_gradient",
        "parameter_gradient",
    )
    resolved = dict(value)
    for field_name in gradient_fields:
        resolved[field_name] = _gradient_from_mapping(value[field_name])
    resolved["parameter_shape"] = tuple(value["parameter_shape"])
    return DeploymentBackwardEvidence(**resolved)
