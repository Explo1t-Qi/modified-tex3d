"""Dense Seed Audit 的完整 raw ``G_s`` artifact/evidence 契约。

本模块是纯 CPU contract：它把每个 source state 的零扰动 Action-only dense
geometry gradient 保存为禁止 pickle 的 NPZ，并提供可在 WSL 独立执行的严格
复核。artifact 只包含原始 ``G_s [N_v,3]`` 与生成它所需的 provenance；禁止
提前写入 Support Seed Score、density、coverage 或 Fixed Vertex Support。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from .action_objective_audit import (
    ACTION_OBJECTIVE_NAME,
    EXPECTED_STATE_IDS,
    ActionObjectiveAuditEvidence,
    evaluate_action_objective_evidence,
)
from .deployment_backward_audit import GradientEvidence


DENSE_SEED_AUDIT_SCHEMA_VERSION: Final[str] = "openvla-dense-seed-audit-v1"
DENSE_SEED_ARTIFACT_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "code_commit",
    "state_id",
    "initial_state_sha256",
    "objective",
    "parameterization",
    "zero_surface_delta_linf",
    "clean_action_token_ids",
    "clean_classes",
    "margins",
    "hinge_values",
    "action_loss",
    "dense_geometry_gradient",
    "mesh_sha256",
    "render_to_geometry_sha256",
    "policy_source_rgb_sha256",
    "effective_view_rgb_sha256",
    "mujoco_instance_alpha_sha256",
    "renderer_visibility_sha256",
    "shared_instance_body_ids",
    "shared_instance_body_names",
)
NUMERIC_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class DenseSeedCaptureMetadata:
    """一个 state 的输入、几何 correspondence 与共享实例 provenance。"""

    mesh_sha256: str
    render_to_geometry_sha256: str
    policy_source_rgb_sha256: str
    effective_view_rgb_sha256: str
    mujoco_instance_alpha_sha256: str
    renderer_visibility_sha256: str
    shared_instance_body_ids: tuple[int, ...]
    shared_instance_body_names: tuple[str, ...]


@dataclass(frozen=True)
class DenseSeedGradientEvidence:
    """一个 raw ``G_s`` NPZ 及其 Objective/scene 绑定。"""

    schema_version: str
    objective_evidence: ActionObjectiveAuditEvidence
    metadata: DenseSeedCaptureMetadata
    artifact_relative_path: str
    artifact_sha256: str
    dense_geometry_gradient_sha256: str


@dataclass(frozen=True)
class DenseSeedGradientDecision:
    """单 state NPZ 与 JSON evidence 的严格判定。"""

    gate_pass: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class DenseSeedGradientSummary:
    """states 0--9 raw dense gradients 的完整性汇总。"""

    gate_pass: bool
    failures: tuple[str, ...]
    state_ids: tuple[int, ...]
    valid_state_count: int
    artifact_count: int
    num_geometry_vertices: int
    total_gradient_value_count: int
    unique_gradient_hash_count: int
    dense_gradient_l2_min: float
    dense_gradient_l2_max: float
    production_support_constructed: bool


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def array_sha256(array: np.ndarray) -> str:
    """按 contiguous dtype/shape/bytes 计算稳定数组 SHA-256。"""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_failures(metadata: DenseSeedCaptureMetadata) -> list[str]:
    failures: list[str] = []
    for field_name in (
        "mesh_sha256",
        "render_to_geometry_sha256",
        "policy_source_rgb_sha256",
        "effective_view_rgb_sha256",
        "mujoco_instance_alpha_sha256",
        "renderer_visibility_sha256",
    ):
        if not _is_sha256(getattr(metadata, field_name)):
            failures.append(f"{field_name} 非法")
    if not metadata.shared_instance_body_ids:
        failures.append("共享纹理实例不得为空")
    if len(metadata.shared_instance_body_ids) != len(
        metadata.shared_instance_body_names
    ):
        failures.append("共享实例 body ID/name 数量不一致")
    if len(set(metadata.shared_instance_body_ids)) != len(
        metadata.shared_instance_body_ids
    ):
        failures.append("共享实例 body IDs 包含重复")
    if any(not name for name in metadata.shared_instance_body_names):
        failures.append("共享实例 body name 不得为空")
    return failures


def _safe_artifact_path(root: Path, relative_path: str) -> Path | None:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def write_dense_seed_state_artifact(
    *,
    output_root: str | Path,
    objective_evidence: ActionObjectiveAuditEvidence,
    dense_geometry_gradient: np.ndarray,
    metadata: DenseSeedCaptureMetadata,
) -> DenseSeedGradientEvidence:
    """独占写入一个 state 的完整 raw ``G_s`` NPZ 并返回索引证据。"""

    objective_decision = evaluate_action_objective_evidence(
        objective_evidence
    )
    if not objective_decision.gate_pass:
        raise ValueError(
            "Dense Seed artifact 拒绝无效 objective evidence: "
            + "; ".join(objective_decision.failures)
        )
    metadata_errors = _metadata_failures(metadata)
    if metadata_errors:
        raise ValueError(
            "Dense Seed capture metadata 无效: " + "; ".join(metadata_errors)
        )
    gradient = np.asarray(dense_geometry_gradient)
    if gradient.dtype != np.float32:
        raise ValueError("dense_geometry_gradient 必须为 float32")
    if gradient.shape != objective_evidence.parameter_shape:
        raise ValueError(
            "dense_geometry_gradient shape 与 dense parameter 不一致"
        )
    if not np.all(np.isfinite(gradient)) or not np.any(gradient != 0.0):
        raise ValueError("dense_geometry_gradient 必须有限非零")
    gradient = np.ascontiguousarray(gradient)
    gradient_sha256 = array_sha256(gradient)
    if gradient_sha256 != objective_evidence.dense_geometry_gradient_sha256:
        raise ValueError("raw G_s 与 objective evidence gradient hash 不一致")

    root = Path(output_root)
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    relative_path = Path(
        "arrays",
        f"state_{objective_evidence.state_id:02d}_dense_action_gradient.npz",
    )
    artifact_path = root / relative_path
    if artifact_path.exists():
        raise FileExistsError(f"拒绝覆盖 Dense Seed artifact: {artifact_path}")
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray(DENSE_SEED_AUDIT_SCHEMA_VERSION),
        code_commit=np.asarray(objective_evidence.code_commit),
        state_id=np.asarray(objective_evidence.state_id, dtype=np.int64),
        initial_state_sha256=np.asarray(
            objective_evidence.initial_state_sha256
        ),
        objective=np.asarray(objective_evidence.objective),
        parameterization=np.asarray(objective_evidence.parameterization),
        zero_surface_delta_linf=np.asarray(
            objective_evidence.zero_surface_delta_linf,
            dtype=np.float64,
        ),
        clean_action_token_ids=np.asarray(
            objective_evidence.clean_action_token_ids,
            dtype=np.int64,
        ),
        clean_classes=np.asarray(
            objective_evidence.clean_classes,
            dtype=np.int64,
        ),
        margins=np.asarray(objective_evidence.margins, dtype=np.float32),
        hinge_values=np.asarray(
            objective_evidence.hinge_values,
            dtype=np.float32,
        ),
        action_loss=np.asarray(objective_evidence.action_loss, dtype=np.float64),
        dense_geometry_gradient=gradient,
        mesh_sha256=np.asarray(metadata.mesh_sha256),
        render_to_geometry_sha256=np.asarray(
            metadata.render_to_geometry_sha256
        ),
        policy_source_rgb_sha256=np.asarray(
            metadata.policy_source_rgb_sha256
        ),
        effective_view_rgb_sha256=np.asarray(
            metadata.effective_view_rgb_sha256
        ),
        mujoco_instance_alpha_sha256=np.asarray(
            metadata.mujoco_instance_alpha_sha256
        ),
        renderer_visibility_sha256=np.asarray(
            metadata.renderer_visibility_sha256
        ),
        shared_instance_body_ids=np.asarray(
            metadata.shared_instance_body_ids,
            dtype=np.int64,
        ),
        shared_instance_body_names=np.asarray(
            metadata.shared_instance_body_names,
            dtype=np.str_,
        ),
    )
    return DenseSeedGradientEvidence(
        schema_version=DENSE_SEED_AUDIT_SCHEMA_VERSION,
        objective_evidence=objective_evidence,
        metadata=metadata,
        artifact_relative_path=str(relative_path),
        artifact_sha256=_file_sha256(artifact_path),
        dense_geometry_gradient_sha256=gradient_sha256,
    )


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"{key} 必须是 scalar")
    return value.item()


def evaluate_dense_seed_gradient_evidence(
    evidence: DenseSeedGradientEvidence,
    *,
    artifact_root: str | Path,
) -> DenseSeedGradientDecision:
    """读取并逐数组复核一个 raw ``G_s`` artifact。"""

    failures: list[str] = []
    if evidence.schema_version != DENSE_SEED_AUDIT_SCHEMA_VERSION:
        failures.append("schema_version 不匹配")
    objective_decision = evaluate_action_objective_evidence(
        evidence.objective_evidence
    )
    failures.extend(
        f"objective: {failure}" for failure in objective_decision.failures
    )
    failures.extend(_metadata_failures(evidence.metadata))
    if (
        evidence.dense_geometry_gradient_sha256
        != evidence.objective_evidence.dense_geometry_gradient_sha256
    ):
        failures.append("raw G_s SHA-256 与 objective evidence 不一致")
    if not _is_sha256(evidence.dense_geometry_gradient_sha256):
        failures.append("raw G_s SHA-256 非法")
    if not _is_sha256(evidence.artifact_sha256):
        failures.append("artifact SHA-256 非法")

    root = Path(artifact_root)
    artifact_path = _safe_artifact_path(root, evidence.artifact_relative_path)
    if artifact_path is None:
        failures.append("artifact_relative_path 非法")
        return DenseSeedGradientDecision(False, tuple(failures))
    if not artifact_path.is_file():
        failures.append("artifact 文件不存在")
        return DenseSeedGradientDecision(False, tuple(failures))
    if _file_sha256(artifact_path) != evidence.artifact_sha256:
        failures.append("artifact SHA-256 不匹配")
        return DenseSeedGradientDecision(False, tuple(failures))

    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            if set(archive.files) != set(DENSE_SEED_ARTIFACT_KEYS):
                failures.append("artifact keys 与冻结 schema 不一致")
                return DenseSeedGradientDecision(False, tuple(failures))
            objective = evidence.objective_evidence
            metadata = evidence.metadata
            scalar_expectations = {
                "schema_version": DENSE_SEED_AUDIT_SCHEMA_VERSION,
                "code_commit": objective.code_commit,
                "state_id": objective.state_id,
                "initial_state_sha256": objective.initial_state_sha256,
                "objective": ACTION_OBJECTIVE_NAME,
                "parameterization": "geometry_vertex_dense",
                "mesh_sha256": metadata.mesh_sha256,
                "render_to_geometry_sha256": (
                    metadata.render_to_geometry_sha256
                ),
                "policy_source_rgb_sha256": metadata.policy_source_rgb_sha256,
                "effective_view_rgb_sha256": (
                    metadata.effective_view_rgb_sha256
                ),
                "mujoco_instance_alpha_sha256": (
                    metadata.mujoco_instance_alpha_sha256
                ),
                "renderer_visibility_sha256": (
                    metadata.renderer_visibility_sha256
                ),
            }
            for key, expected in scalar_expectations.items():
                if _scalar(archive, key) != expected:
                    failures.append(f"artifact {key} 与 evidence 不一致")
            if abs(
                float(_scalar(archive, "zero_surface_delta_linf"))
                - objective.zero_surface_delta_linf
            ) > NUMERIC_TOLERANCE:
                failures.append("artifact zero Surface Delta 与 evidence 不一致")
            if abs(
                float(_scalar(archive, "action_loss"))
                - objective.action_loss
            ) > NUMERIC_TOLERANCE:
                failures.append("artifact Action loss 与 evidence 不一致")

            exact_arrays = {
                "clean_action_token_ids": np.asarray(
                    objective.clean_action_token_ids,
                    dtype=np.int64,
                ),
                "clean_classes": np.asarray(
                    objective.clean_classes,
                    dtype=np.int64,
                ),
                "shared_instance_body_ids": np.asarray(
                    metadata.shared_instance_body_ids,
                    dtype=np.int64,
                ),
                "shared_instance_body_names": np.asarray(
                    metadata.shared_instance_body_names,
                    dtype=np.str_,
                ),
            }
            for key, expected in exact_arrays.items():
                if not np.array_equal(archive[key], expected):
                    failures.append(f"artifact {key} 与 evidence 不一致")
            for key, expected_values in (
                ("margins", objective.margins),
                ("hinge_values", objective.hinge_values),
            ):
                if not np.allclose(
                    archive[key],
                    np.asarray(expected_values, dtype=np.float32),
                    rtol=0.0,
                    atol=NUMERIC_TOLERANCE,
                ):
                    failures.append(f"artifact {key} 与 evidence 不一致")

            gradient = np.asarray(archive["dense_geometry_gradient"])
            if gradient.dtype != np.float32:
                failures.append("raw G_s dtype 不是 float32")
            if gradient.shape != objective.parameter_shape:
                failures.append("raw G_s shape 与 dense parameter 不一致")
            if not np.all(np.isfinite(gradient)) or not np.any(
                gradient != 0.0
            ):
                failures.append("raw G_s 不是有限非零数组")
            if array_sha256(gradient) != evidence.dense_geometry_gradient_sha256:
                failures.append("raw G_s payload SHA-256 不匹配")
            recomputed_stats = GradientEvidence.from_tensor(
                torch.from_numpy(np.ascontiguousarray(gradient))
            )
            if recomputed_stats != objective.dense_geometry_gradient:
                failures.append("raw G_s 统计与 objective evidence 不一致")
    except (OSError, ValueError, KeyError) as error:
        failures.append(f"artifact 无法严格加载: {error}")
    return DenseSeedGradientDecision(
        gate_pass=not failures,
        failures=tuple(failures),
    )


def summarize_dense_seed_gradient_evidence(
    evidence_rows: Sequence[DenseSeedGradientEvidence],
    *,
    artifact_root: str | Path,
) -> DenseSeedGradientSummary:
    """复核恰好 states 0--9 的全部 raw gradients 与共同几何 provenance。"""

    rows = list(evidence_rows)
    state_ids = tuple(row.objective_evidence.state_id for row in rows)
    failures: list[str] = []
    if state_ids != EXPECTED_STATE_IDS:
        failures.append("state IDs 必须精确等于 0-9")
    decisions = [
        evaluate_dense_seed_gradient_evidence(
            row,
            artifact_root=artifact_root,
        )
        for row in rows
    ]
    for row, decision in zip(rows, decisions):
        failures.extend(
            f"state {row.objective_evidence.state_id}: {failure}"
            for failure in decision.failures
        )
    if len({row.metadata.mesh_sha256 for row in rows}) > 1:
        failures.append("所有 states 必须绑定同一 OBJ mesh")
    if len({row.metadata.render_to_geometry_sha256 for row in rows}) > 1:
        failures.append("所有 states 必须绑定同一 render-to-geometry mapping")
    if len(
        {row.objective_evidence.parameter_shape for row in rows}
    ) > 1:
        failures.append("所有 states 的 dense geometry shape 必须一致")
    relative_paths = [row.artifact_relative_path for row in rows]
    if len(set(relative_paths)) != len(relative_paths):
        failures.append("Dense Seed artifact 路径包含重复")

    num_geometry_vertices = (
        int(rows[0].objective_evidence.parameter_shape[0]) if rows else 0
    )
    total_values = sum(
        int(np.prod(row.objective_evidence.parameter_shape)) for row in rows
    )
    gradient_norms = np.asarray(
        [
            row.objective_evidence.dense_geometry_gradient.l2_norm
            for row in rows
        ],
        dtype=np.float64,
    )
    finite_norms = gradient_norms[np.isfinite(gradient_norms)]
    return DenseSeedGradientSummary(
        gate_pass=not failures,
        failures=tuple(failures),
        state_ids=state_ids,
        valid_state_count=sum(decision.gate_pass for decision in decisions),
        artifact_count=len(rows),
        num_geometry_vertices=num_geometry_vertices,
        total_gradient_value_count=total_values,
        unique_gradient_hash_count=len(
            {row.dense_geometry_gradient_sha256 for row in rows}
        ),
        dense_gradient_l2_min=(
            float(finite_norms.min()) if finite_norms.size else float("nan")
        ),
        dense_gradient_l2_max=(
            float(finite_norms.max()) if finite_norms.size else float("nan")
        ),
        production_support_constructed=False,
    )


def write_dense_seed_gradient_evidence(
    path: str | Path,
    evidence_rows: Sequence[DenseSeedGradientEvidence],
    *,
    artifact_root: str | Path,
) -> Path:
    """独占写入 JSONL，每行附带从 NPZ 重算得到的 decision。"""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("x", encoding="utf-8") as handle:
        for evidence in evidence_rows:
            decision = evaluate_dense_seed_gradient_evidence(
                evidence,
                artifact_root=artifact_root,
            )
            payload = {
                "evidence": asdict(evidence),
                "decision": asdict(decision),
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


def _objective_from_mapping(
    value: Mapping[str, Any],
) -> ActionObjectiveAuditEvidence:
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
    return ActionObjectiveAuditEvidence(**resolved)


def load_dense_seed_gradient_evidence(
    path: str | Path,
) -> list[DenseSeedGradientEvidence]:
    """严格读取 JSONL；NPZ 内容由 evaluator 另行逐文件复核。"""

    rows: list[DenseSeedGradientEvidence] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        value: Mapping[str, Any] = payload["evidence"]
        metadata_value: Mapping[str, Any] = value["metadata"]
        metadata = DenseSeedCaptureMetadata(
            mesh_sha256=str(metadata_value["mesh_sha256"]),
            render_to_geometry_sha256=str(
                metadata_value["render_to_geometry_sha256"]
            ),
            policy_source_rgb_sha256=str(
                metadata_value["policy_source_rgb_sha256"]
            ),
            effective_view_rgb_sha256=str(
                metadata_value["effective_view_rgb_sha256"]
            ),
            mujoco_instance_alpha_sha256=str(
                metadata_value["mujoco_instance_alpha_sha256"]
            ),
            renderer_visibility_sha256=str(
                metadata_value["renderer_visibility_sha256"]
            ),
            shared_instance_body_ids=tuple(
                int(item)
                for item in metadata_value["shared_instance_body_ids"]
            ),
            shared_instance_body_names=tuple(
                str(item)
                for item in metadata_value["shared_instance_body_names"]
            ),
        )
        rows.append(
            DenseSeedGradientEvidence(
                schema_version=str(value["schema_version"]),
                objective_evidence=_objective_from_mapping(
                    value["objective_evidence"]
                ),
                metadata=metadata,
                artifact_relative_path=str(value["artifact_relative_path"]),
                artifact_sha256=str(value["artifact_sha256"]),
                dense_geometry_gradient_sha256=str(
                    value["dense_geometry_gradient_sha256"]
                ),
            )
        )
    return rows
