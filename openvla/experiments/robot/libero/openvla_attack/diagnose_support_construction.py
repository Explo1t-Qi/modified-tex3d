"""从已验收 Seed Score 与 Visibility bundle 构造只读 Support candidates。

本命令是 CPU-only evidence 重放：不导入 OpenVLA/OFT、LIBERO、MuJoCo、
nvdiffrast 或 legacy optimizer，不计算新梯度，也不生成可供训练激活的 Fixed
Support。输出包含完整 contribution/candidate NPZ、最差 Primary state 投影图和
manifest；所有候选始终标记 ``fixed_support_frozen=false``。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
OPENVLA_ROOT = THIS_FILE.parents[4]
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
for import_path in (REPOSITORY_ROOT, OPENVLA_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.coverage_evidence import (  # noqa: E402
    transform_premultiplied_evidence,
)
from openvla_attack.seed_score_audit import (  # noqa: E402
    load_seed_score_artifact_arrays,
)
from openvla_attack.support_construction import (  # noqa: E402
    construct_akita_support_candidates,
)
from openvla_attack.support_construction_audit import (  # noqa: E402
    SUPPORT_AUDIT_SCHEMA_VERSION,
    SupportArtifactEvidence,
    SupportAuditProvenance,
    candidate_to_json,
    coverage_diagnostics_to_json,
    evaluate_support_artifacts,
    file_sha256,
    write_candidate_artifact,
    write_coverage_contribution_artifact,
)
from openvla_attack.support_coverage_contribution import (  # noqa: E402
    load_projection_coverage_evidence_npz,
)
from openvla_attack.visibility_evidence import (  # noqa: E402
    A_OBS_MIN_FROZEN,
    RECALL_MIN_FROZEN,
)


@dataclass(frozen=True)
class SupportConstructionAuditConfig:
    """只读 Support audit 的路径与 commit 绑定。"""

    seed_score_dir: str
    visibility_audit_dir: str
    output_dir: str
    code_commit: str


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> SupportConstructionAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_score_dir", required=True)
    parser.add_argument("--visibility_audit_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    return SupportConstructionAuditConfig(**vars(parser.parse_args(argv)))


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("code_commit 必须是40位小写十六进制 Git SHA")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root 必须是 object: {path}")
    return value


def _load_visibility_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("visibility metrics JSONL 为空或包含非 object 行")
    return rows


def _save_alpha(path: Path, alpha: torch.Tensor) -> str:
    array = (
        alpha.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)
    return file_sha256(path)


def _save_worst_state_images(
    *,
    output_dir: Path,
    visibility_root: Path,
    candidate_index: int,
    num_regions: int,
    state_id: int,
    support_mask: np.ndarray,
) -> dict[str, str]:
    """保存最差 Primary 的 MuJoCo、renderer 与 alpha*w effective 投影。"""

    arrays_path = (
        visibility_root / "arrays" / f"state_{state_id:02d}_primary.npz"
    )
    with np.load(arrays_path, allow_pickle=False) as archive:
        alpha_source = torch.from_numpy(
            np.ascontiguousarray(archive["mujoco_alpha_source"])
        )
        corners = archive["geometry_corner_indices"]
        barycentric = archive["barycentric"]
        valid = np.all(corners >= 0, axis=-1)
        safe_corners = np.maximum(corners, 0)
        control = np.where(
            valid,
            (
                barycentric
                * support_mask[safe_corners].astype(np.float32)
            ).sum(axis=-1),
            0.0,
        ).astype(np.float32)[:, None]
        stages = transform_premultiplied_evidence(
            alpha_source,
            alpha_source * torch.from_numpy(control),
        )
        mujoco_effective = torch.from_numpy(
            np.ascontiguousarray(archive["mujoco_alpha_effective"])
        ).sum(dim=0)[0]
        renderer_instances = torch.from_numpy(
            np.ascontiguousarray(archive["renderer_alpha_effective"])
        )
        renderer_effective = 1.0 - torch.prod(
            1.0 - renderer_instances,
            dim=0,
        )[0]
        support_effective = stages.effective.alpha_times_control.sum(dim=0)[0]
    prefix = f"candidate_{candidate_index:02d}_r{num_regions}_worst_state_{state_id:02d}"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mujoco_effective_mask": mujoco_effective,
        "renderer_effective_mask": renderer_effective,
        "support_premultiplied_effective": support_effective,
    }
    hashes: dict[str, str] = {}
    for name, alpha in outputs.items():
        path = image_dir / f"{prefix}_{name}.png"
        if path.exists():
            raise FileExistsError(path)
        hashes[str(path.relative_to(output_dir))] = _save_alpha(path, alpha)
    return hashes


def _validate_inputs(
    *,
    seed_root: Path,
    visibility_root: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    list[dict[str, Any]],
]:
    seed_manifest_path = seed_root / "seed_score_manifest.json"
    seed_manifest = _load_json(seed_manifest_path)
    if not bool(seed_manifest.get("decision", {}).get("gate_pass")):
        raise ValueError("Seed Score manifest 未通过正式 gate")
    if bool(seed_manifest.get("production_support_constructed")):
        raise ValueError("输入 Seed Score 不得包含 production support")
    score_relative_path = str(
        seed_manifest["evidence"]["artifact_relative_path"]
    )
    score_path = (seed_root / score_relative_path).resolve()
    if seed_root.resolve() not in score_path.parents or not score_path.is_file():
        raise ValueError("Seed Score artifact path 非法")
    if file_sha256(score_path) != str(
        seed_manifest["evidence"]["artifact_sha256"]
    ):
        raise ValueError("Seed Score artifact SHA-256 不匹配")

    visibility_manifest_path = visibility_root / "visibility_alignment_manifest.json"
    visibility_manifest = _load_json(visibility_manifest_path)
    if not bool(visibility_manifest.get("summary", {}).get("structural_pass")):
        raise ValueError("Visibility manifest structural_pass=false")
    thresholds = visibility_manifest["metadata"]["threshold_candidates"]
    if float(thresholds["observation_area_min"]) != A_OBS_MIN_FROZEN or float(
        thresholds["recall_min"]
    ) != RECALL_MIN_FROZEN:
        raise ValueError("Visibility manifest 门槛不是正式冻结值")
    metrics_path = visibility_root / "visibility_alignment_metrics.jsonl"
    if file_sha256(metrics_path) != str(
        visibility_manifest["metrics_jsonl_sha256"]
    ):
        raise ValueError("Visibility metrics SHA-256 不匹配")
    visibility_rows = _load_visibility_rows(metrics_path)
    return (
        seed_manifest,
        score_path,
        visibility_manifest,
        metrics_path,
        visibility_rows,
    )


def run_support_construction_audit(
    cfg: SupportConstructionAuditConfig,
) -> Path:
    """运行 CPU-only Support Construction audit 并返回 manifest path。"""

    _validate_commit(cfg.code_commit)
    seed_root = Path(cfg.seed_score_dir)
    visibility_root = Path(cfg.visibility_audit_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "support_coverage_contributions.npz"
    candidate_path = output_dir / "support_candidates.npz"
    manifest_path = output_dir / "support_construction_manifest.json"
    if any(path.exists() for path in (coverage_path, candidate_path, manifest_path)):
        raise FileExistsError("拒绝覆盖已有 Support Construction 权威文件")

    (
        seed_manifest,
        score_path,
        visibility_manifest,
        metrics_path,
        visibility_rows,
    ) = _validate_inputs(seed_root=seed_root, visibility_root=visibility_root)
    score_arrays = load_seed_score_artifact_arrays(score_path)
    expected_state_ids = tuple(int(value) for value in score_arrays.state_ids)
    if tuple(visibility_manifest["metadata"]["state_ids"]) != expected_state_ids:
        raise ValueError("Visibility 与 Seed Score source state IDs 不一致")
    score_mesh_sha = str(
        seed_manifest["evidence"]["provenance"]["mesh_file_sha256"]
    )
    visibility_mesh_sha = str(
        visibility_manifest["metadata"]["asset_sha256"]["mesh"]
    )
    if score_mesh_sha != visibility_mesh_sha:
        raise ValueError("Seed Score 与 Visibility 原始 OBJ SHA-256 不一致")

    row_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in visibility_rows:
        key = (int(row["state_id"]), str(row["view_name"]))
        if key in row_by_key:
            raise ValueError(f"重复 Visibility row: {key}")
        row_by_key[key] = row
    ordered_keys = tuple(
        (state_id, view_name)
        for state_id in expected_state_ids
        for view_name in ("primary", "wrist_source_crop_proxy")
    )
    if set(row_by_key) != set(ordered_keys):
        raise ValueError("Visibility rows 不是 source states × 两个冻结视角")

    coverage_rows = []
    diagnostics = []
    relative_paths: list[str] = []
    npz_hashes: list[str] = []
    for state_id, view_name in ordered_keys:
        row = row_by_key[(state_id, view_name)]
        relative_path = f"arrays/state_{state_id:02d}_{view_name}.npz"
        arrays_path = visibility_root / relative_path
        expected_sha = str(row["arrays_npz_sha256"])
        if file_sha256(arrays_path) != expected_sha:
            raise ValueError(f"Visibility arrays SHA-256 不匹配: {relative_path}")
        evidence, row_diagnostics = load_projection_coverage_evidence_npz(
            arrays_path,
            status=str(row["status"]),  # type: ignore[arg-type]
            num_vertices=len(score_arrays.vertices),
        )
        if evidence.state_id != state_id or evidence.view_name != view_name:
            raise ValueError("Visibility NPZ state/view 与 JSONL 不一致")
        coverage_rows.append(evidence)
        diagnostics.append(row_diagnostics)
        relative_paths.append(relative_path)
        npz_hashes.append(expected_sha)

    provenance = SupportAuditProvenance(
        code_commit=cfg.code_commit,
        seed_score_artifact_sha256=file_sha256(score_path),
        visibility_manifest_sha256=file_sha256(
            visibility_root / "visibility_alignment_manifest.json"
        ),
        visibility_metrics_sha256=file_sha256(metrics_path),
        visibility_npz_relative_paths=tuple(relative_paths),
        visibility_npz_sha256=tuple(npz_hashes),
    )
    coverage_sha = write_coverage_contribution_artifact(
        coverage_path,
        rows=coverage_rows,
        provenance=provenance,
    )
    primary = tuple(row for row in coverage_rows if row.view_name == "primary")
    wrist = tuple(
        row for row in coverage_rows if row.view_name == "wrist_source_crop_proxy"
    )
    result = construct_akita_support_candidates(
        vertices=score_arrays.vertices,
        faces=score_arrays.faces,
        vertex_mass=score_arrays.vertex_mass,
        smoothed_density=score_arrays.smoothed_density,
        primary_evidence=primary,
        wrist_evidence=wrist,
    )
    candidate_sha = write_candidate_artifact(
        candidate_path,
        result=result,
        provenance=provenance,
        coverage_artifact_sha256=coverage_sha,
    )
    artifact_evidence = SupportArtifactEvidence(
        schema_version=SUPPORT_AUDIT_SCHEMA_VERSION,
        provenance=provenance,
        coverage_artifact_relative_path=coverage_path.name,
        coverage_artifact_sha256=coverage_sha,
        candidate_artifact_relative_path=candidate_path.name,
        candidate_artifact_sha256=candidate_sha,
        production_support_constructed=False,
        fixed_support_frozen=False,
    )
    decision = evaluate_support_artifacts(
        artifact_evidence,
        artifact_root=output_dir,
        score_arrays=score_arrays,
    )

    image_hashes: dict[str, str] = {}
    for candidate_index, candidate in enumerate(result.candidates):
        if candidate.worst_primary_state_id is None:
            continue
        image_hashes.update(
            _save_worst_state_images(
                output_dir=output_dir,
                visibility_root=visibility_root,
                candidate_index=candidate_index,
                num_regions=candidate.num_regions,
                state_id=candidate.worst_primary_state_id,
                support_mask=candidate.support_mask,
            )
        )
    manifest = {
        "schema_version": SUPPORT_AUDIT_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "input": {
            "seed_score_manifest_sha256": file_sha256(
                seed_root / "seed_score_manifest.json"
            ),
            "seed_score_code_commit": seed_manifest["code_commit"],
            "visibility_code_commit": visibility_manifest["metadata"]["code_commit"],
            "mesh_file_sha256": score_mesh_sha,
            "source_state_ids": list(expected_state_ids),
        },
        "frozen_rules": {
            "target_area_fraction": result.target_area_fraction,
            "max_regions": result.max_regions,
            "primary_coverage_min": result.primary_coverage_min,
            "smoothing_alpha": result.smoothing_alpha,
            "smoothing_length": result.smoothing_length,
            "region_scheduler": "seed_order_deterministic_round_robin_best_first",
            "marginal_gain": "worst_valid_primary_state_positive_projection_gain",
            "seed_separation": "original_obj_edge_length_geodesic",
        },
        "evidence": asdict(artifact_evidence),
        "decision": asdict(decision),
        "coverage_diagnostics": coverage_diagnostics_to_json(diagnostics),
        "candidates": [candidate_to_json(candidate) for candidate in result.candidates],
        "selected_candidate_index": result.selected_candidate_index,
        "image_sha256": image_hashes,
        "coverage_computed": True,
        "support_candidate_computed": True,
        "production_support_constructed": False,
        "fixed_support_frozen": False,
        "feature_gradient_used": False,
        "wrist_gradient_used": False,
        "oft_used": False,
        "legacy_optimization_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_gate_pass": decision.gate_pass,
                "failures": list(decision.failures),
                "selected_candidate_index": result.selected_candidate_index,
                "candidate_count": len(result.candidates),
                "manifest_sha256": file_sha256(manifest_path),
                "fixed_support_frozen": False,
            },
            sort_keys=True,
        )
    )
    if not decision.gate_pass:
        raise RuntimeError("Support Construction artifact 验收失败；证据已写入")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_support_construction_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
