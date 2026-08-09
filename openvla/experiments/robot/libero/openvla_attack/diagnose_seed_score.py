"""从正式 Dense Seed bundle 生成 Seed Score/density/smoothing evidence。

这是纯 CPU 命令：它不加载 OpenVLA、OFT、LIBERO 环境或 legacy optimizer，
也不生成 Fixed Vertex Support。原始 OBJ 几何会以内嵌数组形式写入 NPZ，使 WSL
能够脱离服务器 LIBERO 资产重新构造 mass/Laplacian 并复核全部中间量。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openvla.experiments.robot.libero.openvla_attack.dense_seed_audit import (
    DENSE_SEED_AUDIT_SCHEMA_VERSION,
    load_dense_seed_gradient_evidence,
    summarize_dense_seed_gradient_evidence,
)
from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    AKITA_SMOOTHING_ALPHA,
    SEED_SCORE_SCHEMA_VERSION,
    build_seed_score_provenance,
    compute_seed_score_arrays,
    evaluate_seed_score_artifact,
    evidence_to_json,
    file_sha256,
    load_dense_gradient_stack,
    write_seed_score_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    load_obj_geometry,
)


@dataclass(frozen=True)
class SeedScoreAuditConfig:
    """Seed Score CPU audit CLI schema。"""

    dense_audit_dir: str
    mesh_path: str
    output_dir: str
    code_commit: str


def _parse_args(argv: Optional[Sequence[str]] = None) -> SeedScoreAuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense_audit_dir", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    return SeedScoreAuditConfig(**vars(parser.parse_args(argv)))


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("code_commit必须为40位小写十六进制Git SHA")


def _load_dense_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DENSE_SEED_AUDIT_SCHEMA_VERSION:
        raise ValueError("Dense manifest schema不匹配")
    if payload.get("summary", {}).get("gate_pass") is not True:
        raise ValueError("Dense manifest未通过Gate")
    if payload.get("production_support_constructed") is not False:
        raise ValueError("Dense manifest production support字段非法")
    return payload


def run_seed_score_audit(cfg: SeedScoreAuditConfig) -> Path:
    """生成并立即从raw Dense NPZ独立复算score artifact。"""

    _validate_commit(cfg.code_commit)
    dense_root = Path(cfg.dense_audit_dir)
    mesh_path = Path(cfg.mesh_path)
    output_root = Path(cfg.output_dir)
    metrics_path = dense_root / "dense_seed_metrics.jsonl"
    dense_manifest_path = dense_root / "dense_seed_manifest.json"
    for path in (metrics_path, dense_manifest_path, mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "seed_score_manifest.json"
    artifact_path = output_root / "seed_score_arrays.npz"
    if manifest_path.exists() or artifact_path.exists():
        raise FileExistsError("拒绝覆盖已有Seed Score权威文件")

    dense_manifest = _load_dense_manifest(dense_manifest_path)
    if dense_manifest.get("metrics_sha256") != file_sha256(metrics_path):
        raise ValueError("Dense metrics SHA-256与manifest不一致")
    rows = load_dense_seed_gradient_evidence(metrics_path)
    dense_summary = summarize_dense_seed_gradient_evidence(
        rows,
        artifact_root=dense_root,
    )
    if not dense_summary.gate_pass:
        raise ValueError(
            "Dense Seed bundle本地重算未通过: "
            + "; ".join(dense_summary.failures)
        )
    if dense_manifest.get("code_commit") != rows[0].objective_evidence.code_commit:
        raise ValueError("Dense manifest/rows code commit不一致")
    expected_artifacts = {
        row.artifact_relative_path: row.artifact_sha256 for row in rows
    }
    if dense_manifest.get("artifact_sha256") != expected_artifacts:
        raise ValueError("Dense artifact inventory与manifest不一致")

    vertices, faces = load_obj_geometry(mesh_path)
    mesh_file_hash = file_sha256(mesh_path)
    gradients = load_dense_gradient_stack(rows, artifact_root=dense_root)
    arrays = compute_seed_score_arrays(
        raw_gradients=gradients,
        state_ids=[row.objective_evidence.state_id for row in rows],
        vertices=vertices,
        faces=faces,
        alpha=AKITA_SMOOTHING_ALPHA,
    )
    provenance = build_seed_score_provenance(
        code_commit=cfg.code_commit,
        dense_metrics_path=metrics_path,
        dense_rows=rows,
        mesh_file_sha256=mesh_file_hash,
        vertices=vertices,
        faces=faces,
    )
    evidence = write_seed_score_artifact(
        output_root=output_root,
        provenance=provenance,
        arrays=arrays,
    )
    decision = evaluate_seed_score_artifact(
        evidence,
        artifact_root=output_root,
        dense_artifact_root=dense_root,
        dense_rows=rows,
    )
    manifest = {
        "schema_version": SEED_SCORE_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "input_dense_manifest_sha256": file_sha256(dense_manifest_path),
        "evidence": evidence_to_json(evidence),
        "decision": asdict(decision),
        "formula": {
            "state_normalization": "G_s / max(abs(G_s))",
            "mean_sensitivity": "mean_s(norm_rgb(G_s_normalized[i]))",
            "direction_consistency": (
                "norm_rgb(mean_s(G_s_normalized[i])) / mean_sensitivity[i]"
            ),
            "raw_score": "mean_sensitivity * direction_consistency",
            "raw_density": "raw_score / barycentric_lumped_vertex_mass",
            "smoothing": "(M + tau*L) d_tilde = M d",
            "alpha": AKITA_SMOOTHING_ALPHA,
            "tau": "alpha^2 * total_surface_area",
        },
        "feature_gradient_used": False,
        "wrist_gradient_used": False,
        "oft_used": False,
        "legacy_optimization_used": False,
        "coverage_computed": False,
        "production_support_constructed": False,
        "stability_threshold_registered": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(manifest["evidence"]["diagnostics"], indent=2, sort_keys=True))
    print(f"[SEED-SCORE-AUDIT] artifact={artifact_path}")
    print(f"[SEED-SCORE-AUDIT] manifest={manifest_path}")
    if not decision.gate_pass:
        raise RuntimeError(
            "Seed Score Audit未通过: " + "; ".join(decision.failures)
        )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_seed_score_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
