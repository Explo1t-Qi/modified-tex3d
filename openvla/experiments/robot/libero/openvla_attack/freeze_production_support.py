"""把已接受 canonical candidate 冻结为不可变 Production Fixed Support。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
OPENVLA_ROOT = THIS_FILE.parents[4]
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
for import_path in (REPOSITORY_ROOT, OPENVLA_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.production_support import (  # noqa: E402
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    ProductionSupportEvidence,
    build_frozen_support_from_accepted_inputs,
    evaluate_production_support_artifact,
    file_sha256,
    write_production_support_artifact,
)


@dataclass(frozen=True)
class FreezeProductionSupportConfig:
    support_audit_dir: str
    seed_score_dir: str
    visibility_audit_dir: str
    output_dir: str
    code_commit: str
    object_name: str = "akita_black_bowl"


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> FreezeProductionSupportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_audit_dir", required=True)
    parser.add_argument("--seed_score_dir", required=True)
    parser.add_argument("--visibility_audit_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--object_name", default="akita_black_bowl")
    return FreezeProductionSupportConfig(**vars(parser.parse_args(argv)))


def run_freeze_production_support(
    cfg: FreezeProductionSupportConfig,
) -> Path:
    """冻结、写入并从全部上游输入独立复核 Production Support。"""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "production_fixed_support.npz"
    manifest_path = output_dir / "production_fixed_support_manifest.json"
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有 Production Fixed Support")
    support_manifest_path = (
        Path(cfg.support_audit_dir) / "support_construction_manifest.json"
    )
    seed_manifest_path = Path(cfg.seed_score_dir) / "seed_score_manifest.json"
    visibility_manifest_path = (
        Path(cfg.visibility_audit_dir) / "visibility_alignment_manifest.json"
    )
    frozen, _, _ = build_frozen_support_from_accepted_inputs(
        code_commit=cfg.code_commit,
        object_name=cfg.object_name,
        support_manifest_path=support_manifest_path,
        seed_score_manifest_path=seed_manifest_path,
        visibility_manifest_path=visibility_manifest_path,
    )
    artifact_sha = write_production_support_artifact(artifact_path, frozen)
    evidence = ProductionSupportEvidence(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        artifact_relative_path=artifact_path.name,
        artifact_sha256=artifact_sha,
        support_mask_sha256=frozen.support_mask_sha256,
        num_geometry_vertices=frozen.num_geometry_vertices,
        num_support_vertices=len(frozen.support_vertex_indices),
        trainable_rgb_scalar_count=3 * len(frozen.support_vertex_indices),
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    decision = evaluate_production_support_artifact(
        evidence,
        artifact_root=output_dir,
        expected=frozen,
    )
    manifest = {
        "schema_version": PRODUCTION_SUPPORT_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "evidence": asdict(evidence),
        "provenance": asdict(frozen.provenance),
        "decision": asdict(decision),
        "selected_candidate_index": frozen.selected_candidate_index,
        "num_regions": frozen.num_regions,
        "seed_vertex_ids": list(frozen.seed_vertex_ids),
        "target_area_fraction": frozen.target_area_fraction,
        "actual_area_fraction": frozen.actual_area_fraction,
        "primary_effective_coverage_min": min(
            frozen.primary_effective_coverage
        ),
        "compact_coordinate_order": frozen.compact_coordinate_order,
        "naturalness_k_nonconstant": frozen.naturalness_k_nonconstant,
        "next_required_gates": [
            "rho_nat_uniform_support_probe",
            "spectral_guard_lambda_calibration_and_state_restore",
        ],
        "production_support_constructed": True,
        "fixed_support_frozen": True,
        "rho_nat_calibrated": False,
        "lambda_spec_calibrated": False,
        "formal_training_allowed": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate_pass": decision.gate_pass,
                "failures": list(decision.failures),
                "num_support_vertices": len(frozen.support_vertex_indices),
                "support_mask_sha256": frozen.support_mask_sha256,
                "artifact_sha256": artifact_sha,
                "manifest_sha256": file_sha256(manifest_path),
                "formal_training_allowed": False,
            },
            sort_keys=True,
        )
    )
    if not decision.gate_pass:
        raise RuntimeError("Production Fixed Support 冻结验收失败")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_freeze_production_support(_parse_args(argv))


if __name__ == "__main__":
    main()
