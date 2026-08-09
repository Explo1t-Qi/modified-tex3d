"""用冻结Production Support校准谱自然性上限 ``rho_nat``。"""

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

from openvla_attack.seed_score_audit import file_sha256  # noqa: E402
from openvla_attack.spectral_naturalness import (  # noqa: E402
    NATURALNESS_CALIBRATION_SCHEMA_VERSION,
    build_rho_nat_calibration,
    evaluate_rho_nat_calibration,
    write_rho_nat_calibration_artifact,
)


@dataclass(frozen=True)
class CalibrateSpectralNaturalnessConfig:
    production_support_path: str
    spectral_basis_path: str
    output_dir: str
    code_commit: str
    object_name: str = "akita_black_bowl"


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> CalibrateSpectralNaturalnessConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--spectral_basis_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--object_name", default="akita_black_bowl")
    return CalibrateSpectralNaturalnessConfig(**vars(parser.parse_args(argv)))


def run_spectral_naturalness_calibration(
    cfg: CalibrateSpectralNaturalnessConfig,
) -> Path:
    """写入校准artifact与manifest，并从两个输入独立复算验收。"""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "spectral_naturalness_calibration.npz"
    manifest_path = output_dir / "spectral_naturalness_manifest.json"
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有谱自然性校准")

    calibration = build_rho_nat_calibration(
        production_support_path=cfg.production_support_path,
        spectral_basis_path=cfg.spectral_basis_path,
        code_commit=cfg.code_commit,
        object_name=cfg.object_name,
    )
    artifact_sha = write_rho_nat_calibration_artifact(
        artifact_path,
        calibration,
    )
    decision = evaluate_rho_nat_calibration(
        artifact_path,
        production_support_path=cfg.production_support_path,
        spectral_basis_path=cfg.spectral_basis_path,
    )
    manifest = {
        "schema_version": NATURALNESS_CALIBRATION_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "artifact_relative_path": artifact_path.name,
        "artifact_sha256": artifact_sha,
        "production_support_artifact_sha256": (
            calibration.production_support_artifact_sha256
        ),
        "spectral_basis_artifact_sha256": (
            calibration.spectral_basis_artifact_sha256
        ),
        "support_mask_sha256": calibration.support_mask_sha256,
        "mesh_array_sha256": calibration.mesh_array_sha256,
        "mass_sha256": calibration.mass_sha256,
        "low_basis_sha256": calibration.low_basis_sha256,
        "low_band_eigenvalues_sha256": (
            calibration.low_band_eigenvalues_sha256
        ),
        "num_geometry_vertices": calibration.num_geometry_vertices,
        "num_support_vertices": calibration.num_support_vertices,
        "naturalness_k_nonconstant": (
            calibration.naturalness_k_nonconstant
        ),
        "num_low_modes_including_constant": (
            calibration.num_low_modes_including_constant
        ),
        "low_band_eigenvalues": (
            calibration.low_band_eigenvalues.tolist()
        ),
        "total_surface_area": calibration.total_surface_area,
        "cutoff_eigenvalue": calibration.cutoff_eigenvalue,
        "dimensionless_cutoff": calibration.dimensionless_cutoff,
        "mass_orthonormality_linf": (
            calibration.mass_orthonormality_linf
        ),
        "constant_mode_linf": calibration.constant_mode_linf,
        "support_probe_sha256": calibration.support_probe_sha256,
        "energy_total": calibration.energy_total,
        "energy_low": calibration.energy_low,
        "energy_high": calibration.energy_high,
        "energy_epsilon": calibration.energy_epsilon,
        "rho_nat": calibration.rho_nat,
        "decision": asdict(decision),
        "rho_nat_calibrated": True,
        "lambda_spec_calibrated": False,
        "formal_training_allowed": False,
        "next_required_gate": (
            "spectral_guard_lambda_calibration_and_state_restore"
        ),
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
                "rho_nat": calibration.rho_nat,
                "artifact_sha256": artifact_sha,
                "manifest_sha256": file_sha256(manifest_path),
                "lambda_spec_calibrated": False,
                "formal_training_allowed": False,
            },
            sort_keys=True,
        )
    )
    if not decision.gate_pass:
        raise RuntimeError("谱自然性rho_nat校准验收失败")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_spectral_naturalness_calibration(_parse_args(argv))


if __name__ == "__main__":
    main()
