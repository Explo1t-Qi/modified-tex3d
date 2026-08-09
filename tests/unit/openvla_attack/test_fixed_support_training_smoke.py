"""Fixed-Support Action+Spectral两步smoke证据契约测试。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

from openvla.experiments.robot.libero.openvla_attack.fixed_support_training_smoke import (
    FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION,
    evaluate_fixed_support_training_smoke_bundle,
)
from openvla.experiments.robot.libero.openvla_attack.production_support import (
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    FrozenProductionSupport,
    ProductionSupportProvenance,
    array_sha256,
    write_production_support_artifact,
)
from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    file_sha256,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_guard_evidence import (
    SPECTRAL_GUARD_SCHEMA_VERSION,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_support(path: Path, *, mesh_file_sha256: str = "9" * 64) -> None:
    mask = np.asarray([False, True, False, True], dtype=np.bool_)
    support = FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=ProductionSupportProvenance(
            code_commit="1" * 40,
            object_name="akita_black_bowl",
            source_support_manifest_sha256="2" * 64,
            source_candidate_artifact_sha256="3" * 64,
            source_coverage_artifact_sha256="4" * 64,
            seed_score_manifest_sha256="5" * 64,
            seed_score_artifact_sha256="6" * 64,
            visibility_manifest_sha256="7" * 64,
            visibility_metrics_sha256="8" * 64,
            mesh_file_sha256=mesh_file_sha256,
            mesh_array_sha256="a" * 64,
            renderer_faces_sha256="b" * 64,
            render_to_geometry_sha256="c" * 64,
        ),
        selected_candidate_index=0,
        num_geometry_vertices=4,
        support_mask=mask,
        support_vertex_indices=np.asarray([1, 3], dtype=np.int64),
        support_mask_sha256=array_sha256(mask),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=1,
        seed_vertex_ids=(1,),
        total_surface_area=10.0,
        target_area_fraction=0.1,
        target_mass=1.0,
        actual_mass=1.0,
        actual_area_fraction=0.1,
        primary_coverage_min_threshold=0.2,
        primary_state_ids=(0,),
        primary_statuses=("valid",),
        primary_source_coverage=(0.3,),
        primary_effective_coverage=(0.3,),
        wrist_state_ids=(),
        wrist_statuses=(),
        wrist_source_coverage=(),
        wrist_effective_coverage=(),
        naturalness_k_nonconstant=128,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    write_production_support_artifact(path, support)


def _write_guard_bundle(
    root: Path,
    *,
    input_hashes: dict[str, str],
    fingerprints: dict[int, str],
) -> Path:
    iterations = []
    frames = []
    for iteration in range(5):
        iterations.append(
            {
                "iteration": iteration,
                "action_loss": 1.0,
                "num_action_frames": 10,
                "action_state_ids": list(range(10)),
                "action_state_fingerprints": list(fingerprints.values()),
                "hinge_active": True,
                "q_t": 2.0,
                "action_gradient_l2": 1.0,
                "spectral_gradient_l2": 2.0,
                "configured_surface_step": 0.1,
                "surface_step_stats": {
                    "direction_surface_max": 1.0,
                    "parameter_scale": 0.1,
                    "projection_scale": 1.0,
                    "step_cap_scale": 1.0,
                    "actual_surface_step": 0.1,
                    "max_abs_delta": 0.1,
                },
            }
        )
        frames.extend(
            {
                "iteration": iteration,
                "state_id": state_id,
                "initial_state_sha256": fingerprints[state_id],
                "action_loss": 1.0,
                "clean_action_token_ids": [1, 2],
                "margins": [1.0, 2.0],
                "hinge_values": [1.0, 2.0],
                "action_gradient_sha256": "d" * 64,
            }
            for state_id in range(10)
        )
    iteration_path = root / "guard_iterations.jsonl"
    frame_path = root / "guard_frames.jsonl"
    _write_jsonl(iteration_path, iterations)
    _write_jsonl(frame_path, frames)
    manifest = {
        "schema_version": SPECTRAL_GUARD_SCHEMA_VERSION,
        "config": {"attack_surface_step": 0.1, "attack_epsilon": 0.5},
        "input_sha256": input_hashes,
        "state_ids": list(range(10)),
        "state_fingerprints": {
            str(key): value for key, value in fingerprints.items()
        },
        "num_iterations": 5,
        "iterations_relative_path": iteration_path.name,
        "iterations_sha256": file_sha256(iteration_path),
        "action_frames_relative_path": frame_path.name,
        "action_frames_sha256": file_sha256(frame_path),
        "selected_window_start": 0,
        "selected_window_end": 4,
        "q_median": 2.0,
        "lambda_spec": 0.05,
        "calibration_status": "calibrated_stable_activation",
        "restore_evidence": {"all": True},
        "objective_components": ["untargeted_clean_action_margin_hinge"],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "rho_nat_calibrated": True,
        "lambda_spec_calibrated": True,
        "formal_training_allowed": False,
        "gate_pass": True,
    }
    path = root / "guard_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_passing_bundle(tmp_path: Path) -> Path:
    rho_path = tmp_path / "rho.npz"
    basis_path = tmp_path / "basis.npz"
    mesh_path = tmp_path / "mesh.obj"
    for path, payload in (
        (rho_path, b"rho"),
        (basis_path, b"basis"),
        (mesh_path, b"mesh"),
    ):
        path.write_bytes(payload)
    clean_path = tmp_path / "clean.png"
    baked_path = tmp_path / "baked.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(clean_path)
    baked = np.zeros((2, 2, 3), dtype=np.uint8)
    baked[0, 0, 0] = 1
    Image.fromarray(baked).save(baked_path)
    support_path = tmp_path / "support.npz"
    _write_support(
        support_path,
        mesh_file_sha256=file_sha256(mesh_path),
    )

    fingerprints = {state_id: f"{state_id:x}" * 64 for state_id in range(10)}
    upstream_hashes = {
        "production_support": file_sha256(support_path),
        "rho_nat_calibration": file_sha256(rho_path),
        "spectral_basis": file_sha256(basis_path),
        "mesh": file_sha256(mesh_path),
        "texture": file_sha256(clean_path),
    }
    guard_path = _write_guard_bundle(
        tmp_path,
        input_hashes=upstream_hashes,
        fingerprints=fingerprints,
    )
    action = np.ones((2, 2, 3), dtype=np.float32)
    spectral = np.zeros_like(action)
    spectral[1] = 2.0
    weighted = spectral * np.float32(0.05)
    total = action + weighted
    deltas = np.zeros((3, 4, 3), dtype=np.float32)
    deltas[1, [1, 3]] = 0.01
    deltas[2, [1, 3]] = 0.02
    arrays_path = tmp_path / "arrays.npz"
    indices = np.asarray([1, 3], dtype=np.int64)
    np.savez_compressed(
        arrays_path,
        action_gradients=action,
        spectral_gradients=spectral,
        weighted_spectral_gradients=weighted,
        total_gradients=total,
        geometry_deltas=deltas,
        support_vertex_indices=indices,
    )
    steps = []
    for step in range(2):
        a_norm = float(np.linalg.norm(action[step].astype(np.float64)))
        s_norm = float(np.linalg.norm(spectral[step].astype(np.float64)))
        w_norm = float(np.linalg.norm(weighted[step].astype(np.float64)))
        t_norm = float(np.linalg.norm(total[step].astype(np.float64)))
        steps.append(
            {
                "step": step,
                "action_loss": 1.0,
                "num_action_frames": 10,
                "action_state_ids": list(range(10)),
                "action_state_fingerprints": list(fingerprints.values()),
                "total_energy": 0.0 if step == 0 else 1.0,
                "high_energy": 0.0 if step == 0 else 0.8,
                "hinge_active": step == 1,
                "action_gradient_l2": a_norm,
                "spectral_gradient_l2": s_norm,
                "weighted_spectral_gradient_l2": w_norm,
                "total_gradient_l2": t_norm,
                "action_spectral_cosine": None if step == 0 else 1.0,
                "weighted_spectral_action_ratio": w_norm / a_norm,
                "combination_residual_linf": 0.0,
                "surface_step_stats": {
                    "direction_surface_max": 1.0,
                    "parameter_scale": 0.1,
                    "projection_scale": 1.0,
                    "step_cap_scale": 1.0,
                    "actual_surface_step": 0.01,
                    "max_abs_delta": 0.01 * (step + 1),
                },
            }
        )
    steps_path = tmp_path / "steps.jsonl"
    _write_jsonl(steps_path, steps)
    frames_path = tmp_path / "frames.jsonl"
    _write_jsonl(
        frames_path,
        [
            {
                "iteration": step,
                "state_id": state_id,
                "initial_state_sha256": fingerprints[state_id],
                "action_loss": 1.0,
                "clean_action_token_ids": [1, 2],
                "margins": [1.0, 2.0],
                "hinge_values": [1.0, 2.0],
            }
            for step in range(2)
            for state_id in range(10)
        ],
    )
    input_paths = {
        "production_support": str(support_path),
        "rho_nat_calibration": str(rho_path),
        "spectral_basis": str(basis_path),
        "spectral_guard_manifest": str(guard_path),
        "mesh": str(mesh_path),
        "texture": str(clean_path),
    }
    input_hashes = {
        name: file_sha256(Path(path)) for name, path in input_paths.items()
    }
    manifest = {
        "schema_version": FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION,
        "code_commit": "e" * 40,
        "config": {"attack_surface_step": 0.1, "attack_epsilon": 0.5},
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "state_ids": list(range(10)),
        "state_fingerprints": {
            str(key): value for key, value in fingerprints.items()
        },
        "lambda_spec": 0.05,
        "support_vertex_indices_sha256": array_sha256(indices),
        "steps_relative_path": steps_path.name,
        "steps_sha256": file_sha256(steps_path),
        "action_frames_relative_path": frames_path.name,
        "action_frames_sha256": file_sha256(frames_path),
        "arrays_relative_path": arrays_path.name,
        "arrays_sha256": file_sha256(arrays_path),
        "baked_texture_relative_path": baked_path.name,
        "baked_texture_sha256": file_sha256(baked_path),
        "active_texture_sha256": file_sha256(baked_path),
        "active_environment_loaded": True,
        "active_observation_sha256": "f" * 64,
        "xml_sha256_before": "1" * 64,
        "xml_sha256_after_restore": "1" * 64,
        "texture_sha256_before": file_sha256(clean_path),
        "texture_sha256_after_restore": file_sha256(clean_path),
        "backup_paths_removed": True,
        "trainer_update_count": 2,
        "objective_components": [
            "untargeted_clean_action_margin_hinge",
            "spectral_naturalness_hinge_squared",
        ],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "rho_nat_calibrated": True,
        "lambda_spec_calibrated": True,
        "formal_training_allowed": True,
        "next_required_gate": "fixed_support_action_spectral_source_training",
        "gate_pass": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_two_step_bundle_recomputes_combination_support_and_assets(
    tmp_path: Path,
) -> None:
    decision = evaluate_fixed_support_training_smoke_bundle(
        _write_passing_bundle(tmp_path)
    )

    assert decision.gate_pass, decision.failures


def test_bundle_rejects_missing_step_one_spectral_gradient(tmp_path: Path) -> None:
    manifest_path = _write_passing_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    arrays_path = tmp_path / manifest["arrays_relative_path"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["spectral_gradients"][1] = 0.0
    np.savez_compressed(arrays_path, **arrays)
    manifest["arrays_sha256"] = file_sha256(arrays_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = evaluate_fixed_support_training_smoke_bundle(manifest_path)

    assert not decision.gate_pass
    assert any("Step 1谱梯度必须非零" in item for item in decision.failures)
    assert any("weighted spectral梯度不可逐值复算" in item for item in decision.failures)


def test_bundle_resolves_relocated_basis_and_uses_asset_provenance(
    tmp_path: Path,
) -> None:
    manifest_path = _write_passing_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    relocated_dir = tmp_path / "spectral_basis"
    relocated_dir.mkdir()
    original_basis = Path(manifest["input_paths"]["spectral_basis"])
    relocated_basis = relocated_dir / original_basis.name
    original_basis.rename(relocated_basis)
    manifest["input_paths"]["spectral_basis"] = (
        f"experiments/spectral_basis/{original_basis.name}"
    )
    mesh_path = Path(manifest["input_paths"]["mesh"])
    texture_path = Path(manifest["input_paths"]["texture"])
    manifest["input_paths"]["mesh"] = "/server/LIBERO/mesh.obj"
    manifest["input_paths"]["texture"] = "/server/LIBERO/texture.png"
    mesh_path.unlink()
    texture_path.unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = evaluate_fixed_support_training_smoke_bundle(manifest_path)

    assert decision.gate_pass, decision.failures


def test_gpu_runner_uses_combined_core_and_excludes_legacy_objectives() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "openvla/experiments/robot/libero/openvla_attack"
        / "diagnose_fixed_support_training_smoke.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "openvla_attack.optimization" not in imported_modules
    assert "openvla_attack.training" not in imported_modules
    assert "apply_action_spectral_gradients" in source
    assert "untargeted_clean_action_margin_hinge" in source
    assert "spectral_naturalness_hinge_squared" in source
    assert "range(2)" in source
