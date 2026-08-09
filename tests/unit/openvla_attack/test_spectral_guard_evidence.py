"""Spectral Guard GPU runner与bundle独立验收测试。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    file_sha256,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_guard_evidence import (
    SPECTRAL_GUARD_SCHEMA_VERSION,
    evaluate_spectral_guard_bundle,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_passing_bundle(tmp_path: Path) -> Path:
    fingerprints = {state_id: f"{state_id:x}" * 64 for state_id in range(10)}
    iterations: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
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
                "configured_surface_step": 0.01,
                "surface_step_stats": {
                    "direction_surface_max": 1.0,
                    "parameter_scale": 0.1,
                    "projection_scale": 1.0,
                    "step_cap_scale": 1.0,
                    "actual_surface_step": 0.01,
                    "max_abs_delta": 0.1,
                },
            }
        )
        frame_rows.extend(
            {
                "iteration": iteration,
                "state_id": state_id,
                "initial_state_sha256": fingerprints[state_id],
                "action_loss": 1.0,
                "action_gradient_sha256": "a" * 64,
            }
            for state_id in range(10)
        )
    iteration_path = tmp_path / "spectral_guard_iterations.jsonl"
    frame_path = tmp_path / "spectral_guard_action_frames.jsonl"
    _write_jsonl(iteration_path, iterations)
    _write_jsonl(frame_path, frame_rows)
    manifest = {
        "schema_version": SPECTRAL_GUARD_SCHEMA_VERSION,
        "config": {
            "attack_surface_step": 0.01,
            "attack_epsilon": 0.5,
        },
        "state_ids": list(range(10)),
        "state_fingerprints": {
            str(state_id): fingerprint
            for state_id, fingerprint in fingerprints.items()
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
        "restore_evidence": {
            "module_state_restored": True,
            "component_states_restored": True,
            "gradient_cache_restored": True,
            "python_rng_restored": True,
            "numpy_rng_restored": True,
            "torch_cpu_rng_restored": True,
            "torch_cuda_rng_restored": True,
        },
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
    manifest_path = tmp_path / "spectral_guard_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_bundle_recomputes_state_completeness_window_lambda_and_steps(
    tmp_path: Path,
) -> None:
    decision = evaluate_spectral_guard_bundle(_write_passing_bundle(tmp_path))

    assert decision.gate_pass, decision.failures


def test_bundle_rejects_duplicate_state_and_surface_step_overflow(
    tmp_path: Path,
) -> None:
    manifest_path = _write_passing_bundle(tmp_path)
    iteration_path = tmp_path / "spectral_guard_iterations.jsonl"
    rows = [json.loads(line) for line in iteration_path.read_text().splitlines()]
    rows[0]["action_state_ids"][-1] = 8
    rows[0]["surface_step_stats"]["actual_surface_step"] = 0.02
    _write_jsonl(iteration_path, rows)
    manifest = json.loads(manifest_path.read_text())
    manifest["iterations_sha256"] = file_sha256(iteration_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    decision = evaluate_spectral_guard_bundle(manifest_path)

    assert not decision.gate_pass
    assert any("states不完整" in failure for failure in decision.failures)
    assert any("Surface step越界" in failure for failure in decision.failures)


def test_gpu_runner_uses_new_objective_and_shared_trainer_not_legacy() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "openvla/experiments/robot/libero/openvla_attack"
        / "diagnose_spectral_guard_calibration.py"
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
    assert "FixedSupportActionTrainerCore" in source
    assert "untargeted_clean_action_margin_hinge" in source
    assert 'tuple(range(10))' in source
    assert "frame_evidence_history" in source
    assert "SurfaceStepStats" not in source  # stats只能来自共享trainer返回值
