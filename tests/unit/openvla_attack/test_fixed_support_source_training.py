"""正式 Fixed-Support source trainer 的 CPU 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from openvla.experiments.robot.libero.openvla_attack.artifacts import (
    AttackArtifactStore,
)
from openvla.experiments.robot.libero.openvla_attack.fixed_support_source_training import (
    ACTION_ONLY_CONTROL_SCHEMA_VERSION,
    FormalTrainingInputs,
    FormalSourceTrainingError,
    load_completed_formal_source_training,
    run_formal_source_training,
)
from openvla.experiments.robot.libero.openvla_attack.action_margin_kappa_smoke import (
    evaluate_action_margin_kappa_smoke_bundle,
)
from openvla.experiments.robot.libero.openvla_attack.formal_source_training_evidence import (
    evaluate_formal_source_training_bundle,
)
from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    file_sha256,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_guard import (
    MeanActionGradient,
    SpectralGuardTerms,
)
from openvla.experiments.robot.libero.openvla_attack.texture_parameterization import (
    FixedSupportTextureParameterization,
    SurfaceStepStats,
    surface_normalized_step_,
)


class _Renderer:
    epsilon = 0.5

    def __init__(self) -> None:
        self.parameterization = FixedSupportTextureParameterization(
            render_to_geometry=torch.arange(4),
            num_geometry_vertices=4,
            support_vertex_indices=torch.tensor([1, 3]),
            epsilon=self.epsilon,
        )

    def get_texture_parameterization_name(self) -> str:
        return "fixed_support"

    def get_texture_param(self) -> nn.Parameter:
        return self.parameterization.coefficients

    def get_geometry_surface_delta(self) -> torch.Tensor:
        return self.parameterization.geometry_delta()

    def step_surface_parameterization_(
        self,
        gradient: torch.Tensor,
        surface_step: float,
    ) -> SurfaceStepStats:
        return surface_normalized_step_(
            self.parameterization,
            gradient,
            surface_step,
        )

    def get_baked_adv_texture(self) -> torch.Tensor:
        value = 0.5 + self.get_texture_param().mean() * 0.1
        return value.expand(1, 4, 4, 3).clamp(0.0, 1.0)


class _Provider:
    def __init__(
        self,
        renderer: _Renderer,
        fingerprints: tuple[str, ...],
        *,
        action_margin_kappa: float = 0.0,
    ) -> None:
        self.renderer = renderer
        self.fingerprints = fingerprints
        self.action_margin_kappa = action_margin_kappa
        self.iteration = 0
        self.rows: tuple[dict[str, object], ...] = ()

    def __call__(self) -> MeanActionGradient:
        iteration = self.iteration
        self.iteration += 1
        margins = [1.0, -1.0]
        hinge_values = [
            max(0.0, margin + self.action_margin_kappa)
            for margin in margins
        ]
        action_loss = float(np.mean(np.asarray(hinge_values)))
        self.rows = tuple(
            {
                "iteration": iteration,
                "state_id": state_id,
                "initial_state_sha256": self.fingerprints[state_id],
                "margins": margins,
                "hinge_values": hinge_values,
                "action_margin_kappa": self.action_margin_kappa,
                "active_token_count": sum(
                    value > 0.0 for value in hinge_values
                ),
                "shallow_crossed_active_count": sum(
                    margin <= 0.0 and hinge > 0.0
                    for margin, hinge in zip(margins, hinge_values)
                ),
                "nonpositive_margin_count": sum(
                    margin <= 0.0 for margin in margins
                ),
                "action_loss": action_loss,
            }
            for state_id in range(10)
        )
        return MeanActionGradient(
            loss=action_loss,
            gradient=torch.ones_like(self.renderer.get_texture_param()),
            num_frames=10,
            state_ids=tuple(range(10)),
            state_fingerprints=self.fingerprints,
        )

    def drain_frame_evidence(self) -> tuple[dict[str, object], ...]:
        rows = self.rows
        self.rows = ()
        return rows


class _Regularizer:
    rho_nat = 0.1

    def __call__(self, delta: torch.Tensor) -> SpectralGuardTerms:
        total = delta.square().sum()
        low = total * 0.25
        high = total - low
        ratio = high / (total.detach() + 1e-6)
        diagnostic = high / (total + 1e-6)
        hinge = torch.relu(ratio - self.rho_nat)
        return SpectralGuardTerms(
            total_energy=total,
            low_energy=low,
            high_energy=high,
            high_ratio=ratio,
            diagnostic_high_ratio=diagnostic,
            hinge=hinge,
            penalty=hinge.square(),
        )


def _inputs(tmp_path: Path) -> FormalTrainingInputs:
    paths = [tmp_path / f"input-{index}" for index in range(5)]
    for path in paths:
        path.write_text("fixture", encoding="utf-8")
    paths[4].write_text(
        json.dumps({"lambda_spec": 0.1, "rho_nat": 0.1}),
        encoding="utf-8",
    )
    return FormalTrainingInputs(
        production_support_path=paths[0],
        rho_nat_calibration_path=paths[1],
        spectral_basis_path=paths[2],
        spectral_guard_manifest_path=paths[3],
        training_smoke_manifest_path=paths[4],
        input_sha256={
            name: file_sha256(path)
            for name, path in zip(
                (
                    "production_support",
                    "rho_nat_calibration",
                    "spectral_basis",
                    "spectral_guard_manifest",
                    "fixed_support_training_smoke_manifest",
                ),
                paths,
            )
        },
        lambda_spec=0.1,
        rho_nat=0.1,
        smoke_config={},
    )


def test_formal_training_writes_incremental_evidence_and_final_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    artifact_store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="formal-test",
        create_attack_directory=True,
    )

    result = run_formal_source_training(
        code_commit="b" * 40,
        task_id=0,
        num_iterations=2,
        renderer=renderer,
        action_provider=_Provider(renderer, fingerprints),
        regularizer=_Regularizer(),
        artifact_store=artifact_store,
        inputs=_inputs(tmp_path),
        state_fingerprints=fingerprints,
        surface_step=0.1,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_iterations"] == 2
    assert manifest["trainer_update_count"] == 2
    assert manifest["feature_loss_computed"] is False
    assert manifest["paired_source_rollout_required"] is True
    assert len(result.steps_path.read_text().splitlines()) == 2
    assert len(result.action_frames_path.read_text().splitlines()) == 20
    assert result.parameter_path.name == "Ep0_Fixed_Support_Delta.pt"
    assert result.baked_texture_path.is_file()


def test_action_only_control_never_computes_spectral_guard_and_records_variant(
    tmp_path,
    monkeypatch,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    artifact_store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="action-only-control-test",
        create_attack_directory=True,
    )

    result = run_formal_source_training(
        code_commit="d" * 40,
        task_id=0,
        num_iterations=2,
        renderer=renderer,
        action_provider=_Provider(renderer, fingerprints),
        regularizer=None,
        artifact_store=artifact_store,
        inputs=_inputs(tmp_path),
        state_fingerprints=fingerprints,
        surface_step=0.1,
        training_variant="action_only_control",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    assert result.training_variant == "action_only_control"
    assert manifest["schema_version"] == ACTION_ONLY_CONTROL_SCHEMA_VERSION
    assert manifest["training_variant"] == "action_only_control"
    assert manifest["objective_components"] == [
        "untargeted_clean_action_margin_hinge"
    ]
    assert manifest["spectral_guard_computed"] is False
    assert manifest["applied_lambda_spec"] == 0.0
    assert manifest["calibrated_lambda_spec"] == pytest.approx(0.1)
    assert all(row["spectral_guard_computed"] is False for row in rows)
    assert all(row["total_energy"] is None for row in rows)
    assert all(row["spectral_gradient_l2"] == 0.0 for row in rows)
    assert all(row["weighted_spectral_action_ratio"] == 0.0 for row in rows)


def test_action_only_kappa_binds_objective_without_spectral_guard(
    tmp_path,
    monkeypatch,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="action-only-kappa-test",
        create_attack_directory=True,
    )
    provider = _Provider(
        renderer,
        fingerprints,
        action_margin_kappa=4.375,
    )

    result = run_formal_source_training(
        code_commit="e" * 40,
        task_id=0,
        num_iterations=2,
        renderer=renderer,
        action_provider=provider,
        regularizer=None,
        artifact_store=store,
        inputs=_inputs(tmp_path),
        state_fingerprints=fingerprints,
        surface_step=0.1,
        training_variant="action_only_kappa",
        action_margin_kappa=4.375,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    assert result.training_variant == "action_only_kappa"
    assert manifest["training_variant"] == "action_only_kappa"
    assert manifest["action_margin_kappa"] == pytest.approx(4.375)
    assert manifest["spectral_guard_computed"] is False
    assert manifest["objective_components"] == [
        "untargeted_clean_action_margin_hinge_kappa"
    ]
    assert all(row["action_margin_kappa"] == pytest.approx(4.375) for row in rows)


def test_action_only_kappa_engineering_smoke_is_explicitly_non_scientific(
    tmp_path,
    monkeypatch,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="action-only-kappa-smoke-test",
        create_attack_directory=True,
    )

    result = run_formal_source_training(
        code_commit="1" * 40,
        task_id=0,
        num_iterations=2,
        renderer=renderer,
        action_provider=_Provider(
            renderer,
            fingerprints,
            action_margin_kappa=4.375,
        ),
        regularizer=None,
        artifact_store=store,
        inputs=_inputs(tmp_path),
        state_fingerprints=fingerprints,
        surface_step=0.1,
        training_variant="action_only_kappa",
        action_margin_kappa=4.375,
        engineering_smoke=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.manifest_path.name == "action_margin_kappa_smoke_manifest.json"
    assert result.steps_path.name == "action_margin_kappa_smoke_steps.jsonl"
    assert manifest["engineering_smoke"] is True
    assert manifest["scientific_gate"] is False
    assert manifest["formal_training_allowed"] is False
    assert manifest["paired_source_rollout_required"] is False
    proof = manifest["shallow_crossing_gradient_proof"]
    assert proof == {
        "margin": -2.0,
        "zero_kappa_hinge": 0.0,
        "zero_kappa_clean_logit_gradient": 0.0,
        "action_margin_kappa": 4.375,
        "kappa_hinge": 2.375,
        "kappa_clean_logit_gradient": 1.0,
        "kappa_best_other_logit_gradient": -1.0,
    }

    smoke_module = (
        "openvla.experiments.robot.libero.openvla_attack."
        "action_margin_kappa_smoke"
    )
    monkeypatch.setattr(
        f"{smoke_module}.evaluate_fixed_support_training_smoke_bundle",
        lambda path: SimpleNamespace(gate_pass=True, failures=()),
    )
    monkeypatch.setattr(
        f"{smoke_module}.load_production_support_artifact",
        lambda path: SimpleNamespace(
            support_vertex_indices=np.asarray([0, 1], dtype=np.int64)
        ),
    )

    decision = evaluate_action_margin_kappa_smoke_bundle(
        result.manifest_path
    )

    assert decision.gate_pass is True
    assert decision.failures == ()
    assert decision.shallow_crossed_active_tokens == 20


def test_formal_training_rejects_provider_kappa_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="kappa-mismatch-test",
        create_attack_directory=True,
    )

    with pytest.raises(FormalSourceTrainingError, match="provider.*κ"):
        run_formal_source_training(
            code_commit="f" * 40,
            task_id=0,
            num_iterations=2,
            renderer=renderer,
            action_provider=_Provider(renderer, fingerprints),
            regularizer=None,
            artifact_store=store,
            inputs=_inputs(tmp_path),
            state_fingerprints=fingerprints,
            surface_step=0.1,
            training_variant="action_only_kappa",
            action_margin_kappa=4.375,
        )


def test_completed_training_recovery_only_returns_existing_artifact_paths(
    tmp_path,
    monkeypatch,
) -> None:
    names = {
        "parameter_relative_path": "parameter.pt",
        "baked_texture_relative_path": "baked.png",
        "loss_history_relative_path": "loss.npy",
        "steps_relative_path": "steps.jsonl",
        "action_frames_relative_path": "frames.jsonl",
    }
    for name in names.values():
        (tmp_path / name).write_bytes(b"existing")
    manifest_path = tmp_path / "formal_source_training_manifest.json"
    manifest_path.write_text(
        json.dumps({**names, "num_iterations": 5000}),
        encoding="utf-8",
    )
    evaluator_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "formal_source_training_evidence."
        "evaluate_formal_source_training_bundle"
    )
    monkeypatch.setattr(
        evaluator_name,
        lambda path: SimpleNamespace(gate_pass=True, failures=()),
    )

    result = load_completed_formal_source_training(manifest_path)

    assert result.num_iterations == 5000
    assert result.baked_texture_path == tmp_path / "baked.png"
    assert result.parameter_path.read_bytes() == b"existing"

    monkeypatch.setattr(
        evaluator_name,
        lambda path: SimpleNamespace(gate_pass=False, failures=("bad",)),
    )
    with pytest.raises(FormalSourceTrainingError, match="bad"):
        load_completed_formal_source_training(manifest_path)


@pytest.mark.parametrize(
    "training_variant",
    ["action_spectral", "action_only_control"],
)
def test_cpu_evaluator_rejects_no_evidence_and_accepts_complete_sequence(
    tmp_path,
    monkeypatch,
    training_variant,
) -> None:
    module_name = (
        "openvla.experiments.robot.libero.openvla_attack."
        "fixed_support_source_training"
    )
    monkeypatch.setattr(
        f"{module_name}.loaded_legacy_optimizer_modules",
        lambda: (),
    )
    renderer = _Renderer()
    fingerprints = tuple(f"{state_id:064x}" for state_id in range(10))
    store = AttackArtifactStore.prepare(
        local_log_dir=tmp_path,
        run_id="evaluator-test",
        create_attack_directory=True,
    )
    result = run_formal_source_training(
        code_commit="c" * 40,
        task_id=0,
        num_iterations=2,
        renderer=renderer,
        action_provider=_Provider(renderer, fingerprints),
        regularizer=(
            _Regularizer() if training_variant == "action_spectral" else None
        ),
        artifact_store=store,
        inputs=_inputs(tmp_path),
        state_fingerprints=fingerprints,
        surface_step=0.1,
        training_variant=training_variant,
    )

    evidence_module = (
        "openvla.experiments.robot.libero.openvla_attack."
        "formal_source_training_evidence"
    )
    monkeypatch.setattr(
        f"{evidence_module}.evaluate_fixed_support_training_smoke_bundle",
        lambda path: SimpleNamespace(gate_pass=True, failures=()),
    )
    monkeypatch.setattr(
        f"{evidence_module}.load_production_support_artifact",
        lambda path: SimpleNamespace(
            support_vertex_indices=np.asarray([1, 3], dtype=np.int64)
        ),
    )

    incomplete = evaluate_formal_source_training_bundle(result.manifest_path)
    assert incomplete.gate_pass is False
    assert any("5000" in failure for failure in incomplete.failures)

    steps = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    frames = [
        json.loads(line) for line in result.action_frames_path.read_text().splitlines()
    ]
    with result.steps_path.open("w", encoding="utf-8") as handle:
        for iteration in range(5000):
            row = {**steps[iteration % 2], "iteration": iteration}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with result.action_frames_path.open("w", encoding="utf-8") as handle:
        for iteration in range(5000):
            for state_id in range(10):
                row = {**frames[state_id], "iteration": iteration}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    losses = np.asarray([steps[index % 2]["action_loss"] for index in range(5000)])
    np.save(result.loss_history_path, losses)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["num_iterations"] = 5000
    manifest["trainer_update_count"] = 5000
    manifest["steps_sha256"] = file_sha256(result.steps_path)
    manifest["action_frames_sha256"] = file_sha256(result.action_frames_path)
    manifest["loss_history_sha256"] = file_sha256(result.loss_history_path)
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    complete = evaluate_formal_source_training_bundle(result.manifest_path)

    assert complete.gate_pass is True
    assert complete.failures == ()

    # 单行真实证据失败不能级联伪装成文件行数或loss history失败。
    rows = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    rows[101]["action_total_cosine"] = 1.01
    with result.steps_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["steps_sha256"] = file_sha256(result.steps_path)
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    invalid_cosine = evaluate_formal_source_training_bundle(
        result.manifest_path
    )

    assert invalid_cosine.gate_pass is False
    expected_cosine_failure = (
        "cosine越界"
        if training_variant == "action_spectral"
        else "total梯度不等于Action"
    )
    assert any(
        expected_cosine_failure in failure
        for failure in invalid_cosine.failures
    )
    assert not any("行数" in failure for failure in invalid_cosine.failures)
    assert not any("loss history" in failure for failure in invalid_cosine.failures)

    # 真实服务器bundle同时暴露了两个边界：float32 cosine在理论1附近可上溢
    # 约5e-7；rsync后上游artifact位于manifest目录的第三层祖先兄弟目录。
    rows[101]["action_total_cosine"] = 1.0000004788342158
    with result.steps_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["steps_sha256"] = file_sha256(result.steps_path)
    manifest["input_paths"] = {
        name: f"/server/tex3d/artifacts/{Path(path).name}"
        for name, path in manifest["input_paths"].items()
    }
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    moved_bundle = evaluate_formal_source_training_bundle(result.manifest_path)

    assert moved_bundle.gate_pass is True
    assert moved_bundle.failures == ()
