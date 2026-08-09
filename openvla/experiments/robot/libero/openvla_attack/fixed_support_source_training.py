"""正式 Fixed-Support Action+Spectral source trainer 与证据契约。

本模块不采集 Feature/wrist，也不导入 legacy ``training``/``optimization``。
调用方提供已冻结的 states 0--9 Action 梯度 provider；每轮严格计算十个 state
的算术平均梯度、Spectral Naturalness 梯度，并通过
:class:`FixedSupportTrainerCore` 执行唯一一次 surface-normalized update。

5000轮证据使用增量 JSONL，避免把逐 state margins 和梯度摘要长期留在内存。
完整逐轮梯度不重复保存：联合公式、Surface step 和 Support 约束已由两步 smoke
逐值验证；正式证据保存每轮可复核统计、最终紧凑参数、loss history 与 bake PNG。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence

import torch

from .artifacts import AttackArtifactStore, OptimizationArtifactPaths
from .fixed_support_training import (
    CombinedGradientUpdate,
    FixedSupportTrainerCore,
    FixedSupportTrainingRenderer,
)
from .fixed_support_training_smoke import (
    evaluate_fixed_support_training_smoke_bundle,
)
from .seed_score_audit import file_sha256
from .spectral_guard import MeanActionGradient, SpectralGuardTerms


FORMAL_SOURCE_TRAINING_SCHEMA_VERSION: Final[str] = (
    "openvla-fixed-support-action-spectral-source-training-v1"
)
EXPECTED_TRAIN_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10))


class FormalSourceTrainingError(RuntimeError):
    """正式训练配置、上游 provenance 或运行证据不满足契约。"""


class FormalActionGradientProvider(Protocol):
    """正式 trainer 使用的完整 states 0--9 Action provider。"""

    def __call__(self) -> MeanActionGradient: ...

    def drain_frame_evidence(self) -> tuple[dict[str, Any], ...]: ...


class FormalSpectralRegularizer(Protocol):
    """冻结 rho_nat 与 basis 的 Spectral Naturalness interface。"""

    rho_nat: float

    def __call__(self, surface_delta: torch.Tensor) -> SpectralGuardTerms: ...


@dataclass(frozen=True)
class FormalTrainingInputs:
    """已经通过两步 smoke 绑定的不可变输入及冻结 lambda。"""

    production_support_path: Path
    rho_nat_calibration_path: Path
    spectral_basis_path: Path
    spectral_guard_manifest_path: Path
    training_smoke_manifest_path: Path
    input_sha256: Mapping[str, str]
    lambda_spec: float
    smoke_config: Mapping[str, Any]


@dataclass(frozen=True)
class FormalSourceTrainingResult:
    """正式训练完成后供入口激活 texture 与运行成对 rollout 的产物。"""

    manifest_path: Path
    parameter_path: Path
    baked_texture_path: Path
    loss_history_path: Path
    steps_path: Path
    action_frames_path: Path
    num_iterations: int


def loaded_legacy_optimizer_modules() -> tuple[str, ...]:
    """返回当前进程中已加载的 legacy Action+Feature 实现。"""

    suffixes = (
        "openvla_attack.optimization",
        "openvla_attack.training",
    )
    return tuple(
        sorted(
            name
            for name in sys.modules
            if any(name.endswith(suffix) for suffix in suffixes)
        )
    )


def _validate_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise FormalSourceTrainingError("code_commit必须是40位小写Git SHA")


def verify_executing_commit(code_commit: str) -> None:
    """要求命令声明的commit等于当前checkout，且没有tracked文件修改。"""

    _validate_commit(code_commit)
    repository_root = Path(__file__).resolve().parents[5]
    try:
        resolved_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalSourceTrainingError("无法核验正式训练Git checkout") from error
    if resolved_commit != code_commit:
        raise FormalSourceTrainingError(
            f"code_commit={code_commit}与当前HEAD={resolved_commit}不一致"
        )
    if tracked_status:
        raise FormalSourceTrainingError("正式训练拒绝使用含tracked修改的checkout")


def load_formal_training_inputs(
    *,
    production_support_path: str | Path,
    rho_nat_calibration_path: str | Path,
    spectral_basis_path: str | Path,
    spectral_guard_manifest_path: str | Path,
    training_smoke_manifest_path: str | Path,
) -> FormalTrainingInputs:
    """复核两步 smoke，并要求它绑定本轮提供的四个上游 artifact。"""

    paths = {
        "production_support": Path(production_support_path).resolve(),
        "rho_nat_calibration": Path(rho_nat_calibration_path).resolve(),
        "spectral_basis": Path(spectral_basis_path).resolve(),
        "spectral_guard_manifest": Path(spectral_guard_manifest_path).resolve(),
    }
    smoke_path = Path(training_smoke_manifest_path).resolve()
    for path in (*paths.values(), smoke_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = evaluate_fixed_support_training_smoke_bundle(smoke_path)
    if not decision.gate_pass:
        raise FormalSourceTrainingError(
            "Fixed-Support两步smoke未通过独立复核: "
            + "; ".join(decision.failures)
        )
    manifest = json.loads(smoke_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("input_sha256", {})
    actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = [
        name
        for name, actual_hash in actual_hashes.items()
        if expected_hashes.get(name) != actual_hash
    ]
    if mismatches:
        raise FormalSourceTrainingError(
            "正式训练输入与已通过smoke不一致: " + ", ".join(mismatches)
        )
    lambda_spec = float(manifest.get("lambda_spec", float("nan")))
    if not math.isfinite(lambda_spec) or not 0.0 < lambda_spec <= 1.0:
        raise FormalSourceTrainingError("smoke中的lambda_spec无效")
    return FormalTrainingInputs(
        production_support_path=paths["production_support"],
        rho_nat_calibration_path=paths["rho_nat_calibration"],
        spectral_basis_path=paths["spectral_basis"],
        spectral_guard_manifest_path=paths["spectral_guard_manifest"],
        training_smoke_manifest_path=smoke_path,
        input_sha256={
            **actual_hashes,
            "fixed_support_training_smoke_manifest": file_sha256(smoke_path),
        },
        lambda_spec=lambda_spec,
        smoke_config=dict(manifest.get("config", {})),
    )


def validate_formal_config_against_smoke(
    cfg: Any,
    inputs: FormalTrainingInputs,
) -> None:
    """拒绝正式运行更换smoke使用的模型、状态采集或Surface语义。"""

    smoke = inputs.smoke_config
    exact_fields = (
        "model_family",
        "task_suite_name",
        "task_id",
        "object_name",
        "num_steps_wait",
        "seed",
        "unnorm_key",
        "load_in_8bit",
        "load_in_4bit",
        "attack_epsilon",
        "attack_surface_step",
    )
    mismatches = [
        field
        for field in exact_fields
        if smoke.get(field) != getattr(cfg, field)
    ]
    smoke_checkpoint = Path(str(smoke.get("pretrained_checkpoint", ""))).resolve()
    current_checkpoint = Path(str(cfg.pretrained_checkpoint)).resolve()
    if smoke_checkpoint != current_checkpoint:
        mismatches.append("pretrained_checkpoint")
    if smoke.get("state_ids") != "0-9":
        mismatches.append("smoke.state_ids")
    if mismatches:
        raise FormalSourceTrainingError(
            "正式训练配置与已通过smoke不一致: "
            + ", ".join(sorted(set(mismatches)))
        )


def _step_row(
    *,
    iteration: int,
    action_sample: MeanActionGradient,
    terms: SpectralGuardTerms,
    update: CombinedGradientUpdate,
    configured_surface_step: float,
) -> dict[str, Any]:
    """将一次联合更新压缩成可独立汇总的有限标量证据。"""

    return {
        "iteration": iteration,
        "action_loss": action_sample.loss,
        "num_action_frames": action_sample.num_frames,
        "action_state_ids": list(action_sample.state_ids),
        "action_state_fingerprints": list(action_sample.state_fingerprints),
        "total_energy": float(terms.total_energy.detach().item()),
        "low_energy": float(terms.low_energy.detach().item()),
        "high_energy": float(terms.high_energy.detach().item()),
        "high_ratio": float(terms.high_ratio.detach().item()),
        "diagnostic_high_ratio": float(
            terms.diagnostic_high_ratio.detach().item()
        ),
        "hinge": float(terms.hinge.detach().item()),
        "penalty": float(terms.penalty.detach().item()),
        "hinge_active": bool(terms.hinge.detach().item() > 0.0),
        "action_gradient_l2": update.action_gradient_l2,
        "spectral_gradient_l2": update.spectral_gradient_l2,
        "weighted_spectral_gradient_l2": (
            update.weighted_spectral_gradient_l2
        ),
        "total_gradient_l2": update.total_gradient_l2,
        "action_spectral_cosine": update.action_spectral_cosine,
        "weighted_spectral_action_ratio": (
            update.weighted_spectral_action_ratio
        ),
        "combination_residual_linf": update.combination_residual_linf,
        "configured_surface_step": configured_surface_step,
        "surface_step_stats": asdict(update.surface_step_stats),
    }


def _write_json_line(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_formal_source_training(
    *,
    code_commit: str,
    task_id: int,
    num_iterations: int,
    renderer: FixedSupportTrainingRenderer,
    action_provider: FormalActionGradientProvider,
    regularizer: FormalSpectralRegularizer,
    artifact_store: AttackArtifactStore,
    inputs: FormalTrainingInputs,
    state_fingerprints: Sequence[str],
    surface_step: float,
) -> FormalSourceTrainingResult:
    """执行正式联合训练并生成 rollout 前不可变 manifest。"""

    _validate_commit(code_commit)
    if num_iterations <= 0:
        raise FormalSourceTrainingError("num_iterations必须为正数")
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式进程已经加载legacy optimizer")
    fingerprints = tuple(state_fingerprints)
    if len(fingerprints) != 10 or len(set(fingerprints)) != 10:
        raise FormalSourceTrainingError("训练state fingerprint必须完整且唯一")

    artifact_store.ensure_attack_directory()
    steps_path = artifact_store.attack_directory / "formal_training_steps.jsonl"
    frames_path = (
        artifact_store.attack_directory / "formal_training_action_frames.jsonl"
    )
    manifest_path = (
        artifact_store.attack_directory / "formal_source_training_manifest.json"
    )
    if any(path.exists() for path in (steps_path, frames_path, manifest_path)):
        raise FileExistsError("拒绝覆盖已有正式Fixed-Support训练证据")

    trainer = FixedSupportTrainerCore(renderer, surface_step=surface_step)
    losses: list[float] = []
    with steps_path.open("x", encoding="utf-8") as steps_handle, frames_path.open(
        "x", encoding="utf-8"
    ) as frames_handle:
        for iteration in range(num_iterations):
            action_sample = action_provider()
            if action_sample.state_ids != EXPECTED_TRAIN_STATE_IDS:
                raise FormalSourceTrainingError(
                    f"iteration {iteration}未完整按序消费states 0-9"
                )
            if action_sample.state_fingerprints != fingerprints:
                raise FormalSourceTrainingError(
                    f"iteration {iteration} state fingerprint绑定漂移"
                )
            frame_rows = action_provider.drain_frame_evidence()
            if len(frame_rows) != 10 or tuple(
                row.get("state_id") for row in frame_rows
            ) != EXPECTED_TRAIN_STATE_IDS:
                raise FormalSourceTrainingError(
                    f"iteration {iteration}逐state证据不完整唯一"
                )
            if any(row.get("iteration") != iteration for row in frame_rows):
                raise FormalSourceTrainingError(
                    f"iteration {iteration}逐state证据轮次错误"
                )

            terms = regularizer(trainer.geometry_delta())
            spectral_gradient = torch.autograd.grad(
                terms.penalty,
                trainer.texture_parameter,
            )[0]
            update = trainer.apply_action_spectral_gradients(
                action_sample.gradient.detach(),
                spectral_gradient.detach(),
                lambda_spec=inputs.lambda_spec,
            )
            row = _step_row(
                iteration=iteration,
                action_sample=action_sample,
                terms=terms,
                update=update,
                configured_surface_step=surface_step,
            )
            numeric_values = (
                row["action_loss"],
                row["total_energy"],
                row["low_energy"],
                row["high_energy"],
                row["high_ratio"],
                row["hinge"],
                row["penalty"],
                row["action_gradient_l2"],
                row["spectral_gradient_l2"],
                row["total_gradient_l2"],
            )
            if not all(math.isfinite(float(value)) for value in numeric_values):
                raise FormalSourceTrainingError(
                    f"iteration {iteration}联合训练统计包含NaN/Inf"
                )
            _write_json_line(steps_handle, row)
            for frame_row in frame_rows:
                _write_json_line(frames_handle, frame_row)
            # 每轮而非每行flush：保留中断定位能力，同时避免50000次额外flush。
            steps_handle.flush()
            frames_handle.flush()
            losses.append(action_sample.loss)

    artifacts: OptimizationArtifactPaths = artifact_store.save_optimization_result(
        episode_index=task_id,
        renderer=renderer,
        loss_history=losses,
    )
    if trainer.update_count != num_iterations:
        raise FormalSourceTrainingError("正式训练Surface update次数不匹配")
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式训练期间加载了legacy optimizer")
    parameter = renderer.get_texture_param().detach()
    if not bool(torch.isfinite(parameter).all()):
        raise FormalSourceTrainingError("正式训练终态参数包含NaN/Inf")
    final_geometry_delta = renderer.get_geometry_surface_delta().detach()

    manifest = {
        "schema_version": FORMAL_SOURCE_TRAINING_SCHEMA_VERSION,
        "code_commit": code_commit,
        "task_id": task_id,
        "train_state_ids": list(EXPECTED_TRAIN_STATE_IDS),
        "train_state_fingerprints": list(fingerprints),
        "num_iterations": num_iterations,
        "trainer_update_count": trainer.update_count,
        "surface_step": surface_step,
        "surface_epsilon": float(renderer.epsilon),
        "lambda_spec": inputs.lambda_spec,
        "rho_nat": regularizer.rho_nat,
        "parameter_shape": list(parameter.shape),
        "final_parameter_linf": float(parameter.abs().amax().item()),
        "final_geometry_delta_linf": float(
            final_geometry_delta.abs().amax().item()
        ),
        "input_paths": {
            "production_support": str(inputs.production_support_path),
            "rho_nat_calibration": str(inputs.rho_nat_calibration_path),
            "spectral_basis": str(inputs.spectral_basis_path),
            "spectral_guard_manifest": str(inputs.spectral_guard_manifest_path),
            "fixed_support_training_smoke_manifest": str(
                inputs.training_smoke_manifest_path
            ),
        },
        "input_sha256": dict(inputs.input_sha256),
        "steps_relative_path": steps_path.name,
        "steps_sha256": file_sha256(steps_path),
        "action_frames_relative_path": frames_path.name,
        "action_frames_sha256": file_sha256(frames_path),
        "parameter_relative_path": artifacts.noise_path.name,
        "parameter_sha256": file_sha256(artifacts.noise_path),
        "baked_texture_relative_path": artifacts.texture_path.name,
        "baked_texture_sha256": file_sha256(artifacts.texture_path),
        "loss_history_relative_path": artifacts.loss_history_path.name,
        "loss_history_sha256": file_sha256(artifacts.loss_history_path),
        "objective_components": [
            "untargeted_clean_action_margin_hinge",
            "spectral_naturalness_hinge_squared",
        ],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "paired_source_rollout_required": True,
        "source_gate_evaluated": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if num_iterations == 5000:
        from .formal_source_training_evidence import (
            evaluate_formal_source_training_bundle,
        )

        decision = evaluate_formal_source_training_bundle(manifest_path)
        if not decision.gate_pass:
            raise FormalSourceTrainingError(
                "正式训练bundle独立复核失败: "
                + "; ".join(decision.failures)
            )
    return FormalSourceTrainingResult(
        manifest_path=manifest_path,
        parameter_path=artifacts.noise_path,
        baked_texture_path=artifacts.texture_path,
        loss_history_path=artifacts.loss_history_path,
        steps_path=steps_path,
        action_frames_path=frames_path,
        num_iterations=num_iterations,
    )


def run_formal_source_training_for_task(
    *,
    cfg: Any,
    task: Any,
    task_description: str,
    initial_states: Sequence[Any],
    initial_state_ids: Sequence[int],
    asset: Any,
    model: Any,
    processor: Any,
    renderer: Any,
    artifact_store: AttackArtifactStore,
) -> FormalSourceTrainingResult:
    """用主入口已创建的模型/renderer采帧并执行正式 source training。

    GPU、LIBERO 与 OpenVLA 专用依赖都延迟到该函数内导入。非 Fixed-Support
    运行不会加载这些正式模块；本函数也会在采帧前后拒绝 legacy optimizer。
    """

    verify_executing_commit(str(cfg.code_commit))
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式采帧前已加载legacy optimizer")
    if tuple(initial_state_ids) != EXPECTED_TRAIN_STATE_IDS:
        raise FormalSourceTrainingError("正式训练必须精确使用states 0-9")
    if len(initial_states) != 10:
        raise FormalSourceTrainingError("正式训练必须收到10个initial states")

    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    from .diagnose_spectral_guard_calibration import (
        AllStateActionGradientProvider,
        capture_action_frame,
    )
    from .image_preprocessing import DifferentiableOpenVLAImageProcessor
    from .policy_view import (
        POLICY_SOURCE_RESOLUTION,
        build_policy_view_transform,
    )
    from .spectral_guard import SpectralNaturalnessRegularizer

    inputs = load_formal_training_inputs(
        production_support_path=cfg.fixed_support_path,
        rho_nat_calibration_path=cfg.rho_nat_calibration_path,
        spectral_basis_path=cfg.spectral_naturalness_basis_path,
        spectral_guard_manifest_path=cfg.spectral_guard_manifest_path,
        training_smoke_manifest_path=(
            cfg.fixed_support_training_smoke_manifest_path
        ),
    )
    validate_formal_config_against_smoke(cfg, inputs)
    image_preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model,
        processor=processor,
    )
    model_height, model_width = image_preprocessor.output_size
    if model_height != model_width:
        raise FormalSourceTrainingError(
            "正式Fixed-Support训练只支持正方形checkpoint输入"
        )
    policy_view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_height,
    )
    frames = tuple(
        capture_action_frame(
            cfg=cfg,
            state_id=int(state_id),
            initial_state=initial_state,
            task=task,
            task_description=task_description,
            asset=asset,
            model=model,
            processor=processor,
            image_preprocessor=image_preprocessor,
            policy_view_transform=policy_view_transform,
            get_libero_dummy_action=get_libero_dummy_action,
            get_libero_env=get_libero_env,
            get_libero_image=get_libero_image,
        )
        for state_id, initial_state in zip(initial_state_ids, initial_states)
    )
    fingerprints = tuple(frame.initial_state_sha256 for frame in frames)
    provider = AllStateActionGradientProvider(
        frames=frames,
        renderer=renderer,
        model=model,
        image_preprocessor=image_preprocessor,
        policy_view_transform=policy_view_transform,
    )
    regularizer = SpectralNaturalnessRegularizer.from_artifacts(
        inputs.rho_nat_calibration_path,
        inputs.spectral_basis_path,
        device=model.device,
        dtype=torch.float32,
    )
    result = run_formal_source_training(
        code_commit=cfg.code_commit,
        task_id=int(cfg.task_id),
        num_iterations=int(cfg.attack_iters),
        renderer=renderer,
        action_provider=provider,
        regularizer=regularizer,
        artifact_store=artifact_store,
        inputs=inputs,
        state_fingerprints=fingerprints,
        surface_step=float(cfg.attack_surface_step),
    )
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式训练期间加载了legacy optimizer")
    return result
