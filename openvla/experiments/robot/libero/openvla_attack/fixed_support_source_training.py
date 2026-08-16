"""正式 Fixed-Support source trainer及Action-only对照证据契约。

本模块不采集 Feature/wrist，也不导入 legacy ``training``/``optimization``。
正式主候选每轮形成十个state的平均Action梯度与Spectral Naturalness梯度；
严格匹配的Action-only control与预注册Action-only+κ干预只提交Action梯度，
不构造或计算谱正则。三条路径都通过 :class:`FixedSupportTrainerCore` 执行唯一
一次surface-normalized update，并共享Surface-L∞ projection。

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
from typing import Any, Final, Literal, Mapping, Optional, Protocol, Sequence, cast

import torch

from .artifacts import AttackArtifactStore, OptimizationArtifactPaths
from .configuration import FROZEN_ACTION_MARGIN_KAPPA
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
ACTION_ONLY_CONTROL_SCHEMA_VERSION: Final[str] = (
    "openvla-fixed-support-action-only-source-control-v1"
)
ACTION_ONLY_KAPPA_SCHEMA_VERSION: Final[str] = (
    "openvla-fixed-support-action-only-kappa-source-training-v1"
)
ACTION_MARGIN_KAPPA_SMOKE_SCHEMA_VERSION: Final[str] = (
    "openvla-action-margin-kappa-engineering-smoke-v1"
)
FormalTrainingVariant = Literal[
    "action_spectral",
    "action_only_control",
    "action_only_kappa",
]
SUPPORTED_FORMAL_TRAINING_VARIANTS: Final[frozenset[str]] = frozenset(
    {"action_spectral", "action_only_control", "action_only_kappa"}
)
EXPECTED_TRAIN_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10))


class FormalSourceTrainingError(RuntimeError):
    """正式训练配置、上游 provenance 或运行证据不满足契约。"""


class FormalActionGradientProvider(Protocol):
    """正式 trainer 使用的完整 states 0--9 Action provider。"""

    action_margin_kappa: float

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
    rho_nat: float
    smoke_config: Mapping[str, Any]
    action_margin_kappa_smoke_manifest_path: Optional[Path] = None


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
    training_variant: FormalTrainingVariant


def _resolve_training_variant(value: object) -> FormalTrainingVariant:
    """校验内部正式训练变体；旧v1 manifest缺省解释为主候选。"""

    resolved = "action_spectral" if value is None else str(value)
    if resolved not in SUPPORTED_FORMAL_TRAINING_VARIANTS:
        raise FormalSourceTrainingError(f"未知正式训练变体: {resolved!r}")
    return cast(FormalTrainingVariant, resolved)


def load_completed_formal_source_training(
    manifest_path: str | Path,
) -> FormalSourceTrainingResult:
    """独立复核并恢复已完成训练的不可变产物引用，不重新执行任何update。"""

    from .formal_source_training_evidence import (
        evaluate_formal_source_training_bundle,
    )

    resolved_manifest = Path(manifest_path).resolve()
    decision = evaluate_formal_source_training_bundle(resolved_manifest)
    if not decision.gate_pass:
        raise FormalSourceTrainingError(
            "已完成正式训练bundle未通过独立复核: "
            + "; ".join(decision.failures)
        )
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    training_variant = _resolve_training_variant(
        manifest.get("training_variant")
    )
    root = resolved_manifest.parent
    result = FormalSourceTrainingResult(
        manifest_path=resolved_manifest,
        parameter_path=root / str(manifest["parameter_relative_path"]),
        baked_texture_path=root / str(
            manifest["baked_texture_relative_path"]
        ),
        loss_history_path=root / str(
            manifest["loss_history_relative_path"]
        ),
        steps_path=root / str(manifest["steps_relative_path"]),
        action_frames_path=root / str(
            manifest["action_frames_relative_path"]
        ),
        num_iterations=int(manifest["num_iterations"]),
        training_variant=training_variant,
    )
    if result.num_iterations != 5000:
        raise FormalSourceTrainingError("恢复入口只接受完整5000轮正式训练")
    return result


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
    action_margin_kappa_smoke_manifest_path: Optional[str | Path] = None,
) -> FormalTrainingInputs:
    """复核基础训练smoke及可选κ smoke，绑定全部上游artifact。"""

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
    rho_nat = float(manifest.get("rho_nat", float("nan")))
    if not math.isfinite(rho_nat) or not 0.0 <= rho_nat <= 1.0:
        raise FormalSourceTrainingError("smoke中的rho_nat无效")
    kappa_smoke_path: Optional[Path] = None
    kappa_smoke_sha256: Optional[str] = None
    if action_margin_kappa_smoke_manifest_path is not None:
        from .action_margin_kappa_smoke import (
            evaluate_action_margin_kappa_smoke_bundle,
        )

        kappa_smoke_path = Path(
            action_margin_kappa_smoke_manifest_path
        ).resolve()
        if not kappa_smoke_path.is_file():
            raise FileNotFoundError(kappa_smoke_path)
        kappa_decision = evaluate_action_margin_kappa_smoke_bundle(
            kappa_smoke_path
        )
        if not kappa_decision.gate_pass:
            raise FormalSourceTrainingError(
                "Action-only+κ两步smoke未通过独立复核: "
                + "; ".join(kappa_decision.failures)
            )
        kappa_smoke_sha256 = file_sha256(kappa_smoke_path)
    return FormalTrainingInputs(
        production_support_path=paths["production_support"],
        rho_nat_calibration_path=paths["rho_nat_calibration"],
        spectral_basis_path=paths["spectral_basis"],
        spectral_guard_manifest_path=paths["spectral_guard_manifest"],
        training_smoke_manifest_path=smoke_path,
        input_sha256={
            **actual_hashes,
            "fixed_support_training_smoke_manifest": file_sha256(smoke_path),
            **(
                {
                    "action_margin_kappa_smoke_manifest": (
                        kappa_smoke_sha256
                    )
                }
                if kappa_smoke_sha256 is not None
                else {}
            ),
        },
        lambda_spec=lambda_spec,
        rho_nat=rho_nat,
        smoke_config=dict(manifest.get("config", {})),
        action_margin_kappa_smoke_manifest_path=kappa_smoke_path,
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
        "training_variant": "action_spectral",
        "spectral_guard_computed": True,
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
        "action_total_cosine": update.action_total_cosine,
        "weighted_spectral_action_ratio": (
            update.weighted_spectral_action_ratio
        ),
        "combination_residual_linf": update.combination_residual_linf,
        "configured_surface_step": configured_surface_step,
        "surface_step_stats": asdict(update.surface_step_stats),
    }


def _action_only_step_row(
    *,
    iteration: int,
    action_sample: MeanActionGradient,
    update: CombinedGradientUpdate,
    configured_surface_step: float,
    training_variant: FormalTrainingVariant,
    action_margin_kappa: float,
) -> dict[str, Any]:
    """保存Action-only单变量对照；谱相关量用null/显式零区分未计算。"""

    return {
        "iteration": iteration,
        "training_variant": training_variant,
        "action_margin_kappa": action_margin_kappa,
        "spectral_guard_computed": False,
        "action_loss": action_sample.loss,
        "num_action_frames": action_sample.num_frames,
        "action_state_ids": list(action_sample.state_ids),
        "action_state_fingerprints": list(action_sample.state_fingerprints),
        "total_energy": None,
        "low_energy": None,
        "high_energy": None,
        "high_ratio": None,
        "diagnostic_high_ratio": None,
        "hinge": None,
        "penalty": None,
        "hinge_active": None,
        "action_gradient_l2": update.action_gradient_l2,
        "spectral_gradient_l2": 0.0,
        "weighted_spectral_gradient_l2": 0.0,
        "total_gradient_l2": update.total_gradient_l2,
        "action_spectral_cosine": None,
        "action_total_cosine": update.action_total_cosine,
        "weighted_spectral_action_ratio": (
            update.weighted_spectral_action_ratio
        ),
        "combination_residual_linf": update.combination_residual_linf,
        "configured_surface_step": configured_surface_step,
        "surface_step_stats": asdict(update.surface_step_stats),
    }


def _write_json_line(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_shallow_crossing_gradient_proof(
    action_margin_kappa: float,
) -> dict[str, float]:
    """用公共objective证明浅越界token只在κ hinge中保持梯度。"""

    from .objective import (
        ACTION_TOKEN_END,
        ACTION_TOKEN_START,
        untargeted_clean_action_margin_hinge,
    )

    base_logits = torch.full(
        (1, 2, ACTION_TOKEN_END),
        -10.0,
        dtype=torch.float32,
    )
    clean_class_index = ACTION_TOKEN_START
    best_other_index = ACTION_TOKEN_START + 1
    base_logits[0, 0, clean_class_index] = -1.0
    base_logits[0, 0, best_other_index] = 1.0
    clean_ids = torch.tensor(
        [[0, ACTION_TOKEN_START]],
        dtype=torch.int64,
    )
    zero_logits = base_logits.clone().requires_grad_(True)
    kappa_logits = base_logits.clone().requires_grad_(True)
    zero_objective = untargeted_clean_action_margin_hinge(
        zero_logits,
        clean_ids,
        action_margin_kappa=0.0,
    )
    kappa_objective = untargeted_clean_action_margin_hinge(
        kappa_logits,
        clean_ids,
        action_margin_kappa=action_margin_kappa,
    )
    zero_gradient = torch.autograd.grad(zero_objective.loss, zero_logits)[0]
    kappa_gradient = torch.autograd.grad(
        kappa_objective.loss,
        kappa_logits,
    )[0]
    return {
        "margin": float(kappa_objective.margins[0].detach().item()),
        "zero_kappa_hinge": float(
            zero_objective.hinge_values[0].detach().item()
        ),
        "zero_kappa_clean_logit_gradient": float(
            zero_gradient[0, 0, clean_class_index].item()
        ),
        "action_margin_kappa": action_margin_kappa,
        "kappa_hinge": float(
            kappa_objective.hinge_values[0].detach().item()
        ),
        "kappa_clean_logit_gradient": float(
            kappa_gradient[0, 0, clean_class_index].item()
        ),
        "kappa_best_other_logit_gradient": float(
            kappa_gradient[0, 0, best_other_index].item()
        ),
    }


def run_formal_source_training(
    *,
    code_commit: str,
    task_id: int,
    num_iterations: int,
    renderer: FixedSupportTrainingRenderer,
    action_provider: FormalActionGradientProvider,
    regularizer: Optional[FormalSpectralRegularizer],
    artifact_store: AttackArtifactStore,
    inputs: FormalTrainingInputs,
    state_fingerprints: Sequence[str],
    surface_step: float,
    training_variant: FormalTrainingVariant = "action_spectral",
    action_margin_kappa: float = 0.0,
    engineering_smoke: bool = False,
) -> FormalSourceTrainingResult:
    """执行正式主候选、Action-only control或κ工程smoke。"""

    _validate_commit(code_commit)
    resolved_variant = _resolve_training_variant(training_variant)
    resolved_kappa = float(action_margin_kappa)
    expected_kappa = (
        FROZEN_ACTION_MARGIN_KAPPA
        if resolved_variant == "action_only_kappa"
        else 0.0
    )
    if not math.isfinite(resolved_kappa) or resolved_kappa != expected_kappa:
        raise FormalSourceTrainingError(
            f"{resolved_variant}要求action_margin_kappa={expected_kappa}"
        )
    if engineering_smoke and (
        resolved_variant != "action_only_kappa" or num_iterations != 2
    ):
        raise FormalSourceTrainingError(
            "κ工程smoke必须是action_only_kappa且恰好执行2轮"
        )
    if (
        resolved_variant == "action_only_kappa"
        and not engineering_smoke
        and inputs.action_margin_kappa_smoke_manifest_path is None
    ):
        raise FormalSourceTrainingError(
            "Action-only+κ正式训练缺少已验收的κ smoke manifest"
        )
    if (
        resolved_variant != "action_only_kappa"
        and inputs.action_margin_kappa_smoke_manifest_path is not None
    ):
        raise FormalSourceTrainingError(
            "κ smoke manifest只能绑定Action-only+κ正式训练"
        )
    provider_kappa = float(action_provider.action_margin_kappa)
    if provider_kappa != resolved_kappa:
        raise FormalSourceTrainingError(
            "Action gradient provider的κ与正式训练配置不一致"
        )
    if resolved_variant == "action_spectral" and regularizer is None:
        raise FormalSourceTrainingError("Action+Spectral主候选缺少regularizer")
    if resolved_variant != "action_spectral" and regularizer is not None:
        raise FormalSourceTrainingError(
            "Action-only训练禁止构造或计算Spectral Guard"
        )
    if regularizer is not None and not math.isclose(
        float(regularizer.rho_nat),
        inputs.rho_nat,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FormalSourceTrainingError("regularizer rho_nat与smoke冻结值不一致")
    if num_iterations <= 0:
        raise FormalSourceTrainingError("num_iterations必须为正数")
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式进程已经加载legacy optimizer")
    fingerprints = tuple(state_fingerprints)
    if len(fingerprints) != 10 or len(set(fingerprints)) != 10:
        raise FormalSourceTrainingError("训练state fingerprint必须完整且唯一")

    artifact_store.ensure_attack_directory()
    steps_filename = (
        "action_margin_kappa_smoke_steps.jsonl"
        if engineering_smoke
        else "formal_training_steps.jsonl"
    )
    frames_filename = (
        "action_margin_kappa_smoke_action_frames.jsonl"
        if engineering_smoke
        else "formal_training_action_frames.jsonl"
    )
    manifest_filename = (
        "action_margin_kappa_smoke_manifest.json"
        if engineering_smoke
        else "formal_source_training_manifest.json"
    )
    steps_path = artifact_store.attack_directory / steps_filename
    frames_path = (
        artifact_store.attack_directory / frames_filename
    )
    manifest_path = (
        artifact_store.attack_directory / manifest_filename
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

            if resolved_variant == "action_spectral":
                assert regularizer is not None
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
            else:
                update = trainer.apply_action_only_gradient(
                    action_sample.gradient.detach()
                )
                row = _action_only_step_row(
                    iteration=iteration,
                    action_sample=action_sample,
                    update=update,
                    configured_surface_step=surface_step,
                    training_variant=resolved_variant,
                    action_margin_kappa=resolved_kappa,
                )
                numeric_values = (
                    row["action_loss"],
                    row["action_gradient_l2"],
                    row["total_gradient_l2"],
                    row["action_total_cosine"],
                    row["weighted_spectral_action_ratio"],
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
            if (
                iteration == 0
                or (iteration + 1) % 10 == 0
                or iteration + 1 == num_iterations
            ):
                stats = update.surface_step_stats
                print(
                    "[FORMAL-SOURCE] "
                    f"variant={resolved_variant} "
                    f"iteration={iteration + 1}/{num_iterations} "
                    f"action_loss={action_sample.loss:.8f} "
                    f"weighted_spec_action_ratio="
                    f"{update.weighted_spectral_action_ratio!s} "
                    f"cos_action_spec={update.action_spectral_cosine!s} "
                    f"cos_action_total={update.action_total_cosine!s} "
                    f"surface_step={stats.actual_surface_step:.8f} "
                    f"surface_linf={stats.max_abs_delta:.8f}",
                    flush=True,
                )

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
        "schema_version": (
            ACTION_MARGIN_KAPPA_SMOKE_SCHEMA_VERSION
            if engineering_smoke
            else (
                FORMAL_SOURCE_TRAINING_SCHEMA_VERSION
                if resolved_variant == "action_spectral"
                else (
                    ACTION_ONLY_CONTROL_SCHEMA_VERSION
                    if resolved_variant == "action_only_control"
                    else ACTION_ONLY_KAPPA_SCHEMA_VERSION
                )
            )
        ),
        "training_variant": resolved_variant,
        "code_commit": code_commit,
        "task_id": task_id,
        "train_state_ids": list(EXPECTED_TRAIN_STATE_IDS),
        "train_state_fingerprints": list(fingerprints),
        "num_iterations": num_iterations,
        "trainer_update_count": trainer.update_count,
        "surface_step": surface_step,
        "surface_epsilon": float(renderer.epsilon),
        "action_margin_kappa": resolved_kappa,
        # ``lambda_spec``保留旧主候选schema的兼容字段；新字段明确区分已校准
        # 权重和本次实际应用权重，防止把control误读为lambda校准失败。
        "lambda_spec": (
            inputs.lambda_spec if resolved_variant == "action_spectral" else 0.0
        ),
        "calibrated_lambda_spec": inputs.lambda_spec,
        "applied_lambda_spec": (
            inputs.lambda_spec if resolved_variant == "action_spectral" else 0.0
        ),
        # 即使control不计算Guard，也记录同一smoke冻结的rho作为对照provenance。
        "rho_nat": inputs.rho_nat,
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
            **(
                {
                    "action_margin_kappa_smoke_manifest": str(
                        inputs.action_margin_kappa_smoke_manifest_path
                    )
                }
                if inputs.action_margin_kappa_smoke_manifest_path is not None
                else {}
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
        "objective_components": (
            [
                "untargeted_clean_action_margin_hinge",
                "spectral_naturalness_hinge_squared",
            ]
            if resolved_variant == "action_spectral"
            else (
                ["untargeted_clean_action_margin_hinge"]
                if resolved_variant == "action_only_control"
                else ["untargeted_clean_action_margin_hinge_kappa"]
            )
        ),
        "spectral_guard_computed": resolved_variant == "action_spectral",
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "engineering_smoke": engineering_smoke,
        "scientific_gate": False if engineering_smoke else None,
        "formal_training_allowed": False if engineering_smoke else None,
        "shallow_crossing_gradient_proof": (
            _build_shallow_crossing_gradient_proof(resolved_kappa)
            if engineering_smoke
            else None
        ),
        "paired_source_rollout_required": not engineering_smoke,
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
        training_variant=resolved_variant,
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
    training_variant = _resolve_training_variant(
        cfg.fixed_support_formal_training_variant
    )
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
        action_margin_kappa_smoke_manifest_path=(
            cfg.action_margin_kappa_smoke_manifest_path
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
        action_margin_kappa=float(cfg.action_margin_kappa),
    )
    regularizer: Optional[FormalSpectralRegularizer]
    if training_variant == "action_spectral":
        regularizer = SpectralNaturalnessRegularizer.from_artifacts(
            inputs.rho_nat_calibration_path,
            inputs.spectral_basis_path,
            device=model.device,
            dtype=torch.float32,
        )
    else:
        # Control仍绑定相同自然性artifact与两步smoke，但不把basis/regularizer
        # 搬到GPU，也不计算任何谱能量或谱梯度。
        regularizer = None
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
        training_variant=training_variant,
        action_margin_kappa=float(cfg.action_margin_kappa),
        engineering_smoke=bool(cfg.fixed_support_kappa_smoke_enabled),
    )
    if cfg.fixed_support_kappa_smoke_enabled:
        from .action_margin_kappa_smoke import (
            evaluate_action_margin_kappa_smoke_bundle,
        )

        smoke_decision = evaluate_action_margin_kappa_smoke_bundle(
            result.manifest_path
        )
        if not smoke_decision.gate_pass:
            raise FormalSourceTrainingError(
                "Action-only+κ两步工程smoke独立复核失败: "
                + "; ".join(smoke_decision.failures)
            )
        print(
            "[KAPPA-SMOKE] engineering_valid=true "
            "scientific_gate=false "
            "shallow_crossed_active_tokens="
            f"{smoke_decision.shallow_crossed_active_tokens}",
            flush=True,
        )
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("正式训练期间加载了legacy optimizer")
    return result


def prepare_completed_formal_training_for_evaluation(
    *,
    cfg: Any,
) -> FormalSourceTrainingResult:
    """核验当前checkout/config和旧训练provenance，放行后置paired rollout。"""

    verify_executing_commit(str(cfg.code_commit))
    manifest_path = cfg.fixed_support_formal_training_manifest_path
    if manifest_path is None:
        raise FormalSourceTrainingError("恢复paired rollout缺少正式training manifest")
    inputs = load_formal_training_inputs(
        production_support_path=cfg.fixed_support_path,
        rho_nat_calibration_path=cfg.rho_nat_calibration_path,
        spectral_basis_path=cfg.spectral_naturalness_basis_path,
        spectral_guard_manifest_path=cfg.spectral_guard_manifest_path,
        training_smoke_manifest_path=(
            cfg.fixed_support_training_smoke_manifest_path
        ),
        action_margin_kappa_smoke_manifest_path=(
            cfg.action_margin_kappa_smoke_manifest_path
        ),
    )
    validate_formal_config_against_smoke(cfg, inputs)
    result = load_completed_formal_source_training(manifest_path)
    requested_variant = _resolve_training_variant(
        cfg.fixed_support_formal_training_variant
    )
    if result.training_variant != requested_variant:
        raise FormalSourceTrainingError(
            "恢复manifest训练变体与当前命令不一致: "
            f"manifest={result.training_variant}, requested={requested_variant}"
        )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_sha256") != dict(inputs.input_sha256):
        raise FormalSourceTrainingError(
            "已完成训练未绑定当前命令提供的同一组冻结artifact"
        )
    if loaded_legacy_optimizer_modules():
        raise FormalSourceTrainingError("恢复paired rollout进程加载了legacy optimizer")
    return result
