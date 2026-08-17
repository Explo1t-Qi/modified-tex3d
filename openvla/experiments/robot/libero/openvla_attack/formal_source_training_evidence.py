"""正式 Fixed-Support source training及Action-only变体的独立CPU验收。

该 evaluator 不导入模型、LIBERO、renderer 或 CUDA。它重新验证上游两步 smoke、
全部文件 SHA-256、5000轮/50000行状态覆盖、逐轮 SurfaceStepStats、最终紧凑参数
预算和 loss history。Action-only还必须证明谱统计未计算、谱贡献严格为零且
total gradient等于Action gradient。服务器路径失效时，会在同步目录的有限祖先
范围内按文件名与SHA-256解析同一artifact，不会因为路径移动跳过provenance。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import numpy as np
import torch

from .action_margin_kappa_smoke import (
    evaluate_action_margin_kappa_smoke_bundle,
    matches_float32_mean,
)
from .configuration import FROZEN_ACTION_MARGIN_KAPPA
from .fixed_support_source_training import (
    ACTION_ONLY_CONTROL_SCHEMA_VERSION,
    ACTION_ONLY_KAPPA_SCHEMA_VERSION,
    EXPECTED_TRAIN_STATE_IDS,
    FORMAL_SOURCE_TRAINING_SCHEMA_VERSION,
)
from .fixed_support_training_smoke import (
    evaluate_fixed_support_training_smoke_bundle,
)
from .production_support import load_production_support_artifact
from .seed_score_audit import file_sha256


FORMAL_NUM_ITERATIONS: Final[int] = 5000
NUMERIC_TOLERANCE: Final[float] = 1e-7
COSINE_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class FormalSourceTrainingDecision:
    """独立复核后的正式训练工程判定；不包含 rollout 效果结论。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}必须是JSON object")
            yield line_number, value


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _resolve_by_hash(
    raw_path: Any,
    *,
    expected_sha256: Any,
    manifest_root: Path,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or not _is_sha256(
        expected_sha256
    ):
        raise ValueError("上游artifact path/hash无效")
    original = Path(raw_path)
    candidates: list[Path] = [original, manifest_root / original]
    # rsync通常把各run放在experiments_inbox的兄弟目录。只搜索manifest根及
    # 两级祖先，避免意外遍历整个仓库或用户目录。
    search_roots = (manifest_root, *tuple(manifest_root.parents)[:5])
    for root in search_roots:
        if root.is_dir():
            candidates.append(root / original.name)
            candidates.extend(root.glob(f"*/{original.name}"))
            candidates.extend(root.glob(f"*/*/{original.name}"))
            # κ工程smoke自身位于另一个run的
            # run_root/attack_artifacts/timestamped_run/manifest；从共同
            # experiments_inbox祖先解析时恰好多一层。保持固定三层上限，
            # 不用无界rglob遍历整个仓库。
            candidates.extend(root.glob(f"*/*/*/{original.name}"))
    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file() and file_sha256(resolved) == expected_sha256:
            return resolved
    raise FileNotFoundError(
        f"找不到SHA-256匹配的同步artifact: {raw_path}"
    )


def evaluate_formal_source_training_bundle(
    manifest_path: str | Path,
) -> FormalSourceTrainingDecision:
    """复核正式训练工程契约，不把 loss 或梯度趋势解释为攻击效果。"""

    failures: list[str] = []
    try:
        resolved_manifest = Path(manifest_path).resolve()
        manifest: Mapping[str, Any] = json.loads(
            resolved_manifest.read_text(encoding="utf-8")
        )
        root = resolved_manifest.parent
        output_paths = {
            "steps": root / str(manifest["steps_relative_path"]),
            "action_frames": root / str(
                manifest["action_frames_relative_path"]
            ),
            "parameter": root / str(manifest["parameter_relative_path"]),
            "baked_texture": root / str(
                manifest["baked_texture_relative_path"]
            ),
            "loss_history": root / str(
                manifest["loss_history_relative_path"]
            ),
        }
        for name, path in output_paths.items():
            if file_sha256(path) != manifest[f"{name}_sha256"]:
                failures.append(f"{name} SHA-256不匹配")

        input_paths = manifest["input_paths"]
        input_hashes = manifest["input_sha256"]
        resolved_inputs = {
            name: _resolve_by_hash(
                input_paths[name],
                expected_sha256=input_hashes[name],
                manifest_root=root,
            )
            for name in (
                "production_support",
                "rho_nat_calibration",
                "spectral_basis",
                "spectral_guard_manifest",
                "fixed_support_training_smoke_manifest",
            )
        }
        if "action_margin_kappa_smoke_manifest" in input_paths:
            resolved_inputs["action_margin_kappa_smoke_manifest"] = (
                _resolve_by_hash(
                    input_paths["action_margin_kappa_smoke_manifest"],
                    expected_sha256=input_hashes.get(
                        "action_margin_kappa_smoke_manifest"
                    ),
                    manifest_root=root,
                )
            )
        smoke_decision = evaluate_fixed_support_training_smoke_bundle(
            resolved_inputs["fixed_support_training_smoke_manifest"]
        )
        smoke_manifest = json.loads(
            resolved_inputs["fixed_support_training_smoke_manifest"].read_text(
                encoding="utf-8"
            )
        )
        if not smoke_decision.gate_pass:
            failures.append("上游两步training smoke未通过独立复核")
        if "action_margin_kappa_smoke_manifest" in resolved_inputs:
            kappa_smoke_decision = (
                evaluate_action_margin_kappa_smoke_bundle(
                    resolved_inputs["action_margin_kappa_smoke_manifest"]
                )
            )
            if not kappa_smoke_decision.gate_pass:
                failures.append("上游Action-only+κ两步smoke未通过独立复核")
        support = load_production_support_artifact(
            resolved_inputs["production_support"]
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return FormalSourceTrainingDecision(
            False,
            (f"正式source training bundle无法加载: {error}",),
        )

    supported_schemas = {
        FORMAL_SOURCE_TRAINING_SCHEMA_VERSION,
        ACTION_ONLY_CONTROL_SCHEMA_VERSION,
        ACTION_ONLY_KAPPA_SCHEMA_VERSION,
    }
    if manifest.get("schema_version") not in supported_schemas:
        failures.append("正式训练schema不匹配")
    raw_variant = manifest.get("training_variant")
    if raw_variant is None and manifest.get("schema_version") == (
        FORMAL_SOURCE_TRAINING_SCHEMA_VERSION
    ):
        # 兼容已经完成并冻结的0aca525主候选bundle。
        training_variant = "action_spectral"
    elif raw_variant in (
        "action_spectral",
        "action_only_control",
        "action_only_kappa",
    ):
        training_variant = str(raw_variant)
    else:
        training_variant = "invalid"
        failures.append("正式训练variant缺失或无效")
    expected_schemas = {
        "action_spectral": FORMAL_SOURCE_TRAINING_SCHEMA_VERSION,
        "action_only_control": ACTION_ONLY_CONTROL_SCHEMA_VERSION,
        "action_only_kappa": ACTION_ONLY_KAPPA_SCHEMA_VERSION,
    }
    expected_schema = expected_schemas.get(training_variant)
    if training_variant != "invalid" and manifest.get(
        "schema_version"
    ) != expected_schema:
        failures.append("正式训练variant与schema不一致")
    has_kappa_smoke_input = (
        "action_margin_kappa_smoke_manifest" in resolved_inputs
    )
    if training_variant == "action_only_kappa":
        if not has_kappa_smoke_input:
            failures.append("Action-only+κ正式训练未绑定已验收κ smoke")
    elif has_kappa_smoke_input:
        failures.append("非κ正式训练不得绑定κ smoke")
    expected_kappa = (
        FROZEN_ACTION_MARGIN_KAPPA
        if training_variant == "action_only_kappa"
        else 0.0
    )
    if manifest.get("action_margin_kappa") != expected_kappa:
        failures.append("正式训练action_margin_kappa错误")
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        failures.append("code_commit不是40位小写Git SHA")
    if manifest.get("train_state_ids") != list(EXPECTED_TRAIN_STATE_IDS):
        failures.append("manifest未固定train states 0-9")
    fingerprints = manifest.get("train_state_fingerprints")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != 10
        or len(set(fingerprints)) != 10
        or not all(_is_sha256(value) for value in fingerprints)
    ):
        failures.append("训练state fingerprint缺失、无效或不唯一")
        fingerprints = []
    if manifest.get("num_iterations") != FORMAL_NUM_ITERATIONS or manifest.get(
        "trainer_update_count"
    ) != FORMAL_NUM_ITERATIONS:
        failures.append("正式训练必须完整执行5000轮且每轮一次Surface update")
    surface_step = manifest.get("surface_step")
    epsilon = manifest.get("surface_epsilon")
    if not _finite(surface_step) or not _finite(epsilon):
        failures.append("surface_step/epsilon非有限")
        resolved_step = resolved_epsilon = 0.0
    else:
        resolved_step = float(surface_step)
        resolved_epsilon = float(epsilon)

    step_losses: list[float] = []
    try:
        for line_number, row in _jsonl_rows(output_paths["steps"]):
            iteration = line_number - 1
            if row.get("iteration") != iteration:
                failures.append(f"step {iteration}索引不连续")
                break
            if row.get("num_action_frames") != 10 or row.get(
                "action_state_ids"
            ) != list(EXPECTED_TRAIN_STATE_IDS):
                failures.append(f"step {iteration}未完整覆盖states 0-9")
                break
            if fingerprints and row.get(
                "action_state_fingerprints"
            ) != fingerprints:
                failures.append(f"step {iteration} state fingerprint漂移")
                break
            common_required_finite = (
                "action_loss",
                "action_gradient_l2",
                "total_gradient_l2",
                "combination_residual_linf",
                "action_total_cosine",
                "weighted_spectral_action_ratio",
            )
            if not all(
                _finite(row.get(name)) for name in common_required_finite
            ):
                failures.append(f"step {iteration}包含NaN/Inf或缺失统计")
                break
            if training_variant == "action_spectral":
                spectral_required_finite = (
                    "total_energy",
                    "low_energy",
                    "high_energy",
                    "high_ratio",
                    "diagnostic_high_ratio",
                    "hinge",
                    "penalty",
                    "spectral_gradient_l2",
                    "weighted_spectral_gradient_l2",
                )
                if not all(
                    _finite(row.get(name))
                    for name in spectral_required_finite
                ):
                    failures.append(
                        f"step {iteration}谱统计包含NaN/Inf或缺失"
                    )
                    break
                if row.get("training_variant") not in (None, "action_spectral"):
                    failures.append(f"step {iteration}训练variant错误")
                    break
                if row.get("spectral_guard_computed") not in (None, True):
                    failures.append(f"step {iteration}谱Guard计算标记错误")
                    break
            elif training_variant in (
                "action_only_control",
                "action_only_kappa",
            ):
                if row.get("training_variant") != training_variant:
                    failures.append(f"step {iteration}训练variant错误")
                    break
                if row.get("action_margin_kappa") != expected_kappa:
                    failures.append(f"step {iteration} κ错误")
                    break
                if row.get("spectral_guard_computed") is not False:
                    failures.append(f"step {iteration}不得计算Spectral Guard")
                    break
                absent_spectral_fields = (
                    "total_energy",
                    "low_energy",
                    "high_energy",
                    "high_ratio",
                    "diagnostic_high_ratio",
                    "hinge",
                    "penalty",
                    "hinge_active",
                    "action_spectral_cosine",
                )
                if any(
                    row.get(name) is not None
                    for name in absent_spectral_fields
                ):
                    failures.append(f"step {iteration}伪造了未计算的谱统计")
                    break
                if any(
                    row.get(name) != 0.0
                    for name in (
                        "spectral_gradient_l2",
                        "weighted_spectral_gradient_l2",
                        "weighted_spectral_action_ratio",
                    )
                ):
                    failures.append(f"step {iteration}Action-only谱贡献不为零")
                    break
                if not math.isclose(
                    float(row["action_gradient_l2"]),
                    float(row["total_gradient_l2"]),
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                ) or not math.isclose(
                    float(row["action_total_cosine"]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=COSINE_TOLERANCE,
                ):
                    failures.append(f"step {iteration}Action-only total梯度不等于Action")
                    break
            if row.get("combination_residual_linf") != 0.0:
                failures.append(f"step {iteration}联合梯度残差不为零")
                break
            if not -1.0 - COSINE_TOLERANCE <= float(
                row["action_total_cosine"]
            ) <= 1.0 + COSINE_TOLERANCE:
                failures.append(f"step {iteration} Action/total cosine越界")
                break
            stats = row.get("surface_step_stats", {})
            if not all(
                _finite(stats.get(name))
                for name in (
                    "direction_surface_max",
                    "parameter_scale",
                    "projection_scale",
                    "step_cap_scale",
                    "actual_surface_step",
                    "max_abs_delta",
                )
            ):
                failures.append(f"step {iteration} SurfaceStepStats无效")
                break
            if stats["actual_surface_step"] > resolved_step + NUMERIC_TOLERANCE:
                failures.append(f"step {iteration} Surface step越界")
                break
            if stats["max_abs_delta"] > resolved_epsilon + NUMERIC_TOLERANCE:
                failures.append(f"step {iteration} Surface-Linf越界")
                break
            step_losses.append(float(row["action_loss"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"step JSONL无法复核: {error}")
    if _line_count(output_paths["steps"]) != FORMAL_NUM_ITERATIONS:
        failures.append(f"step JSONL行数不是{FORMAL_NUM_ITERATIONS}")

    kappa_frame_losses: list[list[float]] = (
        [[] for _ in range(FORMAL_NUM_ITERATIONS)]
        if training_variant == "action_only_kappa"
        else []
    )
    try:
        for line_number, row in _jsonl_rows(output_paths["action_frames"]):
            iteration, state_id = divmod(line_number - 1, 10)
            if row.get("iteration") != iteration or row.get("state_id") != state_id:
                failures.append("逐state Action证据未按iteration/state 0-9排列")
                break
            if fingerprints and row.get("initial_state_sha256") != fingerprints[
                state_id
            ]:
                failures.append("逐state Action fingerprint漂移")
                break
            if training_variant == "action_only_kappa":
                margins = np.asarray(row.get("margins"), dtype=np.float32)
                hinges = np.asarray(
                    row.get("hinge_values"),
                    dtype=np.float32,
                )
                expected_hinges = np.maximum(
                    np.float32(0.0),
                    margins + np.float32(FROZEN_ACTION_MARGIN_KAPPA),
                )
                if (
                    margins.ndim != 1
                    or margins.size == 0
                    or margins.shape != hinges.shape
                    or not np.isfinite(margins).all()
                    or not np.array_equal(hinges, expected_hinges)
                ):
                    failures.append(
                        f"step {iteration} state {state_id} κ hinge不可复算"
                    )
                    break
                active = int(np.count_nonzero(hinges > 0.0))
                shallow = int(
                    np.count_nonzero((margins <= 0.0) & (hinges > 0.0))
                )
                nonpositive = int(np.count_nonzero(margins <= 0.0))
                if (
                    row.get("action_margin_kappa")
                    != FROZEN_ACTION_MARGIN_KAPPA
                    or row.get("active_token_count") != active
                    or row.get("shallow_crossed_active_count") != shallow
                    or row.get("nonpositive_margin_count") != nonpositive
                    or not matches_float32_mean(
                        row.get("action_loss"),
                        hinges,
                    )
                ):
                    failures.append(
                        f"step {iteration} state {state_id} κ统计错误"
                    )
                    break
                kappa_frame_losses[iteration].append(
                    float(row["action_loss"])
                )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"Action frame JSONL无法复核: {error}")
    if _line_count(output_paths["action_frames"]) != FORMAL_NUM_ITERATIONS * 10:
        failures.append("逐state Action证据行数不是50000")
    if training_variant == "action_only_kappa" and len(
        step_losses
    ) == FORMAL_NUM_ITERATIONS:
        for iteration, frame_losses in enumerate(kappa_frame_losses):
            if len(frame_losses) != 10 or not math.isclose(
                step_losses[iteration],
                float(np.mean(frame_losses, dtype=np.float64)),
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            ):
                failures.append(
                    f"step {iteration} loss不是10-state算术均值"
                )
                break

    try:
        parameter = torch.load(
            output_paths["parameter"],
            map_location="cpu",
            weights_only=True,
        )
        expected_shape = (len(support.support_vertex_indices), 3)
        if (
            not isinstance(parameter, torch.Tensor)
            or tuple(parameter.shape) != expected_shape
            or not parameter.is_floating_point()
            or not bool(torch.isfinite(parameter).all())
        ):
            failures.append("最终紧凑参数必须为有限float [N_support,3]")
        else:
            observed_linf = float(parameter.abs().amax().item())
            if observed_linf > resolved_epsilon + NUMERIC_TOLERANCE:
                failures.append("最终紧凑参数超出Surface-Linf预算")
            if not math.isclose(
                observed_linf,
                float(manifest.get("final_parameter_linf", float("nan"))),
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                failures.append("最终参数Linf与manifest不一致")
        loss_history = np.load(output_paths["loss_history"], allow_pickle=False)
        if (
            loss_history.shape != (FORMAL_NUM_ITERATIONS,)
            or not np.isfinite(loss_history).all()
        ):
            failures.append("loss history必须是5000个有限标量")
        elif len(step_losses) == FORMAL_NUM_ITERATIONS and not np.array_equal(
            loss_history,
            np.asarray(step_losses, dtype=loss_history.dtype),
        ):
            failures.append("loss history未逐值绑定5000轮Action loss")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        failures.append(f"最终参数/loss history无法复核: {error}")

    expected_objectives = {
        "action_spectral": [
            "untargeted_clean_action_margin_hinge",
            "spectral_naturalness_hinge_squared",
        ],
        "action_only_control": [
            "untargeted_clean_action_margin_hinge"
        ],
        "action_only_kappa": [
            "untargeted_clean_action_margin_hinge_kappa"
        ],
    }.get(training_variant)
    if manifest.get("objective_components") != expected_objectives:
        failures.append("正式训练objective组成错误")
    if training_variant == "action_spectral":
        if manifest.get("spectral_guard_computed") not in (None, True):
            failures.append("主候选必须计算Spectral Guard")
    elif training_variant in ("action_only_control", "action_only_kappa"):
        if manifest.get("spectral_guard_computed") is not False:
            failures.append("Action-only变体不得计算Spectral Guard")
        if manifest.get("applied_lambda_spec") != 0.0 or manifest.get(
            "lambda_spec"
        ) != 0.0:
            failures.append("Action-only变体实际lambda必须为零")
        calibrated_lambda = manifest.get("calibrated_lambda_spec")
        if not _finite(calibrated_lambda) or not (
            0.0 < float(calibrated_lambda) <= 1.0
        ):
            failures.append("Action-only变体未绑定有效的已校准lambda")
    if not _finite(manifest.get("rho_nat")) or not math.isclose(
        float(manifest.get("rho_nat", float("nan"))),
        float(smoke_manifest.get("rho_nat", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("正式训练rho_nat未绑定同一smoke冻结值")
    manifest_calibrated_lambda = (
        manifest.get("lambda_spec")
        if training_variant == "action_spectral"
        else manifest.get("calibrated_lambda_spec")
    )
    if not _finite(manifest_calibrated_lambda) or not math.isclose(
        float(manifest_calibrated_lambda),
        float(smoke_manifest.get("lambda_spec", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        failures.append("正式训练未绑定同一smoke冻结lambda")
    for name in (
        "feature_loss_computed",
        "wrist_used",
        "oft_loaded",
        "legacy_optimizer_loaded",
        "source_gate_evaluated",
    ):
        if manifest.get(name) is not False:
            failures.append(f"{name}必须为false")
    if manifest.get("paired_source_rollout_required") is not True:
        failures.append("正式训练后未声明paired source rollout门槛")
    return FormalSourceTrainingDecision(not failures, tuple(failures))
