"""正式 Fixed-Support source training bundle 的独立 CPU 验收。

该 evaluator 不导入模型、LIBERO、renderer 或 CUDA。它重新验证上游两步 smoke、
全部文件 SHA-256、5000轮/50000行状态覆盖、逐轮 SurfaceStepStats、最终紧凑参数
预算和 loss history。服务器路径失效时，会在同步目录的有限祖先范围内按文件名
与 SHA-256 解析同一 artifact，不会因为路径移动跳过 provenance。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import numpy as np
import torch

from .fixed_support_source_training import (
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
        smoke_decision = evaluate_fixed_support_training_smoke_bundle(
            resolved_inputs["fixed_support_training_smoke_manifest"]
        )
        if not smoke_decision.gate_pass:
            failures.append("上游两步training smoke未通过独立复核")
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

    if manifest.get("schema_version") != FORMAL_SOURCE_TRAINING_SCHEMA_VERSION:
        failures.append("正式训练schema不匹配")
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
    step_count = 0
    try:
        for line_number, row in _jsonl_rows(output_paths["steps"]):
            iteration = line_number - 1
            step_count += 1
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
            required_finite = (
                "action_loss",
                "total_energy",
                "low_energy",
                "high_energy",
                "high_ratio",
                "diagnostic_high_ratio",
                "hinge",
                "penalty",
                "action_gradient_l2",
                "spectral_gradient_l2",
                "weighted_spectral_gradient_l2",
                "total_gradient_l2",
                "combination_residual_linf",
                "action_total_cosine",
                "weighted_spectral_action_ratio",
            )
            if not all(_finite(row.get(name)) for name in required_finite):
                failures.append(f"step {iteration}包含NaN/Inf或缺失统计")
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
    if step_count != FORMAL_NUM_ITERATIONS:
        failures.append(f"step JSONL行数不是{FORMAL_NUM_ITERATIONS}")

    frame_count = 0
    try:
        for line_number, row in _jsonl_rows(output_paths["action_frames"]):
            frame_count += 1
            iteration, state_id = divmod(line_number - 1, 10)
            if row.get("iteration") != iteration or row.get("state_id") != state_id:
                failures.append("逐state Action证据未按iteration/state 0-9排列")
                break
            if fingerprints and row.get("initial_state_sha256") != fingerprints[
                state_id
            ]:
                failures.append("逐state Action fingerprint漂移")
                break
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"Action frame JSONL无法复核: {error}")
    if frame_count != FORMAL_NUM_ITERATIONS * 10:
        failures.append("逐state Action证据行数不是50000")

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
            or not np.array_equal(
                loss_history,
                np.asarray(step_losses, dtype=loss_history.dtype),
            )
        ):
            failures.append("loss history未逐值绑定5000轮Action loss")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        failures.append(f"最终参数/loss history无法复核: {error}")

    if manifest.get("objective_components") != [
        "untargeted_clean_action_margin_hinge",
        "spectral_naturalness_hinge_squared",
    ]:
        failures.append("正式训练objective组成错误")
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
