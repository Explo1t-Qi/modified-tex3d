"""Action-only+κ 两步工程 smoke 的独立 CPU evidence 验收。

本模块不导入模型、LIBERO、renderer 或 CUDA。它从 manifest 重新核对两轮
正式 Fixed-Support 更新、20 行逐 state Action objective、Surface step/L∞
预算、最终紧凑参数、loss history、bake PNG 与全部 SHA-256。该 smoke 只证明
工程数据流，schema 明确禁止把结果解释为科学 Gate 或放行正式训练。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import numpy as np
import torch
from PIL import Image

from .configuration import FROZEN_ACTION_MARGIN_KAPPA
from .fixed_support_source_training import (
    ACTION_MARGIN_KAPPA_SMOKE_SCHEMA_VERSION,
    EXPECTED_TRAIN_STATE_IDS,
)
from .fixed_support_training_smoke import (
    evaluate_fixed_support_training_smoke_bundle,
)
from .production_support import load_production_support_artifact
from .seed_score_audit import file_sha256


EXPECTED_NUM_STEPS: Final[int] = 2
NUMERIC_TOLERANCE: Final[float] = 1e-7
FLOAT32_REDUCTION_MAX_ULPS: Final[int] = 4


@dataclass(frozen=True)
class ActionMarginKappaSmokeDecision:
    """独立复核结果；``gate_pass`` 仅表示工程证据有效。"""

    gate_pass: bool
    failures: tuple[str, ...]
    shallow_crossed_active_tokens: int


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def matches_float32_mean(
    observed: Any,
    values: np.ndarray,
) -> bool:
    """允许GPU/CPU float32 mean归约顺序造成的有限ULP差异。"""

    if not _finite(observed) or values.dtype != np.float32:
        return False
    resolved_observed = float(observed)
    # 真实provider的loss来自float32 tensor.item()；拒绝无法由float32表示的
    # 任意JSON小数，避免ULP容差掩盖手工篡改。
    if float(np.float32(resolved_observed)) != resolved_observed:
        return False
    expected = float(values.mean(dtype=np.float64))
    reference = np.float32(max(abs(resolved_observed), abs(expected)))
    ulp = abs(float(np.spacing(reference)))
    return abs(resolved_observed - expected) <= (
        FLOAT32_REDUCTION_MAX_ULPS * ulp
    )


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
    """在服务器原路径或rsync后的有限邻域解析同一输入artifact。"""

    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise ValueError("输入artifact path/hash无效")
    original = Path(raw_path)
    candidates: list[Path] = [original, manifest_root / original]
    for root in (manifest_root, *tuple(manifest_root.parents)[:5]):
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
    raise FileNotFoundError(f"找不到SHA-256匹配的输入artifact: {raw_path}")


def evaluate_action_margin_kappa_smoke_bundle(
    manifest_path: str | Path,
) -> ActionMarginKappaSmokeDecision:
    """从磁盘重算 κ objective 与 Surface/artifact 工程契约。"""

    failures: list[str] = []
    shallow_total = 0
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
                raw_path,
                expected_sha256=input_hashes[name],
                manifest_root=root,
            )
            for name, raw_path in input_paths.items()
        }
        old_smoke_path = resolved_inputs[
            "fixed_support_training_smoke_manifest"
        ]
        old_smoke = evaluate_fixed_support_training_smoke_bundle(
            old_smoke_path
        )
        if not old_smoke.gate_pass:
            failures.append("绑定的原Fixed-Support两步smoke无效")
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
        return ActionMarginKappaSmokeDecision(
            False,
            (f"κ工程smoke bundle无法加载: {error}",),
            0,
        )

    expected_manifest = {
        "schema_version": ACTION_MARGIN_KAPPA_SMOKE_SCHEMA_VERSION,
        "training_variant": "action_only_kappa",
        "action_margin_kappa": FROZEN_ACTION_MARGIN_KAPPA,
        "num_iterations": EXPECTED_NUM_STEPS,
        "trainer_update_count": EXPECTED_NUM_STEPS,
        "engineering_smoke": True,
        "scientific_gate": False,
        "formal_training_allowed": False,
        "paired_source_rollout_required": False,
        "spectral_guard_computed": False,
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            failures.append(f"manifest {name}未绑定工程smoke契约")
    if manifest.get("objective_components") != [
        "untargeted_clean_action_margin_hinge_kappa"
    ]:
        failures.append("κ工程smoke objective组成错误")
    expected_gradient_proof = {
        "margin": -2.0,
        "zero_kappa_hinge": 0.0,
        "zero_kappa_clean_logit_gradient": 0.0,
        "action_margin_kappa": FROZEN_ACTION_MARGIN_KAPPA,
        "kappa_hinge": 2.375,
        "kappa_clean_logit_gradient": 1.0,
        "kappa_best_other_logit_gradient": -1.0,
    }
    if manifest.get("shallow_crossing_gradient_proof") != expected_gradient_proof:
        failures.append("浅越界token的κ梯度证明无效")
    fingerprints = manifest.get("train_state_fingerprints")
    if (
        manifest.get("train_state_ids") != list(EXPECTED_TRAIN_STATE_IDS)
        or not isinstance(fingerprints, list)
        or len(fingerprints) != 10
        or len(set(fingerprints)) != 10
    ):
        failures.append("训练states/fingerprint不完整唯一")
        fingerprints = []
    surface_step = manifest.get("surface_step")
    epsilon = manifest.get("surface_epsilon")
    if not _finite(surface_step) or not _finite(epsilon):
        failures.append("Surface step/epsilon无效")
        resolved_step = resolved_epsilon = 0.0
    else:
        resolved_step = float(surface_step)
        resolved_epsilon = float(epsilon)

    step_losses: list[float] = []
    frame_losses: dict[int, list[float]] = {step: [] for step in range(2)}
    try:
        step_rows = list(_jsonl_rows(output_paths["steps"]))
        if len(step_rows) != 2:
            failures.append("κ工程smoke step JSONL必须恰有2行")
        for line_number, row in step_rows:
            step = line_number - 1
            if (
                row.get("iteration") != step
                or row.get("training_variant") != "action_only_kappa"
                or row.get("action_margin_kappa")
                != FROZEN_ACTION_MARGIN_KAPPA
            ):
                failures.append(f"step {step} κ/variant/索引错误")
                continue
            if row.get("action_state_ids") != list(
                EXPECTED_TRAIN_STATE_IDS
            ) or row.get("action_state_fingerprints") != fingerprints:
                failures.append(f"step {step} state绑定错误")
            if row.get("spectral_guard_computed") is not False:
                failures.append(f"step {step}不得计算Spectral Guard")
            for name in (
                "spectral_gradient_l2",
                "weighted_spectral_gradient_l2",
                "weighted_spectral_action_ratio",
            ):
                if row.get(name) != 0.0:
                    failures.append(f"step {step} {name}必须为零")
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
                failures.append(f"step {step} SurfaceStepStats无效")
            elif (
                float(stats["actual_surface_step"])
                > resolved_step + NUMERIC_TOLERANCE
                or float(stats["max_abs_delta"])
                > resolved_epsilon + NUMERIC_TOLERANCE
            ):
                failures.append(f"step {step} Surface预算越界")
            if _finite(row.get("action_loss")):
                step_losses.append(float(row["action_loss"]))
            else:
                failures.append(f"step {step} action_loss无效")

        frame_rows = list(_jsonl_rows(output_paths["action_frames"]))
        if len(frame_rows) != 20:
            failures.append("κ工程smoke Action frame JSONL必须恰有20行")
        for line_number, row in frame_rows:
            step, state_id = divmod(line_number - 1, 10)
            if row.get("iteration") != step or row.get("state_id") != state_id:
                failures.append("逐state evidence未按两轮states 0-9排列")
                break
            if fingerprints and row.get("initial_state_sha256") != fingerprints[
                state_id
            ]:
                failures.append("逐state fingerprint漂移")
                break
            # provider在BF16模型输出提升到float32后计算margin/hinge；CPU复算
            # 必须保持同一float32加法与ReLU舍入，不能悄悄换成float64公式。
            margins = np.asarray(row.get("margins"), dtype=np.float32)
            hinges = np.asarray(row.get("hinge_values"), dtype=np.float32)
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
                failures.append(f"step {step} state {state_id} κ hinge不可复算")
                continue
            active = int(np.count_nonzero(hinges > 0.0))
            shallow = int(np.count_nonzero((margins <= 0.0) & (hinges > 0.0)))
            nonpositive = int(np.count_nonzero(margins <= 0.0))
            if (
                row.get("action_margin_kappa")
                != FROZEN_ACTION_MARGIN_KAPPA
                or row.get("active_token_count") != active
                or row.get("shallow_crossed_active_count") != shallow
                or row.get("nonpositive_margin_count") != nonpositive
                or not matches_float32_mean(row.get("action_loss"), hinges)
            ):
                failures.append(f"step {step} state {state_id} κ统计错误")
            shallow_total += shallow
            frame_losses[step].append(float(row["action_loss"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"κ工程smoke JSONL无法复核: {error}")

    if len(step_losses) == 2:
        for step, step_loss in enumerate(step_losses):
            if len(frame_losses[step]) != 10 or not math.isclose(
                step_loss,
                float(np.mean(frame_losses[step], dtype=np.float64)),
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                failures.append(f"step {step} loss不是10-state算术均值")
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
            or float(parameter.abs().amax().item())
            > resolved_epsilon + NUMERIC_TOLERANCE
        ):
            failures.append("最终紧凑参数shape/数值/Surface-Linf无效")
        loss_history = np.load(output_paths["loss_history"], allow_pickle=False)
        if loss_history.shape != (2,) or not np.array_equal(
            loss_history,
            np.asarray(step_losses, dtype=loss_history.dtype),
        ):
            failures.append("loss history未逐值绑定两轮Action loss")
        with Image.open(output_paths["baked_texture"]) as baked_image:
            baked_image.load()
            if baked_image.mode != "RGB" or min(baked_image.size) <= 0:
                failures.append("bake PNG不是非空RGB纹理")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        failures.append(f"最终参数/loss history无法复核: {error}")

    return ActionMarginKappaSmokeDecision(
        gate_pass=not failures,
        failures=tuple(failures),
        shallow_crossed_active_tokens=shallow_total,
    )
