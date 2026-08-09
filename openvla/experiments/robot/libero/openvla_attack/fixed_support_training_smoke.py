"""Fixed-Support Action+Spectral两步GPU smoke的独立CPU证据验收。

该模块不导入模型、LIBERO或CUDA。它从落盘梯度重新验证联合公式，并从完整
几何Surface Delta验证Support外严格为零、两次Surface step与L∞预算。零点
Step 0必须退化为Action-only；非零Step 1必须真正激活谱梯度。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
from PIL import Image

from .production_support import (
    array_sha256,
    load_production_support_artifact,
)
from .seed_score_audit import file_sha256
from .spectral_guard_evidence import evaluate_spectral_guard_bundle


FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION: Final[str] = (
    "openvla-fixed-support-action-spectral-smoke-v1"
)
TRAINING_SMOKE_ARRAY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_gradients",
        "spectral_gradients",
        "weighted_spectral_gradients",
        "total_gradients",
        "geometry_deltas",
        "support_vertex_indices",
    }
)
NUM_TRAINING_SMOKE_STEPS: Final[int] = 2
NUMERIC_TOLERANCE: Final[float] = 1e-7


@dataclass(frozen=True)
class FixedSupportTrainingSmokeDecision:
    """独立重算后的两步smoke判定。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}必须是JSON object")
            rows.append(value)
    return rows


def _resolve_input_path(raw_path: Any, *, manifest_root: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("上游input path必须是非空字符串")
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (manifest_root / path).resolve()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _l2(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.astype(np.float64, copy=False).reshape(-1)))


def _close(left: float, right: Any) -> bool:
    return _finite_number(right) and math.isclose(
        left,
        float(right),
        # runner的norm由GPU float32 reduction得到；WSL从落盘数组以float64
        # 重算，允许归约顺序造成的小误差，但联合梯度本身仍要求逐值相等。
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


def evaluate_fixed_support_training_smoke_bundle(
    manifest_path: str | Path,
) -> FixedSupportTrainingSmokeDecision:
    """重载两步长表、完整梯度与Surface Delta并执行严格Gate。"""

    failures: list[str] = []
    try:
        resolved_manifest = Path(manifest_path).resolve()
        manifest: Mapping[str, Any] = json.loads(
            resolved_manifest.read_text(encoding="utf-8")
        )
        root = resolved_manifest.parent
        steps_path = root / str(manifest["steps_relative_path"])
        frames_path = root / str(manifest["action_frames_relative_path"])
        arrays_path = root / str(manifest["arrays_relative_path"])
        baked_path = root / str(manifest["baked_texture_relative_path"])
        steps = _load_jsonl(steps_path)
        frames = _load_jsonl(frames_path)
        if file_sha256(steps_path) != manifest["steps_sha256"]:
            failures.append("step JSONL SHA-256不匹配")
        if file_sha256(frames_path) != manifest["action_frames_sha256"]:
            failures.append("action frame JSONL SHA-256不匹配")
        if file_sha256(arrays_path) != manifest["arrays_sha256"]:
            failures.append("完整梯度artifact SHA-256不匹配")
        if file_sha256(baked_path) != manifest["baked_texture_sha256"]:
            failures.append("bake PNG SHA-256不匹配")
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != TRAINING_SMOKE_ARRAY_KEYS:
                raise ValueError("完整梯度artifact keys不匹配")
            arrays = {name: archive[name].copy() for name in archive.files}

        input_paths = manifest["input_paths"]
        input_hashes = manifest["input_sha256"]
        resolved_inputs = {
            name: _resolve_input_path(path, manifest_root=root)
            for name, path in input_paths.items()
        }
        for name in (
            "production_support",
            "rho_nat_calibration",
            "spectral_basis",
            "spectral_guard_manifest",
            "mesh",
            "texture",
        ):
            if file_sha256(resolved_inputs[name]) != input_hashes[name]:
                failures.append(f"上游{name} SHA-256不匹配")
        support = load_production_support_artifact(
            resolved_inputs["production_support"]
        )
        calibration_manifest = json.loads(
            resolved_inputs["spectral_guard_manifest"].read_text(
                encoding="utf-8"
            )
        )
        calibration_root = resolved_inputs["spectral_guard_manifest"].parent
        calibration_frames = _load_jsonl(
            calibration_root
            / str(calibration_manifest["action_frames_relative_path"])
        )
        calibration_decision = evaluate_spectral_guard_bundle(
            resolved_inputs["spectral_guard_manifest"]
        )
        if not calibration_decision.gate_pass:
            failures.append("上游Spectral Guard calibration未通过独立复核")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return FixedSupportTrainingSmokeDecision(
            False,
            (f"Fixed-Support training smoke bundle无法加载: {error}",),
        )

    if manifest.get("schema_version") != FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION:
        failures.append("training smoke schema不匹配")
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        failures.append("code_commit不是40位小写Git SHA")
    if manifest.get("state_ids") != list(range(10)):
        failures.append("manifest未固定states 0-9")
    fingerprints = manifest.get("state_fingerprints", {})
    expected_fingerprints = [
        fingerprints.get(str(state_id)) for state_id in range(10)
    ]
    if not all(_is_sha256(value) for value in expected_fingerprints) or len(
        set(expected_fingerprints)
    ) != 10:
        failures.append("state fingerprint缺失、无效或不唯一")

    lambda_spec = manifest.get("lambda_spec")
    if not _finite_number(lambda_spec) or not 0.0 < float(lambda_spec) <= 1.0:
        failures.append("lambda_spec必须为(0,1]内有限数")
        resolved_lambda = 0.0
    else:
        resolved_lambda = float(lambda_spec)
    if calibration_manifest.get("lambda_spec") != lambda_spec:
        failures.append("lambda_spec与校准manifest不一致")
    if calibration_manifest.get("input_sha256", {}).get(
        "production_support"
    ) != manifest.get("input_sha256", {}).get("production_support"):
        failures.append("校准与smoke未绑定同一Production Support")
    if calibration_manifest.get("input_sha256", {}).get(
        "rho_nat_calibration"
    ) != manifest.get("input_sha256", {}).get("rho_nat_calibration"):
        failures.append("校准与smoke未绑定同一rho_nat artifact")
    if calibration_manifest.get("input_sha256", {}).get(
        "spectral_basis"
    ) != manifest.get("input_sha256", {}).get("spectral_basis"):
        failures.append("校准与smoke未绑定同一谱基")

    action = arrays["action_gradients"]
    spectral = arrays["spectral_gradients"]
    weighted = arrays["weighted_spectral_gradients"]
    total = arrays["total_gradients"]
    deltas = arrays["geometry_deltas"]
    support_indices = arrays["support_vertex_indices"]
    expected_gradient_shape = (
        NUM_TRAINING_SMOKE_STEPS,
        len(support.support_vertex_indices),
        3,
    )
    if any(
        value.dtype != np.float32 or value.shape != expected_gradient_shape
        for value in (action, spectral, weighted, total)
    ):
        failures.append("联合梯度必须为float32 [2,N_support,3]")
    elif not all(
        bool(np.isfinite(value).all())
        for value in (action, spectral, weighted, total)
    ):
        failures.append("联合梯度artifact包含NaN/Inf")
    else:
        expected_weighted = spectral * np.float32(resolved_lambda)
        if not np.array_equal(weighted, expected_weighted):
            failures.append("weighted spectral梯度不可逐值复算")
        if not np.array_equal(total, action + weighted):
            failures.append("g_total不等于g_action+lambda*g_spec")
        if np.count_nonzero(spectral[0]) != 0:
            failures.append("Step 0谱梯度必须严格为零")
        if np.count_nonzero(spectral[1]) == 0:
            failures.append("Step 1谱梯度必须非零")
        if not np.array_equal(total[0], action[0]):
            failures.append("Step 0联合梯度未严格退化为Action梯度")

    if support_indices.dtype != np.int64 or not np.array_equal(
        support_indices,
        support.support_vertex_indices,
    ):
        failures.append("gradient artifact的Support坐标与Production Support不一致")
    if array_sha256(support_indices) != manifest.get(
        "support_vertex_indices_sha256"
    ):
        failures.append("Support vertex indices semantic hash不匹配")
    expected_delta_shape = (
        NUM_TRAINING_SMOKE_STEPS + 1,
        support.num_geometry_vertices,
        3,
    )
    delta_shape_valid = (
        deltas.dtype == np.float32 and deltas.shape == expected_delta_shape
    )
    if not delta_shape_valid:
        failures.append("geometry_deltas必须为float32 [3,N_v,3]")
    elif not bool(np.isfinite(deltas).all()):
        failures.append("geometry_deltas包含NaN/Inf")
    else:
        outside = np.ones(support.num_geometry_vertices, dtype=np.bool_)
        outside[support.support_vertex_indices] = False
        if np.count_nonzero(deltas[0]) != 0:
            failures.append("smoke未从零Surface Delta开始")
        if np.count_nonzero(deltas[:, outside, :]) != 0:
            failures.append("Support外Surface Delta不为零")

    if len(steps) != NUM_TRAINING_SMOKE_STEPS:
        failures.append("smoke必须恰好保存两个训练step")
    config = manifest.get("config", {})
    surface_step = config.get("attack_surface_step")
    epsilon = config.get("attack_epsilon")
    if not _finite_number(surface_step) or not _finite_number(epsilon):
        failures.append("surface_step或epsilon配置无效")
        resolved_step = 0.0
        resolved_epsilon = 0.0
    else:
        resolved_step = float(surface_step)
        resolved_epsilon = float(epsilon)
    for index, row in enumerate(steps[:NUM_TRAINING_SMOKE_STEPS]):
        if row.get("step") != index:
            failures.append(f"Step {index}索引不连续")
        if row.get("num_action_frames") != 10 or row.get(
            "action_state_ids"
        ) != list(range(10)):
            failures.append(f"Step {index}未完整唯一覆盖states 0-9")
        if row.get("action_state_fingerprints") != expected_fingerprints:
            failures.append(f"Step {index} state fingerprint绑定错误")
        if not _finite_number(row.get("action_loss")):
            failures.append(f"Step {index} Action loss非有限")
        if index < action.shape[0]:
            expected_norms = {
                "action_gradient_l2": _l2(action[index]),
                "spectral_gradient_l2": _l2(spectral[index]),
                "weighted_spectral_gradient_l2": _l2(weighted[index]),
                "total_gradient_l2": _l2(total[index]),
            }
            for name, expected in expected_norms.items():
                if not _close(expected, row.get(name)):
                    failures.append(f"Step {index} {name}不可复算")
            action_norm = expected_norms["action_gradient_l2"]
            weighted_norm = expected_norms["weighted_spectral_gradient_l2"]
            if action_norm <= 0.0:
                failures.append(f"Step {index} Action梯度必须非零")
            elif not _close(
                weighted_norm / action_norm,
                row.get("weighted_spectral_action_ratio"),
            ):
                failures.append(f"Step {index} weighted gradient ratio不可复算")
            spectral_norm = expected_norms["spectral_gradient_l2"]
            if spectral_norm == 0.0:
                if row.get("action_spectral_cosine") is not None:
                    failures.append(f"Step {index}零谱梯度的cosine必须为空")
            else:
                cosine = float(
                    np.dot(
                        action[index].astype(np.float64).reshape(-1),
                        spectral[index].astype(np.float64).reshape(-1),
                    )
                    / (action_norm * spectral_norm)
                )
                if not _close(cosine, row.get("action_spectral_cosine")):
                    failures.append(f"Step {index} Action/Spectral cosine不可复算")
        if row.get("combination_residual_linf") != 0.0:
            failures.append(f"Step {index}联合梯度残差不为零")
        stats = row.get("surface_step_stats", {})
        if not all(
            _finite_number(stats.get(name))
            for name in (
                "direction_surface_max",
                "parameter_scale",
                "projection_scale",
                "step_cap_scale",
                "actual_surface_step",
                "max_abs_delta",
            )
        ):
            failures.append(f"Step {index} SurfaceStepStats非有限")
        else:
            if stats["actual_surface_step"] > resolved_step + NUMERIC_TOLERANCE:
                failures.append(f"Step {index} Surface step越界")
            if stats["max_abs_delta"] > resolved_epsilon + NUMERIC_TOLERANCE:
                failures.append(f"Step {index} Surface L∞越界")
            if delta_shape_valid:
                observed_step = float(
                    np.max(np.abs(deltas[index + 1] - deltas[index]))
                )
                observed_linf = float(np.max(np.abs(deltas[index + 1])))
                if not _close(observed_step, stats["actual_surface_step"]):
                    failures.append(f"Step {index} Surface step不可由delta复算")
                if not _close(observed_linf, stats["max_abs_delta"]):
                    failures.append(f"Step {index} Surface L∞不可由delta复算")
        if index == 0 and (
            row.get("hinge_active") is not False
            or row.get("total_energy") != 0.0
            or row.get("high_energy") != 0.0
        ):
            failures.append("Step 0谱项必须严格为零且未激活")
        if index == 1 and row.get("hinge_active") is not True:
            failures.append("Step 1谱hinge必须激活")

    if len(frames) != 20:
        failures.append("逐state action证据必须恰好20行")
    else:
        for step in range(2):
            selected = sorted(
                (row for row in frames if row.get("iteration") == step),
                key=lambda row: row.get("state_id", -1),
            )
            if [row.get("state_id") for row in selected] != list(range(10)):
                failures.append(f"Step {step}逐state action证据不完整唯一")
            elif [
                row.get("initial_state_sha256") for row in selected
            ] != expected_fingerprints:
                failures.append(f"Step {step}逐state fingerprint错误")
        smoke_step_zero = sorted(
            (row for row in frames if row.get("iteration") == 0),
            key=lambda row: row.get("state_id", -1),
        )
        calibration_step_zero = sorted(
            (row for row in calibration_frames if row.get("iteration") == 0),
            key=lambda row: row.get("state_id", -1),
        )
        comparison_fields = (
            "state_id",
            "initial_state_sha256",
            "clean_action_token_ids",
            "margins",
            "hinge_values",
            "action_loss",
        )
        if len(calibration_step_zero) != 10 or any(
            smoke.get(field) != calibration.get(field)
            for smoke, calibration in zip(
                smoke_step_zero,
                calibration_step_zero,
            )
            for field in comparison_fields
        ):
            failures.append("Step 0未逐项复现已通过的Action-only校准零点")

    try:
        clean_pixels = np.asarray(Image.open(resolved_inputs["texture"]).convert("RGB"))
        baked_pixels = np.asarray(Image.open(baked_path).convert("RGB"))
        if clean_pixels.shape != baked_pixels.shape or np.array_equal(
            clean_pixels, baked_pixels
        ):
            failures.append("bake PNG未产生真实像素变化")
    except OSError as error:
        failures.append(f"clean/bake PNG无法读取: {error}")
    if manifest.get("active_texture_sha256") != manifest.get(
        "baked_texture_sha256"
    ):
        failures.append("Active Texture与bake PNG hash不一致")
    if manifest.get("active_environment_loaded") is not True or not _is_sha256(
        manifest.get("active_observation_sha256")
    ):
        failures.append("MuJoCo未成功加载Active Texture环境")
    if manifest.get("xml_sha256_before") != manifest.get(
        "xml_sha256_after_restore"
    ):
        failures.append("XML资产未恢复")
    if manifest.get("texture_sha256_before") != manifest.get(
        "texture_sha256_after_restore"
    ):
        failures.append("真实纹理资产未恢复")
    if manifest.get("backup_paths_removed") is not True:
        failures.append("Runtime Asset backup未删除")
    if manifest.get("trainer_update_count") != 2:
        failures.append("两步smoke必须且只能执行两次Surface update")
    if manifest.get("objective_components") != [
        "untargeted_clean_action_margin_hinge",
        "spectral_naturalness_hinge_squared",
    ]:
        failures.append("正式trainer objective components不匹配")
    if any(
        manifest.get(name) is not False
        for name in (
            "feature_loss_computed",
            "wrist_used",
            "oft_loaded",
            "legacy_optimizer_loaded",
        )
    ):
        failures.append("Feature/wrist/OFT/legacy optimizer进入了training smoke")
    if (
        manifest.get("rho_nat_calibrated") is not True
        or manifest.get("lambda_spec_calibrated") is not True
        or manifest.get("formal_training_allowed") is not True
        or manifest.get("next_required_gate")
        != "fixed_support_action_spectral_source_training"
    ):
        failures.append("training smoke阶段flags或下一门槛错误")
    if manifest.get("gate_pass") is not True:
        failures.append("manifest未声明gate_pass")
    return FixedSupportTrainingSmokeDecision(not failures, tuple(failures))
