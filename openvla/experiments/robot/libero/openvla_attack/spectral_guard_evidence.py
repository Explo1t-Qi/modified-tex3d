"""Spectral Guard GPU Calibration bundle的独立CPU验收。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .seed_score_audit import file_sha256


SPECTRAL_GUARD_SCHEMA_VERSION = "openvla-spectral-guard-calibration-v1"


@dataclass(frozen=True)
class SpectralGuardBundleDecision:
    gate_pass: bool
    failures: tuple[str, ...]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON无效") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}必须是JSON object")
            rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and (
        float(value) > 0.0
    )


def evaluate_spectral_guard_bundle(
    manifest_path: str | Path,
) -> SpectralGuardBundleDecision:
    """重载逐轮/逐state长表并复算完整性、窗口、lambda和Surface Gate。"""

    failures: list[str] = []
    try:
        resolved_manifest = Path(manifest_path).resolve()
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        root = resolved_manifest.parent
        iteration_path = root / manifest["iterations_relative_path"]
        frame_path = root / manifest["action_frames_relative_path"]
        if file_sha256(iteration_path) != manifest["iterations_sha256"]:
            failures.append("iteration JSONL SHA-256不匹配")
        if file_sha256(frame_path) != manifest["action_frames_sha256"]:
            failures.append("action frame JSONL SHA-256不匹配")
        iterations = _load_jsonl(iteration_path)
        frame_rows = _load_jsonl(frame_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return SpectralGuardBundleDecision(
            False,
            (f"Spectral Guard bundle无法加载: {error}",),
        )

    if manifest.get("schema_version") != SPECTRAL_GUARD_SCHEMA_VERSION:
        failures.append("Spectral Guard schema不匹配")
    expected_ids = list(range(10))
    if manifest.get("state_ids") != expected_ids:
        failures.append("manifest未固定states 0-9")
    fingerprint_map = manifest.get("state_fingerprints", {})
    expected_fingerprints = [
        fingerprint_map.get(str(state_id)) for state_id in expected_ids
    ]
    if not all(_is_sha256(value) for value in expected_fingerprints):
        failures.append("manifest state fingerprint无效或缺失")
    if len(set(expected_fingerprints)) != 10:
        failures.append("manifest state fingerprint不唯一")
    num_iterations = manifest.get("num_iterations")
    if num_iterations != len(iterations) or not 1 <= len(iterations) <= 64:
        failures.append("iteration数量与manifest不一致或越界")
    if len(frame_rows) != 10 * len(iterations):
        failures.append("逐state证据数量不是10倍iteration数")

    config = manifest.get("config", {})
    configured_step = config.get("attack_surface_step")
    epsilon = config.get("attack_epsilon")
    q_rows: list[float | None] = []
    active_rows: list[bool] = []
    for iteration_index, row in enumerate(iterations):
        if row.get("iteration") != iteration_index:
            failures.append(f"iteration {iteration_index}索引不连续")
        if row.get("num_action_frames") != 10:
            failures.append(f"iteration {iteration_index}帧数不是10")
        if row.get("action_state_ids") != expected_ids:
            failures.append(f"iteration {iteration_index} states不完整唯一")
        if row.get("action_state_fingerprints") != expected_fingerprints:
            failures.append(f"iteration {iteration_index} fingerprint绑定错误")
        if row.get("configured_surface_step") != configured_step:
            failures.append(f"iteration {iteration_index} surface_step配置漂移")
        stats = row.get("surface_step_stats", {})
        numeric_names = (
            "direction_surface_max",
            "parameter_scale",
            "projection_scale",
            "step_cap_scale",
            "actual_surface_step",
            "max_abs_delta",
        )
        if not all(
            isinstance(stats.get(name), (int, float))
            and math.isfinite(float(stats[name]))
            for name in numeric_names
        ):
            failures.append(f"iteration {iteration_index} SurfaceStepStats非有限")
        else:
            if stats["actual_surface_step"] > configured_step + 1e-7:
                failures.append(f"iteration {iteration_index} Surface step越界")
            if stats["max_abs_delta"] > epsilon + 1e-7:
                failures.append(f"iteration {iteration_index} Surface Linf越界")
        q_value = row.get("q_t")
        q_rows.append(None if q_value is None else float(q_value))
        active_rows.append(bool(row.get("hinge_active")))
        for gradient_name in (
            "action_gradient_l2",
            "spectral_gradient_l2",
        ):
            gradient_norm = row.get(gradient_name)
            if not isinstance(gradient_norm, (int, float)) or not math.isfinite(
                float(gradient_norm)
            ):
                failures.append(
                    f"iteration {iteration_index} {gradient_name}非有限"
                )

        selected_frames = [
            frame
            for frame in frame_rows
            if frame.get("iteration") == iteration_index
        ]
        selected_frames.sort(key=lambda frame: frame.get("state_id", -1))
        if [frame.get("state_id") for frame in selected_frames] != expected_ids:
            failures.append(f"iteration {iteration_index}逐state行不完整唯一")
        elif [
            frame.get("initial_state_sha256") for frame in selected_frames
        ] != expected_fingerprints:
            failures.append(f"iteration {iteration_index}逐state fingerprint错误")
        elif not all(
            _is_sha256(frame.get("action_gradient_sha256"))
            for frame in selected_frames
        ):
            failures.append(f"iteration {iteration_index}梯度hash无效")
        else:
            recomputed_loss = float(
                np.mean(
                    np.asarray(
                        [frame["action_loss"] for frame in selected_frames],
                        dtype=np.float64,
                    )
                )
            )
            if recomputed_loss != row.get("action_loss"):
                failures.append(f"iteration {iteration_index}Action均值不可复算")

    first_window: tuple[int, int] | None = None
    for end in range(4, len(iterations)):
        indices = range(end - 4, end + 1)
        if all(
            active_rows[index]
            and q_rows[index] is not None
            and math.isfinite(float(q_rows[index]))
            and float(q_rows[index]) > 0.0
            and _is_positive_finite(
                iterations[index].get("action_gradient_l2")
            )
            and _is_positive_finite(
                iterations[index].get("spectral_gradient_l2")
            )
            for index in indices
        ):
            first_window = (end - 4, end)
            break
    if first_window is None:
        if manifest.get("calibration_status") != (
            "uncalibrated_no_stable_activation"
        ):
            failures.append("无稳定窗口时calibration status错误")
        if len(iterations) != 64 or manifest.get("lambda_spec") != 1.0:
            failures.append("无稳定窗口时必须运行64轮并冻结lambda=1")
        if manifest.get("q_median") is not None:
            failures.append("无稳定窗口时q_median必须为空")
    else:
        if [
            manifest.get("selected_window_start"),
            manifest.get("selected_window_end"),
        ] != list(first_window):
            failures.append("manifest未选择首个连续5轮稳定窗口")
        q_median = float(
            np.median(
                np.asarray(
                    [float(q_rows[index]) for index in range(first_window[0], first_window[1] + 1)],
                    dtype=np.float64,
                )
            )
        )
        if manifest.get("q_median") != q_median:
            failures.append("q_median不可复算")
        if manifest.get("lambda_spec") != min(1.0, 0.1 / q_median):
            failures.append("lambda_spec不符合冻结公式")
        if manifest.get("calibration_status") != "calibrated_stable_activation":
            failures.append("稳定窗口calibration status错误")

    restore = manifest.get("restore_evidence", {})
    if not restore or not all(value is True for value in restore.values()):
        failures.append("校准状态未全部恢复")
    if manifest.get("objective_components") != [
        "untargeted_clean_action_margin_hinge"
    ]:
        failures.append("校准objective不是唯一Action hinge")
    if any(
        manifest.get(name) is not False
        for name in (
            "feature_loss_computed",
            "wrist_used",
            "oft_loaded",
            "legacy_optimizer_loaded",
        )
    ):
        failures.append("Feature/wrist/OFT/legacy optimizer进入了校准")
    if (
        manifest.get("rho_nat_calibrated") is not True
        or manifest.get("lambda_spec_calibrated") is not True
        or manifest.get("formal_training_allowed") is not False
    ):
        failures.append("Spectral Guard阶段flags错误")
    if manifest.get("gate_pass") is not True:
        failures.append("manifest未声明gate_pass")
    return SpectralGuardBundleDecision(not failures, tuple(failures))
