"""Gate 6g 两个正式 Fixed-Support 终态的只读输入解析。

本模块不导入 renderer、LIBERO 或 OpenVLA。它先复用正式 source training 的
独立 evaluator，再把 Action+Spectral 与 Action-only manifest 收窄为一个共享
Production Support、相同训练 states 和两个不可变 parameter/bake 对。后续
CUDA preflight 与 C/A/B runner 只能消费本模块返回的已验证对象，不能自行猜测
终态文件名或跳过 SHA-256。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from .formal_source_training_evidence import (
    evaluate_formal_source_training_bundle,
)


EXPECTED_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10))
EXPECTED_SURFACE_EPSILON: Final[float] = 128.0 / 255.0
EXPECTED_VARIANTS: Final[tuple[str, ...]] = (
    "action_spectral",
    "action_only_control",
)


class TerminalDeploymentInputError(RuntimeError):
    """正式终态、Support 或其 provenance 不满足 Gate 6g 输入契约。"""


@dataclass(frozen=True)
class TerminalVariantInput:
    """一个正式终态的哈希绑定 parameter/bake 文件。"""

    variant: str
    manifest_path: Path
    manifest_sha256: str
    training_code_commit: str
    parameter_path: Path
    parameter_sha256: str
    baked_texture_path: Path
    baked_texture_sha256: str


@dataclass(frozen=True)
class TerminalDeploymentInputs:
    """两个终态共享的 Production Support 与训练 state 身份。"""

    production_support_path: Path
    production_support_sha256: str
    state_fingerprints: tuple[str, ...]
    variants: Mapping[str, TerminalVariantInput]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_relative_artifact(
    manifest_path: Path,
    relative_path: object,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TerminalDeploymentInputError("正式终态artifact相对路径无效")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise TerminalDeploymentInputError("正式终态artifact必须使用相对路径")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise TerminalDeploymentInputError("正式终态artifact路径逃逸bundle目录")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise TerminalDeploymentInputError(
            f"无法读取正式训练manifest: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TerminalDeploymentInputError("正式训练manifest必须为JSON object")
    return value


def _validate_common_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_variant: str,
    expected_support_sha256: str,
) -> tuple[str, ...]:
    if manifest.get("training_variant") != expected_variant:
        raise TerminalDeploymentInputError(
            f"正式终态variant错误: 期望{expected_variant}"
        )
    if manifest.get("task_id") != 0:
        raise TerminalDeploymentInputError("Gate 6g只接受Spatial task 0终态")
    if manifest.get("train_state_ids") != list(EXPECTED_STATE_IDS):
        raise TerminalDeploymentInputError("正式终态未绑定train states 0--9")
    fingerprints = manifest.get("train_state_fingerprints")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != 10
        or len(set(fingerprints)) != 10
        or not all(_is_sha256(value) for value in fingerprints)
    ):
        raise TerminalDeploymentInputError("训练state fingerprints无效或不唯一")
    if manifest.get("surface_epsilon") != EXPECTED_SURFACE_EPSILON:
        raise TerminalDeploymentInputError("正式终态Surface-Linf预算不匹配")
    input_sha256 = manifest.get("input_sha256")
    if not isinstance(input_sha256, dict) or input_sha256.get(
        "production_support"
    ) != expected_support_sha256:
        raise TerminalDeploymentInputError(
            "正式终态未绑定当前同一Production Support SHA-256"
        )
    return tuple(str(value) for value in fingerprints)


def _resolve_variant(
    manifest_path: Path,
    *,
    expected_variant: str,
    expected_support_sha256: str,
) -> tuple[TerminalVariantInput, tuple[str, ...]]:
    decision = evaluate_formal_source_training_bundle(manifest_path)
    if not decision.gate_pass:
        raise TerminalDeploymentInputError(
            f"{expected_variant}正式训练bundle未通过独立复核: "
            + "; ".join(decision.failures)
        )
    manifest = _load_manifest(manifest_path)
    fingerprints = _validate_common_manifest(
        manifest,
        expected_variant=expected_variant,
        expected_support_sha256=expected_support_sha256,
    )
    parameter_path = _resolve_relative_artifact(
        manifest_path,
        manifest.get("parameter_relative_path"),
    )
    baked_texture_path = _resolve_relative_artifact(
        manifest_path,
        manifest.get("baked_texture_relative_path"),
    )
    parameter_sha256 = _file_sha256(parameter_path)
    baked_texture_sha256 = _file_sha256(baked_texture_path)
    if parameter_sha256 != manifest.get("parameter_sha256"):
        raise TerminalDeploymentInputError(
            f"{expected_variant} terminal parameter SHA-256不匹配"
        )
    if baked_texture_sha256 != manifest.get("baked_texture_sha256"):
        raise TerminalDeploymentInputError(
            f"{expected_variant} terminal bake SHA-256不匹配"
        )
    training_code_commit = manifest.get("code_commit", "")
    return (
        TerminalVariantInput(
            variant=expected_variant,
            manifest_path=manifest_path,
            manifest_sha256=_file_sha256(manifest_path),
            training_code_commit=str(training_code_commit),
            parameter_path=parameter_path,
            parameter_sha256=parameter_sha256,
            baked_texture_path=baked_texture_path,
            baked_texture_sha256=baked_texture_sha256,
        ),
        fingerprints,
    )


def resolve_terminal_deployment_inputs(
    *,
    action_spectral_manifest_path: str | Path,
    action_only_manifest_path: str | Path,
    production_support_path: str | Path,
) -> TerminalDeploymentInputs:
    """复核两个正式训练bundle并返回唯一Gate 6g终态输入集合。"""

    support_path = Path(production_support_path).resolve()
    if not support_path.is_file():
        raise FileNotFoundError(support_path)
    support_sha256 = _file_sha256(support_path)
    manifest_paths = {
        "action_spectral": Path(action_spectral_manifest_path).resolve(),
        "action_only_control": Path(action_only_manifest_path).resolve(),
    }
    variants: dict[str, TerminalVariantInput] = {}
    fingerprint_sets: dict[str, tuple[str, ...]] = {}
    for variant in EXPECTED_VARIANTS:
        terminal_input, fingerprints = _resolve_variant(
            manifest_paths[variant],
            expected_variant=variant,
            expected_support_sha256=support_sha256,
        )
        variants[variant] = terminal_input
        fingerprint_sets[variant] = fingerprints
    if fingerprint_sets["action_spectral"] != fingerprint_sets[
        "action_only_control"
    ]:
        raise TerminalDeploymentInputError(
            "两个正式终态的train state fingerprints不一致"
        )
    return TerminalDeploymentInputs(
        production_support_path=support_path,
        production_support_sha256=support_sha256,
        state_fingerprints=fingerprint_sets["action_spectral"],
        variants=variants,
    )
