"""Gate 6g 终态参数到 canonical bake 的最小可测试数据流。

CUDA 命令负责构造真实 :class:`DifferentiableRenderer`；本模块只依赖其窄
Protocol。它强制通过 ``.pt`` 加载 Fixed-Support 紧凑参数，读取完整几何顶点域
Surface Delta，连续两次调用同一 canonical bake，并把两次结果与 formal bundle
绑定 PNG 的解码 RGB 交给冻结的零容差判定层。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from .terminal_deployment_response_audit import (
    TerminalBakePairingDecision,
    evaluate_terminal_bake_pairing,
    write_json_atomically,
)
from .terminal_deployment_response_inputs import (
    EXPECTED_VARIANTS,
    TerminalDeploymentInputs,
)


TERMINAL_REBAKE_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-rebake-preflight-v1"
)


class TerminalRebakeRenderer(Protocol):
    """终态 re-bake preflight 使用的最小 renderer interface。"""

    def get_texture_parameterization_name(self) -> str: ...

    def load_adversarial_texture(self, path: Path) -> Any: ...

    def get_geometry_surface_delta(self) -> torch.Tensor: ...

    def get_baked_adv_texture(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class TerminalRebakeEvidence:
    """一个终态的完整 Surface Delta 与三张解码 RGB。"""

    geometry_surface_delta: NDArray[np.float32]
    first_rebaked_rgb: NDArray[np.uint8]
    second_rebaked_rgb: NDArray[np.uint8]
    bound_png_rgb: NDArray[np.uint8]
    max_absolute_delta: float
    nonzero_percentage: float
    pairing: TerminalBakePairingDecision


@dataclass(frozen=True)
class TerminalRebakePreflightDecision:
    """独立CPU复算后的双终态preflight结论。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """返回稳定排序、无空白JSON的SHA-256。"""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("preflight artifact相对路径无效")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("preflight artifact必须使用相对路径")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("preflight artifact路径逃逸输出目录")
    return resolved


def _resolve_external_artifact_by_hash(
    root: Path,
    raw_path: object,
    expected_sha256: object,
) -> Path:
    """在服务器绝对路径失效后，于同步目录的有限祖先中按hash重定位。"""

    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not _is_lower_hex(expected_sha256, 64)
    ):
        raise ValueError("外部artifact path/SHA-256无效")
    original = Path(raw_path)
    artifact_name = original.name
    candidates: list[Path] = [original, root / artifact_name]
    search_roots = (root, *tuple(root.parents)[:2])
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        candidates.append(search_root / artifact_name)
        candidates.extend(search_root.glob(f"*/{artifact_name}"))
        candidates.extend(search_root.glob(f"*/*/{artifact_name}"))
        candidates.extend(search_root.glob(f"*/*/*/{artifact_name}"))
    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file() and _file_sha256(resolved) == expected_sha256:
            return resolved
    raise FileNotFoundError(
        f"找不到SHA-256匹配的同步artifact: {raw_path}"
    )


def _canonical_baked_rgb(renderer: TerminalRebakeRenderer) -> NDArray[np.uint8]:
    """复用正式 artifact store 的 nearest-uint8 canonical bake 语义。"""

    baked = renderer.get_baked_adv_texture().detach().to(torch.float32).cpu()
    if (
        baked.ndim != 4
        or baked.shape[0] != 1
        or baked.shape[-1] != 3
        or baked.numel() == 0
        or not bool(torch.isfinite(baked).all())
        or bool(torch.any(baked < 0.0))
        or bool(torch.any(baked > 1.0))
    ):
        raise ValueError("canonical bake必须为有限[0,1] float [1,H,W,3]")
    return np.rint(baked.squeeze(0).numpy() * 255.0).clip(
        0, 255
    ).astype(np.uint8)


def collect_terminal_rebake_evidence(
    renderer: TerminalRebakeRenderer,
    *,
    parameter_path: str | Path,
    bound_png_path: str | Path,
) -> TerminalRebakeEvidence:
    """执行 ``parameter -> Surface Delta -> re-bake×2 -> bound PNG``。"""

    if renderer.get_texture_parameterization_name() != "fixed_support":
        raise ValueError("Gate 6g re-bake只接受fixed_support renderer")
    resolved_parameter = Path(parameter_path)
    if resolved_parameter.suffix != ".pt" or not resolved_parameter.is_file():
        raise ValueError("Gate 6g终态parameter必须是存在的.pt文件")
    resolved_bound = Path(bound_png_path)
    if not resolved_bound.is_file():
        raise FileNotFoundError(resolved_bound)
    load_result = renderer.load_adversarial_texture(resolved_parameter)
    if getattr(load_result, "source_kind", None) != "parameter":
        raise RuntimeError("Gate 6g禁止从bake PNG反推Fixed-Support参数")

    surface_delta_tensor = (
        renderer.get_geometry_surface_delta().detach().to(torch.float32).cpu()
    )
    if (
        surface_delta_tensor.ndim != 2
        or surface_delta_tensor.shape[0] <= 0
        or surface_delta_tensor.shape[1] != 3
        or not bool(torch.isfinite(surface_delta_tensor).all())
    ):
        raise ValueError("Geometry Surface Delta必须为有限float [N_v,3]")
    surface_delta = surface_delta_tensor.numpy().copy()
    first_rgb = _canonical_baked_rgb(renderer)
    second_rgb = _canonical_baked_rgb(renderer)
    with Image.open(resolved_bound) as image:
        bound_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    pairing = evaluate_terminal_bake_pairing(
        first_rebaked_rgb=first_rgb,
        second_rebaked_rgb=second_rgb,
        bound_png_rgb=bound_rgb,
    )
    return TerminalRebakeEvidence(
        geometry_surface_delta=surface_delta,
        first_rebaked_rgb=first_rgb,
        second_rebaked_rgb=second_rgb,
        bound_png_rgb=bound_rgb,
        max_absolute_delta=float(load_result.max_absolute_delta),
        nonzero_percentage=float(load_result.nonzero_percentage),
        pairing=pairing,
    )


def write_terminal_rebake_preflight_bundle(
    evidence_by_variant: Mapping[str, TerminalRebakeEvidence],
    *,
    inputs: TerminalDeploymentInputs,
    code_commit: str,
    config_sha256: str,
    output_directory: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """保存双终态re-bake原始数组，最后原子发布成功manifest。"""

    if tuple(evidence_by_variant) != EXPECTED_VARIANTS:
        raise ValueError("preflight evidence必须按冻结顺序包含两个正式variant")
    if tuple(inputs.variants) != EXPECTED_VARIANTS:
        raise ValueError("preflight inputs必须按冻结顺序包含两个正式variant")
    if not _is_lower_hex(code_commit, 40) or not _is_lower_hex(
        config_sha256, 64
    ):
        raise ValueError("preflight code_commit/config_sha256无效")
    if _file_sha256(inputs.production_support_path) != (
        inputs.production_support_sha256
    ):
        raise ValueError("preflight Production Support SHA-256漂移")
    output_dir = Path(output_directory)
    manifest_path = output_dir / "terminal_rebake_preflight_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"preflight输出目录必须为空: {output_dir}")
    arrays_dir = output_dir / "arrays"
    images_dir = output_dir / "images"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    pairing_manifest: dict[str, dict[str, Any]] = {}
    for variant in EXPECTED_VARIANTS:
        evidence = evidence_by_variant[variant]
        terminal_input = inputs.variants[variant]
        if _file_sha256(terminal_input.parameter_path) != (
            terminal_input.parameter_sha256
        ):
            raise ValueError(f"{variant} terminal parameter SHA-256漂移")
        if _file_sha256(terminal_input.baked_texture_path) != (
            terminal_input.baked_texture_sha256
        ):
            raise ValueError(f"{variant} bound PNG SHA-256漂移")
        arrays_path = arrays_dir / f"{variant}_terminal_rebake.npz"
        np.savez_compressed(
            arrays_path,
            geometry_surface_delta=evidence.geometry_surface_delta,
            first_rebaked_rgb=evidence.first_rebaked_rgb,
            second_rebaked_rgb=evidence.second_rebaked_rgb,
            bound_png_rgb=evidence.bound_png_rgb,
        )
        image_hashes: dict[str, str] = {}
        for role, pixels in (
            ("first_rebake", evidence.first_rebaked_rgb),
            ("second_rebake", evidence.second_rebaked_rgb),
            ("bound_decoded", evidence.bound_png_rgb),
        ):
            image_path = images_dir / f"{variant}_{role}.png"
            Image.fromarray(pixels, mode="RGB").save(image_path)
            image_hashes[str(image_path.relative_to(output_dir))] = (
                _file_sha256(image_path)
            )
        pairing_manifest[variant] = {
            **asdict(evidence.pairing),
            "formal_manifest_path": str(terminal_input.manifest_path),
            "formal_manifest_sha256": terminal_input.manifest_sha256,
            "training_code_commit": terminal_input.training_code_commit,
            "parameter_path": str(terminal_input.parameter_path),
            "parameter_sha256": terminal_input.parameter_sha256,
            "bound_png_path": str(terminal_input.baked_texture_path),
            "bound_png_sha256": terminal_input.baked_texture_sha256,
            "geometry_surface_delta_shape": list(
                evidence.geometry_surface_delta.shape
            ),
            "max_absolute_delta": evidence.max_absolute_delta,
            "nonzero_percentage": evidence.nonzero_percentage,
            "arrays_relative_path": str(arrays_path.relative_to(output_dir)),
            "arrays_sha256": _file_sha256(arrays_path),
            "image_sha256": image_hashes,
        }
    failed_variants = [
        variant
        for variant, evidence in evidence_by_variant.items()
        if not evidence.pairing.gate_pass
    ]
    if failed_variants:
        raise RuntimeError(
            "终态parameter/re-bake/bound PNG严格配对失败: "
            + ", ".join(failed_variants)
        )
    manifest = {
        "schema_version": TERMINAL_REBAKE_PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "expected_variants": list(EXPECTED_VARIANTS),
        "production_support_path": str(inputs.production_support_path),
        "production_support_sha256": inputs.production_support_sha256,
        "state_fingerprints": list(inputs.state_fingerprints),
        "terminal_pairing": pairing_manifest,
        "provenance": {} if provenance is None else dict(provenance),
    }
    candidate_manifest_path = output_dir / "terminal_rebake_preflight_candidate.json"
    write_json_atomically(manifest, output_path=candidate_manifest_path)
    decision = evaluate_terminal_rebake_preflight_bundle(
        candidate_manifest_path
    )
    if not decision.gate_pass:
        candidate_manifest_path.unlink()
        raise RuntimeError(
            "preflight候选manifest独立复核失败: "
            + "; ".join(decision.failures)
        )
    os.replace(candidate_manifest_path, manifest_path)
    directory_descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return manifest_path


def evaluate_terminal_rebake_preflight_bundle(
    manifest_path: str | Path,
) -> TerminalRebakePreflightDecision:
    """独立加载双终态NPZ并复算全部零容差parameter/bake配对。"""

    failures: list[str] = []
    try:
        resolved_manifest = Path(manifest_path).resolve()
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("preflight manifest必须为JSON object")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return TerminalRebakePreflightDecision(
            False,
            (f"preflight manifest无法加载: {error}",),
        )
    root = resolved_manifest.parent
    if manifest.get("schema_version") != TERMINAL_REBAKE_PREFLIGHT_SCHEMA_VERSION:
        failures.append("preflight schema_version不匹配")
    if manifest.get("status") != "complete":
        failures.append("preflight成功manifest status必须为complete")
    if not _is_lower_hex(manifest.get("code_commit"), 40):
        failures.append("preflight code_commit无效")
    if not _is_lower_hex(manifest.get("config_sha256"), 64):
        failures.append("preflight config_sha256无效")
    if manifest.get("expected_variants") != list(EXPECTED_VARIANTS):
        failures.append("preflight variant inventory不匹配")
    fingerprints = manifest.get("state_fingerprints")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != 10
        or len(set(fingerprints)) != 10
        or not all(_is_lower_hex(value, 64) for value in fingerprints)
    ):
        failures.append("preflight state fingerprints无效或不唯一")
    support_sha256 = manifest.get("production_support_sha256")
    try:
        _resolve_external_artifact_by_hash(
            root,
            manifest.get("production_support_path"),
            support_sha256,
        )
    except (OSError, ValueError):
        failures.append("preflight Production Support路径或SHA-256无效")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("preflight provenance缺失")
    else:
        configuration = provenance.get("configuration")
        if isinstance(configuration, dict) and canonical_json_sha256(
            configuration
        ) != manifest.get("config_sha256"):
            failures.append("preflight configuration不能复算config SHA-256")

    raw_pairing = manifest.get("terminal_pairing")
    if not isinstance(raw_pairing, dict) or set(raw_pairing) != set(
        EXPECTED_VARIANTS
    ):
        failures.append("preflight terminal_pairing inventory不匹配")
        raw_pairing = {}
    for variant in EXPECTED_VARIANTS:
        raw_entry = raw_pairing.get(variant)
        if not isinstance(raw_entry, dict):
            failures.append(f"{variant} pairing entry缺失")
            continue
        for role, path_name, hash_name in (
            ("formal manifest", "formal_manifest_path", "formal_manifest_sha256"),
            ("terminal parameter", "parameter_path", "parameter_sha256"),
            ("bound PNG", "bound_png_path", "bound_png_sha256"),
        ):
            expected_hash = raw_entry.get(hash_name)
            try:
                _resolve_external_artifact_by_hash(
                    root,
                    raw_entry.get(path_name),
                    expected_hash,
                )
            except (OSError, ValueError):
                failures.append(f"{variant} {role}路径或SHA-256无效")
        try:
            arrays_path = _resolve_relative(
                root,
                raw_entry.get("arrays_relative_path"),
            )
            if _file_sha256(arrays_path) != raw_entry.get("arrays_sha256"):
                failures.append(f"{variant} arrays SHA-256不匹配")
                continue
            with np.load(arrays_path, allow_pickle=False) as archive:
                geometry_delta = archive["geometry_surface_delta"].copy()
                first_rgb = archive["first_rebaked_rgb"].copy()
                second_rgb = archive["second_rebaked_rgb"].copy()
                bound_rgb = archive["bound_png_rgb"].copy()
            if (
                geometry_delta.dtype != np.float32
                or geometry_delta.ndim != 2
                or geometry_delta.shape[0] <= 0
                or geometry_delta.shape[1] != 3
                or not bool(np.isfinite(geometry_delta).all())
            ):
                failures.append(f"{variant} Geometry Surface Delta无效")
            recomputed = evaluate_terminal_bake_pairing(
                first_rebaked_rgb=first_rgb,
                second_rebaked_rgb=second_rgb,
                bound_png_rgb=bound_rgb,
            )
            for name, value in asdict(recomputed).items():
                if raw_entry.get(name) != value:
                    failures.append(f"{variant} pairing字段{name}不能独立复算")
            if not recomputed.gate_pass:
                failures.append(f"{variant} parameter/re-bake严格配对未通过")
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{variant} preflight arrays无法复算: {error}")
        image_hashes = raw_entry.get("image_sha256")
        if not isinstance(image_hashes, dict) or len(image_hashes) != 3:
            failures.append(f"{variant} preflight PNG inventory无效")
        else:
            for relative_path, expected_hash in image_hashes.items():
                try:
                    image_path = _resolve_relative(root, relative_path)
                    if (
                        not _is_lower_hex(expected_hash, 64)
                        or _file_sha256(image_path) != expected_hash
                    ):
                        failures.append(f"{variant} preflight PNG SHA-256不匹配")
                except (OSError, ValueError) as error:
                    failures.append(f"{variant} preflight PNG无效: {error}")
    return TerminalRebakePreflightDecision(not failures, tuple(failures))
