"""运行 Gate 6g 双终态 CUDA canonical re-bake preflight。

本命令不加载 OpenVLA 或 LIBERO state，也不修改 MuJoCo 资产。它先独立复核两个
5000轮 formal training bundle 与共享 Production Support，再对每个终态执行：

``.pt -> Fixed-Support -> Geometry Surface Delta -> canonical bake x2``

两次 bake 的解码 uint8 RGB 必须逐像素相同，并且都与 formal bundle 已绑定 PNG
逐像素相同。失败时保留 NPZ/PNG 诊断证据，但绝不生成成功 manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

# nvdiffrast context初始化前固定headless backend；本命令本身不启动MuJoCo。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import PIL
import torch


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
OPENVLA_ROOT = THIS_FILE.parents[4]
ROBOT_EXPERIMENT_DIR = THIS_FILE.parents[2]
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
for import_path in (
    REPOSITORY_ROOT,
    OPENVLA_ROOT,
    ROBOT_EXPERIMENT_DIR,
    LIBERO_EXPERIMENT_DIR,
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.assets import (  # noqa: E402
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.terminal_deployment_response_audit import (  # noqa: E402
    write_json_atomically,
)
from openvla_attack.terminal_deployment_response_inputs import (  # noqa: E402
    EXPECTED_SURFACE_EPSILON,
    EXPECTED_VARIANTS,
    TerminalDeploymentInputs,
    resolve_terminal_deployment_inputs,
)
from openvla_attack.terminal_rebake_preflight import (  # noqa: E402
    TERMINAL_REBAKE_PREFLIGHT_SCHEMA_VERSION,
    TerminalRebakeEvidence,
    canonical_json_sha256,
    collect_terminal_rebake_evidence,
    write_terminal_rebake_preflight_bundle,
)


@dataclass(frozen=True)
class TerminalRebakePreflightConfig:
    """双终态preflight CLI配置。"""

    action_spectral_manifest_path: str
    action_only_manifest_path: str
    production_support_path: str
    output_dir: str
    code_commit: str
    object_name: str = "akita_black_bowl"


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> TerminalRebakePreflightConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action_spectral_manifest_path", required=True)
    parser.add_argument("--action_only_manifest_path", required=True)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--object_name", default="akita_black_bowl")
    return TerminalRebakePreflightConfig(**vars(parser.parse_args(argv)))


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkout(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit必须是40位小写Git SHA")
    actual_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != code_commit:
        raise RuntimeError(
            f"--code_commit与当前HEAD不一致: {code_commit} != {actual_commit}"
        )
    if tracked_status:
        raise RuntimeError("Gate 6g preflight拒绝含tracked修改的checkout")


def _failure_record(
    *,
    output_dir: Path,
    stage: str,
    error: BaseException,
    input_sha256: dict[str, str],
) -> None:
    """best-effort保存preflight失败上下文，永不冒充成功manifest。"""

    try:
        write_json_atomically(
            {
                "schema_version": TERMINAL_REBAKE_PREFLIGHT_SCHEMA_VERSION,
                "status": "audit_invalid",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "input_sha256": input_sha256,
            },
            output_path=output_dir / "audit_failed.json",
        )
    except BaseException as record_error:
        print(
            "[GATE-6G-PREFLIGHT] 无法保存best-effort失败记录: "
            f"{record_error}",
            file=sys.stderr,
        )


def run_terminal_rebake_preflight(
    cfg: TerminalRebakePreflightConfig,
) -> Path:
    """运行双终态CUDA re-bake并返回原子发布的成功manifest。"""

    stage = "checkout_validation"
    input_hashes: dict[str, str] = {}
    output_dir = Path(cfg.output_dir)
    try:
        _verify_checkout(cfg.code_commit)
        if not torch.cuda.is_available():
            raise RuntimeError("Gate 6g re-bake preflight需要真实CUDA device")
        if cfg.object_name != "akita_black_bowl":
            raise ValueError("Gate 6g当前只冻结akita_black_bowl")
        if cfg.object_name not in OBJECT_ASSETS:
            raise ValueError(f"未知object_name: {cfg.object_name}")

        stage = "terminal_input_resolution"
        inputs: TerminalDeploymentInputs = resolve_terminal_deployment_inputs(
            action_spectral_manifest_path=(
                cfg.action_spectral_manifest_path
            ),
            action_only_manifest_path=cfg.action_only_manifest_path,
            production_support_path=cfg.production_support_path,
        )
        input_hashes = {
            "production_support": inputs.production_support_sha256,
            **{
                f"{variant}_manifest": terminal_input.manifest_sha256
                for variant, terminal_input in inputs.variants.items()
            },
            **{
                f"{variant}_parameter": terminal_input.parameter_sha256
                for variant, terminal_input in inputs.variants.items()
            },
            **{
                f"{variant}_bake": terminal_input.baked_texture_sha256
                for variant, terminal_input in inputs.variants.items()
            },
        }
        asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
        xml_path = Path(asset["xml"]).resolve()
        mesh_path = Path(asset["mesh"]).resolve()
        texture_path = Path(asset["texture"]).resolve()
        for path in (xml_path, mesh_path, texture_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        asset_sha256 = {
            "xml": _file_sha256(xml_path),
            "mesh": _file_sha256(mesh_path),
            "clean_texture": _file_sha256(texture_path),
        }

        stage = "renderer_construction"
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device="cuda",
            scale_xyz=parse_mesh_scale(xml_path),
            epsilon=EXPECTED_SURFACE_EPSILON,
            texture_parameterization="fixed_support",
            fixed_support_path=inputs.production_support_path,
        ).to("cuda")
        evidence_by_variant: dict[str, TerminalRebakeEvidence] = {}
        for variant in EXPECTED_VARIANTS:
            stage = f"{variant}_rebake"
            terminal_input = inputs.variants[variant]
            evidence_by_variant[variant] = collect_terminal_rebake_evidence(
                renderer,
                parameter_path=terminal_input.parameter_path,
                bound_png_path=terminal_input.baked_texture_path,
            )

        stage = "evidence_persistence"
        config_payload = asdict(cfg)
        manifest_path = write_terminal_rebake_preflight_bundle(
            evidence_by_variant,
            inputs=inputs,
            code_commit=cfg.code_commit,
            config_sha256=canonical_json_sha256(config_payload),
            output_directory=output_dir,
            provenance={
                "configuration": config_payload,
                "object_asset_paths": {
                    "xml": str(xml_path),
                    "mesh": str(mesh_path),
                    "clean_texture": str(texture_path),
                },
                "object_asset_sha256": asset_sha256,
                "framework_versions": {
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "torch": torch.__version__,
                },
                "cuda_device_name": torch.cuda.get_device_name(
                    torch.cuda.current_device()
                ),
                "command": list(sys.argv),
            },
        )
    except BaseException as error:
        _failure_record(
            output_dir=output_dir,
            stage=stage,
            error=error,
            input_sha256=input_hashes,
        )
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": _file_sha256(manifest_path),
            },
            sort_keys=True,
        )
    )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_terminal_rebake_preflight(_parse_args(argv))


if __name__ == "__main__":
    main()
