"""Gate 6g 两个正式训练终态的输入配对测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

import openvla_attack.terminal_deployment_response_inputs as inputs_module  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_formal_bundle(
    root: Path,
    *,
    variant: str,
    support_sha256: str,
) -> Path:
    root.mkdir(parents=True)
    parameter_path = root / "terminal_parameter.pt"
    bake_path = root / "terminal_bake.png"
    parameter_path.write_bytes(f"parameter:{variant}".encode())
    bake_path.write_bytes(f"bake:{variant}".encode())
    manifest = {
        "training_variant": variant,
        "task_id": 0,
        "train_state_ids": list(range(10)),
        "train_state_fingerprints": [
            f"{state_id + 1:064x}" for state_id in range(10)
        ],
        "surface_epsilon": 128.0 / 255.0,
        "input_sha256": {"production_support": support_sha256},
        "parameter_relative_path": parameter_path.name,
        "parameter_sha256": _sha256(parameter_path),
        "baked_texture_relative_path": bake_path.name,
        "baked_texture_sha256": _sha256(bake_path),
    }
    manifest_path = root / "formal_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_resolve_terminal_inputs_binds_two_variants_to_one_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_path = tmp_path / "production_support.npz"
    support_path.write_bytes(b"frozen support")
    support_sha256 = _sha256(support_path)
    spectral_manifest = _write_formal_bundle(
        tmp_path / "spectral",
        variant="action_spectral",
        support_sha256=support_sha256,
    )
    control_manifest = _write_formal_bundle(
        tmp_path / "control",
        variant="action_only_control",
        support_sha256=support_sha256,
    )
    monkeypatch.setattr(
        inputs_module,
        "evaluate_formal_source_training_bundle",
        lambda _: SimpleNamespace(gate_pass=True, failures=()),
    )

    resolved = inputs_module.resolve_terminal_deployment_inputs(
        action_spectral_manifest_path=spectral_manifest,
        action_only_manifest_path=control_manifest,
        production_support_path=support_path,
    )

    assert resolved.production_support_sha256 == support_sha256
    assert tuple(resolved.variants) == (
        "action_spectral",
        "action_only_control",
    )
    assert resolved.variants["action_spectral"].parameter_path.name == (
        "terminal_parameter.pt"
    )
    assert resolved.state_fingerprints == tuple(
        f"{state_id + 1:064x}" for state_id in range(10)
    )


def test_resolve_terminal_inputs_rejects_support_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_path = tmp_path / "production_support.npz"
    support_path.write_bytes(b"actual support")
    spectral_manifest = _write_formal_bundle(
        tmp_path / "spectral",
        variant="action_spectral",
        support_sha256="a" * 64,
    )
    control_manifest = _write_formal_bundle(
        tmp_path / "control",
        variant="action_only_control",
        support_sha256="a" * 64,
    )
    monkeypatch.setattr(
        inputs_module,
        "evaluate_formal_source_training_bundle",
        lambda _: SimpleNamespace(gate_pass=True, failures=()),
    )

    with pytest.raises(
        inputs_module.TerminalDeploymentInputError,
        match="Production Support",
    ):
        inputs_module.resolve_terminal_deployment_inputs(
            action_spectral_manifest_path=spectral_manifest,
            action_only_manifest_path=control_manifest,
            production_support_path=support_path,
        )
