"""Gate 6g 终态 parameter/re-bake 配对的纯行为测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_rebake_preflight import (  # noqa: E402
    collect_terminal_rebake_evidence,
    evaluate_terminal_rebake_preflight_bundle,
    write_terminal_rebake_preflight_bundle,
)
from openvla_attack.terminal_deployment_response_inputs import (  # noqa: E402
    TerminalDeploymentInputs,
    TerminalVariantInput,
)


class _FakeFixedSupportRenderer:
    def __init__(self, baked_rgb: np.ndarray) -> None:
        self._baked = torch.from_numpy(baked_rgb.astype(np.float32) / 255.0)
        self.loaded_paths: list[Path] = []
        self.bake_calls = 0

    def get_texture_parameterization_name(self) -> str:
        return "fixed_support"

    def load_adversarial_texture(self, path: Path) -> SimpleNamespace:
        self.loaded_paths.append(Path(path))
        return SimpleNamespace(
            source_kind="parameter",
            max_absolute_delta=0.25,
            nonzero_percentage=12.5,
        )

    def get_geometry_surface_delta(self) -> torch.Tensor:
        return torch.asarray([[0.25, 0.0, -0.25]], dtype=torch.float32)

    def get_baked_adv_texture(self) -> torch.Tensor:
        self.bake_calls += 1
        return self._baked.unsqueeze(0)


def test_collect_preflight_uses_parameter_and_two_canonical_bakes(
    tmp_path: Path,
) -> None:
    baked_rgb = np.asarray(
        [[[0, 127, 255], [4, 5, 6]], [[7, 8, 9], [250, 251, 252]]],
        dtype=np.uint8,
    )
    bound_path = tmp_path / "bound.png"
    Image.fromarray(baked_rgb, mode="RGB").save(bound_path)
    parameter_path = tmp_path / "terminal.pt"
    parameter_path.write_bytes(b"fake parameter")
    renderer = _FakeFixedSupportRenderer(baked_rgb)

    evidence = collect_terminal_rebake_evidence(
        renderer,
        parameter_path=parameter_path,
        bound_png_path=bound_path,
    )

    assert renderer.loaded_paths == [parameter_path]
    assert renderer.bake_calls == 2
    assert evidence.pairing.gate_pass is True
    assert np.array_equal(evidence.first_rebaked_rgb, baked_rgb)
    assert np.array_equal(evidence.second_rebaked_rgb, baked_rgb)
    assert evidence.geometry_surface_delta.shape == (1, 3)


def test_preflight_success_manifest_is_complete_and_hash_bound(
    tmp_path: Path,
) -> None:
    baked_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    support_path = tmp_path / "support.npz"
    support_path.write_bytes(b"support")
    variants: dict[str, TerminalVariantInput] = {}
    evidence = {}
    for variant in ("action_spectral", "action_only_control"):
        variant_root = tmp_path / variant
        variant_root.mkdir()
        parameter_path = variant_root / "terminal.pt"
        parameter_path.write_bytes(b"parameter")
        bake_path = variant_root / "terminal.png"
        Image.fromarray(baked_rgb, mode="RGB").save(bake_path)
        manifest_path = variant_root / "formal.json"
        manifest_path.write_text("{}", encoding="utf-8")
        variants[variant] = TerminalVariantInput(
            variant=variant,
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            training_code_commit="b" * 40,
            parameter_path=parameter_path,
            parameter_sha256=hashlib.sha256(
                parameter_path.read_bytes()
            ).hexdigest(),
            baked_texture_path=bake_path,
            baked_texture_sha256=hashlib.sha256(
                bake_path.read_bytes()
            ).hexdigest(),
        )
        evidence[variant] = collect_terminal_rebake_evidence(
            _FakeFixedSupportRenderer(baked_rgb),
            parameter_path=parameter_path,
            bound_png_path=bake_path,
        )
    inputs = TerminalDeploymentInputs(
        production_support_path=support_path,
        production_support_sha256=hashlib.sha256(
            support_path.read_bytes()
        ).hexdigest(),
        state_fingerprints=tuple(
            f"{state_id + 1:064x}" for state_id in range(10)
        ),
        variants=variants,
    )

    manifest_path = write_terminal_rebake_preflight_bundle(
        evidence,
        inputs=inputs,
        code_commit="f" * 40,
        config_sha256="1" * 64,
        output_directory=tmp_path / "preflight",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["terminal_pairing"]["action_spectral"]["gate_pass"] is True
    assert set(manifest["terminal_pairing"]) == {
        "action_spectral",
        "action_only_control",
    }
    assert not manifest_path.with_name(manifest_path.name + ".tmp").exists()
    assert evaluate_terminal_rebake_preflight_bundle(manifest_path).gate_pass

    arrays_path = (
        manifest_path.parent
        / manifest["terminal_pairing"]["action_spectral"][
            "arrays_relative_path"
        ]
    )
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")
    tampered = evaluate_terminal_rebake_preflight_bundle(manifest_path)
    assert tampered.gate_pass is False
    assert any("arrays SHA-256" in failure for failure in tampered.failures)


def test_failed_pairing_keeps_arrays_but_never_writes_success_manifest(
    tmp_path: Path,
) -> None:
    clean = np.zeros((2, 2, 3), dtype=np.uint8)
    changed = clean.copy()
    changed[0, 0, 0] = 1
    support_path = tmp_path / "support.npz"
    support_path.write_bytes(b"support")
    variants: dict[str, TerminalVariantInput] = {}
    evidence = {}
    for variant in ("action_spectral", "action_only_control"):
        variant_root = tmp_path / variant
        variant_root.mkdir()
        parameter_path = variant_root / "terminal.pt"
        parameter_path.write_bytes(b"parameter")
        bake_path = variant_root / "terminal.png"
        Image.fromarray(
            changed if variant == "action_spectral" else clean,
            mode="RGB",
        ).save(bake_path)
        manifest_path = variant_root / "formal.json"
        manifest_path.write_text("{}", encoding="utf-8")
        variants[variant] = TerminalVariantInput(
            variant=variant,
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            training_code_commit="b" * 40,
            parameter_path=parameter_path,
            parameter_sha256=hashlib.sha256(
                parameter_path.read_bytes()
            ).hexdigest(),
            baked_texture_path=bake_path,
            baked_texture_sha256=hashlib.sha256(
                bake_path.read_bytes()
            ).hexdigest(),
        )
        evidence[variant] = collect_terminal_rebake_evidence(
            _FakeFixedSupportRenderer(clean),
            parameter_path=parameter_path,
            bound_png_path=bake_path,
        )
    inputs = TerminalDeploymentInputs(
        production_support_path=support_path,
        production_support_sha256=hashlib.sha256(
            support_path.read_bytes()
        ).hexdigest(),
        state_fingerprints=tuple(
            f"{state_id + 1:064x}" for state_id in range(10)
        ),
        variants=variants,
    )
    output_dir = tmp_path / "failed_preflight"

    with pytest.raises(RuntimeError, match="严格配对失败"):
        write_terminal_rebake_preflight_bundle(
            evidence,
            inputs=inputs,
            code_commit="f" * 40,
            config_sha256="1" * 64,
            output_directory=output_dir,
        )

    assert (output_dir / "arrays/action_spectral_terminal_rebake.npz").is_file()
    assert not (
        output_dir / "terminal_rebake_preflight_manifest.json"
    ).exists()
