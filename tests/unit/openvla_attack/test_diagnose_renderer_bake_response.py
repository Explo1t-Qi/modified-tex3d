"""Gate 2R 真实 runner 的无 LIBERO helper 测试。

该文件导入真实 runner，因此需要服务器已有的 nvdiffrast/OpenVLA 依赖；测试
本身不创建 CUDA context、模型或 LIBERO 环境。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
sys.path.insert(0, str(OPENVLA_ROOT))

from openvla_attack.diagnose_renderer_bake_response import (  # noqa: E402
    _combined_evidence_sha256,
    _save_alpha_visualization,
    _save_delta_visualization,
    _save_weighted_scatter,
    _validate_code_commit,
)
from openvla_attack.renderer_bake_response_audit import (  # noqa: E402
    RendererBakeResponseEvidence,
)


def _evidence() -> RendererBakeResponseEvidence:
    alpha = np.asarray([[1.0, 0.5]], dtype=np.float32)
    d_sur = np.asarray(
        [[[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]],
        dtype=np.float32,
    )
    margins = np.asarray([1.0, -0.5], dtype=np.float64)
    return RendererBakeResponseEvidence(
        state_id=0,
        probe_channel="r",
        alpha=alpha,
        d_sur=d_sur,
        d_bake=d_sur.copy(),
        clean_action_margins=margins,
        surrogate_action_margins=margins.copy(),
        bake_action_margins=margins.copy(),
    )


def test_code_commit_requires_full_lowercase_sha() -> None:
    _validate_code_commit("a" * 40)

    with pytest.raises(ValueError, match="40位"):
        _validate_code_commit("abc")
    with pytest.raises(ValueError, match="小写"):
        _validate_code_commit("A" * 40)


def test_evidence_hash_binds_probe_and_unquantized_arrays() -> None:
    evidence = _evidence()
    changed = replace(
        evidence,
        d_bake=evidence.d_bake + np.float32(1e-7),
    )

    assert _combined_evidence_sha256(evidence) == (
        _combined_evidence_sha256(evidence)
    )
    assert _combined_evidence_sha256(evidence) != (
        _combined_evidence_sha256(changed)
    )


def test_visualization_helpers_write_readable_non_authoritative_pngs(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    alpha_path = tmp_path / "case_alpha.png"
    delta_path = tmp_path / "case_d_sur.png"
    scatter_path = tmp_path / "case_weighted_scatter.png"

    fingerprints = (
        _save_alpha_visualization(alpha_path, evidence.alpha),
        _save_delta_visualization(delta_path, evidence.d_sur),
        _save_weighted_scatter(
            scatter_path,
            alpha=evidence.alpha,
            d_sur=evidence.d_sur,
            d_bake=evidence.d_bake,
        ),
    )

    assert all(len(value) == 64 for value in fingerprints)
    assert Image.open(alpha_path).mode == "L"
    assert Image.open(delta_path).mode == "RGB"
    assert Image.open(scatter_path).size == (512, 512)
    scatter = np.asarray(Image.open(scatter_path).convert("RGB"))
    strongly_colored = (
        scatter.max(axis=2).astype(np.int16)
        - scatter.min(axis=2).astype(np.int16)
    ) > 40
    assert int(np.count_nonzero(strongly_colored)) >= 100
