"""真实 Visibility/Alignment runner 的无环境 helper 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
sys.path.insert(0, str(OPENVLA_ROOT))

from openvla_attack.diagnose_visibility_alignment import (  # noqa: E402
    _save_overlay,
    _sha256_array,
    _validate_code_commit,
)


def test_code_commit_requires_full_lowercase_sha() -> None:
    _validate_code_commit("a" * 40)

    with pytest.raises(ValueError, match="40位"):
        _validate_code_commit("abc")
    with pytest.raises(ValueError, match="小写"):
        _validate_code_commit("A" * 40)


def test_array_hash_binds_dtype_shape_and_values() -> None:
    base = np.asarray([[1, 2]], dtype=np.int32)

    assert _sha256_array(base) == _sha256_array(base.copy())
    assert _sha256_array(base) != _sha256_array(base.astype(np.int64))
    assert _sha256_array(base) != _sha256_array(base.reshape(2, 1))


def test_soft_overlay_colors_mujoco_renderer_and_overlap(
    tmp_path: Path,
) -> None:
    mujoco = torch.tensor([[1.0, 0.0, 1.0]])
    renderer = torch.tensor([[0.0, 1.0, 1.0]])
    output_path = tmp_path / "overlay.png"

    digest = _save_overlay(output_path, mujoco, renderer)

    pixels = np.asarray(Image.open(output_path))
    np.testing.assert_array_equal(
        pixels,
        np.asarray(
            [[[0, 255, 0], [255, 0, 0], [255, 255, 0]]],
            dtype=np.uint8,
        ),
    )
    assert len(digest) == 64
