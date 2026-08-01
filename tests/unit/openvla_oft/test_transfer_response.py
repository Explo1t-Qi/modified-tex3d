"""OFT 成对响应诊断的无 GPU 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


OFT_LIBERO_DIR = (
    Path(__file__).resolve().parents[3]
    / "openvla-oft/experiments/robot/libero"
)
sys.path.insert(0, str(OFT_LIBERO_DIR))

from transfer_response import (  # noqa: E402
    compute_action_diagnostics,
    compute_distance,
    extract_primary_siglip_features,
)


class RecordingFeaturizer:
    """记录输入通道并返回合法 patch feature 的测试替身。"""

    def __init__(self) -> None:
        self.last_input: torch.Tensor | None = None

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.last_input = pixel_values
        return pixel_values.mean(dim=(2, 3)).unsqueeze(1)


def test_extracts_siglip_channels_using_checkpoint_order() -> None:
    dino = RecordingFeaturizer()
    siglip = RecordingFeaturizer()
    model = SimpleNamespace(
        config=SimpleNamespace(
            timm_model_ids=("dinov2", "vit_siglip_224"),
        ),
        vision_backbone=SimpleNamespace(
            featurizer=dino,
            fused_featurizer=siglip,
        ),
    )
    pixels = torch.zeros((1, 6, 2, 2), dtype=torch.float32)
    pixels[:, 3:6] = 7.0

    features = extract_primary_siglip_features(model, pixels)

    assert dino.last_input is None
    assert siglip.last_input is not None
    assert torch.equal(siglip.last_input, pixels[:, 3:6])
    assert tuple(features.shape) == (1, 1, 3)


def test_computes_distance_and_action_gripper_flips() -> None:
    clean = np.asarray([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    adversarial = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)

    distance = compute_distance(clean, adversarial)
    actions = compute_action_diagnostics(clean, adversarial)

    assert distance.mean_absolute == pytest.approx(1.25)
    assert distance.max_absolute == pytest.approx(2.0)
    assert actions.gripper_sign_flip_count == 2
    assert actions.gripper_step_count == 2
    assert actions.first_gripper_sign_flip is True


def test_rejects_mismatched_pair_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_distance(np.zeros((2,)), np.zeros((3,)))
