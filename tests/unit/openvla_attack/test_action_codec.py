"""OpenVLA action codec 的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.action_codec import (
    ActionNormalizationStats,
    FloatingArray,
    decode_action_from_generated_ids,
)


class FakeActionModel:
    """只实现 action codec 所需 interface 的测试模型。"""

    vocab_size: int = 32_000
    bin_centers: FloatingArray = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    @staticmethod
    def get_action_dim(unnorm_key: Optional[str]) -> int:
        del unnorm_key
        return 2

    @staticmethod
    def get_action_stats(unnorm_key: Optional[str]) -> ActionNormalizationStats:
        del unnorm_key
        return {
            "q01": np.array([10.0, 100.0], dtype=np.float32),
            "q99": np.array([20.0, 200.0], dtype=np.float32),
            "mask": np.array([True, False]),
        }


def test_generated_tokens_are_decoded_and_selectively_unnormalized() -> None:
    generated_ids = torch.tensor([[7, 31_999, 31_998]], dtype=torch.long)

    action = decode_action_from_generated_ids(
        FakeActionModel(), generated_ids, unnorm_key="fake"
    )

    np.testing.assert_allclose(action, np.array([10.0, 0.0], dtype=np.float32))


def test_all_action_dimensions_are_unnormalized_when_mask_is_absent() -> None:
    class FakeActionModelWithoutMask(FakeActionModel):
        @staticmethod
        def get_action_stats(unnorm_key: Optional[str]) -> ActionNormalizationStats:
            del unnorm_key
            return {
                "q01": np.array([10.0, 100.0], dtype=np.float32),
                "q99": np.array([20.0, 200.0], dtype=np.float32),
            }

    generated_ids = torch.tensor([[7, 31_999, 31_998]], dtype=torch.long)

    action = decode_action_from_generated_ids(
        FakeActionModelWithoutMask(), generated_ids, unnorm_key="fake"
    )

    np.testing.assert_allclose(action, np.array([10.0, 150.0], dtype=np.float32))
