"""OpenVLA 模型输入对齐逻辑的回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.openvla_utils import (
    LLAMA_EMPTY_TOKEN_ID,
    ensure_trailing_empty_token,
)


def test_missing_empty_token_extends_ids_and_attention_mask_together() -> None:
    """补齐 prompt 尾 token 时，序列与 attention mask 必须保持等长。"""
    pixel_values = torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16)
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": pixel_values,
    }

    returned_inputs = ensure_trailing_empty_token(inputs)

    assert returned_inputs is inputs
    assert inputs["input_ids"].tolist() == [[1, 2, LLAMA_EMPTY_TOKEN_ID]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape == (1, 3)
    assert inputs["pixel_values"] is pixel_values
