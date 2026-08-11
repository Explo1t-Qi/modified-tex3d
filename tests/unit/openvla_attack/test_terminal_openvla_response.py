"""Gate 6g OpenVLA 静态动作响应采集的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_openvla_response import (  # noqa: E402
    bfloat16_tensor_to_uint16_bits,
    capture_openvla_action_response,
)


class _FakeOpenVLA:
    vocab_size = 32_000
    bin_centers = np.linspace(-1.0, 1.0, 256, dtype=np.float64)

    def __init__(self) -> None:
        self.teacher_input_ids: torch.Tensor | None = None

    def get_action_dim(self, unnorm_key: str | None) -> int:
        del unnorm_key
        return 2

    def get_action_stats(self, unnorm_key: str | None) -> dict[str, np.ndarray]:
        del unnorm_key
        return {
            "q01": np.asarray([-2.0, -1.0], dtype=np.float64),
            "q99": np.asarray([2.0, 3.0], dtype=np.float64),
            "mask": np.asarray([True, False], dtype=np.bool_),
        }

    def generate(self, **kwargs: object) -> SimpleNamespace:
        prompt = torch.as_tensor(kwargs["input_ids"])
        generated = torch.tensor([[31_745, 31_746]], dtype=torch.long)
        sequences = torch.cat((prompt, generated), dim=1)
        scores = []
        for class_index in (1, 2):
            score = torch.zeros((1, self.vocab_size), dtype=torch.float32)
            score[0, 31_744 + class_index] = 3.0
            scores.append(score)
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        input_ids = torch.as_tensor(kwargs["input_ids"])
        self.teacher_input_ids = input_ids.clone()
        logits = torch.zeros(
            (1, input_ids.shape[1], self.vocab_size),
            dtype=torch.float32,
        )
        # causal positions 1、2 分别预测两个action token。
        logits[0, 1, 31_745] = 4.0
        logits[0, 2, 31_746] = 4.0
        return SimpleNamespace(logits=logits)


def test_capture_response_uses_fixed_clean_teacher_prefix_and_action_slice() -> None:
    model = _FakeOpenVLA()
    prompt_inputs = {
        "input_ids": torch.tensor([[10, 29_871]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    clean_teacher_ids = torch.tensor(
        [[10, 29_871, 31_745, 31_746]],
        dtype=torch.long,
    )
    pixels = torch.zeros((1, 6, 2, 2), dtype=torch.bfloat16)

    response = capture_openvla_action_response(
        model,
        prompt_inputs=prompt_inputs,
        pixel_values=pixels,
        clean_teacher_input_ids=clean_teacher_ids,
        pad_token_id=0,
        unnorm_key="libero_spatial_no_noops",
    )

    assert torch.equal(model.teacher_input_ids, clean_teacher_ids)
    assert response.prompt_input_ids.tolist() == [10, 29_871]
    assert response.teacher_input_ids.tolist() == clean_teacher_ids[0].tolist()
    assert response.generated_token_ids.tolist() == [31_745, 31_746]
    assert response.generated_classes.tolist() == [1, 2]
    assert response.generation_logits.shape == (2, 256)
    assert response.teacher_logits.shape == (2, 256)
    assert np.argmax(response.generation_logits, axis=1).tolist() == [1, 2]
    assert np.argmax(response.teacher_logits, axis=1).tolist() == [1, 2]
    assert response.decoded_action.shape == (2,)
    # 调用方输入不能被generate或采集器原地改写。
    assert prompt_inputs["input_ids"].tolist() == [[10, 29_871]]


def test_bfloat16_bits_preserve_tensor_shape_and_exact_pattern() -> None:
    values = torch.tensor(
        [[1.0, -2.0, 0.0]],
        dtype=torch.bfloat16,
    )

    bits = bfloat16_tensor_to_uint16_bits(values)

    assert bits.dtype == np.uint16
    assert bits.shape == (1, 3)
    assert bits.tolist() == [[0x3F80, 0xC000, 0x0000]]
