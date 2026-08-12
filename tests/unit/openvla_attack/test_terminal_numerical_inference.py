"""Gate 6g numerical inference 的模型边界纯 CPU 测试。"""

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

from openvla_attack.terminal_numerical_inference import (  # noqa: E402
    capture_attribution_repeats,
    capture_fidelity_repeats,
    tensor_from_bfloat16_bits,
)
from openvla_attack.terminal_openvla_response import (  # noqa: E402
    bfloat16_tensor_to_uint16_bits,
)


class _FakeNumericalOpenVLA:
    """记录每次forward的prefix和显式use_cache参数。"""

    vocab_size = 32_000

    def __init__(self) -> None:
        self.forward_calls: list[tuple[list[int], object]] = []

    def get_action_dim(self, unnorm_key: str | None) -> int:
        del unnorm_key
        return 2

    def generate(self, **kwargs: object) -> SimpleNamespace:
        prompt = torch.as_tensor(kwargs["input_ids"])
        use_cache = kwargs.get("use_cache", "default")
        generated = torch.tensor([[31_745, 31_746]], dtype=torch.long)
        sequences = torch.cat((prompt, generated), dim=1)
        scores = []
        for class_index in (1, 2):
            score = torch.zeros((1, self.vocab_size), dtype=torch.float32)
            score[0, 31_744 + class_index] = (
                4.0 if use_cache == "default" else 3.0
            )
            scores.append(score)
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        input_ids = torch.as_tensor(kwargs["input_ids"])
        use_cache = kwargs.get("use_cache", "default")
        self.forward_calls.append((input_ids[0].tolist(), use_cache))
        logits = torch.zeros(
            (1, input_ids.shape[1], self.vocab_size), dtype=torch.float32
        )
        logits[..., 31_745] = 1.0
        logits[..., 31_746] = 2.0
        return SimpleNamespace(logits=logits)


def test_bfloat16_bit_reconstruction_is_exact() -> None:
    original = torch.tensor(
        [[1.0, -2.0, 0.0, 0.125]], dtype=torch.bfloat16
    )
    bits = bfloat16_tensor_to_uint16_bits(original)

    reconstructed = tensor_from_bfloat16_bits(bits, device=torch.device("cpu"))

    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed, original)
    assert np.array_equal(bfloat16_tensor_to_uint16_bits(reconstructed), bits)


def test_capture_separates_generation_and_clean_prefixes() -> None:
    model = _FakeNumericalOpenVLA()
    prompt_ids = torch.tensor([[10, 11]], dtype=torch.long)
    teacher_ids = torch.tensor(
        [[10, 11, 31_745, 31_747]], dtype=torch.long
    )
    pixels = torch.zeros((1, 6, 2, 2), dtype=torch.bfloat16)

    fidelity = capture_fidelity_repeats(
        model,
        prompt_input_ids=prompt_ids,
        teacher_input_ids=teacher_ids,
        pixel_values=pixels,
        pad_token_id=0,
        unnorm_key=None,
        repeat_count=3,
    )
    attribution = capture_attribution_repeats(
        model,
        prompt_input_ids=prompt_ids,
        teacher_input_ids=teacher_ids,
        pixel_values=pixels,
        cached_generated_token_ids=fidelity.generated_token_ids[0],
        pad_token_id=0,
        unnorm_key=None,
        repeat_count=3,
    )

    assert fidelity.generated_token_ids.shape == (3, 2)
    assert fidelity.generation_logits.shape == (3, 2, 256)
    assert fidelity.teacher_logits.shape == (3, 2, 256)
    assert attribution.no_cache_generation_logits.shape == (3, 2, 256)
    assert attribution.generation_prefix_no_cache_logits.shape == (3, 2, 256)
    assert attribution.clean_prefix_no_cache_logits.shape == (3, 2, 256)
    assert attribution.full_teacher_no_cache_logits.shape == (3, 2, 256)

    calls = model.forward_calls
    assert ([10, 11], False) in calls
    assert ([10, 11, 31_745], False) in calls
    assert ([10, 11, 31_747], False) not in calls
    # clean token 0与generation相同；第二个next-token prefix分别为
    # generation的31745和teacher保存的31745。完整teacher仍保留31747尾部。
    assert ([10, 11, 31_745, 31_747], "default") in calls
    assert ([10, 11, 31_745, 31_747], False) in calls
