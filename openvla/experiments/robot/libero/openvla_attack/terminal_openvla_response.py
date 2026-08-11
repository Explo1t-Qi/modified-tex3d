"""Gate 6g source OpenVLA 静态动作响应的模型边界。

模块不加载 checkpoint、processor、renderer 或 LIBERO。调用方提供已经完成最终
BF16 cast 的 ``pixel_values [1,6,H,W]`` 与固定 prompt tensors；本模块执行一次
greedy generation 和一次固定 clean sequence teacher forward，保存完整 action
子词表 logits、生成 token、class mapping 与确定性 codec 解码。

非 Clean 路径必须显式传入 Clean 生成的完整 ``clean_teacher_input_ids``。这样
teacher logits 始终使用共同 clean prefix，不会因 A/B 自回归分叉而改变后续前缀。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Mapping, Optional, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .action_codec import decode_action_from_generated_ids
from .objective import (
    ACTION_TOKEN_END,
    ACTION_TOKEN_START,
    extract_action_token_logits,
)


TensorMapping: TypeAlias = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class OpenVLAActionResponse:
    """一条静态图像路径的权威动作响应数组。"""

    prompt_input_ids: NDArray[np.int64]
    teacher_input_ids: NDArray[np.int64]
    generated_token_ids: NDArray[np.int64]
    generated_classes: NDArray[np.int64]
    generation_logits: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]
    decoded_action: NDArray[np.floating[Any]]


def bfloat16_tensor_to_uint16_bits(
    value: torch.Tensor,
) -> NDArray[np.uint16]:
    """保留shape，把BF16 tensor逐值导出为无符号16 bit pattern。"""

    if value.dtype != torch.bfloat16 or value.numel() == 0:
        raise ValueError("最终processor tensor必须为非空torch.bfloat16")
    signed_bits = (
        value.detach().contiguous().view(torch.int16).cpu().numpy()
    )
    return signed_bits.view(np.uint16).copy()


def _autocast_context(pixel_values: torch.Tensor) -> ContextManager[Any]:
    if pixel_values.device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


def _clone_tensor_mapping(value: TensorMapping) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in value.items()}


def capture_openvla_action_response(
    model: Any,
    *,
    prompt_inputs: TensorMapping,
    pixel_values: torch.Tensor,
    clean_teacher_input_ids: Optional[torch.Tensor],
    pad_token_id: int,
    unnorm_key: Optional[str],
) -> OpenVLAActionResponse:
    """采集一条路径的greedy generation、clean-prefix teacher logits与动作。"""

    if "input_ids" not in prompt_inputs or "attention_mask" not in prompt_inputs:
        raise ValueError("prompt inputs缺少input_ids或attention_mask")
    prompt_ids = prompt_inputs["input_ids"]
    prompt_attention = prompt_inputs["attention_mask"]
    if (
        prompt_ids.ndim != 2
        or prompt_ids.shape[0] != 1
        or prompt_attention.shape != prompt_ids.shape
        or not torch.is_floating_point(pixel_values)
        or pixel_values.dtype != torch.bfloat16
        or pixel_values.ndim != 4
        or pixel_values.shape[0] != 1
    ):
        raise ValueError("prompt或最终BF16 pixel_values shape/dtype无效")
    action_dim = int(model.get_action_dim(unnorm_key))
    if action_dim <= 0:
        raise ValueError("OpenVLA action_dim必须为正数")

    generation_inputs = _clone_tensor_mapping(prompt_inputs)
    generation_inputs["pixel_values"] = pixel_values
    generation_snapshot = _clone_tensor_mapping(generation_inputs)
    with torch.no_grad(), _autocast_context(pixel_values):
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    for name, before in generation_snapshot.items():
        if name not in generation_inputs or not torch.equal(
            before, generation_inputs[name]
        ):
            raise RuntimeError(f"model.generate原地修改输入tensor: {name}")
    if not hasattr(generated, "sequences") or not hasattr(generated, "scores"):
        raise RuntimeError("model.generate未返回sequences/scores")
    sequences = generated.sequences
    scores = tuple(generated.scores)
    if (
        sequences.ndim != 2
        or sequences.shape[0] != 1
        or sequences.shape[1] < prompt_ids.shape[1] + action_dim
        or len(scores) != action_dim
    ):
        raise RuntimeError("OpenVLA generation sequence/score数量与action_dim不一致")
    generated_token_tensor = sequences[0, -action_dim:].to(torch.int64)
    generated_class_tensor = generated_token_tensor - ACTION_TOKEN_START
    if bool(
        torch.any(generated_class_tensor < 0)
        or torch.any(generated_token_tensor >= ACTION_TOKEN_END)
    ):
        raise RuntimeError("OpenVLA生成了action子词表范围外的token")
    generation_action_logits = torch.stack(
        [
            score[0, ACTION_TOKEN_START:ACTION_TOKEN_END]
            for score in scores
        ],
        dim=0,
    ).float()

    teacher_ids = (
        sequences.detach()
        if clean_teacher_input_ids is None
        else clean_teacher_input_ids.detach()
    )
    if teacher_ids.ndim != 2 or teacher_ids.shape[0] != 1:
        raise ValueError("clean teacher input IDs必须为[1,sequence_length]")
    with torch.no_grad(), _autocast_context(pixel_values):
        teacher_outputs = model(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            pixel_values=pixel_values,
            output_hidden_states=False,
        )
    action_tokens = extract_action_token_logits(
        teacher_outputs.logits,
        teacher_ids,
    )
    if action_tokens.logits.shape != (action_dim, ACTION_TOKEN_END - ACTION_TOKEN_START):
        raise RuntimeError("teacher action logits shape与action_dim不一致")
    expected_clean_classes = (
        teacher_ids[0, -action_dim:] - ACTION_TOKEN_START
    ).to(action_tokens.clean_classes.device)
    if not torch.equal(action_tokens.clean_classes, expected_clean_classes):
        raise RuntimeError("teacher causal action slice与clean sequence尾部不一致")
    teacher_action_logits = action_tokens.logits.float()
    if not bool(
        torch.isfinite(generation_action_logits).all()
        and torch.isfinite(teacher_action_logits).all()
    ):
        raise RuntimeError("generation/teacher action logits包含NaN/Inf")
    decoded_action = np.asarray(
        decode_action_from_generated_ids(
            model,
            sequences,
            unnorm_key=unnorm_key,
        )
    ).copy()
    return OpenVLAActionResponse(
        prompt_input_ids=prompt_ids[0].detach().cpu().to(torch.int64).numpy().copy(),
        teacher_input_ids=teacher_ids[0].cpu().to(torch.int64).numpy().copy(),
        generated_token_ids=generated_token_tensor.cpu().numpy().copy(),
        generated_classes=generated_class_tensor.cpu().numpy().copy(),
        generation_logits=generation_action_logits.cpu().numpy().copy(),
        teacher_logits=teacher_action_logits.cpu().numpy().copy(),
        decoded_action=decoded_action,
    )
