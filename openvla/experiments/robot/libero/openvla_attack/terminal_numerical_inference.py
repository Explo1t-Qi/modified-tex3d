"""Gate 6g numerical inference replay 的最小 OpenVLA 模型边界。

模块不加载 checkpoint、processor、renderer 或 LIBERO。调用方提供从原失败
NPZ 的 ``uint16`` bit pattern 逐位恢复出的 BF16 ``pixel_values``、prompt IDs 和
clean teacher IDs。本模块只执行以下路径并返回完整 action-subvocabulary logits：

* 原始默认 cached generation 与原始 full teacher forward；
* 完整 ``use_cache=False`` generation；
* 固定真实 generation prefix 的 no-cache forward；
* 固定 clean prefix 的 no-cache forward；
* 完整 teacher sequence 的显式 no-cache forward。

不在这里判定 Fidelity 或根因；所有逐位比较由纯 CPU evaluator 完成。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .objective import (
    ACTION_TOKEN_END,
    ACTION_TOKEN_START,
    extract_action_token_logits,
)
from .terminal_openvla_response import bfloat16_tensor_to_uint16_bits


Tensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class FidelitySample:
    """同一输入的一次原始模型路径输出。"""

    generated_token_ids: NDArray[np.int64]
    generated_classes: NDArray[np.int64]
    generation_logits: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]


@dataclass(frozen=True)
class FidelityReplay:
    """同一输入原始两条模型路径的重复数组。"""

    generated_token_ids: NDArray[np.int64]
    generated_classes: NDArray[np.int64]
    generation_logits: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]


@dataclass(frozen=True)
class AttributionReplay:
    """同一输入四条补充 execution path 的重复数组。"""

    no_cache_generated_token_ids: NDArray[np.int64]
    no_cache_generation_logits: NDArray[np.float32]
    generation_prefix_no_cache_logits: NDArray[np.float32]
    clean_prefix_no_cache_logits: NDArray[np.float32]
    full_teacher_no_cache_logits: NDArray[np.float32]


def tensor_from_bfloat16_bits(
    bits: NDArray[np.uint16],
    *,
    device: torch.device,
) -> Tensor:
    """从 ``uint16`` bit pattern 逐位恢复同 shape 的 BF16 tensor。"""

    array = np.asarray(bits)
    if array.dtype != np.uint16 or array.size == 0:
        raise ValueError("BF16 bit pattern必须为非空uint16数组")
    signed = np.ascontiguousarray(array).view(np.int16)
    tensor = torch.from_numpy(signed).view(torch.bfloat16).to(device=device)
    if not np.array_equal(bfloat16_tensor_to_uint16_bits(tensor), array):
        raise RuntimeError("BF16 bit pattern device round-trip不一致")
    return tensor


def _autocast_context(pixel_values: Tensor) -> ContextManager[Any]:
    if pixel_values.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _validate_inputs(
    *,
    prompt_input_ids: Tensor,
    teacher_input_ids: Tensor,
    pixel_values: Tensor,
) -> int:
    if (
        prompt_input_ids.ndim != 2
        or prompt_input_ids.shape[0] != 1
        or prompt_input_ids.shape[1] <= 0
        or teacher_input_ids.ndim != 2
        or teacher_input_ids.shape[0] != 1
        or teacher_input_ids.shape[1] <= prompt_input_ids.shape[1]
        or prompt_input_ids.dtype != torch.long
        or teacher_input_ids.dtype != torch.long
    ):
        raise ValueError("prompt/teacher IDs必须为有效int64 [1,sequence]")
    if not torch.equal(
        teacher_input_ids[:, : prompt_input_ids.shape[1]], prompt_input_ids
    ):
        raise ValueError("teacher IDs必须以保存的prompt IDs开头")
    if (
        pixel_values.dtype != torch.bfloat16
        or pixel_values.ndim != 4
        or pixel_values.shape[0] != 1
        or pixel_values.numel() == 0
    ):
        raise ValueError("pixel_values必须为非空BF16 [1,C,H,W]")
    return int(teacher_input_ids.shape[1] - prompt_input_ids.shape[1])


def _generate_once(
    model: Any,
    *,
    prompt_input_ids: Tensor,
    pixel_values: Tensor,
    action_dim: int,
    pad_token_id: int,
    use_cache: Optional[bool],
) -> tuple[Tensor, Tensor, Tensor]:
    kwargs: dict[str, Any] = {
        "input_ids": prompt_input_ids,
        "attention_mask": torch.ones_like(prompt_input_ids),
        "pixel_values": pixel_values,
        "max_new_tokens": action_dim,
        "do_sample": False,
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    with torch.no_grad(), _autocast_context(pixel_values):
        generated = model.generate(**kwargs)
    if not hasattr(generated, "sequences") or not hasattr(generated, "scores"):
        raise RuntimeError("model.generate未返回sequences/scores")
    sequences = generated.sequences
    scores = tuple(generated.scores)
    if (
        sequences.ndim != 2
        or sequences.shape[0] != 1
        or sequences.shape[1] < prompt_input_ids.shape[1] + action_dim
        or len(scores) != action_dim
    ):
        raise RuntimeError("generation sequence/score数量无效")
    token_ids = sequences[0, -action_dim:].to(torch.int64)
    classes = token_ids - ACTION_TOKEN_START
    if bool(
        torch.any(classes < 0)
        or torch.any(token_ids >= ACTION_TOKEN_END)
    ):
        raise RuntimeError("generation包含action子词表外token")
    logits = torch.stack(
        tuple(
            score[0, ACTION_TOKEN_START:ACTION_TOKEN_END]
            for score in scores
        ),
        dim=0,
    ).float()
    if logits.shape != (action_dim, ACTION_TOKEN_END - ACTION_TOKEN_START):
        raise RuntimeError("generation action logits shape无效")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("generation logits包含NaN/Inf")
    return token_ids, classes, logits


def _teacher_once(
    model: Any,
    *,
    teacher_input_ids: Tensor,
    pixel_values: Tensor,
    action_dim: int,
    use_cache: Optional[bool],
) -> Tensor:
    kwargs: dict[str, Any] = {
        "input_ids": teacher_input_ids,
        "attention_mask": torch.ones_like(teacher_input_ids),
        "pixel_values": pixel_values,
        "output_hidden_states": False,
    }
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    with torch.no_grad(), _autocast_context(pixel_values):
        outputs = model(**kwargs)
    action_tokens = extract_action_token_logits(
        outputs.logits,
        teacher_input_ids,
    )
    logits = action_tokens.logits.float()
    if logits.shape != (action_dim, ACTION_TOKEN_END - ACTION_TOKEN_START):
        raise RuntimeError("teacher action logits shape无效")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("teacher logits包含NaN/Inf")
    return logits


def _next_token_logits(
    model: Any,
    *,
    prefix_input_ids: Tensor,
    pixel_values: Tensor,
) -> Tensor:
    """显式no-cache forward最后位置预测的256维action logits。"""

    with torch.no_grad(), _autocast_context(pixel_values):
        outputs = model(
            input_ids=prefix_input_ids,
            attention_mask=torch.ones_like(prefix_input_ids),
            pixel_values=pixel_values,
            output_hidden_states=False,
            use_cache=False,
        )
    logits = outputs.logits
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < prefix_input_ids.shape[1]
        or logits.shape[2] < ACTION_TOKEN_END
    ):
        raise RuntimeError("prefix forward logits shape无效")
    aligned = logits[:, -prefix_input_ids.shape[1] :, :]
    action_logits = aligned[
        0, -1, ACTION_TOKEN_START:ACTION_TOKEN_END
    ].float()
    if not bool(torch.isfinite(action_logits).all()):
        raise RuntimeError("prefix logits包含NaN/Inf")
    return action_logits


def _prefix_logits(
    model: Any,
    *,
    prompt_input_ids: Tensor,
    prefix_token_ids: Tensor,
    pixel_values: Tensor,
) -> Tensor:
    rows: list[Tensor] = []
    for token_index in range(prefix_token_ids.numel()):
        prefix = torch.cat(
            (
                prompt_input_ids,
                prefix_token_ids[:token_index].view(1, -1),
            ),
            dim=1,
        )
        rows.append(
            _next_token_logits(
                model,
                prefix_input_ids=prefix,
                pixel_values=pixel_values,
            )
        )
    return torch.stack(rows, dim=0)


def capture_fidelity_sample(
    model: Any,
    *,
    prompt_input_ids: Tensor,
    teacher_input_ids: Tensor,
    pixel_values: Tensor,
    pad_token_id: int,
    unnorm_key: Optional[str],
) -> FidelitySample:
    """执行一次原始 default generation 与 default teacher forward。"""

    action_dim = _validate_inputs(
        prompt_input_ids=prompt_input_ids,
        teacher_input_ids=teacher_input_ids,
        pixel_values=pixel_values,
    )
    model_action_dim = int(model.get_action_dim(unnorm_key))
    if model_action_dim != action_dim:
        raise RuntimeError(
            f"保存的action_dim与checkpoint不一致: {action_dim} != {model_action_dim}"
        )
    token_ids, classes, generation_logits = _generate_once(
        model,
        prompt_input_ids=prompt_input_ids,
        pixel_values=pixel_values,
        action_dim=action_dim,
        pad_token_id=pad_token_id,
        use_cache=None,
    )
    teacher_logits = _teacher_once(
        model,
        teacher_input_ids=teacher_input_ids,
        pixel_values=pixel_values,
        action_dim=action_dim,
        use_cache=None,
    )
    return FidelitySample(
        generated_token_ids=token_ids.cpu().numpy().copy(),
        generated_classes=classes.cpu().numpy().copy(),
        generation_logits=generation_logits.cpu().numpy().copy(),
        teacher_logits=teacher_logits.cpu().numpy().copy(),
    )


def capture_fidelity_repeats(
    model: Any,
    *,
    prompt_input_ids: Tensor,
    teacher_input_ids: Tensor,
    pixel_values: Tensor,
    pad_token_id: int,
    unnorm_key: Optional[str],
    repeat_count: int,
) -> FidelityReplay:
    """重复原始 default generation 与原始 default teacher forward。"""

    if repeat_count != 3:
        raise ValueError("numerical inference replay固定重复3次")
    token_repeats: list[NDArray[np.int64]] = []
    class_repeats: list[NDArray[np.int64]] = []
    generation_repeats: list[NDArray[np.float32]] = []
    teacher_repeats: list[NDArray[np.float32]] = []
    for _ in range(repeat_count):
        sample = capture_fidelity_sample(
            model,
            prompt_input_ids=prompt_input_ids,
            teacher_input_ids=teacher_input_ids,
            pixel_values=pixel_values,
            pad_token_id=pad_token_id,
            unnorm_key=unnorm_key,
        )
        token_repeats.append(sample.generated_token_ids)
        class_repeats.append(sample.generated_classes)
        generation_repeats.append(sample.generation_logits)
        teacher_repeats.append(sample.teacher_logits)
    return FidelityReplay(
        generated_token_ids=np.stack(token_repeats, axis=0),
        generated_classes=np.stack(class_repeats, axis=0),
        generation_logits=np.stack(generation_repeats, axis=0),
        teacher_logits=np.stack(teacher_repeats, axis=0),
    )


def capture_attribution_repeats(
    model: Any,
    *,
    prompt_input_ids: Tensor,
    teacher_input_ids: Tensor,
    pixel_values: Tensor,
    cached_generated_token_ids: Tensor | NDArray[np.integer],
    pad_token_id: int,
    unnorm_key: Optional[str],
    repeat_count: int,
) -> AttributionReplay:
    """重复完整no-cache generation、双prefix和full-teacher对照。"""

    action_dim = _validate_inputs(
        prompt_input_ids=prompt_input_ids,
        teacher_input_ids=teacher_input_ids,
        pixel_values=pixel_values,
    )
    if repeat_count != 3:
        raise ValueError("numerical inference replay固定重复3次")
    if int(model.get_action_dim(unnorm_key)) != action_dim:
        raise RuntimeError("checkpoint action_dim与保存teacher IDs不一致")
    cached_tokens = torch.as_tensor(cached_generated_token_ids).to(
        device=prompt_input_ids.device,
        dtype=torch.long,
    ).reshape(-1)
    if cached_tokens.numel() != action_dim:
        raise ValueError("cached generated token数量与action_dim不一致")
    clean_tokens = teacher_input_ids[0, -action_dim:]
    no_cache_token_repeats: list[NDArray[np.int64]] = []
    no_cache_generation_repeats: list[NDArray[np.float32]] = []
    generation_prefix_repeats: list[NDArray[np.float32]] = []
    clean_prefix_repeats: list[NDArray[np.float32]] = []
    full_teacher_repeats: list[NDArray[np.float32]] = []
    for _ in range(repeat_count):
        no_cache_ids, _, no_cache_logits = _generate_once(
            model,
            prompt_input_ids=prompt_input_ids,
            pixel_values=pixel_values,
            action_dim=action_dim,
            pad_token_id=pad_token_id,
            use_cache=False,
        )
        generation_prefix = _prefix_logits(
            model,
            prompt_input_ids=prompt_input_ids,
            prefix_token_ids=cached_tokens,
            pixel_values=pixel_values,
        )
        clean_prefix = _prefix_logits(
            model,
            prompt_input_ids=prompt_input_ids,
            prefix_token_ids=clean_tokens,
            pixel_values=pixel_values,
        )
        full_teacher = _teacher_once(
            model,
            teacher_input_ids=teacher_input_ids,
            pixel_values=pixel_values,
            action_dim=action_dim,
            use_cache=False,
        )
        no_cache_token_repeats.append(no_cache_ids.cpu().numpy().copy())
        no_cache_generation_repeats.append(no_cache_logits.cpu().numpy().copy())
        generation_prefix_repeats.append(
            generation_prefix.cpu().numpy().copy()
        )
        clean_prefix_repeats.append(clean_prefix.cpu().numpy().copy())
        full_teacher_repeats.append(full_teacher.cpu().numpy().copy())
    return AttributionReplay(
        no_cache_generated_token_ids=np.stack(
            no_cache_token_repeats, axis=0
        ),
        no_cache_generation_logits=np.stack(
            no_cache_generation_repeats, axis=0
        ),
        generation_prefix_no_cache_logits=np.stack(
            generation_prefix_repeats, axis=0
        ),
        clean_prefix_no_cache_logits=np.stack(
            clean_prefix_repeats, axis=0
        ),
        full_teacher_no_cache_logits=np.stack(
            full_teacher_repeats, axis=0
        ),
    )
