"""Utils for evaluating the OpenVLA policy."""

from __future__ import annotations

import json
import os
import time
from typing import Final, MutableMapping, TypeVar

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LLAMA_EMPTY_TOKEN_ID: Final[int] = 29_871
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Hugging Face 的 BatchFeature/BatchEncoding 都实现了 MutableMapping 接口。
# 使用 TypeVar 可以让函数保留传入容器的具体类型，而不是把返回值退化成普通 dict。
ModelInputsT = TypeVar(
    "ModelInputsT",
    bound=MutableMapping[str, torch.Tensor],
)

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def ensure_trailing_empty_token(inputs: ModelInputsT) -> ModelInputsT:
    """确保 OpenVLA prompt 尾部 token 与 attention mask 同步补齐。

    OpenVLA checkpoint 内的 ``predict_action`` 会在 prompt 末尾缺少 LLaMA
    empty token（ID 29871）时自行追加 ``input_ids``。原实现不会同时扩展已经
    存在的 ``attention_mask``，因此生成第一个 action token 时会出现长度相差
    1 的 causal mask，并最终在 LLaMA attention 中报 shape mismatch。

    本函数在进入 ``predict_action`` 前完成同一项补齐，同时维护下面的不变量：

    - ``input_ids`` shape: ``[batch_size, sequence_length]``
    - ``attention_mask`` shape: ``[batch_size, sequence_length]``
    - 补齐后两个张量的 ``sequence_length`` 始终相同

    函数原地更新 Hugging Face 输入容器并返回同一对象，其他输入（例如
    ``pixel_values``）不会被复制或修改。
    """
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        return inputs

    input_ids: torch.Tensor = inputs["input_ids"]
    attention_mask: torch.Tensor = inputs["attention_mask"]

    # processor 的常规输出 batch_size=1；这里仍按 batch 维统一处理，使类型和
    # shape 约束清晰。若所有样本已有尾 token，则直接返回，避免重复追加。
    has_trailing_empty_token: bool = bool(
        torch.all(input_ids[:, -1] == LLAMA_EMPTY_TOKEN_ID).item()
    )
    if has_trailing_empty_token:
        return inputs

    batch_size: int = input_ids.shape[0]
    trailing_token: torch.Tensor = torch.full(
        (batch_size, 1),
        fill_value=LLAMA_EMPTY_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    trailing_attention: torch.Tensor = torch.ones(
        (batch_size, 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    inputs["input_ids"] = torch.cat((input_ids, trailing_token), dim=1)
    inputs["attention_mask"] = torch.cat(
        (attention_mask, trailing_attention),
        dim=1,
    )
    return inputs


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    attn_impl = "sdpa" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        # eager attention 与当前 OpenVLA 生成路径兼容，并允许获取所需梯度。
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        # unnorm_key 必须与 checkpoint 的数据集统计量一致。
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    # processor 把 RGB 图像和任务文本转换为模型所需张量。
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    inputs = ensure_trailing_empty_token(inputs)

    # Get action.
    action = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
    )
    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    return action
