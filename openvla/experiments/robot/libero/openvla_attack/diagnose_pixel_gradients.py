"""采集 OpenVLA 在真实 clean/adv 输入处的主视角像素梯度。

本诊断不修改纹理，也不运行 LIBERO。它读取 OFT 响应诊断已保存的同一批成对
policy 输入，在当前 K=256 对抗图像处分别计算：

- ``-MSE(adv_siglip, clean_siglip)``；
- ``-MSE(adv_action_logits, clean_action_logits)``。

两项 loss 的梯度下降方向都会继续增大 clean/adv 差异。保存的梯度关于中心裁剪
之前的 RGB 图像，shape ``[states, height, width, 3]``，可以和 OFT 的同坐标
产物直接比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast


THIS_FILE: Path = Path(__file__).resolve()
REPOSITORY_ROOT: Path = THIS_FILE.parents[5]
OPENVLA_ROOT: Path = THIS_FILE.parents[4]
ROBOT_EXPERIMENT_DIR: Path = THIS_FILE.parents[2]
LIBERO_EXPERIMENT_DIR: Path = THIS_FILE.parents[1]
for import_path in (
    REPOSITORY_ROOT,
    OPENVLA_ROOT,
    ROBOT_EXPERIMENT_DIR,
    LIBERO_EXPERIMENT_DIR,
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402
from scripts.vla_pixel_gradient_audit import (  # noqa: E402
    PixelGradientArtifact,
    build_fused_pixel_values,
    differentiable_center_crop,
    tensor_gradient_to_hwc_float64,
    visible_perturbation_mask,
)

from openvla_attack.objective import (  # noqa: E402
    ACTION_TOKEN_END,
    ACTION_TOKEN_START,
)
from openvla_attack.vision_features import (  # noqa: E402
    extract_siglip_patch_features,
)


@dataclass
class SourceGradientConfig:
    """OpenVLA loader 与梯度诊断共同使用的配置。"""

    pretrained_checkpoint: str
    input_dir: str
    output_path: str
    state_ids: Optional[str] = None
    seed: int = 7
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: Optional[str] = "libero_spatial_no_noops"


def _parse_args(argv: Optional[Sequence[str]] = None) -> SourceGradientConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--state_ids", default=None)
    parser.add_argument("--seed", type=int, default=7)
    return SourceGradientConfig(**vars(parser.parse_args(argv)))


def _resolve_state_ids(
    report_state_ids: Sequence[int],
    specification: Optional[str],
) -> tuple[int, ...]:
    available: tuple[int, ...] = tuple(int(value) for value in report_state_ids)
    if specification is None:
        return available
    selected: tuple[int, ...] = tuple(
        int(token.strip()) for token in specification.split(",")
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("state_ids 必须非空且不能重复")
    missing: list[int] = [value for value in selected if value not in available]
    if missing:
        raise ValueError(f"输入目录缺少 states: {missing}")
    return selected


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _processor_statistics(processor: Any) -> tuple[Any, Any]:
    image_processor: Any = processor.image_processor
    means: Any = getattr(image_processor, "means", None)
    stds: Any = getattr(image_processor, "stds", None)
    if means is None or stds is None:
        raise RuntimeError("OpenVLA processor 缺少 fused mean/std")
    return means, stds


def _siglip_index(model: Any) -> int:
    model_ids: tuple[str, ...] = tuple(
        str(value) for value in model.config.timm_model_ids
    )
    indices: list[int] = [
        index for index, model_id in enumerate(model_ids)
        if "siglip" in model_id.lower()
    ]
    if len(indices) != 1:
        raise RuntimeError(f"无法唯一定位 SigLIP 分支: {model_ids}")
    return indices[0]


def _action_logits(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """从 teacher-forced 输出提取7个动作位置的256维 logits。"""
    aligned_logits: torch.Tensor = logits
    if aligned_logits.shape[1] > generated_ids.shape[1]:
        aligned_logits = aligned_logits[:, -generated_ids.shape[1] :, :]
    shifted_logits: torch.Tensor = aligned_logits[:, :-1, :]
    shifted_labels: torch.Tensor = generated_ids[:, 1:].to(logits.device)
    action_mask: torch.Tensor = (
        (shifted_labels >= ACTION_TOKEN_START)
        & (shifted_labels < ACTION_TOKEN_END)
    )
    action_logits: torch.Tensor = shifted_logits[action_mask][
        :, ACTION_TOKEN_START:ACTION_TOKEN_END
    ]
    if action_logits.ndim != 2 or action_logits.shape[0] != 7:
        raise RuntimeError(
            "OpenVLA 应生成7个 action-token logits，实际 shape="
            f"{tuple(action_logits.shape)}"
        )
    return action_logits


def _rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 HWC → float32 ``[1,3,H,W]``。"""
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def run_source_gradient_audit(cfg: SourceGradientConfig) -> Path:
    """运行五状态 OpenVLA 梯度采集并保存 NPZ。"""
    set_seed_everywhere(cfg.seed)
    input_dir: Path = Path(cfg.input_dir).resolve()
    report_path: Path = input_dir / "oft_transfer_response.json"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    state_ids: tuple[int, ...] = _resolve_state_ids(
        report["state_ids"],
        cfg.state_ids,
    )
    task_description: str = str(report["task_description"])
    prompt: str = (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )

    print(f"[INFO] Loading source OpenVLA: {cfg.pretrained_checkpoint}")
    model: Any = get_model(cfg)
    model.eval()
    processor: Any = get_processor(cfg)
    means, stds = _processor_statistics(processor)
    if len(means) != len(model.config.timm_model_ids):
        raise RuntimeError("processor 分支数量与 timm_model_ids 不一致")
    siglip_index: int = _siglip_index(model)
    device: torch.device = model.device

    feature_losses: list[float] = []
    action_losses: list[float] = []
    feature_gradients: list[np.ndarray] = []
    action_gradients: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_dir: Path = input_dir / "paired_inputs"

    for state_id in state_ids:
        print(f"[INFO] Source gradient state {state_id}")
        clean_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_clean_primary.png"
        )
        adversarial_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_adversarial_primary.png"
        )
        clean_tensor: torch.Tensor = _rgb_tensor(clean_rgb, device)
        adversarial_tensor: torch.Tensor = _rgb_tensor(
            adversarial_rgb,
            device,
        ).requires_grad_(True)
        clean_cropped: torch.Tensor = differentiable_center_crop(clean_tensor)
        adversarial_cropped: torch.Tensor = differentiable_center_crop(
            adversarial_tensor
        )
        clean_pixels: torch.Tensor = build_fused_pixel_values(
            clean_cropped,
            means=means,
            stds=stds,
        ).to(torch.bfloat16)
        adversarial_pixels: torch.Tensor = build_fused_pixel_values(
            adversarial_cropped,
            means=means,
            stds=stds,
        ).to(torch.bfloat16)

        text_inputs: Any = processor(
            prompt,
            images=Image.fromarray(clean_rgb),
        ).to(device)
        text_inputs["pixel_values"] = clean_pixels
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            clean_generated_ids: torch.Tensor = model.generate(
                **text_inputs,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            clean_outputs: Any = model(
                input_ids=clean_generated_ids,
                attention_mask=torch.ones_like(clean_generated_ids),
                pixel_values=clean_pixels,
            )
            clean_action_logits: torch.Tensor = _action_logits(
                clean_outputs.logits,
                clean_generated_ids,
            ).detach()
            clean_siglip: torch.Tensor = extract_siglip_patch_features(
                model,
                clean_pixels[:, 3 * siglip_index : 3 * siglip_index + 3],
            ).detach()

        with autocast(dtype=torch.bfloat16):
            adversarial_outputs: Any = model(
                input_ids=clean_generated_ids,
                attention_mask=torch.ones_like(clean_generated_ids),
                pixel_values=adversarial_pixels,
            )
            adversarial_action_logits: torch.Tensor = _action_logits(
                adversarial_outputs.logits,
                clean_generated_ids,
            )
            adversarial_siglip: torch.Tensor = extract_siglip_patch_features(
                model,
                adversarial_pixels[
                    :, 3 * siglip_index : 3 * siglip_index + 3
                ],
            )
            feature_loss: torch.Tensor = -F.mse_loss(
                adversarial_siglip.float(),
                clean_siglip.float(),
            )
            action_loss: torch.Tensor = -F.mse_loss(
                adversarial_action_logits.float(),
                clean_action_logits.float(),
            )

        feature_gradient: torch.Tensor = torch.autograd.grad(
            feature_loss,
            adversarial_tensor,
            retain_graph=True,
        )[0]
        action_gradient: torch.Tensor = torch.autograd.grad(
            action_loss,
            adversarial_tensor,
            retain_graph=False,
        )[0]
        feature_losses.append(float(feature_loss.detach().item()))
        action_losses.append(float(action_loss.detach().item()))
        feature_gradients.append(
            tensor_gradient_to_hwc_float64(feature_gradient)
        )
        action_gradients.append(
            tensor_gradient_to_hwc_float64(action_gradient)
        )
        masks.append(visible_perturbation_mask(clean_rgb, adversarial_rgb))

    artifact = PixelGradientArtifact(
        model_name="openvla",
        checkpoint=str(Path(cfg.pretrained_checkpoint).resolve()),
        state_ids=np.asarray(state_ids, dtype=np.int64),
        feature_losses=np.asarray(feature_losses, dtype=np.float64),
        action_losses=np.asarray(action_losses, dtype=np.float64),
        primary_feature_gradients=np.stack(feature_gradients, axis=0),
        primary_action_gradients=np.stack(action_gradients, axis=0),
        perturbation_masks=np.stack(masks, axis=0),
        metadata={
            "reference_point": "trained_k256_adversarial_input",
            "feature_objective": "negative_siglip_patch_mse",
            "action_objective": "negative_action_token_logit_mse",
            "crop_area": 0.9,
            "input_directory": str(input_dir),
        },
    )
    output_path: Path = artifact.save(cfg.output_path)
    print(f"[DONE] Source pixel gradients: {output_path}")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_source_gradient_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
