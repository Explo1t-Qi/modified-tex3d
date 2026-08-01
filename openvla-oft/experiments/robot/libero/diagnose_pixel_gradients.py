"""采集 OpenVLA-OFT 在真实 clean/adv 输入处的像素梯度。

OFT 保持主视角、腕部和 proprio 的真实推理条件。Feature loss 使用两个视角的
SigLIP patch features，Action loss 使用连续 action-head 输出；二者都是 clean
与 adversarial 响应的负 MSE。主视角梯度保存到与 OpenVLA 相同的 center-crop
前 RGB 坐标，腕部梯度仅作为目标模型上下文诊断，不进入跨模型 cosine。
"""

from __future__ import annotations

import argparse
import json
import os
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
THIS_DIR: Path = THIS_FILE.parent
REPOSITORY_ROOT: Path = THIS_FILE.parents[4]
DEFAULT_EXTERNAL_OFT_ROOT: Path = Path(
    "/home/xiaomengqi/src/github/paper_code/openvla-oft"
)
external_oft_root: Path = Path(
    os.environ.get("OPENVLA_OFT_ROOT", str(DEFAULT_EXTERNAL_OFT_ROOT))
).resolve()
for import_path in (external_oft_root, REPOSITORY_ROOT, THIS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_model,
    set_seed_everywhere,
)
from scripts.vla_pixel_gradient_audit import (  # noqa: E402
    PixelGradientArtifact,
    build_fused_pixel_values,
    differentiable_center_crop,
    tensor_gradient_to_hwc_float64,
    visible_perturbation_mask,
)
from transfer_response import extract_primary_siglip_features  # noqa: E402


@dataclass
class TargetGradientConfig:
    """OFT loader 与像素梯度诊断共同使用的配置。"""

    pretrained_checkpoint: str
    input_dir: str
    output_path: str
    state_ids: Optional[str] = None
    seed: int = 7
    task_suite_name: str = "libero_spatial"
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 1
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: Optional[str] = None


def _parse_args(argv: Optional[Sequence[str]] = None) -> TargetGradientConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--state_ids", default=None)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=7)
    return TargetGradientConfig(**vars(parser.parse_args(argv)))


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


def _rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _processor_statistics(processor: Any) -> tuple[Any, Any]:
    image_processor: Any = processor.image_processor
    means: Any = getattr(image_processor, "means", None)
    stds: Any = getattr(image_processor, "stds", None)
    if means is None or stds is None:
        raise RuntimeError("OFT processor 缺少 fused mean/std")
    return means, stds


def _check_unnorm_key(cfg: TargetGradientConfig, model: Any) -> None:
    key: str = cfg.unnorm_key or cfg.task_suite_name
    norm_stats: Any = getattr(model, "norm_stats", None)
    if not isinstance(norm_stats, dict):
        raise RuntimeError("OFT model 缺少 norm_stats")
    if key not in norm_stats and f"{key}_no_noops" in norm_stats:
        key = f"{key}_no_noops"
    if key not in norm_stats:
        raise ValueError(f"OFT unnorm key 不存在: {key}")
    cfg.unnorm_key = key


def run_target_gradient_audit(cfg: TargetGradientConfig) -> Path:
    """运行五状态 OFT 梯度采集并保存 NPZ。"""
    if cfg.num_images_in_input != 2 or not cfg.use_proprio:
        raise ValueError("诊断固定复现 OFT 双视角+proprio 推理")
    set_seed_everywhere(cfg.seed)
    input_dir: Path = Path(cfg.input_dir).resolve()
    report: dict[str, Any] = json.loads(
        (input_dir / "oft_transfer_response.json").read_text(encoding="utf-8")
    )
    state_ids: tuple[int, ...] = _resolve_state_ids(
        report["state_ids"],
        cfg.state_ids,
    )
    report_by_state: dict[int, dict[str, Any]] = {
        int(record["state_id"]): record for record in report["states"]
    }
    task_description: str = str(report["task_description"])
    prompt: str = (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )

    print(f"[INFO] Loading target OFT: {cfg.pretrained_checkpoint}")
    model: Any = get_model(cfg)
    model.eval()
    processor: Any = get_processor(cfg)
    means, stds = _processor_statistics(processor)
    if len(means) != len(model.config.timm_model_ids):
        raise RuntimeError("processor 分支数量与 timm_model_ids 不一致")
    _check_unnorm_key(cfg, model)
    action_head: torch.nn.Module = get_action_head(cfg, model.llm_dim)
    proprio_projector: torch.nn.Module = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,
    )
    device: torch.device = model.device

    feature_losses: list[float] = []
    action_losses: list[float] = []
    primary_feature_gradients: list[np.ndarray] = []
    primary_action_gradients: list[np.ndarray] = []
    wrist_feature_gradients: list[np.ndarray] = []
    wrist_action_gradients: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_dir: Path = input_dir / "paired_inputs"

    for state_id in state_ids:
        print(f"[INFO] Target gradient state {state_id}")
        state_record: dict[str, Any] = report_by_state[state_id]
        metrics: dict[str, Any] = state_record["metrics"]
        if "robot_state" not in metrics:
            raise RuntimeError(
                "响应报告缺少 robot_state；请用当前版本重新运行 OFT 响应诊断"
            )
        robot_state: np.ndarray = np.asarray(
            metrics["robot_state"],
            dtype=np.float32,
        )
        if robot_state.shape != (8,):
            raise RuntimeError(f"robot_state 应为 [8]，实际 {robot_state.shape}")

        clean_primary_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_clean_primary.png"
        )
        adversarial_primary_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_adversarial_primary.png"
        )
        clean_wrist_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_clean_wrist.png"
        )
        adversarial_wrist_rgb: np.ndarray = _load_rgb(
            image_dir / f"state_{state_id:02d}_adversarial_wrist.png"
        )
        clean_primary: torch.Tensor = _rgb_tensor(clean_primary_rgb, device)
        clean_wrist: torch.Tensor = _rgb_tensor(clean_wrist_rgb, device)
        adversarial_primary: torch.Tensor = _rgb_tensor(
            adversarial_primary_rgb,
            device,
        ).requires_grad_(True)
        adversarial_wrist: torch.Tensor = _rgb_tensor(
            adversarial_wrist_rgb,
            device,
        ).requires_grad_(True)

        clean_primary_pixels: torch.Tensor = build_fused_pixel_values(
            differentiable_center_crop(clean_primary),
            means=means,
            stds=stds,
        ).to(torch.bfloat16)
        clean_wrist_pixels: torch.Tensor = build_fused_pixel_values(
            differentiable_center_crop(clean_wrist),
            means=means,
            stds=stds,
        ).to(torch.bfloat16)
        adversarial_primary_pixels: torch.Tensor = build_fused_pixel_values(
            differentiable_center_crop(adversarial_primary),
            means=means,
            stds=stds,
        ).to(torch.bfloat16)
        adversarial_wrist_pixels: torch.Tensor = build_fused_pixel_values(
            differentiable_center_crop(adversarial_wrist),
            means=means,
            stds=stds,
        ).to(torch.bfloat16)
        clean_pixels: torch.Tensor = torch.cat(
            [clean_primary_pixels, clean_wrist_pixels], dim=1
        )
        adversarial_pixels: torch.Tensor = torch.cat(
            [adversarial_primary_pixels, adversarial_wrist_pixels], dim=1
        )

        text_inputs: Any = processor(
            prompt,
            Image.fromarray(clean_primary_rgb),
        ).to(device)
        assert cfg.unnorm_key is not None
        proprio_stats: dict[str, Any] = model.norm_stats[cfg.unnorm_key][
            "proprio"
        ]
        normalized_proprio: np.ndarray = normalize_proprio(
            robot_state.copy(),
            proprio_stats,
        )
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            _, clean_hidden = model.predict_action(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=clean_pixels,
                unnorm_key=cfg.unnorm_key,
                proprio=normalized_proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                action_head=action_head,
                use_film=cfg.use_film,
            )
            clean_action: torch.Tensor = action_head.predict_action(
                clean_hidden
            ).detach()
            clean_primary_feature: torch.Tensor = (
                extract_primary_siglip_features(
                    model,
                    clean_primary_pixels,
                ).detach()
            )
            clean_wrist_feature: torch.Tensor = (
                extract_primary_siglip_features(
                    model,
                    clean_wrist_pixels,
                ).detach()
            )
            clean_feature: torch.Tensor = torch.cat(
                [clean_primary_feature, clean_wrist_feature], dim=1
            )

        with autocast(dtype=torch.bfloat16):
            _, adversarial_hidden = model.predict_action(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=adversarial_pixels,
                unnorm_key=cfg.unnorm_key,
                proprio=normalized_proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                action_head=action_head,
                use_film=cfg.use_film,
            )
            adversarial_action: torch.Tensor = action_head.predict_action(
                adversarial_hidden
            )
            adversarial_feature: torch.Tensor = torch.cat(
                [
                    extract_primary_siglip_features(
                        model,
                        adversarial_primary_pixels,
                    ),
                    extract_primary_siglip_features(
                        model,
                        adversarial_wrist_pixels,
                    ),
                ],
                dim=1,
            )
            feature_loss: torch.Tensor = -F.mse_loss(
                adversarial_feature.float(),
                clean_feature.float(),
            )
            action_loss: torch.Tensor = -F.mse_loss(
                adversarial_action.float(),
                clean_action.float(),
            )

        feature_primary_gradient: torch.Tensor
        feature_wrist_gradient: torch.Tensor
        feature_primary_gradient, feature_wrist_gradient = torch.autograd.grad(
            feature_loss,
            (adversarial_primary, adversarial_wrist),
            retain_graph=True,
        )
        action_primary_gradient: torch.Tensor
        action_wrist_gradient: torch.Tensor
        action_primary_gradient, action_wrist_gradient = torch.autograd.grad(
            action_loss,
            (adversarial_primary, adversarial_wrist),
            retain_graph=False,
        )
        feature_losses.append(float(feature_loss.detach().item()))
        action_losses.append(float(action_loss.detach().item()))
        primary_feature_gradients.append(
            tensor_gradient_to_hwc_float64(feature_primary_gradient)
        )
        primary_action_gradients.append(
            tensor_gradient_to_hwc_float64(action_primary_gradient)
        )
        wrist_feature_gradients.append(
            tensor_gradient_to_hwc_float64(feature_wrist_gradient)
        )
        wrist_action_gradients.append(
            tensor_gradient_to_hwc_float64(action_wrist_gradient)
        )
        masks.append(
            visible_perturbation_mask(
                clean_primary_rgb,
                adversarial_primary_rgb,
            )
        )

    artifact = PixelGradientArtifact(
        model_name="openvla_oft",
        checkpoint=str(Path(cfg.pretrained_checkpoint).resolve()),
        state_ids=np.asarray(state_ids, dtype=np.int64),
        feature_losses=np.asarray(feature_losses, dtype=np.float64),
        action_losses=np.asarray(action_losses, dtype=np.float64),
        primary_feature_gradients=np.stack(
            primary_feature_gradients, axis=0
        ),
        primary_action_gradients=np.stack(primary_action_gradients, axis=0),
        perturbation_masks=np.stack(masks, axis=0),
        wrist_feature_gradients=np.stack(wrist_feature_gradients, axis=0),
        wrist_action_gradients=np.stack(wrist_action_gradients, axis=0),
        metadata={
            "reference_point": "trained_k256_adversarial_input",
            "feature_objective": "negative_two_view_siglip_patch_mse",
            "action_objective": "negative_normalized_action_head_mse",
            "crop_area": 0.9,
            "input_directory": str(input_dir),
            "proprio_normalized": True,
        },
    )
    output_path: Path = artifact.save(cfg.output_path)
    print(f"[DONE] Target pixel gradients: {output_path}")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_target_gradient_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
