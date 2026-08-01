"""跨 VLA 像素梯度审计的模型无关数据结构与统计。

OpenVLA 与 OpenVLA-OFT 的动作表示和模型调用方式不同，模型前向分别保留在各自
目录。本模块只定义二者共享的输入坐标、NPZ 产物和方向比较：

- 梯度均关于 center-crop 之前的 RGB 主视角，shape ``[S,H,W,3]``；
- 两个模型都最小化负距离，因此保存的 raw gradient 具有相同的攻击语义；
- 只在攻击纹理实际改变的像素 mask 内比较主要结论，同时保留全图对照。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class PixelGradientAuditError(RuntimeError):
    """像素梯度产物的 shape、语义或数值不满足比较约束。"""


@dataclass(frozen=True)
class PixelGradientArtifact:
    """一个模型在多个固定状态上的 Feature/Action 输入梯度。

    ``primary_*_gradients`` 为 float64 ``[S,H,W,3]``；``perturbation_masks``
    为 bool ``[S,H,W]``。OFT 可额外保存腕部梯度，OpenVLA 对应字段为
    ``None``。loss 均是未加权的负距离标量，梯度下降会增大 clean/adv 差异。
    """

    model_name: str
    checkpoint: str
    state_ids: IntArray
    feature_losses: FloatArray
    action_losses: FloatArray
    primary_feature_gradients: FloatArray
    primary_action_gradients: FloatArray
    perturbation_masks: BoolArray
    wrist_feature_gradients: Optional[FloatArray] = None
    wrist_action_gradients: Optional[FloatArray] = None
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def num_samples(self) -> int:
        return int(self.state_ids.shape[0])

    def validate(self) -> None:
        """验证所有逐状态数组、梯度 shape 与有限值。"""
        if not self.model_name:
            raise PixelGradientAuditError("model_name 不能为空")
        if self.state_ids.ndim != 1 or self.state_ids.size == 0:
            raise PixelGradientAuditError("state_ids 必须是非空一维数组")
        if len(set(self.state_ids.tolist())) != self.num_samples:
            raise PixelGradientAuditError("state_ids 不能重复")
        expected_sample_shape: tuple[int, ...] = (self.num_samples,)
        if self.feature_losses.shape != expected_sample_shape:
            raise PixelGradientAuditError("feature_losses 数量与 state 不一致")
        if self.action_losses.shape != expected_sample_shape:
            raise PixelGradientAuditError("action_losses 数量与 state 不一致")

        feature_shape: tuple[int, ...] = self.primary_feature_gradients.shape
        action_shape: tuple[int, ...] = self.primary_action_gradients.shape
        if (
            len(feature_shape) != 4
            or feature_shape[-1] != 3
            or feature_shape != action_shape
            or feature_shape[0] != self.num_samples
        ):
            raise PixelGradientAuditError(
                "主视角梯度必须具有相同的 [S,H,W,3] shape，实际为 "
                f"{feature_shape} 和 {action_shape}"
            )
        if self.perturbation_masks.shape != feature_shape[:3]:
            raise PixelGradientAuditError(
                "perturbation mask 应为 [S,H,W]，实际为 "
                f"{self.perturbation_masks.shape}"
            )
        for name, wrist_gradient in (
            ("wrist_feature_gradients", self.wrist_feature_gradients),
            ("wrist_action_gradients", self.wrist_action_gradients),
        ):
            if wrist_gradient is not None and wrist_gradient.shape != feature_shape:
                raise PixelGradientAuditError(
                    f"{name} 应与主视角梯度 shape 一致"
                )
        floating_arrays: list[np.ndarray] = [
            self.feature_losses,
            self.action_losses,
            self.primary_feature_gradients,
            self.primary_action_gradients,
        ]
        if self.wrist_feature_gradients is not None:
            floating_arrays.append(self.wrist_feature_gradients)
        if self.wrist_action_gradients is not None:
            floating_arrays.append(self.wrist_action_gradients)
        if not all(np.isfinite(array).all() for array in floating_arrays):
            raise PixelGradientAuditError("像素梯度产物包含 NaN/Inf")
        if not np.any(self.perturbation_masks):
            raise PixelGradientAuditError("所有状态的攻击可见 mask 均为空")

    def save(self, path: str | Path) -> Path:
        """保存压缩 NPZ；可选腕部梯度用零长度数组表达缺失。"""
        self.validate()
        resolved_path: Path = Path(path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json: str = json.dumps(
            dict(self.metadata or {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        np.savez_compressed(
            resolved_path,
            model_name=np.asarray(self.model_name),
            checkpoint=np.asarray(self.checkpoint),
            state_ids=self.state_ids,
            feature_losses=self.feature_losses,
            action_losses=self.action_losses,
            primary_feature_gradients=self.primary_feature_gradients,
            primary_action_gradients=self.primary_action_gradients,
            perturbation_masks=self.perturbation_masks,
            wrist_feature_gradients=(
                self.wrist_feature_gradients
                if self.wrist_feature_gradients is not None
                else np.empty((0,), dtype=np.float64)
            ),
            wrist_action_gradients=(
                self.wrist_action_gradients
                if self.wrist_action_gradients is not None
                else np.empty((0,), dtype=np.float64)
            ),
            metadata_json=np.asarray(metadata_json),
        )
        return resolved_path

    @classmethod
    def load(cls, path: str | Path) -> "PixelGradientArtifact":
        """读取并验证由 :meth:`save` 生成的 NPZ。"""
        resolved_path: Path = Path(path).resolve()
        with np.load(resolved_path, allow_pickle=False) as archive:
            wrist_feature: FloatArray = np.asarray(
                archive["wrist_feature_gradients"],
                dtype=np.float64,
            )
            wrist_action: FloatArray = np.asarray(
                archive["wrist_action_gradients"],
                dtype=np.float64,
            )
            artifact = cls(
                model_name=str(archive["model_name"].item()),
                checkpoint=str(archive["checkpoint"].item()),
                state_ids=np.asarray(archive["state_ids"], dtype=np.int64),
                feature_losses=np.asarray(
                    archive["feature_losses"], dtype=np.float64
                ),
                action_losses=np.asarray(
                    archive["action_losses"], dtype=np.float64
                ),
                primary_feature_gradients=np.asarray(
                    archive["primary_feature_gradients"], dtype=np.float64
                ),
                primary_action_gradients=np.asarray(
                    archive["primary_action_gradients"], dtype=np.float64
                ),
                perturbation_masks=np.asarray(
                    archive["perturbation_masks"], dtype=np.bool_
                ),
                wrist_feature_gradients=(
                    None if wrist_feature.size == 0 else wrist_feature
                ),
                wrist_action_gradients=(
                    None if wrist_action.size == 0 else wrist_action
                ),
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
        artifact.validate()
        return artifact


def differentiable_center_crop(
    images: torch.Tensor,
    *,
    crop_area: float = 0.9,
) -> torch.Tensor:
    """复现 OFT/OpenVLA 的中心裁剪，并保留到输入 RGB 的梯度。

    Args:
        images: float NCHW ``[B,3,H,W]``，值域通常为 ``[0,1]``。
        crop_area: 裁剪面积比例；高宽比例为 ``sqrt(crop_area)``。

    Returns:
        与输入同 shape 的 float tensor。``align_corners=True`` 对应
        TensorFlow ``crop_and_resize`` 以 box 两端为采样中心的坐标定义。
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"center crop 输入应为 [B,3,H,W]，实际 {images.shape}")
    if not 0.0 < crop_area <= 1.0:
        raise ValueError("crop_area 必须位于 (0,1]")
    batch_size, _, height, width = images.shape
    scale: float = float(np.sqrt(crop_area))
    y_coordinates: torch.Tensor = torch.linspace(
        -scale,
        scale,
        steps=height,
        dtype=images.dtype,
        device=images.device,
    )
    x_coordinates: torch.Tensor = torch.linspace(
        -scale,
        scale,
        steps=width,
        dtype=images.dtype,
        device=images.device,
    )
    grid_y, grid_x = torch.meshgrid(
        y_coordinates,
        x_coordinates,
        indexing="ij",
    )
    grid: torch.Tensor = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def build_fused_pixel_values(
    rgb_images: torch.Tensor,
    *,
    means: Sequence[Sequence[float]],
    stds: Sequence[Sequence[float]],
) -> torch.Tensor:
    """按 checkpoint processor 顺序把 RGB 映射为 fused 视觉输入。

    ``rgb_images`` 为 float ``[B,3,H,W]``；返回 ``[B,3*branches,H,W]``。
    每个 mean/std 必须含三个通道。该函数不猜测 DINO/SigLIP 顺序，调用方直接
    传入 processor 保存的顺序。
    """
    if rgb_images.ndim != 4 or rgb_images.shape[1] != 3:
        raise ValueError("RGB 输入必须是 [B,3,H,W]")
    if len(means) == 0 or len(means) != len(stds):
        raise ValueError("means/stds 分支数量必须相同且非零")
    normalized_branches: list[torch.Tensor] = []
    for branch_mean, branch_std in zip(means, stds):
        if len(branch_mean) != 3 or len(branch_std) != 3:
            raise ValueError("每个视觉分支必须提供三个通道的 mean/std")
        mean_tensor: torch.Tensor = rgb_images.new_tensor(branch_mean).view(
            1, 3, 1, 1
        )
        std_tensor: torch.Tensor = rgb_images.new_tensor(branch_std).view(
            1, 3, 1, 1
        )
        if torch.any(std_tensor <= 0):
            raise ValueError("视觉归一化 std 必须为正数")
        normalized_branches.append((rgb_images - mean_tensor) / std_tensor)
    return torch.cat(normalized_branches, dim=1)


def tensor_gradient_to_hwc_float64(gradient: torch.Tensor) -> FloatArray:
    """把有限的 ``[1,3,H,W]`` tensor 转为 float64 ``[H,W,3]``。"""
    if gradient.ndim != 4 or gradient.shape[0] != 1 or gradient.shape[1] != 3:
        raise PixelGradientAuditError(
            f"像素梯度应为 [1,3,H,W]，实际 {tuple(gradient.shape)}"
        )
    if not torch.isfinite(gradient).all():
        raise PixelGradientAuditError("像素梯度包含 NaN/Inf")
    return np.asarray(
        gradient.detach().float().cpu().permute(0, 2, 3, 1)[0].numpy(),
        dtype=np.float64,
    )


def visible_perturbation_mask(
    clean_rgb: np.ndarray,
    adversarial_rgb: np.ndarray,
    *,
    threshold: int = 1,
) -> BoolArray:
    """从成对 uint8 RGB 构造攻击实际可见区域 ``[H,W]``。"""
    clean: np.ndarray = np.asarray(clean_rgb)
    adversarial: np.ndarray = np.asarray(adversarial_rgb)
    if clean.shape != adversarial.shape or clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError("clean/adv RGB 必须具有相同的 [H,W,3] shape")
    absolute_difference: np.ndarray = np.abs(
        adversarial.astype(np.int16) - clean.astype(np.int16)
    )
    return np.asarray(
        np.max(absolute_difference, axis=2) > threshold,
        dtype=np.bool_,
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_flat: np.ndarray = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat: np.ndarray = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator: float = float(
        np.linalg.norm(first_flat) * np.linalg.norm(second_flat)
    )
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.clip(np.dot(first_flat, second_flat) / denominator, -1, 1))


def _masked_gradient(gradient: FloatArray, mask: BoolArray) -> FloatArray:
    return np.asarray(gradient[mask], dtype=np.float64)


def _norm_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """返回两个梯度范数之比；零分母时用0表示没有可比较信号。"""
    denominator_norm: float = float(np.linalg.norm(denominator))
    if denominator_norm <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.linalg.norm(numerator) / denominator_norm)


def compare_pixel_gradient_artifacts(
    source: PixelGradientArtifact,
    target: PixelGradientArtifact,
    *,
    source_action_weight: float = 0.1,
    source_feature_weight: float = 4.0,
) -> dict[str, Any]:
    """比较同状态、同输入坐标上的 source/target 梯度方向。"""
    source.validate()
    target.validate()
    if not np.array_equal(source.state_ids, target.state_ids):
        raise PixelGradientAuditError("source/target state IDs 不一致")
    if source.primary_feature_gradients.shape != target.primary_feature_gradients.shape:
        raise PixelGradientAuditError("source/target 主视角梯度 shape 不一致")
    if not np.array_equal(source.perturbation_masks, target.perturbation_masks):
        raise PixelGradientAuditError("source/target perturbation mask 不一致")
    if not np.isfinite(source_action_weight) or not np.isfinite(
        source_feature_weight
    ):
        raise PixelGradientAuditError("source objective 权重必须有限")
    if source_action_weight < 0.0 or source_feature_weight < 0.0:
        raise PixelGradientAuditError("source objective 权重不能为负数")
    if source_action_weight == 0.0 and source_feature_weight == 0.0:
        raise PixelGradientAuditError("source objective 权重不能同时为0")

    state_records: list[dict[str, Any]] = []
    for sample_index, state_id in enumerate(source.state_ids.tolist()):
        mask: BoolArray = source.perturbation_masks[sample_index]
        source_feature: FloatArray = source.primary_feature_gradients[
            sample_index
        ]
        target_feature: FloatArray = target.primary_feature_gradients[
            sample_index
        ]
        source_action: FloatArray = source.primary_action_gradients[sample_index]
        target_action: FloatArray = target.primary_action_gradients[sample_index]
        source_weighted_objective: FloatArray = (
            source_action_weight * source_action
            + source_feature_weight * source_feature
        )
        state_record: dict[str, Any] = {
                "state_id": int(state_id),
                "visible_pixel_count": int(np.count_nonzero(mask)),
                "cross_model_feature_cosine_masked": _cosine(
                    _masked_gradient(source_feature, mask),
                    _masked_gradient(target_feature, mask),
                ),
                "cross_model_action_cosine_masked": _cosine(
                    _masked_gradient(source_action, mask),
                    _masked_gradient(target_action, mask),
                ),
                "source_feature_target_action_cosine_masked": _cosine(
                    _masked_gradient(source_feature, mask),
                    _masked_gradient(target_action, mask),
                ),
                "source_action_target_feature_cosine_masked": _cosine(
                    _masked_gradient(source_action, mask),
                    _masked_gradient(target_feature, mask),
                ),
                "source_weighted_objective_target_action_cosine_masked": (
                    _cosine(
                        _masked_gradient(source_weighted_objective, mask),
                        _masked_gradient(target_action, mask),
                    )
                ),
                "source_feature_action_cosine_masked": _cosine(
                    _masked_gradient(source_feature, mask),
                    _masked_gradient(source_action, mask),
                ),
                "target_feature_action_cosine_masked": _cosine(
                    _masked_gradient(target_feature, mask),
                    _masked_gradient(target_action, mask),
                ),
                "cross_model_feature_cosine_full": _cosine(
                    source_feature,
                    target_feature,
                ),
                "cross_model_action_cosine_full": _cosine(
                    source_action,
                    target_action,
                ),
        }
        if (
            target.wrist_feature_gradients is not None
            and target.wrist_action_gradients is not None
        ):
            state_record.update(
                {
                    "target_wrist_to_primary_feature_norm_ratio": _norm_ratio(
                        target.wrist_feature_gradients[sample_index],
                        target_feature,
                    ),
                    "target_wrist_to_primary_action_norm_ratio": _norm_ratio(
                        target.wrist_action_gradients[sample_index],
                        target_action,
                    ),
                }
            )
        state_records.append(state_record)

    metric_names: tuple[str, ...] = tuple(
        key
        for key in state_records[0]
        if key not in {"state_id", "visible_pixel_count"}
    )
    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values: list[float] = [
            float(record[metric_name]) for record in state_records
        ]
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "schema_version": 1,
        "source_model": source.model_name,
        "target_model": target.model_name,
        "source_checkpoint": source.checkpoint,
        "target_checkpoint": target.checkpoint,
        "source_objective_weights": {
            "action": source_action_weight,
            "feature": source_feature_weight,
        },
        "state_ids": source.state_ids.tolist(),
        "states": state_records,
        "aggregate": aggregate,
    }
