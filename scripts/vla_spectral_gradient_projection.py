"""跨模型像素梯度投影到共同谱系数空间的数据结构与统计。

模型前向已经在像素梯度阶段完成。本模块不依赖 LIBERO、nvdiffrast 或具体
VLA，只保存 renderer VJP 的结果并比较方向：

``pixel gradient [H,W,3] --J_renderer^T--> coefficient gradient [K,3]``。

OFT 的主视角与腕部使用同一组物理谱系数，所以目标模型的总系数梯度是两视角
VJP 之和。它只用于机制诊断；source-only 训练不得读取该目标梯度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import torch
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class SpectralProjectionError(RuntimeError):
    """谱系数梯度产物的 shape、状态或数值不满足约束。"""


class SpectralVJPRenderer(Protocol):
    """多实例 renderer VJP 所需的最小强类型接口。"""

    def get_texture_param(self) -> torch.Tensor:
        """返回共享纹理参数，shape ``[K,3]``。"""
        ...

    def render(
        self,
        mvp: torch.Tensor,
        resolution: tuple[int, int],
        *,
        model_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 RGB ``[1,H,W,3]`` 与 visibility ``[1,H,W,1]``。"""
        ...


@dataclass(frozen=True)
class RenderInstance:
    """一个共享纹理物体实例的相机变换。

    ``mvp`` shape 为 ``[4,4]``；``model_rotation`` shape 为 ``[3,3]``。
    不同实例的变换不同，但 renderer 内的 ``[K,3]`` 参数是同一个 tensor。
    """

    mvp: torch.Tensor
    model_rotation: torch.Tensor


def _pixel_gradient_tensor(
    gradient: FloatArray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """float HWC ``[H,W,3]`` → renderer device 上的 float32 NHWC。"""
    array: np.ndarray = np.asarray(gradient, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SpectralProjectionError(
            f"像素梯度应为 [H,W,3]，实际 {array.shape}"
        )
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).unsqueeze(0)


def project_scene_pixel_gradients(
    *,
    renderer: SpectralVJPRenderer,
    instances: Sequence[RenderInstance],
    pixel_gradients: Sequence[FloatArray],
) -> tuple[list[FloatArray], BoolArray]:
    """把一个共享纹理资产的全部可见实例投影到同一参数空间。

    对每个实例分别 rasterize，再把 RGB 对共享系数的 Jacobian 相加。若场景中
    同一 PNG 被两个 bowl 引用，这正是直接激活 PNG 时的物理数据流；只渲染第一
    个 body 会漏掉另一个 bowl 的 ``dL/dC``。

    Args:
        renderer: 所有实例共享同一 ``[K,3]`` 参数的可微 renderer。
        instances: 各物体实例的 MVP 和世界旋转；不能为空。
        pixel_gradients: 同一相机输入上的一组 ``[H,W,3]`` 梯度。

    Returns:
        每个像素梯度对应的 float64 ``[K,3]`` VJP，以及全部实例 visibility
        的并集 bool ``[H,W]``。
    """
    if not instances:
        raise SpectralProjectionError("render instances 不能为空")
    if not pixel_gradients:
        raise SpectralProjectionError("pixel_gradients 不能为空")
    height, width, channels = pixel_gradients[0].shape
    expected_pixel_shape: tuple[int, int, int] = (height, width, 3)
    if channels != 3 or any(
        gradient.shape != expected_pixel_shape for gradient in pixel_gradients
    ):
        raise SpectralProjectionError(
            "同一视角的像素梯度必须共享 [H,W,3] shape"
        )

    rendered_images: list[torch.Tensor] = []
    visibility_masks: list[torch.Tensor] = []
    for instance in instances:
        rendered_rgb: torch.Tensor
        visibility: torch.Tensor
        rendered_rgb, visibility = renderer.render(
            instance.mvp,
            resolution=(height, width),
            model_rot=instance.model_rotation,
        )
        if rendered_rgb.shape != (1, height, width, 3):
            raise SpectralProjectionError(
                f"renderer RGB shape 非法: {tuple(rendered_rgb.shape)}"
            )
        if visibility.shape != (1, height, width, 1):
            raise SpectralProjectionError(
                f"renderer mask shape 非法: {tuple(visibility.shape)}"
            )
        rendered_images.append(rendered_rgb)
        visibility_masks.append(visibility > 0.5)

    # 同一个纹理参数在每个物体实例中重复出现，所以对 RGB 求和再做一次 VJP
    # 等价于逐实例计算 J_i^T g 后求和；不会复制或平均共享系数梯度。
    combined_rendered_rgb: torch.Tensor = torch.stack(
        rendered_images,
        dim=0,
    ).sum(dim=0)
    combined_visibility: torch.Tensor = torch.stack(
        visibility_masks,
        dim=0,
    ).any(dim=0)
    parameter: torch.Tensor = renderer.get_texture_param()
    coefficient_gradients: list[FloatArray] = []
    for gradient_index, pixel_gradient in enumerate(pixel_gradients):
        coefficient_gradient: torch.Tensor = torch.autograd.grad(
            outputs=combined_rendered_rgb,
            inputs=parameter,
            grad_outputs=_pixel_gradient_tensor(
                pixel_gradient,
                device=combined_rendered_rgb.device,
            ),
            retain_graph=gradient_index < len(pixel_gradients) - 1,
            create_graph=False,
        )[0]
        coefficient_gradients.append(
            np.asarray(
                coefficient_gradient.detach().double().cpu().numpy(),
                dtype=np.float64,
            )
        )
    visibility_mask: BoolArray = np.asarray(
        combined_visibility.detach().cpu().numpy()[0, ..., 0],
        dtype=np.bool_,
    )
    return coefficient_gradients, visibility_mask


@dataclass(frozen=True)
class SpectralGradientProjectionArtifact:
    """同一批状态上的 source/target 谱系数梯度。

    六个梯度数组均为 float64 ``[S,K,3]``。renderer/observed mask 为 bool
    ``[S,H,W]``，只用于验证可微轮廓是否覆盖真实纹理产生变化的位置。
    """

    state_ids: IntArray
    source_primary_feature: FloatArray
    source_primary_action: FloatArray
    target_primary_feature: FloatArray
    target_primary_action: FloatArray
    target_wrist_feature: FloatArray
    target_wrist_action: FloatArray
    primary_renderer_masks: BoolArray
    wrist_renderer_masks: BoolArray
    primary_observed_masks: BoolArray
    wrist_observed_masks: BoolArray
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def num_samples(self) -> int:
        return int(self.state_ids.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.source_primary_feature.shape[1])

    def validate(self) -> None:
        """验证状态、``[S,K,3]`` 梯度、``[S,H,W]`` mask 与有限值。"""
        if self.state_ids.ndim != 1 or self.state_ids.size == 0:
            raise SpectralProjectionError("state_ids 必须是非空一维数组")
        if len(set(self.state_ids.tolist())) != self.num_samples:
            raise SpectralProjectionError("state_ids 不能重复")
        expected_gradient_shape: tuple[int, int, int] = (
            self.num_samples,
            self.num_basis,
            3,
        )
        if self.num_basis <= 0:
            raise SpectralProjectionError("谱基数量必须为正数")
        gradient_arrays: tuple[FloatArray, ...] = (
            self.source_primary_feature,
            self.source_primary_action,
            self.target_primary_feature,
            self.target_primary_action,
            self.target_wrist_feature,
            self.target_wrist_action,
        )
        if any(array.shape != expected_gradient_shape for array in gradient_arrays):
            raise SpectralProjectionError(
                "所有系数梯度必须具有相同 [S,K,3] shape，期望 "
                f"{expected_gradient_shape}"
            )
        if not all(np.isfinite(array).all() for array in gradient_arrays):
            raise SpectralProjectionError("系数梯度包含 NaN/Inf")

        mask_arrays: tuple[BoolArray, ...] = (
            self.primary_renderer_masks,
            self.wrist_renderer_masks,
            self.primary_observed_masks,
            self.wrist_observed_masks,
        )
        mask_shape: tuple[int, ...] = self.primary_renderer_masks.shape
        if len(mask_shape) != 3 or mask_shape[0] != self.num_samples:
            raise SpectralProjectionError("renderer mask 必须为 [S,H,W]")
        if any(array.shape != mask_shape for array in mask_arrays):
            raise SpectralProjectionError("四组 mask 的 [S,H,W] shape 必须相同")
        if not np.any(self.primary_renderer_masks):
            raise SpectralProjectionError("所有状态的主视角 renderer mask 均为空")

    def save(self, path: str | Path) -> Path:
        """保存压缩 NPZ；模型 checkpoint 不重复写入本产物。"""
        self.validate()
        resolved_path: Path = Path(path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved_path,
            state_ids=self.state_ids,
            source_primary_feature=self.source_primary_feature,
            source_primary_action=self.source_primary_action,
            target_primary_feature=self.target_primary_feature,
            target_primary_action=self.target_primary_action,
            target_wrist_feature=self.target_wrist_feature,
            target_wrist_action=self.target_wrist_action,
            primary_renderer_masks=self.primary_renderer_masks,
            wrist_renderer_masks=self.wrist_renderer_masks,
            primary_observed_masks=self.primary_observed_masks,
            wrist_observed_masks=self.wrist_observed_masks,
            metadata_json=np.asarray(
                json.dumps(
                    dict(self.metadata or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        )
        return resolved_path

    @classmethod
    def load(cls, path: str | Path) -> "SpectralGradientProjectionArtifact":
        """读取并验证 :meth:`save` 生成的 NPZ。"""
        with np.load(Path(path).resolve(), allow_pickle=False) as archive:
            artifact = cls(
                state_ids=np.asarray(archive["state_ids"], dtype=np.int64),
                source_primary_feature=np.asarray(
                    archive["source_primary_feature"], dtype=np.float64
                ),
                source_primary_action=np.asarray(
                    archive["source_primary_action"], dtype=np.float64
                ),
                target_primary_feature=np.asarray(
                    archive["target_primary_feature"], dtype=np.float64
                ),
                target_primary_action=np.asarray(
                    archive["target_primary_action"], dtype=np.float64
                ),
                target_wrist_feature=np.asarray(
                    archive["target_wrist_feature"], dtype=np.float64
                ),
                target_wrist_action=np.asarray(
                    archive["target_wrist_action"], dtype=np.float64
                ),
                primary_renderer_masks=np.asarray(
                    archive["primary_renderer_masks"], dtype=np.bool_
                ),
                wrist_renderer_masks=np.asarray(
                    archive["wrist_renderer_masks"], dtype=np.bool_
                ),
                primary_observed_masks=np.asarray(
                    archive["primary_observed_masks"], dtype=np.bool_
                ),
                wrist_observed_masks=np.asarray(
                    archive["wrist_observed_masks"], dtype=np.bool_
                ),
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
        artifact.validate()
        return artifact


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_flat: FloatArray = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat: FloatArray = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator: float = float(
        np.linalg.norm(first_flat) * np.linalg.norm(second_flat)
    )
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(
        np.clip(np.dot(first_flat, second_flat) / denominator, -1.0, 1.0)
    )


def _norm_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_norm: float = float(np.linalg.norm(denominator))
    if denominator_norm <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.linalg.norm(numerator) / denominator_norm)


def _mask_overlap(
    renderer_mask: BoolArray,
    observed_mask: BoolArray,
) -> dict[str, float]:
    """比较 renderer 可见轮廓与真实 clean/adv 变化区域。

    observed mask 不是完整物体分割，因此重点看 ``observed_recall``：真实发生纹理
    变化的像素有多少落在 renderer Jacobian 可达区域中。
    """
    intersection: int = int(np.count_nonzero(renderer_mask & observed_mask))
    union: int = int(np.count_nonzero(renderer_mask | observed_mask))
    renderer_count: int = int(np.count_nonzero(renderer_mask))
    observed_count: int = int(np.count_nonzero(observed_mask))
    return {
        "iou": float(intersection / union) if union else 0.0,
        "observed_recall": (
            float(intersection / observed_count) if observed_count else 0.0
        ),
        "renderer_precision": (
            float(intersection / renderer_count) if renderer_count else 0.0
        ),
    }


def _mode_energy_summary(gradients: FloatArray) -> dict[str, Any]:
    """汇总 ``[S,K,3]`` 梯度在谱模态上的平均平方能量。"""
    energy: FloatArray = np.mean(np.sum(np.square(gradients), axis=2), axis=0)
    total: float = float(np.sum(energy))
    if total <= np.finfo(np.float64).tiny:
        normalized: FloatArray = np.zeros_like(energy)
        effective_count: float = 0.0
    else:
        normalized = energy / total
        effective_count = float(1.0 / np.sum(np.square(normalized)))
    top_count: int = min(10, normalized.size)
    top_indices: IntArray = np.argsort(normalized)[::-1][:top_count].astype(
        np.int64
    )
    cutoffs: tuple[int, ...] = tuple(
        value for value in (32, 64, 128, 256) if value <= normalized.size
    )
    return {
        "effective_basis_count": effective_count,
        "prefix_energy_fraction": {
            str(cutoff): float(np.sum(normalized[:cutoff]))
            for cutoff in cutoffs
        },
        "top_basis_indices_zero_based": top_indices.tolist(),
        "top_basis_energy_fraction": normalized[top_indices].tolist(),
    }


def summarize_spectral_projection(
    artifact: SpectralGradientProjectionArtifact,
    *,
    source_action_weight: float,
    source_feature_weight: float,
) -> dict[str, Any]:
    """比较 source、target 主视角及 target 双视角的谱系数方向。"""
    artifact.validate()
    weights: tuple[float, float] = (
        float(source_action_weight),
        float(source_feature_weight),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in weights):
        raise SpectralProjectionError("source objective 权重必须为有限非负数")
    if weights == (0.0, 0.0):
        raise SpectralProjectionError("source objective 权重不能同时为0")

    source_weighted: FloatArray = (
        source_action_weight * artifact.source_primary_action
        + source_feature_weight * artifact.source_primary_feature
    )
    target_combined_feature: FloatArray = (
        artifact.target_primary_feature + artifact.target_wrist_feature
    )
    target_combined_action: FloatArray = (
        artifact.target_primary_action + artifact.target_wrist_action
    )

    state_records: list[dict[str, Any]] = []
    for sample_index, state_id in enumerate(artifact.state_ids.tolist()):
        source_feature: FloatArray = artifact.source_primary_feature[sample_index]
        source_action: FloatArray = artifact.source_primary_action[sample_index]
        target_primary_feature: FloatArray = artifact.target_primary_feature[
            sample_index
        ]
        target_primary_action: FloatArray = artifact.target_primary_action[
            sample_index
        ]
        target_wrist_feature: FloatArray = artifact.target_wrist_feature[
            sample_index
        ]
        target_wrist_action: FloatArray = artifact.target_wrist_action[sample_index]
        state_records.append(
            {
                "state_id": int(state_id),
                "cross_model_primary_feature_cosine": _cosine(
                    source_feature, target_primary_feature
                ),
                "cross_model_primary_action_cosine": _cosine(
                    source_action, target_primary_action
                ),
                "source_feature_target_combined_action_cosine": _cosine(
                    source_feature, target_combined_action[sample_index]
                ),
                "source_weighted_target_primary_action_cosine": _cosine(
                    source_weighted[sample_index], target_primary_action
                ),
                "source_weighted_target_combined_action_cosine": _cosine(
                    source_weighted[sample_index],
                    target_combined_action[sample_index],
                ),
                "source_feature_action_cosine": _cosine(
                    source_feature, source_action
                ),
                "target_combined_feature_action_cosine": _cosine(
                    target_combined_feature[sample_index],
                    target_combined_action[sample_index],
                ),
                "target_wrist_primary_feature_norm_ratio": _norm_ratio(
                    target_wrist_feature, target_primary_feature
                ),
                "target_wrist_primary_action_norm_ratio": _norm_ratio(
                    target_wrist_action, target_primary_action
                ),
                "primary_mask_overlap": _mask_overlap(
                    artifact.primary_renderer_masks[sample_index],
                    artifact.primary_observed_masks[sample_index],
                ),
                "wrist_mask_overlap": _mask_overlap(
                    artifact.wrist_renderer_masks[sample_index],
                    artifact.wrist_observed_masks[sample_index],
                ),
            }
        )

    flat_metric_names: tuple[str, ...] = (
        "cross_model_primary_feature_cosine",
        "cross_model_primary_action_cosine",
        "source_feature_target_combined_action_cosine",
        "source_weighted_target_primary_action_cosine",
        "source_weighted_target_combined_action_cosine",
        "source_feature_action_cosine",
        "target_combined_feature_action_cosine",
        "target_wrist_primary_feature_norm_ratio",
        "target_wrist_primary_action_norm_ratio",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in flat_metric_names:
        values: list[float] = [
            float(record[metric_name]) for record in state_records
        ]
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    for view_name in ("primary", "wrist"):
        for overlap_name in ("iou", "observed_recall", "renderer_precision"):
            values = [
                float(record[f"{view_name}_mask_overlap"][overlap_name])
                for record in state_records
            ]
            aggregate[f"{view_name}_mask_{overlap_name}"] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return {
        "schema_version": 1,
        "diagnostic": "cross_model_spectral_coefficient_projection",
        "target_gradient_role": "diagnostic_only_not_training_or_selection",
        "state_ids": artifact.state_ids.tolist(),
        "num_basis": artifact.num_basis,
        "source_objective_weights": {
            "action": source_action_weight,
            "feature": source_feature_weight,
        },
        "states": state_records,
        "aggregate": aggregate,
        "mode_energy": {
            "source_feature": _mode_energy_summary(
                artifact.source_primary_feature
            ),
            "source_action": _mode_energy_summary(
                artifact.source_primary_action
            ),
            "source_weighted": _mode_energy_summary(source_weighted),
            "target_combined_feature": _mode_energy_summary(
                target_combined_feature
            ),
            "target_combined_action": _mode_energy_summary(
                target_combined_action
            ),
        },
        "metadata": dict(artifact.metadata or {}),
    }
