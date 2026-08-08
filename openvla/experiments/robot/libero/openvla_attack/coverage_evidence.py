"""Support coverage 的连续、非负、预乘 evidence 空间变换。

RGB deployment path 使用 Pillow bicubic 与 uint8 exact forward；coverage 证据
具有不同的数值语义。每个 MuJoCo 实例的可见权重 ``alpha_k`` 以及预乘控制量
``alpha_k * w_{k,S}`` 必须先在 512 Policy Source 上构造，再共同执行非负 area
downsampling 和与部署一致的 center-crop 几何。只有完成变换后才能跨实例求和。

本模块不读取 segmentation、mesh 或 support，也不导入 LIBERO/nvdiffrast。它只
保护以下不变量：有限、``0 <= alpha*w <= alpha <= 1``、跨实例
``sum_k alpha_k <= 1``，以及零/全控制和 support 单调性。任何阶段违反约束都
fail-fast；实现不会用事后 clamp 隐藏上游对应错误。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Optional

import torch
import torch.nn.functional as torch_functional

from experiments.robot.openvla_image_transform import (
    CenterCropSpecification,
    torch_center_crop_float,
)

from .policy_view import (
    POLICY_PRE_CROP_RESOLUTION,
    POLICY_SOURCE_RESOLUTION,
)


COVERAGE_INVARIANT_ATOL: Final[float] = 1e-6
COVERAGE_DENOMINATOR_EPSILON: Final[float] = 1e-12


@dataclass(frozen=True)
class CoverageEvidenceSpecification:
    """连续 evidence 的 source→pre-crop→effective 空间契约。"""

    source_resolution: int = POLICY_SOURCE_RESOLUTION
    pre_crop_resolution: int = POLICY_PRE_CROP_RESOLUTION
    center_crop: CenterCropSpecification = field(
        default_factory=CenterCropSpecification
    )
    invariant_atol: float = COVERAGE_INVARIANT_ATOL

    def __post_init__(self) -> None:
        if self.source_resolution <= 0 or self.pre_crop_resolution <= 0:
            raise ValueError("coverage evidence resolution 必须为正数")
        if self.center_crop.input_resolution != self.pre_crop_resolution:
            raise ValueError(
                "coverage pre-crop resolution 与 center-crop input 不一致"
            )
        if not math.isfinite(self.invariant_atol) or self.invariant_atol < 0:
            raise ValueError("coverage invariant_atol 必须为有限非负数")


@dataclass(frozen=True)
class PremultipliedEvidenceStage:
    """一个空间阶段的逐实例 alpha 与预乘控制量。

    两个 tensor 均为 float32 ``[num_instances, 1, height, width]``；batch 维
    表示共享 Active Texture 的物体实例，不能解释为普通样本 batch。
    """

    alpha: torch.Tensor
    alpha_times_control: torch.Tensor


@dataclass(frozen=True)
class PremultipliedEvidenceStages:
    """source、area-downsampled pre-crop 与 effective 三阶段证据。"""

    source: PremultipliedEvidenceStage
    pre_crop: PremultipliedEvidenceStage
    effective: PremultipliedEvidenceStage


@dataclass(frozen=True)
class CoverageSummary:
    """变换后按实际可见投影质量加权的 coverage 汇总。"""

    coverage: Optional[float]
    numerator: float
    equivalent_visible_pixels: float
    observation_area: float
    per_instance_coverage: tuple[Optional[float], ...]
    per_instance_equivalent_visible_pixels: tuple[float, ...]


def _validate_stage(
    stage: PremultipliedEvidenceStage,
    *,
    expected_resolution: int,
    invariant_atol: float,
    stage_name: str,
) -> None:
    """校验一个阶段的 shape、有限性、预乘偏序和实例 union 上界。"""

    alpha = stage.alpha
    premultiplied = stage.alpha_times_control
    if not isinstance(alpha, torch.Tensor) or not isinstance(
        premultiplied,
        torch.Tensor,
    ):
        raise TypeError(f"{stage_name} coverage evidence 必须是 torch.Tensor")
    if alpha.dtype != torch.float32 or premultiplied.dtype != torch.float32:
        raise TypeError(f"{stage_name} coverage evidence 必须使用 float32")
    if alpha.device != premultiplied.device:
        raise ValueError(f"{stage_name} alpha/premultiplied device 不一致")
    if alpha.shape != premultiplied.shape:
        raise ValueError(f"{stage_name} alpha/premultiplied shape 不一致")
    expected_tail = (1, expected_resolution, expected_resolution)
    if alpha.ndim != 4 or tuple(alpha.shape[1:]) != expected_tail:
        raise ValueError(
            f"{stage_name} coverage evidence shape 必须为 "
            f"[num_instances,{expected_tail[0]},{expected_tail[1]},"
            f"{expected_tail[2]}]，收到 {tuple(alpha.shape)}"
        )
    if alpha.shape[0] <= 0:
        raise ValueError(f"{stage_name} coverage evidence 不得没有实例")
    if not bool(torch.isfinite(alpha).all()) or not bool(
        torch.isfinite(premultiplied).all()
    ):
        raise ValueError(f"{stage_name} coverage evidence 包含 NaN/Inf")
    if bool((alpha < -invariant_atol).any()) or bool(
        (alpha > 1.0 + invariant_atol).any()
    ):
        raise ValueError(f"{stage_name} alpha 超出 [0,1] 容差")
    if bool((premultiplied < -invariant_atol).any()) or bool(
        (premultiplied > alpha + invariant_atol).any()
    ):
        raise ValueError(
            f"{stage_name} 未满足 0 <= alpha*control <= alpha"
        )
    union_alpha = alpha.sum(dim=0)
    if bool((union_alpha > 1.0 + invariant_atol).any()):
        raise ValueError(f"{stage_name} 多实例 alpha 不是互斥可见分区")


def transform_premultiplied_evidence(
    alpha: torch.Tensor,
    alpha_times_control: torch.Tensor,
    *,
    specification: CoverageEvidenceSpecification = (
        CoverageEvidenceSpecification()
    ),
) -> PremultipliedEvidenceStages:
    """共同变换逐实例 ``alpha`` 与 ``alpha*control``，全程检查偏序。

    Args:
        alpha: float32 ``[num_instances,1,Hsource,Wsource]``，表示 MuJoCo
            front-most 可见权重。
        alpha_times_control: 同 shape 预乘量；coverage 时 control 为 renderer
            barycentric ``w_S``，其他 alignment 诊断可传对应非负控制量。
        specification: 固定 source/pre-crop/crop 几何和容差。

    Returns:
        三阶段 tensor。返回值保留输入的 device；area/crop 均为确定性浮点算子，
        不执行 uint8 量化、Normalize 或 clamp。
    """

    source = PremultipliedEvidenceStage(alpha, alpha_times_control)
    _validate_stage(
        source,
        expected_resolution=specification.source_resolution,
        invariant_atol=specification.invariant_atol,
        stage_name="source",
    )

    output_size = (
        specification.pre_crop_resolution,
        specification.pre_crop_resolution,
    )
    pre_crop = PremultipliedEvidenceStage(
        alpha=torch_functional.interpolate(
            alpha,
            size=output_size,
            mode="area",
        ),
        alpha_times_control=torch_functional.interpolate(
            alpha_times_control,
            size=output_size,
            mode="area",
        ),
    )
    _validate_stage(
        pre_crop,
        expected_resolution=specification.pre_crop_resolution,
        invariant_atol=specification.invariant_atol,
        stage_name="pre_crop",
    )

    effective = PremultipliedEvidenceStage(
        alpha=torch_center_crop_float(
            pre_crop.alpha,
            specification=specification.center_crop,
        ),
        alpha_times_control=torch_center_crop_float(
            pre_crop.alpha_times_control,
            specification=specification.center_crop,
        ),
    )
    _validate_stage(
        effective,
        expected_resolution=specification.center_crop.output_resolution,
        invariant_atol=specification.invariant_atol,
        stage_name="effective",
    )
    return PremultipliedEvidenceStages(
        source=source,
        pre_crop=pre_crop,
        effective=effective,
    )


def summarize_effective_coverage(
    effective: PremultipliedEvidenceStage,
    *,
    invariant_atol: float = COVERAGE_INVARIANT_ATOL,
    denominator_epsilon: float = COVERAGE_DENOMINATOR_EPSILON,
) -> CoverageSummary:
    """在 effective view 中按实例可见面积加权汇总 coverage。

    ``coverage=None`` 明确表示 union 观测为零；调用方随后结合冻结的
    ``A_obs_min`` 与 alignment recall 判定四类 Visibility Evidence Status。
    本函数不自行猜测状态或把缺失观测解释为 coverage=0。
    """

    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("coverage denominator_epsilon 必须为有限正数")
    output_resolution = int(effective.alpha.shape[-1])
    _validate_stage(
        effective,
        expected_resolution=output_resolution,
        invariant_atol=invariant_atol,
        stage_name="effective_summary",
    )

    instance_denominators = effective.alpha.sum(dim=(1, 2, 3))
    instance_numerators = effective.alpha_times_control.sum(
        dim=(1, 2, 3)
    )
    per_instance_coverage = tuple(
        (
            float(
                numerator.item()
                / (denominator.item() + denominator_epsilon)
            )
            if float(denominator.item()) > 0.0
            else None
        )
        for numerator, denominator in zip(
            instance_numerators,
            instance_denominators,
            strict=True,
        )
    )
    numerator = float(instance_numerators.sum().item())
    equivalent_visible_pixels = float(instance_denominators.sum().item())
    coverage = (
        numerator / (equivalent_visible_pixels + denominator_epsilon)
        if equivalent_visible_pixels > 0.0
        else None
    )
    height = int(effective.alpha.shape[-2])
    width = int(effective.alpha.shape[-1])
    return CoverageSummary(
        coverage=coverage,
        numerator=numerator,
        equivalent_visible_pixels=equivalent_visible_pixels,
        observation_area=equivalent_visible_pixels / float(height * width),
        per_instance_coverage=per_instance_coverage,
        per_instance_equivalent_visible_pixels=tuple(
            float(value.item()) for value in instance_denominators
        ),
    )
