"""OpenVLA 部署路径中的 Policy Pre-Crop Canvas 纯计算契约。

录像帧和 policy 输入可以来自同一个 MuJoCo observation，但两者的分辨率属于
不同职责。正式 Akita 路径固定先生成 512×512 policy-source RGB，再以显式
Pillow bicubic 缩放为 checkpoint 期望的 224×224 uint8 canvas；录像分辨率变化
不得隐式改变该结果。

本模块不导入 LIBERO 或模型，可以在 WSL CPU 环境中独立测试。纯 numpy helper
固定 exact uint8 stages；:class:`DifferentiablePolicyViewTransform` 再将同一
Pillow/TensorFlow forward 与 PyTorch surrogate 组合为 BPDA，供后续 collector、
coverage 与 Attack Training 共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
import torch.nn.functional as torch_functional
from numpy.typing import NDArray
from PIL import Image

from experiments.robot.openvla_image_transform import (
    CenterCropSpecification,
    deployment_center_crop_uint8,
    torch_center_crop_float,
)


POLICY_SOURCE_RESOLUTION: Final[int] = 512
POLICY_PRE_CROP_RESOLUTION: Final[int] = 224


def build_policy_view_transform(
    *,
    source_resolution: int,
    model_input_resolution: int,
) -> "DifferentiablePolicyViewTransform":
    """按显式 source/model 尺寸构造统一 deployment transform。

    正式运行固定为 ``512→224``；参数化尺寸只用于无 LIBERO 单元测试和未来
    checkpoint 尺寸校验，crop area 仍与 OpenVLA 部署语义固定为 ``0.9``。
    """

    return DifferentiablePolicyViewTransform(
        DeploymentViewSpecification(
            policy_canvas=PolicyPreCropSpecification(
                source_resolution=source_resolution,
                canvas_resolution=model_input_resolution,
            ),
            center_crop=CenterCropSpecification(
                input_resolution=model_input_resolution,
                output_resolution=model_input_resolution,
                crop_area=0.9,
            ),
        )
    )


@dataclass(frozen=True)
class PolicyPreCropSpecification:
    """Policy Pre-Crop Canvas 的不可变空间 specification。

    Attributes:
        source_resolution: 输入 policy-source 正方形 RGB 的边长。
        canvas_resolution: 输出 pre-crop 正方形 RGB 的边长。

    正式 OpenVLA Spatial 配置为 512→224。字段保留显式值是为了让纯单元测试
    能使用小图验证坐标和插值语义，而不分配无意义的大数组。
    """

    source_resolution: int = POLICY_SOURCE_RESOLUTION
    canvas_resolution: int = POLICY_PRE_CROP_RESOLUTION

    def __post_init__(self) -> None:
        if self.source_resolution <= 0:
            raise ValueError("policy source resolution 必须为正数")
        if self.canvas_resolution <= 0:
            raise ValueError("policy canvas resolution 必须为正数")


@dataclass(frozen=True)
class DeploymentViewSpecification:
    """Policy Source 到 Effective View 的完整不可变空间契约。"""

    policy_canvas: PolicyPreCropSpecification = field(
        default_factory=PolicyPreCropSpecification
    )
    center_crop: CenterCropSpecification = field(
        default_factory=CenterCropSpecification
    )

    def __post_init__(self) -> None:
        if (
            self.policy_canvas.canvas_resolution
            != self.center_crop.input_resolution
        ):
            raise ValueError(
                "Policy Pre-Crop Canvas resolution 与 center-crop input "
                "resolution 必须一致"
            )


@dataclass(frozen=True)
class ExactDeploymentViewStages:
    """完整 deployment spatial path 的三个 uint8 HWC RGB stage。"""

    source_rgb: NDArray[np.uint8]
    pre_crop_rgb: NDArray[np.uint8]
    effective_view_rgb: NDArray[np.uint8]


@dataclass(frozen=True)
class DifferentiableDeploymentViewStages:
    """完整 BPDA deployment path 的两个可微 NCHW RGB stage。

    ``pre_crop_canvas`` 与 ``effective_view`` 的 forward 分别逐值等于
    Pillow resize 和 TensorFlow center crop 的 uint8 部署结果（再除以255）；
    backward 则沿 PyTorch bicubic resize 与 Gate 2C crop surrogate 回到
    Policy Source。Gate 2E 可对两个非叶子 tensor 调用 ``retain_grad``，从而
    分别验证 crop 前后的梯度，而不需要重新构造另一张 autograd graph。
    """

    pre_crop_canvas: torch.Tensor
    effective_view: torch.Tensor


def resize_policy_pre_crop_canvas(
    source_rgb: NDArray[np.uint8],
    *,
    specification: PolicyPreCropSpecification,
) -> NDArray[np.uint8]:
    """把 uint8 policy-source RGB 显式 bicubic resize 为 pre-crop canvas。

    Args:
        source_rgb: uint8 HWC RGB，shape 必须为
            ``[source_resolution, source_resolution, 3]``。
        specification: 固定输入/输出分辨率。

    Returns:
        独立、C-contiguous、可写的 uint8 HWC RGB，shape 为
        ``[canvas_resolution, canvas_resolution, 3]``。

    Raises:
        TypeError: 输入不是 numpy uint8 array。
        ValueError: layout、channel 或空间尺寸与 specification 不一致。

    Pillow 对普通 RGB 的默认 filter 历史上是 bicubic，但这里仍显式指定，避免
    library 默认值或图像 mode 变化造成无 provenance 的部署输入漂移。
    """

    if not isinstance(source_rgb, np.ndarray):
        raise TypeError("policy source RGB 必须是 numpy array")
    if source_rgb.dtype != np.uint8:
        raise TypeError(
            "policy source RGB 必须为 uint8，收到 "
            f"{source_rgb.dtype}"
        )
    expected_shape: tuple[int, int, int] = (
        specification.source_resolution,
        specification.source_resolution,
        3,
    )
    if source_rgb.shape != expected_shape:
        raise ValueError(
            "policy source RGB shape 与 specification 不一致："
            f"{source_rgb.shape} != {expected_shape}"
        )

    source_image: Image.Image = Image.fromarray(
        np.ascontiguousarray(source_rgb),
    )
    canvas_image: Image.Image = source_image.resize(
        (
            specification.canvas_resolution,
            specification.canvas_resolution,
        ),
        resample=Image.Resampling.BICUBIC,
    )
    # np.asarray(PIL.Image) 可能返回只读 view；正式调用方后续会交给 TF/PIL，
    # 因此显式复制，固定为可写且连续的 uint8 数组。
    canvas_rgb: NDArray[np.uint8] = np.array(
        canvas_image,
        dtype=np.uint8,
        copy=True,
        order="C",
    )
    if canvas_rgb.shape != (
        specification.canvas_resolution,
        specification.canvas_resolution,
        3,
    ):
        raise RuntimeError("Pillow policy canvas resize 返回了意外 shape")
    return canvas_rgb


def build_exact_deployment_view_stages(
    source_rgb: NDArray[np.uint8],
    *,
    specification: DeploymentViewSpecification,
) -> ExactDeploymentViewStages:
    """构造 rollout 真值路径的三个可保存 uint8 RGB stage。

    顺序固定为 Policy Source → Pillow RGB bicubic Pre-Crop Canvas → TensorFlow
    center crop Effective View。每个返回数组都是独立、可写、C-contiguous 的
    uint8 HWC RGB，避免审计保存时被调用方后续原地修改。
    """

    pre_crop_rgb: NDArray[np.uint8] = resize_policy_pre_crop_canvas(
        source_rgb,
        specification=specification.policy_canvas,
    )
    effective_view_rgb: NDArray[np.uint8] = deployment_center_crop_uint8(
        pre_crop_rgb,
        specification=specification.center_crop,
    )
    return ExactDeploymentViewStages(
        source_rgb=np.array(
            source_rgb,
            dtype=np.uint8,
            copy=True,
            order="C",
        ),
        pre_crop_rgb=np.array(
            pre_crop_rgb,
            dtype=np.uint8,
            copy=True,
            order="C",
        ),
        effective_view_rgb=np.array(
            effective_view_rgb,
            dtype=np.uint8,
            copy=True,
            order="C",
        ),
    )


@dataclass(frozen=True)
class DifferentiablePolicyViewTransform:
    """完整 Deployment View 的 exact-forward / surrogate-backward BPDA。

    输入是连续 float32 NCHW Policy Source，shape 为
    ``[batch_size, 3, source_resolution, source_resolution]``。forward 先量化为
    uint8，再逐样本调用 :func:`build_exact_deployment_view_stages`；backward
    使用 PyTorch bicubic+antialias 生成 Pre-Crop Canvas，再调用 Gate 2C 已通过
    的 center-crop surrogate。

    该 interface 只生成未归一化的 [0,1] Effective View；checkpoint fused
    normalization 仍由 ``DifferentiableOpenVLAImageProcessor`` 负责。
    """

    specification: DeploymentViewSpecification = field(
        default_factory=DeploymentViewSpecification
    )

    def _validate_source(self, source_nchw: torch.Tensor) -> None:
        if not isinstance(source_nchw, torch.Tensor):
            raise TypeError("Policy Source 必须是 torch.Tensor")
        if source_nchw.dtype != torch.float32:
            raise TypeError(
                "Policy Source BPDA 必须使用 float32，收到 "
                f"{source_nchw.dtype}"
            )
        expected_tail: tuple[int, int, int] = (
            3,
            self.specification.policy_canvas.source_resolution,
            self.specification.policy_canvas.source_resolution,
        )
        if (
            source_nchw.ndim != 4
            or tuple(source_nchw.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "Policy Source shape 与 specification 不一致："
                f"{tuple(source_nchw.shape)}，期望尾部 {expected_tail}"
            )
        if source_nchw.shape[0] <= 0:
            raise ValueError("Policy Source batch 不得为空")

    @staticmethod
    def _bpda_forward(
        exact_forward: torch.Tensor,
        surrogate: torch.Tensor,
    ) -> torch.Tensor:
        if exact_forward.shape != surrogate.shape:
            raise ValueError(
                "Policy View BPDA exact/surrogate shape 不一致："
                f"{tuple(exact_forward.shape)} != {tuple(surrogate.shape)}"
            )
        return exact_forward + (surrogate - surrogate.detach())

    def _exact_batches(
        self,
        source_nchw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 float32 NCHW exact pre-crop/effective-view batch。"""

        source_uint8_nhwc: NDArray[np.uint8] = (
            source_nchw.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        exact_stages: tuple[ExactDeploymentViewStages, ...] = tuple(
            build_exact_deployment_view_stages(
                source_rgb,
                specification=self.specification,
            )
            for source_rgb in source_uint8_nhwc
        )

        def to_source_tensor(
            images: tuple[NDArray[np.uint8], ...],
        ) -> torch.Tensor:
            stacked_nhwc: NDArray[np.uint8] = np.stack(images, axis=0)
            return (
                torch.from_numpy(stacked_nhwc)
                .permute(0, 3, 1, 2)
                .to(device=source_nchw.device, dtype=torch.float32)
                .div(255.0)
            )

        return (
            to_source_tensor(
                tuple(stage.pre_crop_rgb for stage in exact_stages)
            ),
            to_source_tensor(
                tuple(stage.effective_view_rgb for stage in exact_stages)
            ),
        )

    def _surrogate_pre_crop_canvas(
        self,
        source_nchw: torch.Tensor,
    ) -> torch.Tensor:
        """返回连续 PyTorch bicubic+antialias Policy Canvas。"""

        canvas_resolution = (
            self.specification.policy_canvas.canvas_resolution
        )
        return torch_functional.interpolate(
            source_nchw,
            size=(canvas_resolution, canvas_resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    def build_pre_crop_canvas(
        self,
        source_nchw: torch.Tensor,
    ) -> torch.Tensor:
        """返回 exact-forward、surrogate-backward 的 float32 NCHW Canvas。"""

        return self.build_stages(source_nchw).pre_crop_canvas

    def build_stages(
        self,
        source_nchw: torch.Tensor,
    ) -> DifferentiableDeploymentViewStages:
        """在同一 autograd graph 中返回 Pre-Crop 与 Effective View。

        center-crop surrogate 读取已经做过第一层 BPDA 的
        ``pre_crop_canvas``。因此该中间 tensor 的 forward 是 Pillow exact，
        对 source 的 backward 是 PyTorch resize surrogate；其 ``.grad`` 又能
        直接表示来自 center crop/checkpoint processor 的上游信号。
        """

        self._validate_source(source_nchw)
        exact_pre_crop, exact_effective_view = self._exact_batches(
            source_nchw
        )
        surrogate_pre_crop = self._surrogate_pre_crop_canvas(source_nchw)
        pre_crop_canvas = self._bpda_forward(
            exact_pre_crop,
            surrogate_pre_crop,
        )
        surrogate_effective_view = torch_center_crop_float(
            pre_crop_canvas,
            specification=self.specification.center_crop,
        )
        effective_view = self._bpda_forward(
            exact_effective_view,
            surrogate_effective_view,
        )
        return DifferentiableDeploymentViewStages(
            pre_crop_canvas=pre_crop_canvas,
            effective_view=effective_view,
        )

    def build_effective_view(
        self,
        source_nchw: torch.Tensor,
    ) -> torch.Tensor:
        """返回完整 exact-forward、surrogate-backward Effective View。"""

        return self.build_stages(source_nchw).effective_view
