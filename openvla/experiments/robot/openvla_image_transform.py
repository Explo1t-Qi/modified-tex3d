"""OpenVLA 部署与攻击训练共享的 Effective View 图像变换。

OpenVLA checkpoint 在读取 Policy Pre-Crop Canvas 后，还会执行训练时约定的
中心裁剪并 resize 回模型输入分辨率。这个步骤同时决定部署时模型真正看到的
有效视野，以及攻击训练时梯度应当经过的空间变换。

本模块明确区分两条职责一致、数值目的不同的路径：

* :func:`deployment_center_crop_uint8` 使用 TensorFlow 的 checkpoint 精确路径，
  输入和输出均为 uint8 HWC RGB；
* :func:`torch_center_crop_float` 使用相同归一化坐标的 PyTorch 双线性采样，
  为连续 NCHW tensor 提供可微 surrogate。

两条路径共享 :class:`CenterCropSpecification`，从而防止 crop 面积、输入尺寸或
输出尺寸在 rollout、coverage 审计和攻击训练之间静默漂移。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import tensorflow as tf
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class CenterCropSpecification:
    """中心裁剪与 resize 的不可变几何 specification。

    Attributes:
        input_resolution: 输入正方形图像边长。
        output_resolution: 输出正方形图像边长。
        crop_area: 中心裁剪面积占原图面积的比例。裁剪边长比例因此为
            ``sqrt(crop_area)``。
    """

    input_resolution: int = 224
    output_resolution: int = 224
    crop_area: float = 0.9

    def __post_init__(self) -> None:
        if self.input_resolution <= 0:
            raise ValueError("center-crop input resolution 必须为正数")
        if self.output_resolution <= 0:
            raise ValueError("center-crop output resolution 必须为正数")
        if not math.isfinite(self.crop_area):
            raise ValueError("center-crop area 必须为有限数")
        if not 0.0 < self.crop_area <= 1.0:
            raise ValueError("center-crop area 必须位于 (0, 1]")


def _validate_tensorflow_image(
    image: tf.Tensor,
    *,
    specification: CenterCropSpecification,
) -> int:
    """校验 float HWC/NHWC TensorFlow 输入并返回静态 rank。"""

    if image.dtype != tf.float32:
        raise TypeError(
            "TensorFlow center-crop oracle 输入必须为 float32 tensor，收到 "
            f"{image.dtype.name}"
        )
    rank: int | None = image.shape.rank
    if rank not in (3, 4):
        raise ValueError(
            "TensorFlow center-crop 输入必须为 HWC 或 NHWC，收到 rank "
            f"{rank}"
        )

    expected_tail: tuple[int, int, int] = (
        specification.input_resolution,
        specification.input_resolution,
        3,
    )
    static_tail: tuple[int | None, ...] = tuple(image.shape[-3:])
    for actual_dimension, expected_dimension in zip(
        static_tail,
        expected_tail,
        strict=True,
    ):
        if (
            actual_dimension is not None
            and actual_dimension != expected_dimension
        ):
            raise ValueError(
                "TensorFlow center-crop 输入 shape 与 specification 不一致："
                f"{image.shape}，期望尾部为 {expected_tail}"
            )

    # 静态 shape 在 eager 单元测试中足够，但 graph tracing 时可能包含 None。
    # 动态断言仍保护 H/W/C，不让错误尺寸悄悄进入插值。
    tf.debugging.assert_equal(
        tf.shape(image)[-3:],
        tf.constant(expected_tail, dtype=tf.int32),
        message="TensorFlow center-crop 输入动态 shape 与 specification 不一致",
    )
    return rank


@lru_cache(maxsize=16)
def _tensorflow_float32_crop_box(crop_area: float) -> tuple[float, float]:
    """返回由 TensorFlow float32 真值路径计算的 ``(start, end)``。

    TensorFlow 2.15 CPU 与 PyTorch 2.2 CPU 对 ``sqrt(float32(0.9))`` 的最后一位
    舍入并不相同。稀疏 impulse 的 relative L2 会放大这一个 ULP 的 box 差异。
    crop box 本来就是固定的非学习参数，因此在构造 PyTorch sampling weights
    时复用 oracle 产生的两个 float32 常量，既保持跨框架几何一致，也不引入
    TensorFlow 到训练 autograd graph。
    """

    crop_area_tensor = tf.constant(crop_area, dtype=tf.float32)
    side_scale = tf.sqrt(crop_area_tensor)
    offset = (tf.constant(1.0, dtype=tf.float32) - side_scale) / tf.constant(
        2.0,
        dtype=tf.float32,
    )
    box_end = offset + side_scale
    return float(offset.numpy()), float(box_end.numpy())


def tensorflow_center_crop_float(
    image: tf.Tensor,
    *,
    specification: CenterCropSpecification,
) -> tf.Tensor:
    """使用 checkpoint 精确 TensorFlow 语义执行中心裁剪和 resize。

    Args:
        image: float32 HWC 或 NHWC tensor，shape 尾部为
            ``[input_resolution, input_resolution, 3]``。数值通常位于 [0, 1]，
            但本函数不做 clip，以保留线性算子及其真实梯度。
        specification: 输入尺寸、输出尺寸和裁剪面积。

    Returns:
        与输入 rank 对应的浮点 HWC/NHWC tensor，空间 shape 为
        ``[output_resolution, output_resolution]``。

    ``tf.image.crop_and_resize`` 的归一化 box 端点映射到输入像素中心 0 与
    ``input_resolution - 1``。输出采样包含两个端点；PyTorch surrogate 会
    直接复现该像素坐标与双线性插值顺序。
    """

    rank: int = _validate_tensorflow_image(
        image,
        specification=specification,
    )
    image_nhwc: tf.Tensor = image[None, ...] if rank == 3 else image
    batch_size: tf.Tensor = tf.shape(image_nhwc)[0]

    coordinate_dtype: tf.dtypes.DType = image.dtype
    side_scale: tf.Tensor = tf.sqrt(
        tf.cast(specification.crop_area, coordinate_dtype)
    )
    offset: tf.Tensor = (
        tf.cast(1.0, coordinate_dtype) - side_scale
    ) / tf.cast(2.0, coordinate_dtype)
    single_box: tf.Tensor = tf.stack(
        (
            offset,
            offset,
            offset + side_scale,
            offset + side_scale,
        )
    )
    bounding_boxes: tf.Tensor = tf.tile(
        single_box[None, :],
        tf.stack((batch_size, tf.constant(1, dtype=tf.int32))),
    )

    output_nhwc: tf.Tensor = tf.image.crop_and_resize(
        image_nhwc,
        boxes=bounding_boxes,
        box_indices=tf.range(batch_size),
        crop_size=(
            specification.output_resolution,
            specification.output_resolution,
        ),
        method="bilinear",
        extrapolation_value=0.0,
    )
    return output_nhwc[0] if rank == 3 else output_nhwc


def deployment_center_crop_uint8(
    image_rgb: NDArray[np.uint8],
    *,
    specification: CenterCropSpecification,
) -> NDArray[np.uint8]:
    """执行 OpenVLA rollout 的精确 uint8 center-crop 路径。

    Args:
        image_rgb: uint8 HWC RGB，shape 为
            ``[input_resolution, input_resolution, 3]``。
        specification: 输入尺寸、输出尺寸和裁剪面积。

    Returns:
        独立、C-contiguous、可写的 uint8 HWC RGB，空间尺寸为
        ``output_resolution``。

    转换顺序固定为 ``uint8 -> float32 [0,1] -> TF crop_and_resize -> clip ->
    uint8``，与历史 OpenVLA evaluation 代码一致。不能用 Pillow crop 或整数
    边界替代，否则会改变亚像素采样坐标。
    """

    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("deployment center-crop 输入必须是 numpy array")
    if image_rgb.dtype != np.uint8:
        raise TypeError(
            "deployment center-crop 输入必须为 uint8，收到 "
            f"{image_rgb.dtype}"
        )
    expected_shape: tuple[int, int, int] = (
        specification.input_resolution,
        specification.input_resolution,
        3,
    )
    if image_rgb.shape != expected_shape:
        raise ValueError(
            "deployment center-crop 输入 shape 与 specification 不一致："
            f"{image_rgb.shape} != {expected_shape}"
        )

    image_uint8: tf.Tensor = tf.convert_to_tensor(
        np.ascontiguousarray(image_rgb)
    )
    image_float: tf.Tensor = tf.image.convert_image_dtype(
        image_uint8,
        tf.float32,
    )
    cropped_float: tf.Tensor = tensorflow_center_crop_float(
        image_float,
        specification=specification,
    )
    cropped_uint8: tf.Tensor = tf.image.convert_image_dtype(
        tf.clip_by_value(cropped_float, 0.0, 1.0),
        tf.uint8,
        saturate=True,
    )
    result: NDArray[np.uint8] = np.array(
        cropped_uint8.numpy(),
        dtype=np.uint8,
        copy=True,
        order="C",
    )
    return result


def torch_center_crop_float(
    image_nchw: torch.Tensor,
    *,
    specification: CenterCropSpecification,
) -> torch.Tensor:
    """使用 PyTorch 可微 surrogate 复现 TensorFlow center-crop 几何。

    Args:
        image_nchw: float32 tensor，语义为连续 RGB，shape 为
            ``[batch_size, 3, input_resolution, input_resolution]``，device
            与 dtype 由调用方决定。
        specification: 输入尺寸、输出尺寸和裁剪面积。

    Returns:
        同 dtype、同 device 的 NCHW tensor，shape 为
        ``[batch_size, 3, output_resolution, output_resolution]``。

    本函数只替代部署变换的 backward。正式 BPDA forward 仍必须使用
    :func:`deployment_center_crop_uint8` 产生的精确 uint8 结果。
    """

    if not isinstance(image_nchw, torch.Tensor):
        raise TypeError("PyTorch center-crop 输入必须是 torch.Tensor")
    if image_nchw.dtype != torch.float32:
        raise TypeError(
            "PyTorch center-crop surrogate 输入必须为 float32 tensor，收到 "
            f"{image_nchw.dtype}"
        )
    if image_nchw.ndim != 4:
        raise ValueError(
            "PyTorch center-crop 输入必须为 NCHW rank-4 tensor，收到 "
            f"shape {tuple(image_nchw.shape)}"
        )
    expected_tail: tuple[int, int, int] = (
        3,
        specification.input_resolution,
        specification.input_resolution,
    )
    if tuple(image_nchw.shape[1:]) != expected_tail:
        raise ValueError(
            "PyTorch center-crop 输入 shape 与 specification 不一致："
            f"{tuple(image_nchw.shape)}，期望尾部为 {expected_tail}"
        )

    coordinate_dtype: torch.dtype = torch.float32
    coordinate_device: torch.device = image_nchw.device
    two = torch.tensor(
        2.0,
        dtype=coordinate_dtype,
        device=coordinate_device,
    )
    box_start_value: float
    box_end_value: float
    box_start_value, box_end_value = _tensorflow_float32_crop_box(
        specification.crop_area
    )
    box_start = torch.tensor(
        box_start_value,
        dtype=coordinate_dtype,
        device=coordinate_device,
    )
    box_end = torch.tensor(
        box_end_value,
        dtype=coordinate_dtype,
        device=coordinate_device,
    )

    def build_pixel_coordinates(output_resolution: int) -> torch.Tensor:
        """按 TF CropAndResize CPU kernel 的运算顺序构造像素坐标。"""

        input_extent = torch.tensor(
            specification.input_resolution - 1,
            dtype=coordinate_dtype,
            device=coordinate_device,
        )
        if output_resolution == 1:
            return ((box_start + box_end) / two * input_extent)[None]
        output_extent = torch.tensor(
            output_resolution - 1,
            dtype=coordinate_dtype,
            device=coordinate_device,
        )
        coordinate_scale = (
            (box_end - box_start) * input_extent / output_extent
        )
        output_indices = torch.arange(
            output_resolution,
            dtype=coordinate_dtype,
            device=coordinate_device,
        )
        return box_start * input_extent + output_indices * coordinate_scale

    pixel_coordinates = build_pixel_coordinates(
        specification.output_resolution
    )
    lower_indices = torch.floor(pixel_coordinates).to(dtype=torch.long)
    upper_indices = torch.ceil(pixel_coordinates).to(dtype=torch.long)
    lower_indices = torch.clamp(
        lower_indices,
        min=0,
        max=specification.input_resolution - 1,
    )
    upper_indices = torch.clamp(
        upper_indices,
        min=0,
        max=specification.input_resolution - 1,
    )
    lerp = pixel_coordinates - lower_indices.to(dtype=coordinate_dtype)

    # 正方形中心 crop 的 x/y 坐标相同。下面显式复现 TensorFlow kernel 的
    # x-linear interpolation 后 y-linear interpolation 顺序。相比 affine_grid +
    # grid_sample，这也避免了 normalized→pixel 的第二次坐标舍入；高频 checker
    # 和 impulse case 对一个 ULP 的坐标差异非常敏感。
    top_left = image_nchw[
        :,
        :,
        lower_indices[:, None],
        lower_indices[None, :],
    ]
    top_right = image_nchw[
        :,
        :,
        lower_indices[:, None],
        upper_indices[None, :],
    ]
    bottom_left = image_nchw[
        :,
        :,
        upper_indices[:, None],
        lower_indices[None, :],
    ]
    bottom_right = image_nchw[
        :,
        :,
        upper_indices[:, None],
        upper_indices[None, :],
    ]
    x_lerp = lerp[None, None, None, :]
    y_lerp = lerp[None, None, :, None]
    top = top_left + (top_right - top_left) * x_lerp
    bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
    return top + (bottom - top) * y_lerp
