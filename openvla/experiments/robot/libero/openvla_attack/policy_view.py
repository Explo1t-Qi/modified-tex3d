"""OpenVLA 部署路径中的 Policy Pre-Crop Canvas 纯计算契约。

录像帧和 policy 输入可以来自同一个 MuJoCo observation，但两者的分辨率属于
不同职责。正式 Akita 路径固定先生成 512×512 policy-source RGB，再以显式
Pillow bicubic 缩放为 checkpoint 期望的 224×224 uint8 canvas；录像分辨率变化
不得隐式改变该结果。

本模块只处理 center crop 之前的确定性 uint8 resize，不导入 LIBERO、模型或
TensorFlow，因而可以在 WSL 的最小 CPU 环境中独立测试。Deployment Effective
View Transform 将在后续纵切中消费这里的输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image


POLICY_SOURCE_RESOLUTION: Final[int] = 512
POLICY_PRE_CROP_RESOLUTION: Final[int] = 224


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
