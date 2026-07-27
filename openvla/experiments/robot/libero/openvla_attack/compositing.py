"""可微渲染前景与 LIBERO 相机背景的图像合成逻辑。

DifferentiableRenderer 输出 channel-last 的 RGB 与可见性 mask，而 OpenVLA
图像预处理使用 channel-first tensor。本模块集中处理 layout 转换、alpha 合成
和取值范围约束，调用方不需要重复了解 renderer 的输出格式。
"""

from __future__ import annotations

from typing import Optional, Protocol, TypeAlias, TypedDict

import torch


Tensor: TypeAlias = torch.Tensor
ImageResolution: TypeAlias = tuple[int, int]


class ForegroundRenderer(Protocol):
    """图像合成所需的最小可微 renderer interface。"""

    def render(
        self,
        mvp: Tensor,
        resolution: ImageResolution,
        *,
        model_rot: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """返回 NHWC 前景 RGB 和 NHWC 单通道可见性 mask。"""
        ...


class _RequiredSingleViewFrame(TypedDict):
    """构造单个对抗视图必需的训练帧字段。"""

    mvp: Tensor
    bg_tensor: Tensor


class SingleViewFrame(_RequiredSingleViewFrame, total=False):
    """单视图合成所需的最小 frame data。

    ``mvp`` 的形状为 ``[4, 4]``；两个背景 tensor 均为 NCHW，形状为
    ``[1, 3, height, width]``；``model_rot`` 的形状为 ``[3, 3]``。训练帧
    可以包含更多模型输入字段，但本 module 不读取它们。
    """

    bg_tensor_no_obj: Optional[Tensor]
    model_rot: Optional[Tensor]


def composite_foreground(
    foreground_rgb: Tensor,
    foreground_mask: Tensor,
    background_rgb: Tensor,
) -> Tensor:
    """按照 renderer mask 将对抗前景覆盖到相机背景。

    Args:
        foreground_rgb: renderer 输出的前景 RGB，浮点 tensor，形状为
            ``[batch_size, height, width, 3]``，layout 为 NHWC。
        foreground_mask: renderer 输出的前景可见性 mask，浮点 tensor，形状为
            ``[batch_size, height, width, 1]``。当前 renderer 产生 0/1 mask，
            该计算同时兼容 ``[0, 1]`` 内的软 mask。
        background_rgb: LIBERO 相机背景，浮点 tensor，形状为
            ``[batch_size, 3, height, width]``，layout 为 NCHW。三个输入应位于
            同一个 device，并使用可进行乘加运算的兼容 dtype。

    Returns:
        合成后的 NCHW RGB tensor，形状为
        ``[batch_size, 3, height, width]``，数值截断到 ``[0, 1]``。计算保留
        ``foreground_rgb`` 到 renderer 参数的梯度。
    """
    # mask_nchw: [batch_size, 1, height, width]
    mask_nchw: Tensor = foreground_mask.permute(0, 3, 1, 2)

    # foreground_nchw: [batch_size, 3, height, width]
    foreground_nchw: Tensor = foreground_rgb.permute(0, 3, 1, 2)

    # 前景区域取 renderer RGB，背景区域取去除目标物体后的相机画面。
    composited_rgb: Tensor = (
        foreground_nchw * mask_nchw + background_rgb * (1 - mask_nchw)
    )
    return torch.clamp(composited_rgb, 0.0, 1.0)


def render_and_composite(
    renderer: ForegroundRenderer,
    background_rgb: Tensor,
    mvp: Optional[Tensor],
    resolution: ImageResolution = (256, 256),
    model_rotation: Optional[Tensor] = None,
) -> Tensor:
    """渲染当前物体姿态，并把对抗前景覆盖到相机背景。

    该函数封装 ``renderer.render()`` 和 ``composite_foreground()``；调用方只需
    提供当前视角及两种背景，不需要理解 renderer 返回的 NHWC 格式。

    Args:
        renderer: 满足 :class:`ForegroundRenderer` interface 的可微 renderer。
        background_rgb: NCHW 相机背景，形状为
            ``[batch_size, 3, height, width]``。
        mvp: model-view-projection 矩阵，形状为 ``[4, 4]``。为 ``None`` 表示
            当前帧未定位到目标物体，此时不会调用 renderer。
        resolution: renderer 输出的 ``(height, width)``。
        model_rotation: 目标物体的世界旋转矩阵，形状为 ``[3, 3]``；renderer
            用它把法线旋转到世界坐标系。缺省时沿用 renderer 的局部坐标法线。

    Returns:
        NCHW 合成 RGB，形状为 ``[batch_size, 3, height, width]``。当 ``mvp``
        为 ``None`` 时原样返回 ``background_rgb``，保持现有 rollout 回退行为。
    """
    if mvp is None:
        return background_rgb

    # foreground_rgb: [batch_size, height, width, 3]
    # foreground_mask: [batch_size, height, width, 1]
    foreground_rgb: Tensor
    foreground_mask: Tensor
    foreground_rgb, foreground_mask = renderer.render(
        mvp,
        resolution=resolution,
        model_rot=model_rotation,
    )
    return composite_foreground(foreground_rgb, foreground_mask, background_rgb)


def build_single_view_samples(
    renderer: ForegroundRenderer,
    frame: SingleViewFrame,
    render_resolution: int,
) -> list[Tensor]:
    """从一个训练帧构造当前实现唯一的对抗视图。

    Args:
        renderer: 满足 :class:`ForegroundRenderer` interface 的可微 renderer。
        frame: 单视图渲染需要的最小训练帧字段。调用方必须先排除
            ``mvp=None`` 的帧，再把已经收窄为 ``Tensor`` 的 MVP 传入这里。
        render_resolution: 正方形 renderer 输出的边长，单位为像素。

    Returns:
        只含一个 NCHW RGB tensor 的列表，每个 tensor 形状为
        ``[1, 3, render_resolution, render_resolution]``。保留列表 interface 是
        为了让当前训练循环统一遍历样本；本函数本身明确只表达单视图策略，
        后续 EoT/TAAO 多视图策略应通过新的采样实现替换它。

    Notes:
        如果提供了去除目标物体后的背景 ``bg_tensor_no_obj``，必须优先使用它，
        否则 renderer 前景以外仍会残留原物体，形成重影。字段缺失或值为
        ``None`` 时回退到原始相机背景 ``bg_tensor``。
    """
    foreground_rgb: Tensor
    foreground_mask: Tensor
    foreground_rgb, foreground_mask = renderer.render(
        frame["mvp"],
        resolution=(render_resolution, render_resolution),
        model_rot=frame.get("model_rot"),
    )

    background_without_target: Optional[Tensor] = frame.get("bg_tensor_no_obj")
    composition_background: Tensor = (
        background_without_target
        if background_without_target is not None
        else frame["bg_tensor"]
    )
    composited_rgb: Tensor = composite_foreground(
        foreground_rgb, foreground_mask, composition_background
    )
    return [composited_rgb]
