"""把 nvdiffrast 光栅输出解码为原始几何 Support 的像素控制权重。

nvdiffrast raster 最后一维固定为 ``(u, v, z/w, triangle_id)``：有效像素的
``triangle_id`` 从 1 开始，0 表示背景；一个 face 三个 corner 的重心权重依次
为 ``(u, v, 1-u-v)``。本模块将 face corner 的 renderer 顶点先通过严格的
``render_to_geometry`` 映射回原始 OBJ 几何顶点，再计算

``w_S(p) = sum_c beta_c(p) * 1[geometry_vertex_c in S]``。

背景像素同时返回 ``valid_mask=False`` 与 ``support_control=0``。调用方必须保留
valid mask，不能把背景控制量单独解释为“可见但未覆盖”。本模块是纯 tensor
契约，不导入 nvdiffrast，也不创建 CUDA context，因此角点顺序、face ID 偏移、
UV seam 映射和 Support 单调性都可以在 CPU 上回归测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch


RASTER_BARYCENTRIC_ATOL: Final[float] = 1e-6


@dataclass(frozen=True)
class RasterSupportCorrespondence:
    """逐像素 renderer correspondence 与 Support 控制量。

    Attributes:
        triangle_indices: int64 ``[batch,height,width]``；背景为 -1。
        barycentric: float32 ``[batch,height,width,3]``；背景为全零，有效像素
            的最后一维严格对应 face corner 0、1、2。
        geometry_corner_indices: int64 ``[batch,height,width,3]``；有效像素
            是 face corner 对应的原始 OBJ 几何顶点，背景为 -1。
        valid_mask: bool ``[batch,1,height,width]``；仅 triangle ID 大于零。
        support_control: float32 ``[batch,1,height,width]``；有效像素为
            ``w_S``，背景为零。
    """

    triangle_indices: torch.Tensor
    barycentric: torch.Tensor
    geometry_corner_indices: torch.Tensor
    valid_mask: torch.Tensor
    support_control: torch.Tensor


def _validate_integer_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} 必须使用整数 dtype")
    if value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"{name} 必须为非空 {ndim} 维 tensor")


def decode_raster_support_correspondence(
    raster: torch.Tensor,
    renderer_faces: torch.Tensor,
    render_to_geometry: torch.Tensor,
    geometry_support: torch.Tensor,
    *,
    barycentric_atol: float = RASTER_BARYCENTRIC_ATOL,
) -> RasterSupportCorrespondence:
    """解码 raster，并计算一个几何顶点 Support 的逐像素控制权重。

    Args:
        raster: nvdiffrast float32 ``[batch,height,width,4]`` 输出。
        renderer_faces: integer ``[num_faces,3]`` renderer face corner 索引。
        render_to_geometry: integer ``[num_render_vertices]`` 严格 seam 映射。
        geometry_support: bool ``[num_geometry_vertices]`` Support membership。
        barycentric_atol: 重心坐标与整数 triangle ID 的数值容差。

    Returns:
        与输入 raster 同 device 的 correspondence。此函数不 clamp 非法光栅
        值；越界、非整数 face ID 或非法重心坐标均 fail-fast。
    """

    if not isinstance(raster, torch.Tensor):
        raise TypeError("raster 必须是 torch.Tensor")
    if raster.dtype != torch.float32:
        raise TypeError("raster 必须使用 float32")
    if (
        raster.ndim != 4
        or raster.shape[0] <= 0
        or raster.shape[1] <= 0
        or raster.shape[2] <= 0
        or raster.shape[3] != 4
    ):
        raise ValueError("raster shape 必须为非空 [batch,height,width,4]")
    if not bool(torch.isfinite(raster).all()):
        raise ValueError("raster 包含 NaN/Inf")
    if not math.isfinite(barycentric_atol) or barycentric_atol < 0.0:
        raise ValueError("barycentric_atol 必须为有限非负数")

    _validate_integer_tensor(renderer_faces, name="renderer_faces", ndim=2)
    if renderer_faces.shape[1] != 3:
        raise ValueError("renderer_faces shape 必须为 [num_faces,3]")
    _validate_integer_tensor(
        render_to_geometry,
        name="render_to_geometry",
        ndim=1,
    )
    if not isinstance(geometry_support, torch.Tensor):
        raise TypeError("geometry_support 必须是 torch.Tensor")
    if geometry_support.dtype != torch.bool:
        raise TypeError("geometry_support 必须使用 bool dtype")
    if geometry_support.ndim != 1 or geometry_support.numel() == 0:
        raise ValueError("geometry_support 必须为非空一维 tensor")
    if not (
        renderer_faces.device
        == render_to_geometry.device
        == geometry_support.device
        == raster.device
    ):
        raise ValueError("raster、拓扑映射与 Support 必须位于同一 device")

    renderer_faces_long: torch.Tensor = renderer_faces.to(dtype=torch.long)
    render_to_geometry_long: torch.Tensor = render_to_geometry.to(
        dtype=torch.long
    )
    if int(renderer_faces_long.min().item()) < 0 or int(
        renderer_faces_long.max().item()
    ) >= int(render_to_geometry_long.shape[0]):
        raise ValueError("renderer_faces 包含越界 renderer 顶点索引")
    if int(render_to_geometry_long.min().item()) < 0 or int(
        render_to_geometry_long.max().item()
    ) >= int(geometry_support.shape[0]):
        raise ValueError("render_to_geometry 包含越界几何顶点索引")

    raw_triangle_ids: torch.Tensor = raster[..., 3]
    rounded_triangle_ids: torch.Tensor = torch.round(raw_triangle_ids)
    if bool(
        ((raw_triangle_ids - rounded_triangle_ids).abs() > barycentric_atol).any()
    ):
        raise ValueError("raster triangle ID 不是容差内整数")
    triangle_ids: torch.Tensor = rounded_triangle_ids.to(dtype=torch.long)
    if bool((triangle_ids < 0).any()) or bool(
        (triangle_ids > renderer_faces_long.shape[0]).any()
    ):
        raise ValueError("raster triangle ID 超出 0..num_faces")

    valid_nhw: torch.Tensor = triangle_ids > 0
    triangle_indices: torch.Tensor = torch.where(
        valid_nhw,
        triangle_ids - 1,
        torch.full_like(triangle_ids, -1),
    )
    barycentric_raw: torch.Tensor = torch.stack(
        (
            raster[..., 0],
            raster[..., 1],
            1.0 - raster[..., 0] - raster[..., 1],
        ),
        dim=-1,
    )
    valid_barycentric: torch.Tensor = barycentric_raw[valid_nhw]
    if valid_barycentric.numel() > 0:
        if bool((valid_barycentric < -barycentric_atol).any()) or bool(
            (valid_barycentric > 1.0 + barycentric_atol).any()
        ):
            raise ValueError("有效 raster 重心坐标超出 [0,1] 容差")
        barycentric_sums: torch.Tensor = valid_barycentric.sum(dim=-1)
        if bool(
            ((barycentric_sums - 1.0).abs() > barycentric_atol).any()
        ):
            raise ValueError("有效 raster 重心坐标之和不为 1")
    barycentric: torch.Tensor = torch.where(
        valid_nhw.unsqueeze(-1),
        barycentric_raw,
        torch.zeros_like(barycentric_raw),
    )

    safe_triangle_indices: torch.Tensor = triangle_indices.clamp_min(0)
    renderer_corner_indices: torch.Tensor = renderer_faces_long[
        safe_triangle_indices
    ]
    geometry_corners_raw: torch.Tensor = render_to_geometry_long[
        renderer_corner_indices
    ]
    geometry_corner_indices: torch.Tensor = torch.where(
        valid_nhw.unsqueeze(-1),
        geometry_corners_raw,
        torch.full_like(geometry_corners_raw, -1),
    )
    corner_in_support: torch.Tensor = geometry_support[
        geometry_corners_raw
    ].to(dtype=torch.float32)
    support_control_nhw: torch.Tensor = (
        barycentric * corner_in_support
    ).sum(dim=-1)
    valid_mask: torch.Tensor = valid_nhw.unsqueeze(1)
    support_control: torch.Tensor = support_control_nhw.unsqueeze(1)
    return RasterSupportCorrespondence(
        triangle_indices=triangle_indices,
        barycentric=barycentric,
        geometry_corner_indices=geometry_corner_indices,
        valid_mask=valid_mask,
        support_control=support_control,
    )
