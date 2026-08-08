"""MuJoCo object-type/object-id segmentation 到逐实例可见 alpha 的解析层。

正式 visibility evidence 不能把 segmentation 的裸整数直接当成 geom ID。本模块
要求输入保留 backend 的两个通道 ``[object_type, object_id]``，只对显式 GEOM
类型解析 ID，再沿 ``geom_bodyid`` 和 ``body_parentid`` 映射到目标实例根 body 的
唯一子树。输出 hard alpha 表示同一 ID 图中的 front-most 可见实例，因此逐像素
互斥；抗锯齿 soft 权重只允许由后续 area/crop evidence transform 产生。

本模块不调用 renderer，也不推进 simulation。真实采集器后续负责在同一静止
state transaction 内产生已经使用统一相机方向变换的 segmentation array，并把
backend/version 与本解析结果共同写入审计 provenance。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetInstanceRoot:
    """共享 Active Texture 的一个目标实例根 body。"""

    body_id: int
    body_name: str


@dataclass(frozen=True)
class InstanceGeometryMapping:
    """一个实例根 body 的完整递归 body/geom 集合。"""

    body_id: int
    body_name: str
    subtree_body_ids: tuple[int, ...]
    geometry_ids: tuple[int, ...]
    geometry_body_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ParsedInstanceSegmentation:
    """显式 schema 验证后的逐实例 hard visibility evidence。"""

    # float32 CPU [num_instances,1,height,width]，值严格属于 {0,1}。
    instance_alpha: torch.Tensor
    mappings: tuple[InstanceGeometryMapping, ...]
    object_type_histogram: Mapping[int, int]
    object_id_histograms_by_type: Mapping[int, Mapping[int, int]]
    geometry_id_histogram: Mapping[int, int]
    non_geometry_pixel_count: int
    non_target_geometry_pixel_count: int


def _validate_model_topology(
    model: Any,
) -> tuple[
    int,
    int,
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """读取并校验解析所需的最小 MuJoCo model topology。"""

    required_attributes = ("nbody", "ngeom", "body_parentid", "geom_bodyid")
    missing = [name for name in required_attributes if not hasattr(model, name)]
    if missing:
        raise TypeError(f"MuJoCo model 缺少 segmentation topology: {missing}")
    number_of_bodies = int(model.nbody)
    number_of_geometries = int(model.ngeom)
    if number_of_bodies <= 0 or number_of_geometries <= 0:
        raise ValueError("MuJoCo model nbody/ngeom 必须为正数")
    body_parent_ids = np.asarray(model.body_parentid, dtype=np.int64)
    geometry_body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
    if body_parent_ids.shape != (number_of_bodies,):
        raise ValueError("body_parentid shape 与 nbody 不一致")
    if geometry_body_ids.shape != (number_of_geometries,):
        raise ValueError("geom_bodyid shape 与 ngeom 不一致")
    if np.any(geometry_body_ids < 0) or np.any(
        geometry_body_ids >= number_of_bodies
    ):
        raise ValueError("geom_bodyid 包含越界 body ID")
    for body_id in range(1, number_of_bodies):
        parent_id = int(body_parent_ids[body_id])
        if parent_id < 0 or parent_id >= number_of_bodies:
            raise ValueError("body_parentid 包含越界 parent ID")
        if parent_id == body_id:
            raise ValueError("非 world body 不能以自身作为 parent")
    return (
        number_of_bodies,
        number_of_geometries,
        body_parent_ids,
        geometry_body_ids,
    )


def _descendant_body_ids(
    root_body_id: int,
    body_parent_ids: NDArray[np.int64],
) -> tuple[int, ...]:
    """确定性返回 root 及全部递归 descendants，按 body ID 排序。"""

    descendants: set[int] = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(body_parent_ids):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return tuple(sorted(descendants))


def build_instance_geometry_mappings(
    model: Any,
    instance_roots: Sequence[TargetInstanceRoot],
) -> tuple[InstanceGeometryMapping, ...]:
    """构造互不重叠且每个实例非空的 body-subtree→geom 映射。"""

    number_of_bodies, _, body_parent_ids, geometry_body_ids = (
        _validate_model_topology(model)
    )
    if not instance_roots:
        raise ValueError("目标实例根 body 列表不得为空")
    root_ids = [root.body_id for root in instance_roots]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("目标实例根 body ID 重复")

    mappings: list[InstanceGeometryMapping] = []
    claimed_body_ids: dict[int, int] = {}
    for instance_index, root in enumerate(instance_roots):
        if not 0 <= root.body_id < number_of_bodies:
            raise ValueError(f"目标实例 root body ID 越界: {root.body_id}")
        if not root.body_name:
            raise ValueError("目标实例 body_name 不得为空")
        subtree_body_ids = _descendant_body_ids(
            root.body_id,
            body_parent_ids,
        )
        for body_id in subtree_body_ids:
            if body_id in claimed_body_ids:
                raise ValueError(
                    "目标实例 body subtrees 重叠：body "
                    f"{body_id} 同时属于实例 {claimed_body_ids[body_id]} "
                    f"与 {instance_index}"
                )
            claimed_body_ids[body_id] = instance_index
        subtree_set = set(subtree_body_ids)
        geometry_ids = tuple(
            int(geometry_id)
            for geometry_id, body_id in enumerate(geometry_body_ids)
            if int(body_id) in subtree_set
        )
        if not geometry_ids:
            raise ValueError(
                f"目标实例 {root.body_name!r} 的 body subtree 没有 geom"
            )
        mappings.append(
            InstanceGeometryMapping(
                body_id=root.body_id,
                body_name=root.body_name,
                subtree_body_ids=subtree_body_ids,
                geometry_ids=geometry_ids,
                geometry_body_pairs=tuple(
                    (
                        geometry_id,
                        int(geometry_body_ids[geometry_id]),
                    )
                    for geometry_id in geometry_ids
                ),
            )
        )
    return tuple(mappings)


def _histogram(values: NDArray[np.int64]) -> dict[int, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        int(value): int(count)
        for value, count in zip(unique_values, counts, strict=True)
    }


def parse_instance_segmentation(
    segmentation: np.ndarray,
    *,
    model: Any,
    instance_roots: Sequence[TargetInstanceRoot],
    geom_object_type: int,
) -> ParsedInstanceSegmentation:
    """把 oriented MuJoCo segmentation ID 图解析为逐实例 hard alpha。

    Args:
        segmentation: 整数 HWC ``[height,width,2]``；最后一维依次为 backend
            object type 与 object ID。
        model: 提供 ``nbody/ngeom/body_parentid/geom_bodyid`` 的 MuJoCo model。
        instance_roots: 第一个命中关键词组中的全部共享纹理实例根 body。
        geom_object_type: 当前 backend 显式报告的 ``OBJ_GEOM`` 整数常量；调用方
            必须从 backend 读取并记录，不能在本模块硬编码。

    Returns:
        CPU float32 ``instance_alpha`` 及可持久化 mapping/histogram。
    """

    if not isinstance(segmentation, np.ndarray):
        raise TypeError("MuJoCo segmentation 必须是 numpy array")
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError("MuJoCo segmentation shape 必须为 [H,W,2]")
    if segmentation.shape[0] <= 0 or segmentation.shape[1] <= 0:
        raise ValueError("MuJoCo segmentation 空间尺寸必须为正数")
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError("MuJoCo segmentation object type/ID 必须为整数")
    if not isinstance(geom_object_type, int):
        raise TypeError("geom_object_type 必须是 backend 整数常量")

    _, number_of_geometries, _, _ = _validate_model_topology(model)
    mappings = build_instance_geometry_mappings(model, instance_roots)
    segmentation_int64 = np.asarray(segmentation, dtype=np.int64)
    object_types = segmentation_int64[..., 0]
    object_ids = segmentation_int64[..., 1]
    geometry_pixels = object_types == geom_object_type
    geometry_ids = object_ids[geometry_pixels]
    if geometry_ids.size > 0 and (
        np.any(geometry_ids < 0)
        or np.any(geometry_ids >= number_of_geometries)
    ):
        raise ValueError("GEOM segmentation pixel 包含越界 geom ID")

    geometry_to_instance: dict[int, int] = {}
    for instance_index, mapping in enumerate(mappings):
        for geometry_id in mapping.geometry_ids:
            if geometry_id in geometry_to_instance:
                raise RuntimeError("一个 geom 同时映射到多个目标实例")
            geometry_to_instance[geometry_id] = instance_index

    height, width = object_types.shape
    alpha_numpy = np.zeros(
        (len(mappings), 1, height, width),
        dtype=np.float32,
    )
    non_target_geometry_pixel_count = 0
    for geometry_id, pixel_count in _histogram(geometry_ids).items():
        matching_instance = geometry_to_instance.get(geometry_id)
        geometry_mask = geometry_pixels & (object_ids == geometry_id)
        if matching_instance is None:
            non_target_geometry_pixel_count += pixel_count
            continue
        alpha_numpy[matching_instance, 0, geometry_mask] = 1.0

    # 由单张 front-most ID 图拆分，理论上必然互斥；仍显式断言，防止后续实现
    # 改成多次 render 后悄悄破坏这一语义。
    if np.any(alpha_numpy.sum(axis=0) > 1.0):
        raise RuntimeError("解析后的逐实例 segmentation alpha 不互斥")
    return ParsedInstanceSegmentation(
        instance_alpha=torch.from_numpy(alpha_numpy),
        mappings=mappings,
        object_type_histogram=_histogram(object_types.reshape(-1)),
        object_id_histograms_by_type={
            object_type: _histogram(
                object_ids[object_types == object_type]
            )
            for object_type in _histogram(object_types.reshape(-1))
        },
        geometry_id_histogram=_histogram(geometry_ids),
        non_geometry_pixel_count=int((~geometry_pixels).sum()),
        non_target_geometry_pixel_count=non_target_geometry_pixel_count,
    )
