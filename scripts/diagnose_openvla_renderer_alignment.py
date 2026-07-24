#!/usr/bin/env python3
"""量化 OpenVLA 攻击 renderer 与 MuJoCo 目标物体轮廓的对齐程度。

该脚本不加载 VLA 权重，只创建一个 LIBERO 环境和 nvdiffrast renderer。它在
固定任务、固定初始状态上比较若干 model-space position offset，并输出：

* MuJoCo segmentation 参考 mask；
* 每个候选 offset 的 nvdiffrast mask 与红/绿/黄叠加图；
* IoU、轮廓面积、中心点和包围盒等 JSON 指标。

默认把 renderer 当前缺省 offset 与零偏移、历史偏移进行比较，并要求当前缺省
值达到给定 IoU。它既可用于定位错位，也可作为修改 renderer 几何语义后的 GPU
回归检查。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

# 必须在导入 LIBERO/Robosuite 之前固定 headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
LIBERO_EXPERIMENT_DIR: Path = (
    REPO_ROOT / "openvla" / "experiments" / "robot" / "libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)

sys.path.insert(0, LIBERO_ROOT)

from libero.libero import benchmark  # noqa: E402
from libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    compute_render_mvp,
    find_target_body_pose,
)


BooleanMask = NDArray[np.bool_]
Uint8Image = NDArray[np.uint8]


class MaskGeometry(TypedDict):
    """一个二值轮廓的像素几何统计。"""

    area: int
    centroid_xy: Optional[list[float]]
    bbox_xyxy: Optional[list[int]]


class AlignmentMetrics(TypedDict):
    """MuJoCo 参考 mask 与 renderer mask 的对齐指标。"""

    iou: float
    reference_coverage: float
    rendered_precision: float
    intersection: int
    union: int
    reference: MaskGeometry
    rendered: MaskGeometry
    rendered_minus_reference_centroid_xy: Optional[list[float]]


def _parse_offset(raw_value: str) -> tuple[str, list[float]]:
    """解析 ``name:x,y,z`` 命令行参数。"""
    try:
        name, raw_xyz = raw_value.split(":", maxsplit=1)
        offset_xyz: list[float] = [
            float(component) for component in raw_xyz.split(",")
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "offset 必须使用 name:x,y,z 格式"
        ) from error

    if (
        not name
        or len(offset_xyz) != 3
        or not np.isfinite(offset_xyz).all()
    ):
        raise argparse.ArgumentTypeError(
            "offset 必须包含名称和三个有限浮点数"
        )
    return name, offset_xyz


def _get_simulation(env: Any) -> Any:
    """兼容 LIBERO wrapper 与直接暴露 sim 的环境。"""
    return env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim


def _descendant_body_ids(simulation: Any, root_body_id: int) -> list[int]:
    """返回根 body 及其全部递归子 body ID。"""
    descendant_ids: set[int] = {int(root_body_id)}
    changed: bool = True
    while changed:
        changed = False
        body_id: int
        parent_id: int
        for body_id, parent_id in enumerate(
            np.asarray(simulation.model.body_parentid)
        ):
            if int(parent_id) in descendant_ids and body_id not in descendant_ids:
                descendant_ids.add(body_id)
                changed = True
    return sorted(descendant_ids)


def _geometry_ids_for_bodies(
    simulation: Any,
    body_ids: Sequence[int],
) -> list[int]:
    """返回附着在指定 body 集合上的全部 MuJoCo geom ID。"""
    body_id_set: set[int] = set(body_ids)
    return [
        int(geometry_id)
        for geometry_id, geometry_body_id in enumerate(
            np.asarray(simulation.model.geom_bodyid)
        )
        if int(geometry_body_id) in body_id_set
    ]


def _render_reference_mask(
    simulation: Any,
    resolution: int,
    target_geometry_ids: Sequence[int],
) -> tuple[BooleanMask, dict[int, int]]:
    """通过 MuJoCo segmentation 获得参考轮廓及各目标 geom 的可见像素数。"""
    segmentation: NDArray[np.integer[Any]] = np.asarray(
        simulation.render(
            width=resolution,
            height=resolution,
            camera_name="agentview",
            mode="offscreen",
            segmentation=True,
        )
    )
    # 与 get_libero_image 保持一致：相机原始数据同时翻转两个空间轴。
    segmentation = segmentation[::-1, ::-1]
    if segmentation.ndim == 3 and segmentation.shape[2] >= 2:
        geometry_ids: NDArray[np.integer[Any]] = segmentation[..., 1]
    elif segmentation.ndim == 2:
        geometry_ids = segmentation
    else:
        raise RuntimeError(
            f"无法识别 MuJoCo segmentation shape: {segmentation.shape}"
        )

    reference_mask: BooleanMask = np.isin(
        geometry_ids,
        np.asarray(target_geometry_ids),
    )
    if not reference_mask.any():
        raise RuntimeError("目标物体 segmentation mask 为空")
    visible_pixel_counts: dict[int, int] = {
        int(geometry_id): int((geometry_ids == geometry_id).sum())
        for geometry_id in target_geometry_ids
        if (geometry_ids == geometry_id).any()
    }
    return reference_mask, visible_pixel_counts


def _mask_geometry(mask: BooleanMask) -> MaskGeometry:
    """计算 mask 面积、中心点与闭区间包围盒。"""
    y_coordinates: NDArray[np.int64]
    x_coordinates: NDArray[np.int64]
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) == 0:
        return {
            "area": 0,
            "centroid_xy": None,
            "bbox_xyxy": None,
        }
    return {
        "area": int(len(x_coordinates)),
        "centroid_xy": [
            float(x_coordinates.mean()),
            float(y_coordinates.mean()),
        ],
        "bbox_xyxy": [
            int(x_coordinates.min()),
            int(y_coordinates.min()),
            int(x_coordinates.max()),
            int(y_coordinates.max()),
        ],
    }


def _measure_alignment(
    reference_mask: BooleanMask,
    rendered_mask: BooleanMask,
) -> AlignmentMetrics:
    """计算两个同尺寸二值 mask 的 IoU 与几何偏差。"""
    if reference_mask.shape != rendered_mask.shape:
        raise ValueError(
            "mask shape 不一致: "
            f"{reference_mask.shape} != {rendered_mask.shape}"
        )

    intersection: int = int(
        np.logical_and(reference_mask, rendered_mask).sum()
    )
    union: int = int(np.logical_or(reference_mask, rendered_mask).sum())
    reference_geometry: MaskGeometry = _mask_geometry(reference_mask)
    rendered_geometry: MaskGeometry = _mask_geometry(rendered_mask)
    reference_area: int = reference_geometry["area"]
    rendered_area: int = rendered_geometry["area"]

    reference_centroid: Optional[list[float]] = reference_geometry["centroid_xy"]
    rendered_centroid: Optional[list[float]] = rendered_geometry["centroid_xy"]
    centroid_delta: Optional[list[float]] = None
    if reference_centroid is not None and rendered_centroid is not None:
        centroid_delta = [
            rendered_centroid[axis] - reference_centroid[axis]
            for axis in range(2)
        ]

    return {
        "iou": float(intersection / union) if union else 0.0,
        # coverage 对系统性平移很敏感，但允许 renderer 因缺少 MuJoCo depth
        # 而在遮挡边界产生少量额外像素；precision 单独暴露这些多画区域。
        "reference_coverage": (
            float(intersection / reference_area) if reference_area else 0.0
        ),
        "rendered_precision": (
            float(intersection / rendered_area) if rendered_area else 0.0
        ),
        "intersection": intersection,
        "union": union,
        "reference": reference_geometry,
        "rendered": rendered_geometry,
        "rendered_minus_reference_centroid_xy": centroid_delta,
    }


def _mask_overlay(
    reference_mask: BooleanMask,
    rendered_mask: BooleanMask,
) -> Uint8Image:
    """生成绿=MuJoCo、红=renderer、黄=重合区域的 RGB 诊断图。"""
    overlay: Uint8Image = np.zeros(
        (*reference_mask.shape, 3),
        dtype=np.uint8,
    )
    overlay[np.logical_and(reference_mask, ~rendered_mask)] = [0, 220, 0]
    overlay[np.logical_and(~reference_mask, rendered_mask)] = [230, 0, 0]
    overlay[np.logical_and(reference_mask, rendered_mask)] = [255, 220, 0]
    return overlay


def _homogeneous_matrix(
    position_xyz: NDArray[np.floating[Any]],
    rotation_3x3: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """把 MuJoCo position/rotation 组成 float64 齐次矩阵。"""
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_3x3
    matrix[:3, 3] = position_xyz
    return matrix


def _visual_geometry_diagnostics(
    simulation: Any,
    target_geometry_ids: Sequence[int],
    body_world_matrix: NDArray[np.float64],
) -> dict[str, dict[str, Any]]:
    """记录 MuJoCo 实际可视 geom 相对 body 的姿态和编译后 mesh bounds。"""
    diagnostics: dict[str, dict[str, Any]] = {}
    body_world_inverse: NDArray[np.float64] = np.linalg.inv(body_world_matrix)
    geometry_id: int
    for geometry_id in target_geometry_ids:
        if int(simulation.model.geom_group[geometry_id]) != 1:
            continue

        geometry_world_matrix: NDArray[np.float64] = _homogeneous_matrix(
            np.asarray(
                simulation.data.geom_xpos[geometry_id],
                dtype=np.float64,
            ),
            np.asarray(
                simulation.data.geom_xmat[geometry_id],
                dtype=np.float64,
            ).reshape(3, 3),
        )
        geometry_info: dict[str, Any] = {
            "body_to_geometry_matrix": (
                body_world_inverse @ geometry_world_matrix
            ).tolist(),
        }

        mesh_id: int = int(simulation.model.geom_dataid[geometry_id])
        geometry_info["mesh_id"] = mesh_id
        if mesh_id >= 0 and hasattr(simulation.model, "mesh_vert"):
            vertex_start: int = int(simulation.model.mesh_vertadr[mesh_id])
            vertex_count: int = int(simulation.model.mesh_vertnum[mesh_id])
            compiled_vertices: NDArray[np.float64] = np.asarray(
                simulation.model.mesh_vert[
                    vertex_start : vertex_start + vertex_count
                ],
                dtype=np.float64,
            )
            geometry_info["compiled_vertex_count"] = vertex_count
            geometry_info["compiled_vertex_bounds"] = [
                compiled_vertices.min(axis=0).tolist(),
                compiled_vertices.max(axis=0).tolist(),
            ]
            if hasattr(simulation.model, "mesh_pos"):
                geometry_info["mesh_pos"] = np.asarray(
                    simulation.model.mesh_pos[mesh_id],
                    dtype=np.float64,
                ).tolist()
            if hasattr(simulation.model, "mesh_quat"):
                geometry_info["mesh_quat_wxyz"] = np.asarray(
                    simulation.model.mesh_quat[mesh_id],
                    dtype=np.float64,
                ).tolist()
        diagnostics[str(geometry_id)] = geometry_info
    return diagnostics


def main() -> None:
    """运行一次固定状态的真实 renderer 对齐诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--task-suite-name")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--offset",
        action="append",
        type=_parse_offset,
        help="候选 name:x,y,z；可重复传入",
    )
    parser.add_argument(
        "--minimum-default-iou",
        type=float,
        default=0.85,
        help="默认 offset 的最低 IoU；允许缺少 depth test 导致的遮挡边界余量",
    )
    parser.add_argument(
        "--minimum-reference-coverage",
        type=float,
        default=0.98,
        help="MuJoCo 可见轮廓至少有多少比例被 renderer 覆盖",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("该对齐诊断需要 CUDA/nvdiffrast")
    if arguments.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知物体: {arguments.object_name!r}")

    object_spec: ObjectAssetSpec = OBJECT_ASSETS[arguments.object_name]
    task_suite_name: str = (
        arguments.task_suite_name or object_spec["task_suite"]
    )
    task_id: int = (
        arguments.task_id
        if arguments.task_id is not None
        else object_spec["task_id"]
    )
    task_suite: Any = benchmark.get_benchmark_dict()[task_suite_name]()
    task: Any = task_suite.get_task(task_id)
    initial_states: Sequence[Any] = task_suite.get_task_init_states(task_id)
    if not 0 <= arguments.init_state_id < len(initial_states):
        raise ValueError(
            f"init-state 超出范围: {arguments.init_state_id}; "
            f"共有 {len(initial_states)} 个状态"
        )

    device: torch.device = torch.device("cuda:0")
    env: Any
    task_description: str
    env, task_description = get_libero_env(
        task,
        model_family="openvla",
        resolution=arguments.resolution,
    )
    try:
        env.reset()
        observation: Any = env.set_init_state(
            initial_states[arguments.init_state_id]
        )
        env.env.sim.forward()
        for _ in range(arguments.num_steps_wait):
            observation, _, _, _ = env.step(
                get_libero_dummy_action("openvla")
            )

        clean_image: Uint8Image = get_libero_image(
            observation,
            arguments.resolution,
        )
        target_pose = find_target_body_pose(
            env,
            object_spec["search"],
            device,
        )
        if target_pose.body_id < 0:
            raise RuntimeError("没有找到目标 MuJoCo body")

        simulation: Any = _get_simulation(env)
        descendant_body_ids: list[int] = _descendant_body_ids(
            simulation,
            target_pose.body_id,
        )
        target_geometry_ids: list[int] = _geometry_ids_for_bodies(
            simulation,
            descendant_body_ids,
        )
        reference_mask: BooleanMask
        visible_geometry_pixel_counts: dict[int, int]
        reference_mask, visible_geometry_pixel_counts = _render_reference_mask(
            simulation,
            arguments.resolution,
            target_geometry_ids,
        )
        body_world_matrix: NDArray[np.float64] = (
            target_pose.model_matrix.detach().cpu().numpy().astype(np.float64)
        )
        visual_geometry_diagnostics: dict[str, dict[str, Any]] = (
            _visual_geometry_diagnostics(
                simulation,
                target_geometry_ids,
                body_world_matrix,
            )
        )

        renderer = DifferentiableRenderer(
            mesh_path=object_spec["mesh"],
            orig_texture_path=object_spec["texture"],
            device=device,
            scale_xyz=parse_mesh_scale(object_spec["xml"]),
        ).to(device)
        configured_default_offset: list[float] = (
            renderer.pos_offset.detach().cpu().tolist()
        )
        candidate_offsets: list[tuple[str, list[float]]] = (
            arguments.offset
            or [
                ("configured_default", configured_default_offset),
                ("zero_reference", [0.0, 0.0, 0.0]),
                ("legacy_reference", [0.02, 0.01, 0.025]),
            ]
        )
        mvp: torch.Tensor = compute_render_mvp(
            env,
            target_pose.model_matrix,
            resolution=(arguments.resolution, arguments.resolution),
        )

        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean_image).save(
            arguments.output_dir / "mujoco_clean.png"
        )
        Image.fromarray((reference_mask * 255).astype(np.uint8)).save(
            arguments.output_dir / "mujoco_target_mask.png"
        )

        results: dict[str, dict[str, Any]] = {}
        candidate_name: str
        offset_xyz: list[float]
        for candidate_name, offset_xyz in candidate_offsets:
            renderer.pos_offset.copy_(
                torch.tensor(
                    offset_xyz,
                    dtype=torch.float32,
                    device=device,
                )
            )
            with torch.no_grad():
                _, rendered_mask_tensor = renderer.render(
                    mvp,
                    resolution=(arguments.resolution, arguments.resolution),
                    model_rot=target_pose.model_matrix[:3, :3],
                )
            rendered_mask: BooleanMask = (
                rendered_mask_tensor[0, ..., 0].detach().cpu().numpy() > 0.5
            )
            metrics: AlignmentMetrics = _measure_alignment(
                reference_mask,
                rendered_mask,
            )
            results[candidate_name] = {
                "offset_xyz": offset_xyz,
                **metrics,
            }
            Image.fromarray((rendered_mask * 255).astype(np.uint8)).save(
                arguments.output_dir / f"{candidate_name}_mask.png"
            )
            Image.fromarray(
                _mask_overlay(reference_mask, rendered_mask)
            ).save(arguments.output_dir / f"{candidate_name}_overlay.png")

        report: dict[str, Any] = {
            "task_description": task_description,
            "object_name": arguments.object_name,
            "task_suite_name": task_suite_name,
            "task_id": task_id,
            "init_state_id": arguments.init_state_id,
            "settle_steps": arguments.num_steps_wait,
            "resolution": arguments.resolution,
            "target_body_id": target_pose.body_id,
            "target_body_name": target_pose.body_name,
            "body_world_matrix": body_world_matrix.tolist(),
            "descendant_body_ids": descendant_body_ids,
            "target_geometry_ids": target_geometry_ids,
            "visible_geometry_pixel_counts": visible_geometry_pixel_counts,
            "visual_geometry_diagnostics": visual_geometry_diagnostics,
            "candidates": results,
        }
        report_path: Path = arguments.output_dir / "alignment_metrics.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"诊断产物已保存到 {arguments.output_dir}")

        configured_result: Optional[dict[str, Any]] = results.get(
            "configured_default"
        )
        if configured_result is not None and (
            configured_result["iou"] < arguments.minimum_default_iou
            or configured_result["reference_coverage"]
            < arguments.minimum_reference_coverage
        ):
            raise RuntimeError(
                "renderer 默认位置未通过对齐回归: "
                f"IoU={configured_result['iou']:.6f} "
                f"(最低 {arguments.minimum_default_iou:.6f}), "
                "reference coverage="
                f"{configured_result['reference_coverage']:.6f} "
                f"(最低 {arguments.minimum_reference_coverage:.6f})"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
