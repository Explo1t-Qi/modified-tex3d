"""LIBERO/MuJoCo 场景到可微 renderer 坐标的数据转换。

训练帧采集、在线测试和最终评估都会读取同一套 MuJoCo body pose，并把它转换
为 nvdiffrast 使用的 model-view-projection（MVP）矩阵。本模块集中这些知识，
使调用方不必重复理解 MuJoCo 命名接口、四元数顺序、相机外参和背景渲染时的
临时 alpha 修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeAlias

import numpy as np
import torch
from scipy.spatial.transform import Rotation


Tensor: TypeAlias = torch.Tensor
ImageResolution: TypeAlias = tuple[int, int]
SearchKeywords: TypeAlias = Sequence[Sequence[str]]


@dataclass(frozen=True)
class TargetBodyPose:
    """目标 MuJoCo body 的世界姿态。

    Attributes:
        model_matrix: ``float32`` tensor，shape ``[4, 4]``，位于调用方指定的
            device。左上角 ``[3, 3]`` 是物体世界旋转，最后一列前三项是世界
            平移。
        body_id: MuJoCo body ID；``-1`` 表示没有找到匹配目标。
        body_name: 匹配到的 MuJoCo body 名称；未找到时为 ``None``。
    """

    model_matrix: Tensor
    body_id: int
    body_name: Optional[str]


def _get_simulation(env: Any) -> Any:
    """兼容 LIBERO wrapper 与直接暴露 ``sim`` 的测试/仿真环境。"""
    return env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim


def _get_mujoco_name(model: Any, object_id: int, object_type: str) -> Optional[str]:
    """兼容不同 mujoco-py 版本的 ID 到名称查询接口。"""
    short_type_map: dict[str, str] = {
        "texture": "tex",
        "material": "mat",
        "geom": "geom",
        "body": "body",
        "camera": "cam",
    }
    short_type: str = short_type_map.get(object_type, object_type)
    typed_lookup_name: str = f"{short_type}_id2name"

    if hasattr(model, typed_lookup_name):
        try:
            name: Optional[str] = getattr(model, typed_lookup_name)(object_id)
            return name
        except Exception:
            pass

    if hasattr(model, "id2name"):
        try:
            fallback_name: Optional[str] = model.id2name(object_id, object_type)
            return fallback_name
        except Exception:
            pass
    return None


def find_target_body_pose(
    env: Any,
    search_keywords: SearchKeywords,
    device: torch.device,
) -> TargetBodyPose:
    """按关键词优先级定位目标 body，并读取其世界姿态。

    ``search_keywords`` 中每个内层序列表示一次“全部包含”的名称匹配，外层顺序
    表示回退优先级。例如 ``(("akita", "bowl"), ("bowl",))`` 会优先寻找同时
    包含前两个词的 body，找不到时再放宽到 ``"bowl"``。

    未找到目标时沿用原实现：返回 ``body_id=-1``，并使用 z=0.85 的单位矩阵
    作为占位姿态。调用方必须根据 ``body_id`` 决定是否真正进行前景渲染。
    """
    simulation: Any = _get_simulation(env)
    target_body_id: int = -1
    target_body_name: Optional[str] = None

    keyword_group: Sequence[str]
    for keyword_group in search_keywords:
        if hasattr(simulation.model, "nbody"):
            body_id: int
            for body_id in range(simulation.model.nbody):
                body_name: Optional[str] = _get_mujoco_name(
                    simulation.model,
                    body_id,
                    "body",
                )
                if body_name is None or "vis" in body_name or "site" in body_name:
                    continue
                if all(keyword in body_name for keyword in keyword_group):
                    target_body_id = body_id
                    target_body_name = body_name
                    break
        if target_body_id != -1:
            break

    if target_body_id == -1:
        print(
            f"[WARNING] Could not find target body for {search_keywords}. "
            "Using fallback matrix."
        )
        fallback_matrix: Tensor = torch.eye(
            4,
            dtype=torch.float32,
            device=device,
        )
        fallback_matrix[2, 3] = 0.85
        return TargetBodyPose(fallback_matrix, -1, None)

    # MuJoCo position: float array [3]；quaternion: [w, x, y, z]。
    body_position: np.ndarray = np.asarray(
        simulation.data.body_xpos[target_body_id],
        dtype=np.float32,
    )
    body_quaternion: np.ndarray = np.asarray(
        simulation.data.body_xquat[target_body_id],
        dtype=np.float32,
    )
    rotation: Rotation = Rotation.from_quat(
        [
            body_quaternion[1],
            body_quaternion[2],
            body_quaternion[3],
            body_quaternion[0],
        ]
    )

    model_matrix_numpy: np.ndarray = np.eye(4, dtype=np.float32)
    model_matrix_numpy[:3, :3] = rotation.as_matrix().astype(np.float32)
    model_matrix_numpy[:3, 3] = body_position
    model_matrix: Tensor = torch.from_numpy(model_matrix_numpy).to(device)
    return TargetBodyPose(model_matrix, target_body_id, target_body_name)


def compute_render_mvp(
    env: Any,
    model_matrix: Tensor,
    resolution: ImageResolution = (256, 256),
    *,
    projection_flip_x: float = -1.0,
    projection_flip_y: float = -1.0,
) -> Tensor:
    """把 MuJoCo agentview 相机与物体姿态组合成 renderer 的 MVP。

    Args:
        env: LIBERO 环境或直接暴露 ``sim`` 的 MuJoCo wrapper。
        model_matrix: ``float32`` tensor，shape ``[4, 4]``。
        resolution: ``(width, height)``；当前实验使用正方形分辨率。
        projection_flip_x: 投影矩阵 x 轴方向，保留原实现默认值 ``-1``。
        projection_flip_y: 投影矩阵 y 轴方向，保留原实现默认值 ``-1``。

    Returns:
        与 ``model_matrix`` 同 device 的 ``float32`` MVP tensor，
        shape ``[4, 4]``。
    """
    simulation: Any = _get_simulation(env)
    width: int
    height: int
    width, height = resolution

    camera_id: int = 0
    try:
        camera_id = simulation.model.camera_name2id("agentview")
    except Exception:
        print("[WARNING] Camera 'agentview' not found, using camera 0.")

    # cam_xpos: [3]；cam_xmat: 展平的 [3, 3] camera-to-world 旋转矩阵。
    camera_position: np.ndarray = np.asarray(
        simulation.data.cam_xpos[camera_id],
        dtype=np.float32,
    )
    camera_rotation: np.ndarray = np.asarray(
        simulation.data.cam_xmat[camera_id],
        dtype=np.float32,
    ).reshape(3, 3)

    view_rotation: np.ndarray = camera_rotation.T
    view_matrix_numpy: np.ndarray = np.eye(4, dtype=np.float32)
    view_matrix_numpy[:3, :3] = view_rotation
    view_matrix_numpy[:3, 3] = -(view_rotation @ camera_position)

    field_of_view_y_degrees: float = float(
        simulation.model.cam_fovy[camera_id]
    )
    aspect_ratio: float = float(width) / float(height)
    near_plane: float = 0.01
    far_plane: float = 10.0
    focal_scale: float = 1.0 / np.tan(
        np.deg2rad(field_of_view_y_degrees) / 2.0
    )

    projection_matrix_numpy: np.ndarray = np.zeros((4, 4), dtype=np.float32)
    projection_matrix_numpy[0, 0] = (
        projection_flip_x * focal_scale / aspect_ratio
    )
    projection_matrix_numpy[1, 1] = projection_flip_y * focal_scale
    projection_matrix_numpy[2, 2] = (
        (far_plane + near_plane) / (near_plane - far_plane)
    )
    projection_matrix_numpy[2, 3] = (
        (2 * far_plane * near_plane) / (near_plane - far_plane)
    )
    projection_matrix_numpy[3, 2] = -1.0

    projection_matrix: Tensor = torch.from_numpy(
        projection_matrix_numpy
    ).to(model_matrix.device)
    view_matrix: Tensor = torch.from_numpy(view_matrix_numpy).to(
        model_matrix.device
    )
    return projection_matrix @ view_matrix @ model_matrix


def render_background_without_target(
    env: Any,
    body_id: int,
    resolution: int,
) -> Optional[np.ndarray]:
    """临时隐藏目标 body，渲染不含目标物体的 RGB 背景。

    返回 ``uint8`` HWC array，shape
    ``[resolution, resolution, 3]``。函数只修改属于 ``body_id`` 的 geom
    alpha，并在 ``finally`` 中恢复，因此不会改变物理状态。没有关联 geom 时
    返回 ``None``。
    """
    simulation: Any = _get_simulation(env)
    geometry_ids: list[int] = [
        geometry_id
        for geometry_id in range(simulation.model.ngeom)
        if simulation.model.geom_bodyid[geometry_id] == body_id
    ]
    if not geometry_ids:
        return None

    original_alphas: list[float] = [
        float(simulation.model.geom_rgba[geometry_id, 3])
        for geometry_id in geometry_ids
    ]
    try:
        geometry_id: int
        for geometry_id in geometry_ids:
            simulation.model.geom_rgba[geometry_id, 3] = 0.0
        background: np.ndarray = simulation.render(
            width=resolution,
            height=resolution,
            camera_name="agentview",
            mode="offscreen",
        )
        # 保留现有 LIBERO 相机方向转换：同时翻转垂直和水平轴。
        background = background[::-1, ::-1]
    finally:
        for geometry_id, original_alpha in zip(
            geometry_ids,
            original_alphas,
        ):
            simulation.model.geom_rgba[geometry_id, 3] = original_alpha
    return background
