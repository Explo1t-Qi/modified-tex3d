"""MuJoCo visibility render 与静止场景证据事务。

RGB、instance segmentation、目标 pose 和 renderer correspondence 必须来自同一
个不推进环境的静止 state。本模块提供两项边界能力：

* 严格调用 MuJoCo offscreen segmentation，并复用 policy RGB 的双轴翻转；
* 在 evidence transaction 前后验证 time/qpos/qvel 逐值相等、目标 body pose
  在 ``1e-12`` 候选容差内不变。

旧脚本曾接受二维裸 ID；这里明确只接受整数 ``[H,W,2]`` object-type/object-ID
schema。backend 不符合时 fail-fast，不把裸 ID 猜成 geom ID。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Iterator, Optional

import numpy as np

from .visibility_segmentation import (
    ParsedInstanceSegmentation,
    TargetInstanceRoot,
    parse_instance_segmentation,
)


STATE_POSE_ATOL_CANDIDATE: Final[float] = 1e-12


@dataclass(frozen=True)
class MujocoBackendIdentity:
    """segmentation 产物必须记录的 simulation/backend 身份。"""

    simulation_class: str
    model_class: str
    mujoco_version: Optional[str]
    mujoco_py_version: Optional[str]
    robosuite_version: Optional[str]


@dataclass(frozen=True)
class CapturedInstanceSegmentation:
    """已统一相机方向的原始 ID 图与逐实例 hard alpha。"""

    oriented_segmentation: np.ndarray
    parsed: ParsedInstanceSegmentation
    backend: MujocoBackendIdentity
    camera_name: str
    resolution: int
    geom_object_type: int


@dataclass(frozen=True)
class StaticSceneSnapshot:
    """一次静止场景快照；数组均为拥有数据的副本。"""

    simulation_time: float
    qpos: np.ndarray
    qvel: np.ndarray
    body_ids: tuple[int, ...]
    body_xpos: np.ndarray
    body_xquat: np.ndarray
    fingerprint_sha256: str


@dataclass
class StaticSceneEvidenceTransaction:
    """context 退出后填充的静止性审计结果。"""

    before: StaticSceneSnapshot
    after: Optional[StaticSceneSnapshot] = None
    verified: bool = False
    maximum_position_delta: Optional[float] = None
    maximum_quaternion_delta: Optional[float] = None


def _get_simulation(env: Any) -> Any:
    return env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim


def _installed_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def identify_mujoco_backend(simulation: Any) -> MujocoBackendIdentity:
    """读取 simulation/model 类和常见 MuJoCo wrapper 发行版版本。"""

    simulation_class = (
        f"{type(simulation).__module__}.{type(simulation).__qualname__}"
    )
    model = simulation.model
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    return MujocoBackendIdentity(
        simulation_class=simulation_class,
        model_class=model_class,
        mujoco_version=_installed_version("mujoco"),
        mujoco_py_version=_installed_version("mujoco-py"),
        robosuite_version=_installed_version("robosuite"),
    )


def resolve_mujoco_geom_object_type() -> int:
    """从当前官方 ``mujoco`` binding 读取 GEOM enum，禁止魔数。"""

    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "当前环境无法导入 mujoco，必须由 runner 显式提供 GEOM object type"
        ) from error
    try:
        return int(mujoco.mjtObj.mjOBJ_GEOM)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("当前 mujoco binding 未暴露 mjtObj.mjOBJ_GEOM") from error


def capture_instance_segmentation(
    env: Any,
    instance_roots: tuple[TargetInstanceRoot, ...],
    *,
    camera_name: str,
    resolution: int,
    geom_object_type: Optional[int] = None,
) -> CapturedInstanceSegmentation:
    """单次渲染 front-most segmentation，并严格拆分目标实例 alpha。"""

    if not camera_name:
        raise ValueError("segmentation camera_name 不能为空")
    if resolution <= 0:
        raise ValueError("segmentation resolution 必须为正数")
    if geom_object_type is not None and (
        not isinstance(geom_object_type, int)
        or isinstance(geom_object_type, bool)
    ):
        raise TypeError("geom_object_type 必须是 backend 整数常量")
    resolved_geom_object_type: int = (
        resolve_mujoco_geom_object_type()
        if geom_object_type is None
        else geom_object_type
    )
    simulation: Any = _get_simulation(env)
    raw_segmentation: np.ndarray = np.asarray(
        simulation.render(
            width=resolution,
            height=resolution,
            camera_name=camera_name,
            mode="offscreen",
            segmentation=True,
        )
    )
    if raw_segmentation.ndim != 3 or raw_segmentation.shape != (
        resolution,
        resolution,
        2,
    ):
        raise RuntimeError(
            "MuJoCo segmentation backend 必须返回 "
            f"[{resolution},{resolution},2]，实际为 "
            f"shape={raw_segmentation.shape}, dtype={raw_segmentation.dtype}"
        )
    if not np.issubdtype(raw_segmentation.dtype, np.integer):
        raise RuntimeError(
            "MuJoCo segmentation object type/ID 必须为整数，实际为 "
            f"{raw_segmentation.dtype}"
        )
    # 与 get_libero_image/render_background_without_targets 完全相同的方向转换。
    oriented_segmentation: np.ndarray = raw_segmentation[
        ::-1,
        ::-1,
    ].copy()
    parsed = parse_instance_segmentation(
        oriented_segmentation,
        model=simulation.model,
        instance_roots=instance_roots,
        geom_object_type=resolved_geom_object_type,
    )
    return CapturedInstanceSegmentation(
        oriented_segmentation=oriented_segmentation,
        parsed=parsed,
        backend=identify_mujoco_backend(simulation),
        camera_name=camera_name,
        resolution=resolution,
        geom_object_type=resolved_geom_object_type,
    )


def _update_array_hash(
    digest: Any,
    *,
    name: str,
    value: np.ndarray,
) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def capture_static_scene_snapshot(
    env: Any,
    body_ids: tuple[int, ...],
) -> StaticSceneSnapshot:
    """复制 simulation state 与目标 pose，并计算可复查 SHA-256。"""

    simulation: Any = _get_simulation(env)
    if not body_ids:
        raise ValueError("static transaction body_ids 不得为空")
    if len(set(body_ids)) != len(body_ids):
        raise ValueError("static transaction body_ids 不得重复")
    number_of_bodies: int = int(simulation.model.nbody)
    if any(body_id < 0 or body_id >= number_of_bodies for body_id in body_ids):
        raise ValueError("static transaction body_id 越界")

    simulation_time: float = float(simulation.data.time)
    if not math.isfinite(simulation_time):
        raise ValueError("simulation time 不是有限值")
    qpos = np.asarray(simulation.data.qpos).copy()
    qvel = np.asarray(simulation.data.qvel).copy()
    body_xpos = np.asarray(simulation.data.body_xpos)[list(body_ids)].copy()
    body_xquat = np.asarray(simulation.data.body_xquat)[list(body_ids)].copy()
    for name, value in (
        ("qpos", qpos),
        ("qvel", qvel),
        ("body_xpos", body_xpos),
        ("body_xquat", body_xquat),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"static transaction {name} 包含 NaN/Inf")

    digest = hashlib.sha256()
    _update_array_hash(
        digest,
        name="simulation_time",
        value=np.asarray([simulation_time], dtype=np.float64),
    )
    _update_array_hash(digest, name="qpos", value=qpos)
    _update_array_hash(digest, name="qvel", value=qvel)
    _update_array_hash(
        digest,
        name="body_ids",
        value=np.asarray(body_ids, dtype=np.int64),
    )
    _update_array_hash(digest, name="body_xpos", value=body_xpos)
    _update_array_hash(digest, name="body_xquat", value=body_xquat)
    return StaticSceneSnapshot(
        simulation_time=simulation_time,
        qpos=qpos,
        qvel=qvel,
        body_ids=body_ids,
        body_xpos=body_xpos,
        body_xquat=body_xquat,
        fingerprint_sha256=digest.hexdigest(),
    )


@contextmanager
def static_scene_evidence_transaction(
    env: Any,
    body_ids: tuple[int, ...],
    *,
    pose_atol: float = STATE_POSE_ATOL_CANDIDATE,
) -> Iterator[StaticSceneEvidenceTransaction]:
    """验证 context 内所有 evidence capture 都没有推进或改变场景。"""

    if not math.isfinite(pose_atol) or pose_atol < 0.0:
        raise ValueError("state pose_atol 必须为有限非负数")
    transaction = StaticSceneEvidenceTransaction(
        before=capture_static_scene_snapshot(env, body_ids)
    )
    caught_error: Optional[BaseException] = None
    try:
        yield transaction
    except BaseException as error:
        caught_error = error

    after = capture_static_scene_snapshot(env, body_ids)
    transaction.after = after
    time_unchanged = (
        after.simulation_time == transaction.before.simulation_time
    )
    qpos_unchanged = np.array_equal(after.qpos, transaction.before.qpos)
    qvel_unchanged = np.array_equal(after.qvel, transaction.before.qvel)
    position_delta = np.abs(
        after.body_xpos - transaction.before.body_xpos
    )
    quaternion_delta = np.abs(
        after.body_xquat - transaction.before.body_xquat
    )
    transaction.maximum_position_delta = float(
        position_delta.max(initial=0.0)
    )
    transaction.maximum_quaternion_delta = float(
        quaternion_delta.max(initial=0.0)
    )
    pose_unchanged = (
        transaction.maximum_position_delta <= pose_atol
        and transaction.maximum_quaternion_delta <= pose_atol
    )
    transaction.verified = bool(
        time_unchanged and qpos_unchanged and qvel_unchanged and pose_unchanged
    )
    if not transaction.verified:
        raise RuntimeError(
            "static scene evidence transaction 改变了环境状态: "
            f"time_equal={time_unchanged}, qpos_equal={qpos_unchanged}, "
            f"qvel_equal={qvel_unchanged}, "
            f"max_xpos_delta={transaction.maximum_position_delta}, "
            f"max_xquat_delta={transaction.maximum_quaternion_delta}"
        ) from caught_error
    if caught_error is not None:
        raise caught_error
