"""OpenVLA 攻击中 MuJoCo 场景读取逻辑的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.scene import (
    compute_render_mvp,
    find_target_body_pose,
    find_target_body_poses,
    render_background_without_target,
)


class FakeMujocoModel:
    """只实现场景 module 读取字段的 MuJoCo model fake。"""

    nbody: int = 4
    ngeom: int = 2
    geom_bodyid: np.ndarray = np.array([0, 2], dtype=np.int32)
    geom_rgba: np.ndarray = np.ones((2, 4), dtype=np.float32)
    cam_fovy: np.ndarray = np.asarray([45.0, 60.0], dtype=np.float32)

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return (
            "world",
            "robot0_base",
            "akita_black_bowl_1_main",
            "akita_black_bowl_2_main",
        )[body_id]

    @staticmethod
    def camera_name2id(camera_name: str) -> int:
        return {"agentview": 0, "robot0_eye_in_hand": 1}[camera_name]


class FakeMujocoSimulation:
    """提供 body pose 与离屏渲染的最小仿真 fake。"""

    def __init__(self) -> None:
        self.model = FakeMujocoModel()
        self.data = SimpleNamespace(
            body_xpos=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                dtype=np.float32,
            ),
            # MuJoCo quaternion 顺序为 [w, x, y, z]。
            body_xquat=np.array(
                [[1.0, 0.0, 0.0, 0.0]] * 4,
                dtype=np.float32,
            ),
            cam_xpos=np.asarray(
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                dtype=np.float32,
            ),
            cam_xmat=np.asarray(
                [np.eye(3), np.eye(3)],
                dtype=np.float32,
            ).reshape(2, 9),
        )
        self.alpha_seen_during_render: float | None = None

    def render(
        self,
        *,
        width: int,
        height: int,
        camera_name: str,
        mode: str,
    ) -> np.ndarray:
        del camera_name, mode
        self.alpha_seen_during_render = float(self.model.geom_rgba[1, 3])
        return np.arange(height * width * 3, dtype=np.uint8).reshape(
            height, width, 3
        )


def test_scene_finds_target_pose_and_restores_hidden_geometry() -> None:
    simulation = FakeMujocoSimulation()
    env = SimpleNamespace(sim=simulation)

    target = find_target_body_pose(
        env,
        search_keywords=(("akita", "bowl"),),
        device=torch.device("cpu"),
    )

    assert target.body_id == 2
    assert target.body_name == "akita_black_bowl_1_main"
    torch.testing.assert_close(
        target.model_matrix[:3, 3],
        torch.tensor([0.1, 0.2, 0.3]),
    )
    torch.testing.assert_close(target.model_matrix[:3, :3], torch.eye(3))

    background = render_background_without_target(
        env,
        body_id=target.body_id,
        resolution=2,
    )

    assert background is not None
    assert simulation.alpha_seen_during_render == 0.0
    assert simulation.model.geom_rgba[1, 3] == 1.0


def test_scene_finds_all_instances_for_first_matching_keyword_group() -> None:
    """共享同一纹理的两个 bowl body 必须同时进入物理纹理 Jacobian。"""
    simulation = FakeMujocoSimulation()
    env = SimpleNamespace(sim=simulation)

    targets = find_target_body_poses(
        env,
        search_keywords=(("akita", "bowl"), ("bowl",)),
        device=torch.device("cpu"),
    )

    assert [target.body_id for target in targets] == [2, 3]
    assert [target.body_name for target in targets] == [
        "akita_black_bowl_1_main",
        "akita_black_bowl_2_main",
    ]


def test_compute_render_mvp_selects_requested_camera() -> None:
    """腕部与主视角必须使用各自外参与 FOV，不能静默共用 agentview。"""
    simulation = FakeMujocoSimulation()
    env = SimpleNamespace(sim=simulation)
    model_matrix = torch.eye(4, dtype=torch.float32)

    primary_mvp = compute_render_mvp(env, model_matrix)
    wrist_mvp = compute_render_mvp(
        env,
        model_matrix,
        camera_name="robot0_eye_in_hand",
    )

    assert primary_mvp.shape == (4, 4)
    assert wrist_mvp.shape == (4, 4)
    assert not torch.allclose(primary_mvp, wrist_mvp)
