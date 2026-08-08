"""MuJoCo visibility capture 与静止事务的 CPU fake 测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.visibility_capture import (  # noqa: E402
    capture_instance_segmentation,
    capture_static_scene_snapshot,
    static_scene_evidence_transaction,
)
from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
)


class _FakeModel:
    nbody = 3
    ngeom = 3
    body_parentid = np.array([0, 0, 0], dtype=np.int32)
    geom_bodyid = np.array([0, 1, 2], dtype=np.int32)


class _FakeSimulation:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.data = SimpleNamespace(
            time=1.25,
            qpos=np.array([0.1, 0.2], dtype=np.float64),
            qvel=np.array([0.3, 0.4], dtype=np.float64),
            body_xpos=np.zeros((3, 3), dtype=np.float64),
            body_xquat=np.tile(
                np.array([1.0, 0.0, 0.0, 0.0]),
                (3, 1),
            ),
        )
        self.rendered_arguments: dict[str, object] = {}
        self.segmentation = np.array(
            [
                [[0, -1], [5, 1]],
                [[5, 2], [5, 1]],
            ],
            dtype=np.int32,
        )

    def render(self, **kwargs: object) -> np.ndarray:
        self.rendered_arguments = kwargs
        return self.segmentation.copy()


def _environment() -> SimpleNamespace:
    return SimpleNamespace(sim=_FakeSimulation())


def test_segmentation_capture_is_strict_oriented_and_instance_aware() -> None:
    env = _environment()

    captured = capture_instance_segmentation(
        env,
        (
            TargetInstanceRoot(1, "target_1"),
            TargetInstanceRoot(2, "target_2"),
        ),
        camera_name="agentview",
        resolution=2,
        geom_object_type=5,
    )

    assert env.sim.rendered_arguments == {
        "width": 2,
        "height": 2,
        "camera_name": "agentview",
        "mode": "offscreen",
        "segmentation": True,
    }
    np.testing.assert_array_equal(
        captured.oriented_segmentation,
        env.sim.segmentation[::-1, ::-1],
    )
    assert captured.parsed.instance_alpha[:, 0].tolist() == [
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
    assert captured.geom_object_type == 5
    assert captured.backend.simulation_class.endswith("._FakeSimulation")


@pytest.mark.parametrize(
    "invalid_segmentation",
    [
        np.zeros((2, 2), dtype=np.int32),
        np.zeros((2, 2, 2), dtype=np.float32),
    ],
)
def test_segmentation_capture_rejects_backend_schema(
    invalid_segmentation: np.ndarray,
) -> None:
    env = _environment()
    env.sim.segmentation = invalid_segmentation

    with pytest.raises(RuntimeError, match="segmentation"):
        capture_instance_segmentation(
            env,
            (TargetInstanceRoot(1, "target"),),
            camera_name="agentview",
            resolution=2,
            geom_object_type=5,
        )


def test_static_scene_snapshot_is_stable_and_owns_array_copies() -> None:
    env = _environment()
    snapshot = capture_static_scene_snapshot(env, (1, 2))

    env.sim.data.qpos[0] = 9.0

    assert snapshot.qpos.tolist() == [0.1, 0.2]
    assert len(snapshot.fingerprint_sha256) == 64


def test_static_transaction_accepts_read_only_capture() -> None:
    env = _environment()

    with static_scene_evidence_transaction(env, (1, 2)) as transaction:
        _ = env.sim.render(width=2, height=2)

    assert transaction.verified
    assert transaction.after is not None
    assert (
        transaction.after.fingerprint_sha256
        == transaction.before.fingerprint_sha256
    )


@pytest.mark.parametrize("field", ["time", "qpos", "qvel"])
def test_static_transaction_rejects_physics_state_change(field: str) -> None:
    env = _environment()

    with pytest.raises(RuntimeError, match="改变了环境状态"):
        with static_scene_evidence_transaction(env, (1, 2)):
            if field == "time":
                env.sim.data.time += 0.01
            else:
                getattr(env.sim.data, field)[0] += 0.01


def test_static_transaction_uses_pose_tolerance_only_for_body_pose() -> None:
    env = _environment()

    with static_scene_evidence_transaction(
        env,
        (1,),
        pose_atol=1e-12,
    ) as transaction:
        env.sim.data.body_xpos[1, 0] += 5e-13

    assert transaction.verified
    assert transaction.maximum_position_delta == pytest.approx(5e-13)


def test_static_transaction_rejects_pose_above_tolerance() -> None:
    env = _environment()

    with pytest.raises(RuntimeError, match="max_xpos_delta"):
        with static_scene_evidence_transaction(
            env,
            (1,),
            pose_atol=1e-12,
        ):
            env.sim.data.body_xpos[1, 0] += 2e-12


def test_original_error_propagates_when_state_remains_static() -> None:
    env = _environment()

    with pytest.raises(LookupError, match="capture failed"):
        with static_scene_evidence_transaction(env, (1,)):
            raise LookupError("capture failed")
