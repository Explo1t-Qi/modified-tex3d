"""OpenVLA 攻击训练帧采集 module 的行为测试。"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

# Robosuite 的 Numba decorator 默认尝试写 site-packages cache；本测试不需要 JIT。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import openvla_attack.frame_collection as frame_collection
from openvla_attack.frame_collection import TrainingFrameCollector
from openvla_attack.scene import TargetBodyPose


@dataclass
class FakeFrameCollectionConfig:
    """覆盖训练帧采集所读取的配置字段。"""

    model_family: str = "openvla"
    num_steps_wait: int = 0
    num_train_init_states: int = 1
    train_frames_per_state: int = 1
    frame_collect_with_policy: bool = False
    collect_grasp_frames: bool = False
    grasp_pre_frames: int = 2
    grasp_post_frames: int = 0
    grasp_max_steps: int = 5
    grasp_qpos_threshold: float = 0.02
    photometric_calib_frames: int = 1
    unnorm_key: Optional[str] = "fake"


class FakeProcessorInputs(dict[str, torch.Tensor]):
    """模拟 Hugging Face BatchFeature 的 mapping 与 ``to`` interface。"""

    def to(self, device: torch.device) -> "FakeProcessorInputs":
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class FakeProcessor:
    """记录 prompt 和 PIL 图像，并返回固定 token 输入。"""

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.calls: list[tuple[str, Image.Image]] = []

    def __call__(
        self,
        prompt: str,
        *,
        images: Image.Image,
    ) -> FakeProcessorInputs:
        self.calls.append((prompt, images))
        return FakeProcessorInputs(
            {
                "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
                "pixel_values": torch.zeros((1, 6, 2, 2)),
            }
        )


class FakeImagePreprocessor:
    """测试用 identity-resize、DINO/SigLIP 双分支预处理。"""

    output_size: tuple[int, int] = (2, 2)
    siglip_index: int = 1

    def build_fused_pixel_values(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((image, image), dim=1)

    def build_siglip_pixel_values(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return image


class FakeModel:
    """实现 collector 使用的生成与 hidden-state 前向 interface。"""

    device: torch.device = torch.device("cpu")

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.siglip_input_shapes: list[tuple[int, ...]] = []
        self.config = SimpleNamespace(
            timm_model_ids=[
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            ]
        )

        class FakeSigLIP(nn.Module):
            def __init__(self, owner: "FakeModel") -> None:
                super().__init__()
                self.owner = owner

            def forward(
                self,
                pixel_values: torch.Tensor,
            ) -> torch.Tensor:
                self.owner.siglip_input_shapes.append(
                    tuple(pixel_values.shape)
                )
                return (
                    pixel_values.float()
                    .mean(dim=(2, 3))
                    .unsqueeze(1)
                )

        self.vision_backbone = SimpleNamespace(
            featurizer=nn.Identity(),
            fused_featurizer=FakeSigLIP(self),
        )

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        return torch.tensor([[10, 11, 12, 13]], dtype=torch.long)

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.forward_calls.append(kwargs)
        hidden = torch.full((1, 4, 3), 0.25)
        return SimpleNamespace(hidden_states=[hidden])


class FakeLightingRenderer:
    """记录光照校准输入的 renderer fake。"""

    def __init__(self) -> None:
        self.calibration_calls: list[
            tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]
        ] = []

    def calibrate_lighting(
        self,
        mvp: torch.Tensor,
        camera_rgb: torch.Tensor,
        *,
        ema: float,
        model_rot: torch.Tensor,
    ) -> None:
        self.calibration_calls.append((mvp, camera_rgb, ema, model_rot))


class FakeEnvironment:
    """记录采集流程中的环境推进与关闭。"""

    def __init__(self, observation: dict[str, np.ndarray]) -> None:
        self.observation = observation
        self.actions: list[list[float]] = []
        self.closed: bool = False
        self.forward_calls: int = 0
        self.env = SimpleNamespace(
            sim=SimpleNamespace(forward=self._record_forward)
        )

    def _record_forward(self) -> None:
        self.forward_calls += 1

    def reset(self) -> None:
        return None

    def set_init_state(self, initial_state: object) -> dict[str, np.ndarray]:
        del initial_state
        return self.observation

    def step(
        self,
        action: list[float],
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        self.actions.append(action)
        return self.observation, 0.0, False, {}

    def close(self) -> None:
        self.closed = True


def test_collector_builds_typed_frame_calibrates_and_closes_environment(
    monkeypatch,
) -> None:
    observation = {
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }
    env = FakeEnvironment(observation)
    camera_image = np.full((2, 2, 3), 128, dtype=np.uint8)
    background_without_target = np.full((2, 2, 3), 64, dtype=np.uint8)
    target_pose = TargetBodyPose(
        model_matrix=torch.eye(4),
        body_id=2,
        body_name="akita_black_bowl",
    )

    monkeypatch.setattr(
        frame_collection,
        "get_libero_env",
        lambda task, model_family, resolution: (env, "unused"),
    )
    monkeypatch.setattr(
        frame_collection,
        "get_libero_dummy_action",
        lambda model_family: [0.0] * 7,
    )
    monkeypatch.setattr(
        frame_collection,
        "get_libero_image",
        lambda current_observation, resolution: camera_image,
    )
    monkeypatch.setattr(
        frame_collection,
        "get_image_resize_size",
        lambda cfg: 2,
    )
    monkeypatch.setattr(
        frame_collection,
        "find_target_body_pose",
        lambda current_env, keywords, device: target_pose,
    )
    monkeypatch.setattr(
        frame_collection,
        "compute_render_mvp",
        lambda current_env, model_matrix, resolution: torch.eye(4),
    )
    monkeypatch.setattr(
        frame_collection,
        "render_background_without_target",
        lambda current_env, body_id, resolution: background_without_target,
    )
    monkeypatch.setattr(
        frame_collection,
        "decode_action_from_generated_ids",
        lambda model, generated_ids, unnorm_key: np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        frame_collection,
        "autocast",
        lambda **kwargs: nullcontext(),
    )

    model = FakeModel()
    processor = FakeProcessor()
    renderer = FakeLightingRenderer()
    collector = TrainingFrameCollector(
        cfg=FakeFrameCollectionConfig(),
        model=model,
        processor=processor,
        renderer=renderer,
        image_preprocessor=FakeImagePreprocessor(),
        search_keywords=(("akita", "bowl"),),
        feature_objective="siglip_patch",
        render_resolution=2,
    )

    frames = collector.collect(
        task=object(),
        task_description="Pick up the bowl",
        fallback_initial_state=object(),
        initial_states=[object()],
        initial_state_ids=[17],
    )

    assert len(frames) == 1
    frame = frames[0]
    assert frame["bg_tensor"].shape == (1, 3, 2, 2)
    assert frame["bg_tensor_no_obj"] is not None
    assert frame["bg_tensor_no_obj"].shape == (1, 3, 2, 2)
    torch.testing.assert_close(frame["mvp"], torch.eye(4))
    torch.testing.assert_close(frame["model_rot"], torch.eye(3))
    assert frame["clean_output_ids"].tolist() == [[10, 11, 12, 13]]
    assert frame["prompt_ids"].tolist() == [[10, 11]]
    np.testing.assert_allclose(
        frame["clean_action"],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
    )
    assert frame["executed_action"] is None
    assert frame["clean_hidden"].shape == (1, 4, 3)
    assert frame["clean_siglip_features"] is not None
    assert frame["clean_siglip_features"].shape == (1, 1, 3)
    assert frame["shared_texture_views"] == ()
    assert frame["initial_state_id"] == 17
    assert frame["collection_step_index"] == 0
    assert model.siglip_input_shapes == [(1, 3, 2, 2)]
    assert frame["processor_pixel_values"].shape == (1, 6, 2, 2)

    assert processor.calls[0][0] == (
        "In: What action should the robot take to pick up the bowl?\nOut:"
    )
    assert len(renderer.calibration_calls) == 1
    assert renderer.calibration_calls[0][2] == 0.0
    assert env.actions == [[0.0] * 7]
    assert env.forward_calls == 1
    assert env.closed is True


def test_collector_builds_dual_views_with_all_shared_texture_instances(
    monkeypatch,
) -> None:
    observation = {
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }
    env = FakeEnvironment(observation)
    primary_image = np.full((2, 2, 3), 128, dtype=np.uint8)
    wrist_image = np.full((2, 2, 3), 96, dtype=np.uint8)
    first_pose = TargetBodyPose(torch.eye(4), 2, "akita_black_bowl_1_main")
    second_matrix = torch.eye(4)
    second_matrix[0, 3] = 1.0
    second_pose = TargetBodyPose(
        second_matrix,
        3,
        "akita_black_bowl_2_main",
    )

    monkeypatch.setattr(
        frame_collection,
        "get_libero_env",
        lambda task, model_family, resolution: (env, "unused"),
    )
    monkeypatch.setattr(
        frame_collection,
        "get_libero_dummy_action",
        lambda model_family: [0.0] * 7,
    )
    monkeypatch.setattr(
        frame_collection,
        "get_libero_image",
        lambda current_observation, resolution: primary_image,
    )
    monkeypatch.setattr(
        frame_collection,
        "get_libero_wrist_image",
        lambda current_observation, resolution: wrist_image,
    )
    monkeypatch.setattr(frame_collection, "get_image_resize_size", lambda cfg: 2)
    monkeypatch.setattr(
        frame_collection,
        "find_target_body_poses",
        lambda current_env, keywords, device: (first_pose, second_pose),
    )
    monkeypatch.setattr(
        frame_collection,
        "compute_render_mvp",
        lambda current_env, model_matrix, resolution, camera_name="agentview": (
            model_matrix.clone()
            if camera_name == "agentview"
            else model_matrix.clone() * 2.0
        ),
    )
    monkeypatch.setattr(
        frame_collection,
        "render_background_without_target",
        lambda current_env, body_id, resolution: np.zeros(
            (2, 2, 3), dtype=np.uint8
        ),
    )
    hidden_calls: list[tuple[tuple[int, ...], str]] = []

    def fake_hide_all(
        current_env: object,
        body_ids: tuple[int, ...],
        resolution: int,
        *,
        camera_name: str,
    ) -> np.ndarray:
        del current_env
        assert body_ids == (2, 3)
        hidden_calls.append((body_ids, camera_name))
        value = 32 if camera_name == "agentview" else 16
        return np.full((resolution, resolution, 3), value, dtype=np.uint8)

    monkeypatch.setattr(
        frame_collection,
        "render_background_without_targets",
        fake_hide_all,
    )
    monkeypatch.setattr(
        frame_collection,
        "decode_action_from_generated_ids",
        lambda model, generated_ids, unnorm_key: np.zeros(7, dtype=np.float32),
    )
    monkeypatch.setattr(
        frame_collection,
        "autocast",
        lambda **kwargs: nullcontext(),
    )

    model = FakeModel()
    collector = TrainingFrameCollector(
        cfg=FakeFrameCollectionConfig(),
        model=model,
        processor=FakeProcessor(),
        renderer=FakeLightingRenderer(),
        image_preprocessor=FakeImagePreprocessor(),
        search_keywords=(("akita", "bowl"),),
        feature_objective="siglip_patch",
        feature_view_mode="primary_wrist",
        render_resolution=2,
    )
    frames = collector.collect(
        task=object(),
        task_description="Pick up the bowl",
        fallback_initial_state=object(),
        initial_states=[object()],
        initial_state_ids=[0],
    )

    assert len(frames) == 1
    views = frames[0]["shared_texture_views"]
    assert [view["view_name"] for view in views] == ["primary", "wrist"]
    assert [len(view["instances"]) for view in views] == [2, 2]
    assert hidden_calls == [
        ((2, 3), "agentview"),
        ((2, 3), "robot0_eye_in_hand"),
    ]
    assert model.siglip_input_shapes == [
        (1, 3, 2, 2),
        (1, 3, 2, 2),
    ]
