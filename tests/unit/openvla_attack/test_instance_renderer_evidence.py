"""共享纹理多实例 renderer evidence 编排的纯 CPU 测试。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    render_shared_texture_instances,
)


@dataclass(frozen=True)
class _FakeEvidence:
    adversarial_rgb: torch.Tensor
    clean_rgb: torch.Tensor
    visibility_mask: torch.Tensor
    raster: torch.Tensor


class _FakeRenderer:
    def __init__(self, *, inconsistent_mask: bool = False) -> None:
        self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.mapping = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.inconsistent_mask = inconsistent_mask
        self.calls: list[int] = []

    def get_render_to_geometry_mapping(self) -> torch.Tensor:
        return self.mapping

    def render_evidence(
        self,
        mvp: torch.Tensor,
        resolution: tuple[int, int],
        model_rot: torch.Tensor | None = None,
    ) -> _FakeEvidence:
        del model_rot
        instance_index = int(mvp[0, 3].item())
        self.calls.append(instance_index)
        height, width = resolution
        adversarial = torch.full(
            (1, height, width, 3),
            0.6 + 0.1 * instance_index,
        )
        clean = torch.full((1, height, width, 3), 0.5)
        raster = torch.zeros((1, height, width, 4))
        raster[..., 0] = 1.0
        raster[:, instance_index, :, 3] = 1.0
        mask = (raster[..., 3:] > 0).float()
        if self.inconsistent_mask:
            mask.zero_()
        return _FakeEvidence(adversarial, clean, mask, raster)


def _instances() -> tuple[TextureRenderInstance, ...]:
    first_mvp = torch.eye(4)
    second_mvp = torch.eye(4)
    second_mvp[0, 3] = 1.0
    return (
        {"mvp": first_mvp, "model_rot": torch.eye(3)},
        {"mvp": second_mvp, "model_rot": torch.eye(3)},
    )


def test_instances_are_stacked_in_input_order_and_layout() -> None:
    renderer = _FakeRenderer()

    evidence = render_shared_texture_instances(
        renderer,
        _instances(),
        resolution=(2, 3),
    )

    assert renderer.calls == [0, 1]
    assert evidence.adversarial_rgb.shape == (2, 3, 2, 3)
    assert evidence.clean_rgb.shape == (2, 3, 2, 3)
    assert evidence.visibility_mask.shape == (2, 1, 2, 3)
    assert evidence.raster.shape == (2, 2, 3, 4)
    torch.testing.assert_close(
        evidence.adversarial_rgb[:, 0, 0, 0],
        torch.tensor([0.6, 0.7]),
    )
    assert evidence.visibility_mask[0, 0, 0].all()
    assert evidence.visibility_mask[1, 0, 1].all()


def test_support_decode_reuses_bound_faces_and_seam_mapping() -> None:
    evidence = render_shared_texture_instances(
        _FakeRenderer(),
        _instances(),
        resolution=(2, 3),
    )

    correspondence = evidence.decode_support(
        torch.tensor([True, False, False]),
    )

    # Fake raster 在有效像素使用 (u,v)=(1,0)，因此选择 face corner 0。
    assert correspondence.support_control[0, 0, 0].tolist() == [1.0] * 3
    assert correspondence.support_control[1, 0, 1].tolist() == [1.0] * 3
    assert not bool(correspondence.support_control[0, 0, 1].any())
    assert not bool(correspondence.support_control[1, 0, 0].any())


def test_public_mask_must_match_raw_raster_validity() -> None:
    with pytest.raises(ValueError, match="mask 与 raster valid 不一致"):
        render_shared_texture_instances(
            _FakeRenderer(inconsistent_mask=True),
            _instances(),
            resolution=(2, 3),
        )


def test_trainable_clean_evidence_fails_fast() -> None:
    class TrainableCleanRenderer(_FakeRenderer):
        def render_evidence(
            self,
            mvp: torch.Tensor,
            resolution: tuple[int, int],
            model_rot: torch.Tensor | None = None,
        ) -> _FakeEvidence:
            evidence = super().render_evidence(mvp, resolution, model_rot)
            return _FakeEvidence(
                evidence.adversarial_rgb,
                evidence.clean_rgb.requires_grad_(True),
                evidence.visibility_mask,
                evidence.raster,
            )

    with pytest.raises(ValueError, match="clean renderer evidence 未断开梯度"):
        render_shared_texture_instances(
            TrainableCleanRenderer(),
            _instances(),
            resolution=(2, 3),
        )
