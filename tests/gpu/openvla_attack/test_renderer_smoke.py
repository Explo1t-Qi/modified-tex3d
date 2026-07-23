"""OpenVLA nvdiffrast renderer 的真实 GPU smoke。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.assets import OBJECT_ASSETS, parse_mesh_scale
from openvla_attack.renderer import DifferentiableRenderer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="该 smoke 需要 nvdiffrast CUDA context",
)


def test_real_bowl_render_backward_and_texture_bake() -> None:
    """使用真实 LIBERO bowl 资产验证 renderer 的最短 CUDA 数据流。"""
    bowl_asset = OBJECT_ASSETS["akita_black_bowl"]
    mesh_path = Path(bowl_asset["mesh"])
    texture_path = Path(bowl_asset["texture"])
    xml_path = Path(bowl_asset["xml"])
    assert mesh_path.is_file()
    assert texture_path.is_file()
    assert xml_path.is_file()

    device = torch.device("cuda:0")
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=device,
        scale_xyz=parse_mesh_scale(xml_path),
    )

    # identity MVP 足以验证 rasterizer 调用和 tensor 数据流；这里不检查相机对齐。
    mvp = torch.eye(4, dtype=torch.float32, device=device)  # [4, 4]
    model_rotation = torch.eye(3, dtype=torch.float32, device=device)  # [3, 3]
    adversarial_rgb, clean_rgb, visibility_mask = renderer.render(
        mvp,
        resolution=(64, 64),
        return_clean=True,
        model_rot=model_rotation,
    )

    assert adversarial_rgb.shape == (1, 64, 64, 3)
    assert clean_rgb.shape == (1, 64, 64, 3)
    assert visibility_mask.shape == (1, 64, 64, 1)
    assert torch.isfinite(adversarial_rgb).all()
    assert torch.isfinite(clean_rgb).all()

    adversarial_rgb.mean().backward()
    assert renderer.adv_noise.grad is not None
    assert torch.isfinite(renderer.adv_noise.grad).all()

    baked_texture = renderer.get_baked_adv_texture()
    assert baked_texture.shape == (1, renderer.tex_h, renderer.tex_w, 3)
    assert torch.isfinite(baked_texture).all()
