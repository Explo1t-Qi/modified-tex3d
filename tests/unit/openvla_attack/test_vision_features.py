"""共享 SigLIP feature 访问边界的 CPU 回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from openvla.experiments.robot.libero.openvla_attack.vision_features import (
    extract_siglip_patch_features,
    get_siglip_featurizer,
)


class _UnexpectedFeaturizer(nn.Module):
    """若测试错误选择 DINO 分支则立即失败。"""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        del pixel_values
        raise AssertionError("不应调用 DINO featurizer")


class _DifferentiableSigLIPFeaturizer(nn.Module):
    """构造保留输入梯度的最小 patch feature。"""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pooled: [batch_size, channels]；复制为两个 patch。
        pooled: torch.Tensor = pixel_values.mean(dim=(2, 3))
        return pooled.unsqueeze(1).expand(-1, 2, -1)


def _fake_openvla() -> SimpleNamespace:
    siglip_featurizer = _DifferentiableSigLIPFeaturizer()
    return SimpleNamespace(
        config=SimpleNamespace(
            timm_model_ids=[
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            ]
        ),
        vision_backbone=SimpleNamespace(
            featurizer=_UnexpectedFeaturizer(),
            fused_featurizer=siglip_featurizer,
        ),
    )


def test_siglip_branch_is_selected_from_checkpoint_model_ids() -> None:
    model = _fake_openvla()

    assert (
        get_siglip_featurizer(model)
        is model.vision_backbone.fused_featurizer
    )


def test_siglip_patch_features_keep_gradient_to_three_channel_input() -> None:
    model = _fake_openvla()
    pixels = torch.full(
        (1, 3, 4, 4),
        0.25,
        dtype=torch.float32,
        requires_grad=True,
    )

    features = extract_siglip_patch_features(model, pixels)
    features.sum().backward()

    assert features.shape == (1, 2, 3)
    assert pixels.grad is not None
    assert bool(torch.all(pixels.grad != 0))


def test_siglip_feature_extraction_rejects_six_channel_input() -> None:
    with pytest.raises(ValueError, match="通道数必须为 3"):
        extract_siglip_patch_features(
            _fake_openvla(),
            torch.zeros((1, 6, 4, 4)),
        )
