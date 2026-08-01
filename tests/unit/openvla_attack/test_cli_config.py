"""OpenVLA attack CLI 配置与强类型边界的回归测试。"""

from __future__ import annotations

import pytest
from draccus.parsers import decoding

from openvla.experiments.robot.libero.openvla_attack.configuration import (
    GenerateConfig,
    resolve_feature_objective,
    resolve_feature_view_mode,
    resolve_texture_parameterization,
)


def test_draccus_decodes_spectral_parameterization_from_cli_string() -> None:
    """复现用户命令的真实 Draccus dataclass decode seam。"""
    config = decoding.decode(
        GenerateConfig,
        {"texture_parameterization": "spectral"},
    )

    assert config.texture_parameterization == "spectral"
    assert resolve_texture_parameterization(
        config.texture_parameterization
    ) == "spectral"


def test_runtime_boundary_rejects_unknown_parameterization() -> None:
    with pytest.raises(ValueError, match="未知纹理参数化"):
        resolve_texture_parameterization("frequency_magic")


def test_draccus_decodes_and_runtime_narrows_siglip_objective() -> None:
    config = decoding.decode(
        GenerateConfig,
        {"feature_objective": "siglip_patch"},
    )

    assert config.feature_objective == "siglip_patch"
    assert resolve_feature_objective(
        config.feature_objective
    ) == "siglip_patch"


def test_runtime_boundary_rejects_unknown_feature_objective() -> None:
    with pytest.raises(ValueError, match="未知 feature objective"):
        resolve_feature_objective("target_model_magic")


def test_draccus_decodes_and_runtime_narrows_dual_feature_views() -> None:
    config = decoding.decode(
        GenerateConfig,
        {"feature_view_mode": "primary_wrist"},
    )

    assert config.feature_view_mode == "primary_wrist"
    assert resolve_feature_view_mode(config.feature_view_mode) == "primary_wrist"


def test_runtime_boundary_rejects_unknown_feature_view_mode() -> None:
    with pytest.raises(ValueError, match="未知 feature view mode"):
        resolve_feature_view_mode("target_model_views")


def test_draccus_decodes_source_only_spectral_gradient_audit_fields() -> None:
    config = decoding.decode(
        GenerateConfig,
        {
            "spectral_gradient_audit_enabled": True,
            "spectral_gradient_audit_only": True,
            "spectral_gradient_audit_top_k": 128,
        },
    )

    assert config.spectral_gradient_audit_enabled is True
    assert config.spectral_gradient_audit_only is True
    assert config.spectral_gradient_audit_top_k == 128
