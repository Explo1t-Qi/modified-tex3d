"""OpenVLA attack CLI 配置与强类型边界的回归测试。"""

from __future__ import annotations

import pytest
from draccus.parsers import decoding

from openvla.experiments.robot.libero.openvla_attack.configuration import (
    GenerateConfig,
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
