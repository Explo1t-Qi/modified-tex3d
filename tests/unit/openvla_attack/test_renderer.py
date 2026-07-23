"""OpenVLA 可微 renderer module 的无 GPU 接口测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.renderer import DifferentiableRenderer


def test_renderer_module_can_be_imported_without_creating_cuda_context() -> None:
    """导入类定义不应提前构造 nvdiffrast CUDA context。"""
    assert issubclass(DifferentiableRenderer, nn.Module)
