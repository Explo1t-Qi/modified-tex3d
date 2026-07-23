"""OpenVLA 对抗纹理实验的模型专用实现。

``attack_openvla.py`` 只保留跨 task 的实验编排；本包集中攻击目标、场景坐标、
图像合成、renderer、实验产物与单 episode 评估逻辑。这里不在包初始化时导入
``evaluation``、``frame_collection``、``optimization`` 和 ``artifacts``，
避免只使用纯计算 module 的调用方被迫加载 LIBERO/Robosuite 或图像/视频 I/O。
"""

from .action_codec import decode_action_from_generated_ids
from .assets import OBJECT_ASSETS, parse_mesh_scale
from .compositing import (
    build_single_view_samples,
    composite_foreground,
    render_and_composite,
)
from .objective import get_attack_loss

__all__: list[str] = [
    "build_single_view_samples",
    "composite_foreground",
    "decode_action_from_generated_ids",
    "get_attack_loss",
    "OBJECT_ASSETS",
    "parse_mesh_scale",
    "render_and_composite",
]
