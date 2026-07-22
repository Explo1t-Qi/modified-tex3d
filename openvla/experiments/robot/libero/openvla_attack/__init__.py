"""OpenVLA 对抗纹理实验的模型专用逻辑。

该包只承载 OpenVLA 特有的攻击语义，不负责 LIBERO 环境创建、模型加载或
实验流程编排。这样纯计算逻辑可以脱离 GPU 和仿真环境单独验证。
"""

from .action_codec import decode_action_from_generated_ids
from .objective import get_attack_loss

__all__: list[str] = ["decode_action_from_generated_ids", "get_attack_loss"]
