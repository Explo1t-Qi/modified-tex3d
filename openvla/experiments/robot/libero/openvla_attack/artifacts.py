"""OpenVLA 攻击实验的路径约定与产物序列化。

本 module 集中隐藏三类容易散落到实验入口的知识：

1. run log 与 ``attack_artifacts/<run-id>`` 的目录结构；
2. 历史实验依赖的文件命名规则；
3. renderer tensor 到 ``.pt``、``.npy``、PNG 和 MP4 的转换方式。

调用方只需要携带一个 :class:`AttackArtifactStore`，不再自行拼接文件名或重复
处理 device、shape 和 dtype。这里不负责修改 LIBERO XML，也不负责覆盖或恢复
MuJoCo 的真实纹理；这些运行时资源具有独立的事务语义，应由另一个 module
管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, TextIO, TypeAlias

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PathLike: TypeAlias = str | Path


class ArtifactRenderer(Protocol):
    """保存攻击产物所需的最小 renderer interface。"""

    def get_texture_param(self) -> nn.Parameter:
        """返回可学习纹理参数，shape 为 ``[parameter_count, 3]``。"""
        ...

    def get_texture_parameterization_name(self) -> str:
        """返回当前纹理参数化的稳定名称。"""
        ...

    def get_baked_adv_texture(self) -> torch.Tensor:
        """返回 float NHWC texture，shape 为 ``[1, tex_h, tex_w, 3]``。"""
        ...


class VideoWriter(Protocol):
    """``imageio`` writer 在本 module 使用的最小 interface。"""

    def append_data(self, frame: np.ndarray) -> None:
        """追加一个 uint8 HWC RGB frame。"""
        ...

    def close(self) -> None:
        """完成编码并关闭输出文件。"""
        ...


@dataclass(frozen=True)
class OptimizationArtifactPaths:
    """一次优化完成后生成的三个核心产物路径。"""

    noise_path: Path
    texture_path: Path
    loss_history_path: Path


@dataclass(frozen=True)
class LiveSnapshotPaths:
    """一次 live-test 前保存的纹理和可学习噪声路径。"""

    texture_path: Path
    noise_path: Path


class AttackArtifactStore:
    """管理一次 OpenVLA 攻击运行中需要长期保留的目录与文件格式。

    :meth:`prepare` 是 run 级 interface，其余保存方法是 task/iteration 级
    interface。实现内部统一保留历史文件名，因此调用方不需要了解
    ``Ep``、``task_``、iteration 补零等约定。
    """

    def __init__(self, *, local_log_dir: Path, run_id: str) -> None:
        self.run_log_path: Path = local_log_dir / f"{run_id}.txt"
        self.attack_directory: Path = (
            local_log_dir / "attack_artifacts" / run_id
        )

    @classmethod
    def prepare(
        cls,
        *,
        local_log_dir: PathLike,
        run_id: str,
        create_attack_directory: bool,
    ) -> "AttackArtifactStore":
        """创建 run 根目录，并按现有配置选择是否提前创建攻击目录。

        ``local_log_dir`` 始终创建，因为评估文本无论是否启用攻击都需要写入。
        攻击目录的提前创建条件继续由入口决定；训练开始时也可以调用
        :meth:`ensure_attack_directory`，对应重构前训练函数的 ``makedirs``。
        """
        resolved_log_dir: Path = Path(local_log_dir)
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        store: AttackArtifactStore = cls(
            local_log_dir=resolved_log_dir,
            run_id=run_id,
        )
        if create_attack_directory:
            store.ensure_attack_directory()
        return store

    def ensure_attack_directory(self) -> None:
        """确保当前 run 的攻击产物目录存在。"""
        self.attack_directory.mkdir(parents=True, exist_ok=True)

    def open_run_log(self) -> TextIO:
        """以覆盖模式打开本次运行的评估文本，调用方负责关闭。"""
        return self.run_log_path.open("w")

    def gradient_log_path(self, *, episode_index: int) -> Path:
        """返回 optimizer 应写入的逐 iteration 梯度日志路径。"""
        return self.attack_directory / (
            f"Ep{episode_index}_gradient_log.txt"
        )

    @staticmethod
    def _baked_texture_to_image(
        renderer: ArtifactRenderer,
    ) -> Image.Image:
        """把 renderer 的 NHWC float texture 转成历史格式的 RGB PNG。

        renderer 返回 ``[1, tex_h, tex_w, 3]``、值域 ``[0, 1]``。转换时舍入到
        最近的 uint8 texel；这样从原始 uint8 PNG 读取后再做零增量 bake，能够
        精确恢复原像素，而不会因浮点除法后的向零截断产生偶发 ``-1``。
        """
        # baked_texture: float CPU NHWC [1, tex_h, tex_w, 3]。
        baked_texture: torch.Tensor = (
            renderer.get_baked_adv_texture().squeeze(0).cpu()
        )
        texture_pixels: np.ndarray = np.rint(
            baked_texture.numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        return Image.fromarray(texture_pixels)

    def _save_baked_texture(
        self,
        *,
        path: Path,
        renderer: ArtifactRenderer,
    ) -> Path:
        """将当前 bake 后的纹理保存到给定路径并返回该路径。"""
        self._baked_texture_to_image(renderer).save(path)
        return path

    @staticmethod
    def _parameter_artifact_tag(renderer: ArtifactRenderer) -> str:
        """把 adapter 名称转换为可读且稳定的参数产物标签。"""
        parameterization_name: str = (
            renderer.get_texture_parameterization_name()
        )
        tags: dict[str, str] = {
            "legacy_vertex": "Vertex_Noise",
            "geometry_vertex": "Geometry_Vertex_Delta",
            "spectral": "Spectral_Coefficients",
        }
        if parameterization_name not in tags:
            raise ValueError(
                f"未知纹理参数化产物名称: {parameterization_name}"
            )
        return tags[parameterization_name]

    def save_optimization_result(
        self,
        *,
        episode_index: int,
        renderer: ArtifactRenderer,
        loss_history: Sequence[float],
    ) -> OptimizationArtifactPaths:
        """保存优化终态噪声、UV texture 与逐轮 loss。

        参数保持 float tensor：legacy/Geometry 为 ``[V 或 N, 3]``，
        Spectral 为 ``[K, 3]``；
        ``loss_history`` 保存为一维 NumPy array，shape ``[num_iters]``。
        """
        parameter_tag: str = self._parameter_artifact_tag(renderer)
        noise_path: Path = self.attack_directory / (
            f"Ep{episode_index}_{parameter_tag}.pt"
        )
        texture_path: Path = self.attack_directory / (
            f"Ep{episode_index}_UV_Map.png"
        )
        loss_history_path: Path = self.attack_directory / (
            f"Ep{episode_index}_loss_history.npy"
        )

        torch.save(
            renderer.get_texture_param().detach().cpu(),
            noise_path,
        )
        self._save_baked_texture(path=texture_path, renderer=renderer)
        # loss_array: float64 [num_iters]，保持 np.array(list[float]) 的旧格式。
        loss_array: np.ndarray = np.array(loss_history)
        np.save(loss_history_path, loss_array)
        return OptimizationArtifactPaths(
            noise_path=noise_path,
            texture_path=texture_path,
            loss_history_path=loss_history_path,
        )

    def save_live_snapshot(
        self,
        *,
        episode_index: int,
        iteration: int,
        renderer: ArtifactRenderer,
    ) -> LiveSnapshotPaths:
        """保存 live-test 使用的 bake 纹理与当前优化参数。"""
        iteration_tag: str = f"iter{iteration:04d}"
        texture_path: Path = self.attack_directory / (
            f"Ep{episode_index}_LiveTexture_{iteration_tag}.png"
        )
        noise_path: Path = self.attack_directory / (
            f"Ep{episode_index}_noise_{iteration_tag}.pt"
        )
        self._save_baked_texture(path=texture_path, renderer=renderer)
        # parameter: float CPU [parameter_count, 3]。
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            noise_path,
        )
        return LiveSnapshotPaths(
            texture_path=texture_path,
            noise_path=noise_path,
        )

    def save_loaded_texture(
        self,
        *,
        task_id: int,
        renderer: ArtifactRenderer,
    ) -> Path:
        """保存从 ``load_texture_path`` 恢复后用于注入 MuJoCo 的纹理。"""
        texture_path: Path = self.attack_directory / (
            f"task_{task_id}_adv_tex_loaded.png"
        )
        return self._save_baked_texture(
            path=texture_path,
            renderer=renderer,
        )

    def save_trained_texture(
        self,
        *,
        task_id: int,
        timestamp: str,
        renderer: ArtifactRenderer,
    ) -> Path:
        """保存当前 task 训练后用于 rollout 的最终对抗纹理。"""
        texture_path: Path = self.attack_directory / (
            f"task_{task_id}_adv_texture_{timestamp}.png"
        )
        return self._save_baked_texture(
            path=texture_path,
            renderer=renderer,
        )

    def save_live_test_video(
        self,
        *,
        episode_index: int,
        iteration: int,
        success: bool,
        frames: Sequence[np.ndarray],
        fps: int = 30,
    ) -> Path:
        """把 live-test RGB 帧写为 MP4，并保留历史命名规则。

        每帧应为 uint8 HWC RGB array，shape ``[height, width, 3]``。当前行为
        与旧实现一致：这里不改变颜色通道、不缩放，也不补空帧。
        """
        video_path: Path = self.attack_directory / (
            f"Ep{episode_index}_LiveTest_iter{iteration:04d}_"
            f"success={success}.mp4"
        )
        writer: VideoWriter = imageio.get_writer(video_path, fps=fps)
        try:
            frame: np.ndarray
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return video_path
