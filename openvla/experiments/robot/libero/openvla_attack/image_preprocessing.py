"""与 OpenVLA checkpoint processor 语义一致的可微图像预处理。

OpenVLA 的 fused vision backbone 不是“任意两个三通道 tensor 的拼接”。六通道
顺序、每个分支的 mean/std、resize interpolation 和 antialias 都由 checkpoint
processor 定义，并与 ``model.config.timm_model_ids`` 一一对应。本模块从这两个
真实对象构造不可变 specification，再组合精确 PIL forward 与 PyTorch surrogate
gradient，使 renderer RGB 到 Action/Feature loss 的梯度保持连续。

当前实现只接受 OpenVLA Spatial checkpoint 使用的 ``resize-naive``、两个相同
输出尺寸、bicubic+antialias 配置。forward 先把合成 RGB 按部署语义量化成
uint8，再调用 PIL bicubic；backward 则使用连续 PyTorch tensor resize 的梯度，
即显式的 BPDA/straight-through estimator。这样模型实际看到的像素与真实
processor 一致，同时 renderer 仍能收到有意义的代理梯度。遇到其他 checkpoint
时显式失败，禁止静默退回历史手写顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from numpy.typing import NDArray
from PIL import Image


RGBStatistics = tuple[float, float, float]
ImageSize = tuple[int, int]


class ImagePreprocessingModel(Protocol):
    """只读取视觉分支稳定标识的 OpenVLA model interface。"""

    config: Any


class ImagePreprocessingProcessor(Protocol):
    """只读取 image_processor 配置的 Hugging Face processor interface。"""

    image_processor: Any


@dataclass(frozen=True)
class VisionBranchPreprocessing:
    """一个 fused vision 分支的归一化 specification。"""

    model_id: str
    mean: RGBStatistics
    std: RGBStatistics


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} 必须是序列")
    return tuple(value)


def _rgb_statistics(value: Any, *, name: str) -> RGBStatistics:
    values: tuple[Any, ...] = _sequence(value, name=name)
    if len(values) != 3:
        raise ValueError(f"{name} 必须包含3个 RGB 数值")
    result: RGBStatistics = (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )
    if not all(torch.isfinite(torch.tensor(result)).tolist()):
        raise ValueError(f"{name} 包含 NaN/Inf")
    return result


def _image_size(value: Any, *, name: str) -> ImageSize:
    if isinstance(value, int):
        size: ImageSize = (int(value), int(value))
    else:
        values: tuple[Any, ...] = _sequence(value, name=name)
        if len(values) != 2:
            raise ValueError(f"{name} 必须是 [height,width]")
        size = (int(values[0]), int(values[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{name} 必须为正数")
    return size


def _is_bicubic(interpolation: Any) -> bool:
    """兼容 JSON 的整数3、字符串和 torchvision InterpolationMode。"""
    value: Any = getattr(interpolation, "value", interpolation)
    return value == 3 or str(value).lower() == "bicubic"


@dataclass(frozen=True)
class DifferentiableOpenVLAImageProcessor:
    """把 float NCHW RGB 映射为 checkpoint 顺序的 fused pixel values。

    ``branches`` 的顺序与 ``timm_model_ids`` 和真实 processor 完全一致。当前
    checkpoint 中是 ``DINOv2 → SigLIP``，但实现不硬编码这个位置；SigLIP
    单分支由 model ID 动态定位。

    这里的“可微”指 BPDA：PIL/uint8 路径决定 forward 数值，tensor bicubic
    路径只提供 backward surrogate。它不是在声称 PIL resize 本身可微。
    """

    branches: tuple[VisionBranchPreprocessing, ...]
    output_size: ImageSize
    siglip_index: int
    antialias: bool = True

    @classmethod
    def from_checkpoint(
        cls,
        *,
        model: ImagePreprocessingModel,
        processor: ImagePreprocessingProcessor,
    ) -> "DifferentiableOpenVLAImageProcessor":
        """从已加载 checkpoint 的 model/processor 构造并严格校验配置。"""
        model_ids: tuple[Any, ...] = _sequence(
            getattr(model.config, "timm_model_ids", None),
            name="model.config.timm_model_ids",
        )
        resolved_model_ids: tuple[str, ...] = tuple(
            str(model_id) for model_id in model_ids
        )
        if len(resolved_model_ids) != 2:
            raise ValueError(
                "当前 OpenVLA fused preprocessor 要求恰有两个视觉分支"
            )

        image_processor: Any = processor.image_processor
        resize_strategy: Any = getattr(
            image_processor,
            "image_resize_strategy",
            None,
        )
        if resize_strategy != "resize-naive":
            raise ValueError(
                "当前可微预处理只支持 image_resize_strategy='resize-naive'"
            )
        input_sizes: tuple[Any, ...] = _sequence(
            getattr(image_processor, "input_sizes", None),
            name="image_processor.input_sizes",
        )
        resize_parameters: tuple[Any, ...] = _sequence(
            getattr(image_processor, "tvf_resize_params", None),
            name="image_processor.tvf_resize_params",
        )
        crop_parameters: tuple[Any, ...] = _sequence(
            getattr(image_processor, "tvf_crop_params", None),
            name="image_processor.tvf_crop_params",
        )
        normalize_parameters: tuple[Any, ...] = _sequence(
            getattr(image_processor, "tvf_normalize_params", None),
            name="image_processor.tvf_normalize_params",
        )
        branch_count: int = len(resolved_model_ids)
        if not all(
            len(values) == branch_count
            for values in (
                input_sizes,
                resize_parameters,
                crop_parameters,
                normalize_parameters,
            )
        ):
            raise ValueError("processor 配置数量与 timm_model_ids 不一致")

        output_sizes: list[ImageSize] = []
        branches: list[VisionBranchPreprocessing] = []
        for index, model_id in enumerate(resolved_model_ids):
            input_size_values: tuple[Any, ...] = _sequence(
                input_sizes[index],
                name=f"input_sizes[{index}]",
            )
            if len(input_size_values) != 3 or int(input_size_values[0]) != 3:
                raise ValueError("每个视觉分支 input_size 必须为 [3,H,W]")
            input_size: ImageSize = (
                int(input_size_values[1]),
                int(input_size_values[2]),
            )
            resize_config: Any = resize_parameters[index]
            crop_config: Any = crop_parameters[index]
            normalize_config: Any = normalize_parameters[index]
            if not isinstance(resize_config, dict):
                raise ValueError("tvf_resize_params 每项必须为 dict")
            if not isinstance(crop_config, dict):
                raise ValueError("tvf_crop_params 每项必须为 dict")
            if not isinstance(normalize_config, dict):
                raise ValueError("tvf_normalize_params 每项必须为 dict")
            if not _is_bicubic(resize_config.get("interpolation")):
                raise ValueError("processor-equivalent resize 必须为 bicubic")
            if resize_config.get("antialias") is not True:
                raise ValueError("processor-equivalent resize 必须启用 antialias")
            resize_size: ImageSize = _image_size(
                resize_config.get("size"),
                name=f"tvf_resize_params[{index}].size",
            )
            crop_size: ImageSize = _image_size(
                crop_config.get("output_size"),
                name=f"tvf_crop_params[{index}].output_size",
            )
            if resize_size != input_size or crop_size != input_size:
                raise ValueError(
                    "resize-naive 的 resize/crop/input size 必须完全一致"
                )
            output_sizes.append(input_size)
            mean: RGBStatistics = _rgb_statistics(
                normalize_config.get("mean"),
                name=f"tvf_normalize_params[{index}].mean",
            )
            std: RGBStatistics = _rgb_statistics(
                normalize_config.get("std"),
                name=f"tvf_normalize_params[{index}].std",
            )
            if any(value <= 0.0 for value in std):
                raise ValueError("视觉归一化 std 必须为有限正数")
            branches.append(
                VisionBranchPreprocessing(
                    model_id=model_id,
                    mean=mean,
                    std=std,
                )
            )
        if len(set(output_sizes)) != 1:
            raise ValueError("当前 fused backbone 要求两个分支输出尺寸相同")

        siglip_indices: list[int] = [
            index
            for index, model_id in enumerate(resolved_model_ids)
            if "siglip" in model_id.lower()
        ]
        if len(siglip_indices) != 1:
            raise ValueError("timm_model_ids 中必须恰有一个 SigLIP 分支")
        return cls(
            branches=tuple(branches),
            output_size=output_sizes[0],
            siglip_index=siglip_indices[0],
        )

    def resize_rgb(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """返回 backward 使用的连续 bicubic+antialias surrogate。

        输入/输出均为 float NCHW RGB。正式模型 forward 不应直接使用本方法的
        返回值，而应调用 ``build_fused_pixel_values`` 或
        ``build_siglip_pixel_values``，由它们组合精确 PIL forward 与代理梯度。
        """
        if rgb_images.ndim != 4 or rgb_images.shape[1] != 3:
            raise ValueError(
                "OpenVLA RGB 输入必须为 [batch,3,height,width]，收到 "
                f"{tuple(rgb_images.shape)}"
            )
        if not torch.is_floating_point(rgb_images):
            raise ValueError("OpenVLA RGB 输入必须为浮点 tensor")
        if tuple(rgb_images.shape[-2:]) == self.output_size:
            return rgb_images
        return F.interpolate(
            rgb_images,
            size=self.output_size,
            mode="bicubic",
            align_corners=False,
            antialias=self.antialias,
        )

    def _exact_pil_resized_rgb(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """执行无梯度的部署 uint8→PIL bicubic forward。

        Args:
            rgb_images: float ``[B,3,H,W]``，通常位于 GPU，值域应为
                ``[0,1]``。越界值先 clamp；乘255后舍入到最近整数，与纹理 PNG
                bake 和 LIBERO policy uint8 输入语义一致。

        Returns:
            与输入同 device/dtype 的 ``[B,3,Hout,Wout]``。该 tensor 本身不
            携带梯度，调用方必须通过 ``_bpda_forward`` 接入 surrogate。
        """
        if rgb_images.ndim != 4 or rgb_images.shape[1] != 3:
            raise ValueError(
                "OpenVLA RGB 输入必须为 [batch,3,height,width]，收到 "
                f"{tuple(rgb_images.shape)}"
            )
        if not torch.is_floating_point(rgb_images):
            raise ValueError("OpenVLA RGB 输入必须为浮点 tensor")
        # rgb_uint8_nhwc: CPU uint8 [B,H,W,3]。round 使用 ties-to-even，和
        # NumPy rint 纹理 bake 一致；clean uint8/255 输入可无损往返。
        rgb_uint8_nhwc: NDArray[np.uint8] = (
            rgb_images.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        exact_images: list[torch.Tensor] = []
        image_array: NDArray[np.uint8]
        for image_array in rgb_uint8_nhwc:
            pil_image: Image.Image = Image.fromarray(
                image_array,
                mode="RGB",
            )
            resized_image: Image.Image = TVF.resize(
                pil_image,
                size=self.output_size,
                interpolation=Image.Resampling.BICUBIC,
                max_size=None,
                antialias=self.antialias,
            )
            exact_images.append(TVF.to_tensor(resized_image))
        exact_batch: torch.Tensor = torch.stack(exact_images, dim=0)
        return exact_batch.to(
            device=rgb_images.device,
            dtype=rgb_images.dtype,
        )

    @staticmethod
    def _bpda_forward(
        exact_forward: torch.Tensor,
        surrogate: torch.Tensor,
    ) -> torch.Tensor:
        """forward 取 exact、backward 对 surrogate 求导。"""
        if exact_forward.shape != surrogate.shape:
            raise ValueError(
                "BPDA exact/surrogate shape 不一致："
                f"{tuple(exact_forward.shape)} != {tuple(surrogate.shape)}"
            )
        # 括号内 forward 严格为零；autograd 只保留 surrogate 的梯度。
        return exact_forward + (surrogate - surrogate.detach())

    def normalize_resized_branch(
        self,
        resized_rgb: torch.Tensor,
        branch_index: int,
    ) -> torch.Tensor:
        """归一化已 resize RGB，返回 ``[B,3,Hout,Wout]``。"""
        if not 0 <= branch_index < len(self.branches):
            raise IndexError(f"视觉分支索引越界: {branch_index}")
        if tuple(resized_rgb.shape[-2:]) != self.output_size:
            raise ValueError(
                "归一化前 RGB 尺寸不匹配："
                f"{tuple(resized_rgb.shape[-2:])} != {self.output_size}"
            )
        branch: VisionBranchPreprocessing = self.branches[branch_index]
        mean: torch.Tensor = resized_rgb.new_tensor(branch.mean).view(
            1, 3, 1, 1
        )
        std: torch.Tensor = resized_rgb.new_tensor(branch.std).view(
            1, 3, 1, 1
        )
        return (resized_rgb - mean) / std

    def build_fused_pixel_values(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """按 checkpoint 分支顺序返回 BPDA ``[B,6,Hout,Wout]``。"""
        surrogate_rgb: torch.Tensor = self.resize_rgb(rgb_images)
        exact_rgb: torch.Tensor = self._exact_pil_resized_rgb(rgb_images)
        surrogate_branches: tuple[torch.Tensor, ...] = tuple(
            self.normalize_resized_branch(surrogate_rgb, branch_index)
            for branch_index in range(len(self.branches))
        )
        exact_branches: tuple[torch.Tensor, ...] = tuple(
            self.normalize_resized_branch(exact_rgb, branch_index)
            for branch_index in range(len(self.branches))
        )
        return self._bpda_forward(
            torch.cat(exact_branches, dim=1),
            torch.cat(surrogate_branches, dim=1),
        )

    def build_siglip_pixel_values(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """返回模型配置所指 SigLIP BPDA 分支 ``[B,3,Hout,Wout]``。"""
        surrogate_rgb: torch.Tensor = self.resize_rgb(rgb_images)
        exact_rgb: torch.Tensor = self._exact_pil_resized_rgb(rgb_images)
        surrogate_siglip: torch.Tensor = self.normalize_resized_branch(
            surrogate_rgb,
            self.siglip_index,
        )
        exact_siglip: torch.Tensor = self.normalize_resized_branch(
            exact_rgb,
            self.siglip_index,
        )
        return self._bpda_forward(exact_siglip, surrogate_siglip)
