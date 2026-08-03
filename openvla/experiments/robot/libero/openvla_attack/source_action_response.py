"""固定源 OpenVLA 纹理的单步动作响应诊断。

当前攻击用 teacher-forced 的“对称 action bin 交叉熵”优化纹理，但该标量下降
不等价于贪心生成 token 改变，更不等价于连续机器人动作改变。本模块在完全
相同的训练初始观测上并列测量两条数据流：

``clean/adv RGB -> teacher-forced logits -> 对称 bin CE / 决策 margin``

``clean/adv RGB -> greedy generation -> action token -> 连续 action``

模块只读取一份已经训练完成的谱系数，不更新纹理，也不激活 MuJoCo 运行资产。
因此结果只用于定位源模型代理目标与单步策略决策是否对齐，不能替代闭环 rollout。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.cuda.amp import autocast

from .action_codec import (
    FloatingArray,
    OpenVLAActionModel,
    decode_action_from_generated_ids,
)
from .compositing import (
    ForegroundRenderer,
    MultiInstanceViewFrame,
    build_multi_instance_view_sample,
)
from .frame_collection import TrainingFrame
from .image_preprocessing import DifferentiableOpenVLAImageProcessor
from .objective import (
    ACTION_TOKEN_END,
    ACTION_TOKEN_START,
    ActionTokenLogits,
    extract_action_token_logits,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SourceActionResponseError(RuntimeError):
    """动作响应诊断的输入、参考纹理或模型输出不满足约束。"""


class SourceActionResponseModel(OpenVLAActionModel, Protocol):
    """诊断所需的最小 OpenVLA 模型 interface。"""

    device: torch.device

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """返回完整生成序列，shape ``[1, prompt_length + action_dim]``。"""
        ...

    def __call__(self, **kwargs: Any) -> Any:
        """返回至少包含 ``logits`` 的模型前向结果。"""
        ...


class SourceActionResponseRenderer(ForegroundRenderer, Protocol):
    """固定谱纹理合成主视角所需的 renderer interface。"""

    def reset_texture(self) -> None:
        """把谱系数恢复为零 Surface Delta。"""
        ...

    def get_texture_param(self) -> nn.Parameter:
        """返回谱系数，shape ``[K, 3]``。"""
        ...

    def get_texture_parameterization_name(self) -> str:
        """返回稳定参数化名称；本诊断要求 ``spectral``。"""
        ...


@dataclass(frozen=True)
class ActionResponseSample:
    """一个固定观测上的 teacher-forced 与 greedy-generation 响应。

    所有逐动作维数组 shape 均为 ``[action_dim]``。margin 定义为前者减后者；
    ``target_minus_best_other_logit > 0`` 才表示对称 target 已成为 action 子词表
    argmax，而不只是 CE 相对变小。
    """

    clean_generated_token_ids: IntArray
    adversarial_generated_token_ids: IntArray
    processor_generated_token_ids: IntArray
    clean_actions: FloatArray
    adversarial_actions: FloatArray
    clean_classes: IntArray
    symmetric_target_classes: IntArray
    clean_teacher_argmax_classes: IntArray
    adversarial_teacher_argmax_classes: IntArray
    processor_teacher_argmax_classes: IntArray
    clean_decision_margin: FloatArray
    adversarial_decision_margin: FloatArray
    processor_decision_margin: FloatArray
    clean_target_ce: float
    adversarial_target_ce: float
    clean_target_minus_clean_logit: FloatArray
    adversarial_target_minus_clean_logit: FloatArray
    clean_target_minus_best_other_logit: FloatArray
    adversarial_target_minus_best_other_logit: FloatArray
    clean_target_minus_clean_probability: FloatArray
    adversarial_target_minus_clean_probability: FloatArray
    clean_target_minus_best_other_probability: FloatArray
    adversarial_target_minus_best_other_probability: FloatArray
    processor_pixel_mae: float
    processor_pixel_linf: float

    @property
    def token_hamming_count(self) -> int:
        return int(
            np.count_nonzero(
                self.clean_generated_token_ids
                != self.adversarial_generated_token_ids
            )
        )

    @property
    def collector_clean_token_ids(self) -> IntArray:
        """帧采集器保存、并被攻击损失用作 label 的 clean token ``[A]``。"""
        return self.clean_classes + ACTION_TOKEN_START

    @property
    def clean_regeneration_hamming_count(self) -> int:
        """可微预处理重生成与 collector clean label 的差异数。"""
        return int(
            np.count_nonzero(
                self.clean_generated_token_ids
                != self.collector_clean_token_ids
            )
        )

    @property
    def processor_equivalence_hamming_count(self) -> int:
        """真实 processor 与可微预处理生成 token 的差异数。"""
        return int(
            np.count_nonzero(
                self.processor_generated_token_ids
                != self.clean_generated_token_ids
            )
        )

    @property
    def action_l2(self) -> float:
        return float(
            np.linalg.norm(self.adversarial_actions - self.clean_actions)
        )

    @property
    def action_linf(self) -> float:
        return float(
            np.max(np.abs(self.adversarial_actions - self.clean_actions))
        )


def _selected_values(values: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """从 ``[A,256]`` 中按每行 class 取值，返回 ``[A]``。"""
    return values.gather(1, classes.unsqueeze(1)).squeeze(1)


def _best_other_values(
    values: torch.Tensor,
    excluded_classes: torch.Tensor,
) -> torch.Tensor:
    """排除指定 class 后返回每个动作维的最大值，shape ``[A]``。"""
    masked_values: torch.Tensor = values.clone()
    masked_values.scatter_(
        1,
        excluded_classes.unsqueeze(1),
        float("-inf"),
    )
    return masked_values.amax(dim=1)


def _top1_minus_top2(values: torch.Tensor) -> torch.Tensor:
    """返回每行 top1−top2 logit margin，shape ``[A]``。"""
    if values.ndim != 2 or values.shape[1] < 2:
        raise SourceActionResponseError(
            f"决策 margin 要求 [A,num_classes>=2]，收到 {tuple(values.shape)}"
        )
    top_two: torch.Tensor = values.topk(k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def _as_float64_numpy(tensor: torch.Tensor) -> FloatArray:
    return tensor.detach().to(torch.float64).cpu().numpy()


def _as_int64_numpy(tensor: torch.Tensor) -> IntArray:
    return tensor.detach().to(torch.int64).cpu().numpy()


def compute_action_response_sample(
    *,
    clean_action_logits: torch.Tensor,
    adversarial_action_logits: torch.Tensor,
    processor_action_logits: torch.Tensor,
    clean_classes: torch.Tensor,
    symmetric_target_classes: torch.Tensor,
    clean_generated_token_ids: torch.Tensor,
    adversarial_generated_token_ids: torch.Tensor,
    processor_generated_token_ids: torch.Tensor,
    clean_actions: FloatingArray,
    adversarial_actions: FloatingArray,
    processor_pixel_mae: float,
    processor_pixel_linf: float,
) -> ActionResponseSample:
    """由模型输出计算一个状态的可复查诊断量。

    Args:
        两个 logits: 浮点 ``[action_dim, 256]``。
        两个 class: 整数 ``[action_dim]``，来自实际攻击目标的 clean labels。
        三个 generated token: 整数 ``[action_dim]``，分别来自可微 clean、
            adversarial 与真实 processor clean 贪心生成。
        两个 action: 浮点 NumPy ``[action_dim]``，由同一 codec 反归一化。
    """
    expected_shape: tuple[int, int] = (
        int(clean_classes.numel()),
        ACTION_TOKEN_END - ACTION_TOKEN_START,
    )
    if tuple(clean_action_logits.shape) != expected_shape:
        raise SourceActionResponseError(
            "clean action logits shape 不匹配："
            f"{tuple(clean_action_logits.shape)} != {expected_shape}"
        )
    if tuple(adversarial_action_logits.shape) != expected_shape:
        raise SourceActionResponseError(
            "adversarial action logits shape 不匹配："
            f"{tuple(adversarial_action_logits.shape)} != {expected_shape}"
        )
    if tuple(processor_action_logits.shape) != expected_shape:
        raise SourceActionResponseError(
            "processor action logits shape 不匹配："
            f"{tuple(processor_action_logits.shape)} != {expected_shape}"
        )
    action_dim: int = expected_shape[0]
    for name, tensor in (
        ("symmetric_target_classes", symmetric_target_classes),
        ("clean_generated_token_ids", clean_generated_token_ids),
        ("adversarial_generated_token_ids", adversarial_generated_token_ids),
        ("processor_generated_token_ids", processor_generated_token_ids),
    ):
        if tuple(tensor.shape) != (action_dim,):
            raise SourceActionResponseError(
                f"{name} shape 必须为 {(action_dim,)}，收到 {tuple(tensor.shape)}"
            )
    clean_action_array: FloatArray = np.asarray(
        clean_actions,
        dtype=np.float64,
    )
    adversarial_action_array: FloatArray = np.asarray(
        adversarial_actions,
        dtype=np.float64,
    )
    if clean_action_array.shape != (action_dim,) or (
        adversarial_action_array.shape != (action_dim,)
    ):
        raise SourceActionResponseError("连续 action shape 与 action_dim 不一致")

    def teacher_metrics(
        action_logits: torch.Tensor,
    ) -> tuple[
        float,
        IntArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        logits_float: torch.Tensor = action_logits.float()
        probabilities: torch.Tensor = torch.softmax(logits_float, dim=1)
        target_logits: torch.Tensor = _selected_values(
            logits_float,
            symmetric_target_classes,
        )
        clean_logits: torch.Tensor = _selected_values(
            logits_float,
            clean_classes,
        )
        target_probabilities: torch.Tensor = _selected_values(
            probabilities,
            symmetric_target_classes,
        )
        clean_probabilities: torch.Tensor = _selected_values(
            probabilities,
            clean_classes,
        )
        best_other_logits: torch.Tensor = _best_other_values(
            logits_float,
            symmetric_target_classes,
        )
        best_other_probabilities: torch.Tensor = _best_other_values(
            probabilities,
            symmetric_target_classes,
        )
        target_ce: float = float(
            F.cross_entropy(
                logits_float,
                symmetric_target_classes,
            ).item()
        )
        return (
            target_ce,
            _as_int64_numpy(logits_float.argmax(dim=1)),
            _as_float64_numpy(target_logits - clean_logits),
            _as_float64_numpy(target_logits - best_other_logits),
            _as_float64_numpy(target_probabilities - clean_probabilities),
            _as_float64_numpy(
                target_probabilities - best_other_probabilities
            ),
            _as_float64_numpy(_top1_minus_top2(logits_float)),
        )

    clean_metrics = teacher_metrics(clean_action_logits)
    adversarial_metrics = teacher_metrics(adversarial_action_logits)
    processor_metrics = teacher_metrics(processor_action_logits)
    return ActionResponseSample(
        clean_generated_token_ids=_as_int64_numpy(
            clean_generated_token_ids
        ),
        adversarial_generated_token_ids=_as_int64_numpy(
            adversarial_generated_token_ids
        ),
        processor_generated_token_ids=_as_int64_numpy(
            processor_generated_token_ids
        ),
        clean_actions=clean_action_array,
        adversarial_actions=adversarial_action_array,
        clean_classes=_as_int64_numpy(clean_classes),
        symmetric_target_classes=_as_int64_numpy(
            symmetric_target_classes
        ),
        clean_teacher_argmax_classes=clean_metrics[1],
        adversarial_teacher_argmax_classes=adversarial_metrics[1],
        processor_teacher_argmax_classes=processor_metrics[1],
        clean_decision_margin=clean_metrics[6],
        adversarial_decision_margin=adversarial_metrics[6],
        processor_decision_margin=processor_metrics[6],
        clean_target_ce=clean_metrics[0],
        adversarial_target_ce=adversarial_metrics[0],
        clean_target_minus_clean_logit=clean_metrics[2],
        adversarial_target_minus_clean_logit=adversarial_metrics[2],
        clean_target_minus_best_other_logit=clean_metrics[3],
        adversarial_target_minus_best_other_logit=adversarial_metrics[3],
        clean_target_minus_clean_probability=clean_metrics[4],
        adversarial_target_minus_clean_probability=adversarial_metrics[4],
        clean_target_minus_best_other_probability=clean_metrics[5],
        adversarial_target_minus_best_other_probability=adversarial_metrics[5],
        processor_pixel_mae=float(processor_pixel_mae),
        processor_pixel_linf=float(processor_pixel_linf),
    )


@dataclass(frozen=True)
class SourceActionResponsePaths:
    """一次动作响应诊断的完整数组、逐状态表格与摘要路径。"""

    npz_path: Path
    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class SourceActionResponseResult:
    """跨固定状态堆叠的源模型动作响应结果。

    ``state_ids``、``step_indices`` 与逐状态标量 shape 为 ``[S]``；token、
    action、class、margin 数组 shape 均为 ``[S, action_dim]``。
    """

    state_ids: IntArray
    step_indices: IntArray
    samples: tuple[ActionResponseSample, ...]
    reference_path: str
    reference_sha256: str

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def action_dim(self) -> int:
        return int(self.samples[0].clean_actions.shape[0])

    def _stack(self, field_name: str) -> NDArray[np.generic]:
        return np.stack(
            [getattr(sample, field_name) for sample in self.samples],
            axis=0,
        )

    def _summary(self) -> dict[str, Any]:
        token_hamming: FloatArray = np.asarray(
            [sample.token_hamming_count for sample in self.samples],
            dtype=np.float64,
        )
        clean_regeneration_hamming: FloatArray = np.asarray(
            [
                sample.clean_regeneration_hamming_count
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        processor_equivalence_hamming: FloatArray = np.asarray(
            [
                sample.processor_equivalence_hamming_count
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        processor_pixel_mae: FloatArray = np.asarray(
            [sample.processor_pixel_mae for sample in self.samples],
            dtype=np.float64,
        )
        processor_pixel_linf: FloatArray = np.asarray(
            [sample.processor_pixel_linf for sample in self.samples],
            dtype=np.float64,
        )
        action_l2: FloatArray = np.asarray(
            [sample.action_l2 for sample in self.samples],
            dtype=np.float64,
        )
        action_linf: FloatArray = np.asarray(
            [sample.action_linf for sample in self.samples],
            dtype=np.float64,
        )
        clean_ce: FloatArray = np.asarray(
            [sample.clean_target_ce for sample in self.samples],
            dtype=np.float64,
        )
        adversarial_ce: FloatArray = np.asarray(
            [sample.adversarial_target_ce for sample in self.samples],
            dtype=np.float64,
        )
        adversarial_target_margin: FloatArray = self._stack(
            "adversarial_target_minus_best_other_logit"
        ).astype(np.float64)
        adversarial_generated_ids: IntArray = self._stack(
            "adversarial_generated_token_ids"
        ).astype(np.int64)
        clean_generated_ids: IntArray = self._stack(
            "clean_generated_token_ids"
        ).astype(np.int64)
        clean_teacher_argmax: IntArray = self._stack(
            "clean_teacher_argmax_classes"
        ).astype(np.int64)
        adversarial_teacher_argmax: IntArray = self._stack(
            "adversarial_teacher_argmax_classes"
        ).astype(np.int64)
        processor_teacher_argmax: IntArray = self._stack(
            "processor_teacher_argmax_classes"
        ).astype(np.int64)
        first_divergence_indices: list[int] = []
        first_divergence_clean_margins: list[float] = []
        first_divergence_processor_margins: list[float] = []
        first_divergence_teacher_matches: list[bool] = []
        sample: ActionResponseSample
        for sample in self.samples:
            divergent_indices: NDArray[np.int64] = np.flatnonzero(
                sample.processor_generated_token_ids
                != sample.clean_generated_token_ids
            ).astype(np.int64)
            if divergent_indices.size == 0:
                continue
            token_index: int = int(divergent_indices[0])
            first_divergence_indices.append(token_index)
            first_divergence_clean_margins.append(
                float(sample.clean_decision_margin[token_index])
            )
            first_divergence_processor_margins.append(
                float(sample.processor_decision_margin[token_index])
            )
            first_divergence_teacher_matches.append(
                int(sample.processor_teacher_argmax_classes[token_index])
                == int(
                    sample.processor_generated_token_ids[token_index]
                    - ACTION_TOKEN_START
                )
            )
        target_token_ids: IntArray = (
            self._stack("symmetric_target_classes").astype(np.int64)
            + ACTION_TOKEN_START
        )
        return {
            "diagnostic_scope": "source_openvla_fixed_initial_observations",
            "interpretation_scope": (
                "single_step_attack_training_preprocessing; "
                "not_deployment_center_crop_or_closed_loop"
            ),
            "num_samples": self.num_samples,
            "action_dim": self.action_dim,
            "state_ids": self.state_ids.tolist(),
            "step_indices": self.step_indices.tolist(),
            "reference": {
                "path": self.reference_path,
                "sha256": self.reference_sha256,
            },
            "processor_equivalence": {
                "pixel_values_mae_mean": float(processor_pixel_mae.mean()),
                "pixel_values_linf_max": float(processor_pixel_linf.max()),
                "mean_token_hamming_count": float(
                    processor_equivalence_hamming.mean()
                ),
                "exact_token_match_fraction": float(
                    np.mean(processor_equivalence_hamming == 0)
                ),
                "first_divergence": {
                    "num_mismatched_samples": len(
                        first_divergence_indices
                    ),
                    "mean_token_index": (
                        float(np.mean(first_divergence_indices))
                        if first_divergence_indices
                        else None
                    ),
                    "clean_margin_mean": (
                        float(np.mean(first_divergence_clean_margins))
                        if first_divergence_clean_margins
                        else None
                    ),
                    "processor_margin_mean": (
                        float(np.mean(first_divergence_processor_margins))
                        if first_divergence_processor_margins
                        else None
                    ),
                    "teacher_matches_generated_fraction": (
                        float(np.mean(first_divergence_teacher_matches))
                        if first_divergence_teacher_matches
                        else None
                    ),
                },
            },
            "teacher_forced_first_token_consistency": {
                "clean_match_fraction": float(
                    np.mean(
                        clean_teacher_argmax[:, 0]
                        == clean_generated_ids[:, 0] - ACTION_TOKEN_START
                    )
                ),
                "adversarial_match_fraction": float(
                    np.mean(
                        adversarial_teacher_argmax[:, 0]
                        == adversarial_generated_ids[:, 0]
                        - ACTION_TOKEN_START
                    )
                ),
                "processor_match_fraction": float(
                    np.mean(
                        processor_teacher_argmax[:, 0]
                        == (
                            self._stack("processor_generated_token_ids")[
                                :, 0
                            ]
                            - ACTION_TOKEN_START
                        )
                    )
                ),
            },
            "greedy_generation": {
                "collector_vs_differentiable_clean_mean_token_hamming_count": float(
                    clean_regeneration_hamming.mean()
                ),
                "collector_vs_differentiable_clean_exact_match_fraction": float(
                    np.mean(clean_regeneration_hamming == 0)
                ),
                "mean_token_hamming_count": float(token_hamming.mean()),
                "states_with_any_token_change_fraction": float(
                    np.mean(token_hamming > 0)
                ),
                "mean_action_l2": float(action_l2.mean()),
                "mean_action_linf": float(action_linf.mean()),
                "generated_symmetric_target_fraction": float(
                    np.mean(adversarial_generated_ids == target_token_ids)
                ),
            },
            "teacher_forced_proxy": {
                "clean_symmetric_target_ce_mean": float(clean_ce.mean()),
                "adversarial_symmetric_target_ce_mean": float(
                    adversarial_ce.mean()
                ),
                "symmetric_target_ce_reduction_mean": float(
                    (clean_ce - adversarial_ce).mean()
                ),
                "adversarial_symmetric_target_argmax_fraction": float(
                    np.mean(adversarial_target_margin > 0.0)
                ),
                "adversarial_target_minus_best_other_logit_mean": float(
                    adversarial_target_margin.mean()
                ),
            },
        }

    def save(
        self,
        *,
        output_directory: str | Path,
        task_id: int,
    ) -> SourceActionResponsePaths:
        """保存详细 NPZ、逐状态 CSV 和可直接阅读的 JSON 摘要。"""
        directory: Path = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem: str = f"Ep{task_id}_Source_Action_Response"
        paths = SourceActionResponsePaths(
            npz_path=directory / f"{stem}.npz",
            csv_path=directory / f"{stem}.csv",
            json_path=directory / f"{stem}.json",
        )
        array_fields: tuple[str, ...] = (
            "clean_generated_token_ids",
            "adversarial_generated_token_ids",
            "processor_generated_token_ids",
            "clean_actions",
            "adversarial_actions",
            "clean_classes",
            "symmetric_target_classes",
            "clean_teacher_argmax_classes",
            "adversarial_teacher_argmax_classes",
            "processor_teacher_argmax_classes",
            "clean_decision_margin",
            "adversarial_decision_margin",
            "processor_decision_margin",
            "clean_target_minus_clean_logit",
            "adversarial_target_minus_clean_logit",
            "clean_target_minus_best_other_logit",
            "adversarial_target_minus_best_other_logit",
            "clean_target_minus_clean_probability",
            "adversarial_target_minus_clean_probability",
            "clean_target_minus_best_other_probability",
            "adversarial_target_minus_best_other_probability",
        )
        payload: dict[str, NDArray[np.generic]] = {
            "state_ids": self.state_ids,
            "step_indices": self.step_indices,
            "clean_target_ce": np.asarray(
                [sample.clean_target_ce for sample in self.samples],
                dtype=np.float64,
            ),
            "adversarial_target_ce": np.asarray(
                [sample.adversarial_target_ce for sample in self.samples],
                dtype=np.float64,
            ),
            "token_hamming_count": np.asarray(
                [sample.token_hamming_count for sample in self.samples],
                dtype=np.int64,
            ),
            "clean_regeneration_hamming_count": np.asarray(
                [
                    sample.clean_regeneration_hamming_count
                    for sample in self.samples
                ],
                dtype=np.int64,
            ),
            "processor_equivalence_hamming_count": np.asarray(
                [
                    sample.processor_equivalence_hamming_count
                    for sample in self.samples
                ],
                dtype=np.int64,
            ),
            "processor_pixel_mae": np.asarray(
                [sample.processor_pixel_mae for sample in self.samples],
                dtype=np.float64,
            ),
            "processor_pixel_linf": np.asarray(
                [sample.processor_pixel_linf for sample in self.samples],
                dtype=np.float64,
            ),
            "action_l2": np.asarray(
                [sample.action_l2 for sample in self.samples],
                dtype=np.float64,
            ),
            "action_linf": np.asarray(
                [sample.action_linf for sample in self.samples],
                dtype=np.float64,
            ),
        }
        payload.update(
            {field_name: self._stack(field_name) for field_name in array_fields}
        )
        payload["collector_clean_token_ids"] = self._stack(
            "collector_clean_token_ids"
        )
        np.savez_compressed(paths.npz_path, **payload)

        with paths.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "state_id",
                    "step_index",
                    "token_hamming_count",
                    "clean_regeneration_hamming_count",
                    "processor_equivalence_hamming_count",
                    "processor_pixel_mae",
                    "processor_pixel_linf",
                    "processor_first_divergence_index",
                    "processor_first_divergence_clean_margin",
                    "processor_first_divergence_processor_margin",
                    "action_l2",
                    "action_linf",
                    "clean_symmetric_target_ce",
                    "adversarial_symmetric_target_ce",
                    "adversarial_target_argmax_fraction",
                )
            )
            for state_id, step_index, sample in zip(
                self.state_ids,
                self.step_indices,
                self.samples,
            ):
                divergent_indices: NDArray[np.int64] = np.flatnonzero(
                    sample.processor_generated_token_ids
                    != sample.clean_generated_token_ids
                ).astype(np.int64)
                first_divergence_index: int = (
                    int(divergent_indices[0])
                    if divergent_indices.size > 0
                    else -1
                )
                first_divergence_clean_margin: Optional[float] = (
                    float(
                        sample.clean_decision_margin[
                            first_divergence_index
                        ]
                    )
                    if first_divergence_index >= 0
                    else None
                )
                first_divergence_processor_margin: Optional[float] = (
                    float(
                        sample.processor_decision_margin[
                            first_divergence_index
                        ]
                    )
                    if first_divergence_index >= 0
                    else None
                )
                writer.writerow(
                    (
                        int(state_id),
                        int(step_index),
                        sample.token_hamming_count,
                        sample.clean_regeneration_hamming_count,
                        sample.processor_equivalence_hamming_count,
                        sample.processor_pixel_mae,
                        sample.processor_pixel_linf,
                        first_divergence_index,
                        first_divergence_clean_margin,
                        first_divergence_processor_margin,
                        sample.action_l2,
                        sample.action_linf,
                        sample.clean_target_ce,
                        sample.adversarial_target_ce,
                        float(
                            np.mean(
                                sample.adversarial_target_minus_best_other_logit
                                > 0.0
                            )
                        ),
                    )
                )
        paths.json_path.write_text(
            json.dumps(self._summary(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return paths


class SourceActionResponseAuditor:
    """在固定训练帧上比较零纹理与一份已训练谱纹理的动作响应。"""

    def __init__(
        self,
        *,
        model: SourceActionResponseModel,
        renderer: SourceActionResponseRenderer,
        image_preprocessor: DifferentiableOpenVLAImageProcessor,
        reference_parameter_path: str | Path,
        unnorm_key: Optional[str],
        pad_token_id: Optional[int],
        render_resolution: int = 256,
    ) -> None:
        self._model: SourceActionResponseModel = model
        self._renderer: SourceActionResponseRenderer = renderer
        self._image_preprocessor: DifferentiableOpenVLAImageProcessor = (
            image_preprocessor
        )
        self._reference_path: Path = Path(
            reference_parameter_path
        ).resolve()
        self._unnorm_key: Optional[str] = unnorm_key
        self._pad_token_id: Optional[int] = pad_token_id
        self._render_resolution: int = render_resolution

    @staticmethod
    def _primary_view(frame: TrainingFrame) -> MultiInstanceViewFrame:
        primary_views: list[MultiInstanceViewFrame] = [
            view
            for view in frame["shared_texture_views"]
            if view["view_name"] == "primary"
        ]
        if len(primary_views) != 1:
            raise SourceActionResponseError(
                "动作响应诊断要求每帧恰有一个 multi-instance primary 视角"
            )
        return primary_views[0]

    def _pixel_values(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """复现攻击 Action loss 的 checkpoint-order 六通道输入。"""
        return self._image_preprocessor.build_fused_pixel_values(
            image
        ).to(torch.bfloat16)

    def _generate(
        self,
        *,
        frame: TrainingFrame,
        pixel_values: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        generation_arguments: dict[str, Any] = {
            "input_ids": frame["prompt_ids"],
            "attention_mask": torch.ones_like(frame["prompt_ids"]),
            "pixel_values": pixel_values,
            "max_new_tokens": action_dim,
            "do_sample": False,
        }
        if self._pad_token_id is not None:
            generation_arguments["pad_token_id"] = self._pad_token_id
        return self._model.generate(**generation_arguments)

    def _sample(
        self,
        *,
        frame: TrainingFrame,
        clean_pixel_values: torch.Tensor,
        adversarial_pixel_values: torch.Tensor,
        processor_pixel_values: torch.Tensor,
    ) -> ActionResponseSample:
        clean_labels: torch.Tensor = frame["clean_output_ids"]
        action_dim: int = self._model.get_action_dim(self._unnorm_key)
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            clean_outputs: Any = self._model(
                input_ids=clean_labels,
                attention_mask=torch.ones_like(clean_labels),
                pixel_values=clean_pixel_values,
                output_hidden_states=False,
            )
            adversarial_outputs: Any = self._model(
                input_ids=clean_labels,
                attention_mask=torch.ones_like(clean_labels),
                pixel_values=adversarial_pixel_values,
                output_hidden_states=False,
            )
            processor_outputs: Any = self._model(
                input_ids=clean_labels,
                attention_mask=torch.ones_like(clean_labels),
                pixel_values=processor_pixel_values,
                output_hidden_states=False,
            )
            clean_generated_ids: torch.Tensor = self._generate(
                frame=frame,
                pixel_values=clean_pixel_values,
                action_dim=action_dim,
            )
            adversarial_generated_ids: torch.Tensor = self._generate(
                frame=frame,
                pixel_values=adversarial_pixel_values,
                action_dim=action_dim,
            )
            processor_generated_ids: torch.Tensor = self._generate(
                frame=frame,
                pixel_values=processor_pixel_values,
                action_dim=action_dim,
            )

        clean_teacher: ActionTokenLogits = extract_action_token_logits(
            clean_outputs.logits,
            clean_labels,
        )
        adversarial_teacher: ActionTokenLogits = extract_action_token_logits(
            adversarial_outputs.logits,
            clean_labels,
        )
        processor_teacher: ActionTokenLogits = extract_action_token_logits(
            processor_outputs.logits,
            clean_labels,
        )
        if not torch.equal(
            clean_teacher.clean_classes,
            adversarial_teacher.clean_classes,
        ):
            raise SourceActionResponseError("clean/adv teacher labels 不一致")
        if not torch.equal(
            clean_teacher.clean_classes,
            processor_teacher.clean_classes,
        ):
            raise SourceActionResponseError(
                "clean/processor teacher labels 不一致"
            )
        if clean_teacher.logits.shape[0] != action_dim:
            raise SourceActionResponseError(
                "teacher-forced action token 数与 checkpoint action_dim 不一致："
                f"{clean_teacher.logits.shape[0]} != {action_dim}"
            )

        clean_action_tokens: torch.Tensor = clean_generated_ids[0, -action_dim:]
        adversarial_action_tokens: torch.Tensor = (
            adversarial_generated_ids[0, -action_dim:]
        )
        processor_action_tokens: torch.Tensor = (
            processor_generated_ids[0, -action_dim:]
        )
        for name, token_ids in (
            ("clean", clean_action_tokens),
            ("adversarial", adversarial_action_tokens),
            ("processor", processor_action_tokens),
        ):
            valid_tokens: torch.Tensor = (
                (token_ids >= ACTION_TOKEN_START)
                & (token_ids < ACTION_TOKEN_END)
            )
            if not bool(valid_tokens.all().item()):
                raise SourceActionResponseError(
                    f"{name} greedy generation 产生非 action token："
                    f"{token_ids.detach().cpu().tolist()}"
                )

        clean_action: FloatingArray = decode_action_from_generated_ids(
            self._model,
            clean_generated_ids,
            self._unnorm_key,
        )
        adversarial_action: FloatingArray = decode_action_from_generated_ids(
            self._model,
            adversarial_generated_ids,
            self._unnorm_key,
        )
        return compute_action_response_sample(
            clean_action_logits=clean_teacher.logits,
            adversarial_action_logits=adversarial_teacher.logits,
            processor_action_logits=processor_teacher.logits,
            clean_classes=clean_teacher.clean_classes,
            symmetric_target_classes=(
                clean_teacher.symmetric_target_classes
            ),
            clean_generated_token_ids=clean_action_tokens,
            adversarial_generated_token_ids=adversarial_action_tokens,
            processor_generated_token_ids=processor_action_tokens,
            clean_actions=clean_action,
            adversarial_actions=adversarial_action,
            processor_pixel_mae=float(
                (
                    processor_pixel_values.float()
                    - clean_pixel_values.float()
                ).abs().mean().item()
            ),
            processor_pixel_linf=float(
                (
                    processor_pixel_values.float()
                    - clean_pixel_values.float()
                ).abs().amax().item()
            ),
        )

    def run(
        self,
        frames: Sequence[TrainingFrame],
    ) -> SourceActionResponseResult:
        """逐帧比较 clean/adv 响应，并在结束后恢复零谱系数。"""
        if self._renderer.get_texture_parameterization_name() != "spectral":
            raise SourceActionResponseError("动作响应诊断只支持 spectral 参数化")
        if not self._reference_path.is_file():
            raise SourceActionResponseError(
                f"动作响应参考谱系数不存在：{self._reference_path}"
            )
        texture_parameter: nn.Parameter = self._renderer.get_texture_param()
        loaded: object = torch.load(
            self._reference_path,
            map_location=texture_parameter.device,
            weights_only=True,
        )
        if not isinstance(loaded, torch.Tensor):
            raise SourceActionResponseError("动作响应参考 .pt 必须直接保存 Tensor")
        reference: torch.Tensor = loaded.to(
            device=texture_parameter.device,
            dtype=texture_parameter.dtype,
        )
        if reference.shape != texture_parameter.shape:
            raise SourceActionResponseError(
                "参考谱系数 shape 不匹配："
                f"{tuple(reference.shape)} != {tuple(texture_parameter.shape)}"
            )
        if not bool(torch.isfinite(reference).all().item()):
            raise SourceActionResponseError("参考谱系数包含 NaN/Inf")

        state_ids: list[int] = []
        step_indices: list[int] = []
        samples: list[ActionResponseSample] = []
        try:
            for frame in frames:
                primary_view: MultiInstanceViewFrame = self._primary_view(frame)
                clean_image: torch.Tensor = primary_view["bg_tensor"]
                self._renderer.reset_texture()
                clean_pixel_values: torch.Tensor = self._pixel_values(
                    clean_image,
                )
                with torch.no_grad():
                    texture_parameter.copy_(reference)
                adversarial_image: torch.Tensor = (
                    build_multi_instance_view_sample(
                        self._renderer,
                        primary_view,
                        self._render_resolution,
                    )
                )
                adversarial_pixel_values: torch.Tensor = self._pixel_values(
                    adversarial_image,
                )
                sample: ActionResponseSample = self._sample(
                    frame=frame,
                    clean_pixel_values=clean_pixel_values,
                    adversarial_pixel_values=adversarial_pixel_values,
                    processor_pixel_values=frame["processor_pixel_values"],
                )
                state_ids.append(frame["initial_state_id"])
                step_indices.append(frame["collection_step_index"])
                samples.append(sample)
        finally:
            self._renderer.reset_texture()
        if not samples:
            raise SourceActionResponseError("没有可用于动作响应诊断的训练帧")
        return SourceActionResponseResult(
            state_ids=np.asarray(state_ids, dtype=np.int64),
            step_indices=np.asarray(step_indices, dtype=np.int64),
            samples=tuple(samples),
            reference_path=str(self._reference_path),
            reference_sha256=hashlib.sha256(
                self._reference_path.read_bytes()
            ).hexdigest(),
        )
