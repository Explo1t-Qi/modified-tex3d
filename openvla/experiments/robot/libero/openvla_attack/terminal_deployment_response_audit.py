"""Gate 6g 终态部署响应的纯 CPU 审计核心。

本模块不导入 OpenVLA、LIBERO、renderer 或 CUDA。第一条公共接口只消费三条
静态动作响应：Clean ``C``、训练期 Renderer Delta Composition ``A`` 和真实
MuJoCo Active Texture ``B``。分类只依据相对于 Clean 的首次自回归因果分歧；
首次分歧以后的 token 已使用不同前缀，因此完整序列相等性不作为主判据。

``teacher_logits`` 是固定 clean prefix 下的 action-vocabulary logits，浮点
``[action_dim, num_action_classes]``；``generated_classes`` 是同一 action 子词表内
的整数 class，shape ``[action_dim]``。tie 严格按保存 logits 与逐行最大值逐值
相等判定，不使用数值容差。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
)

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]
TerminalResponseClassification: TypeAlias = Literal[
    "no_training_response",
    "deployment_lost",
    "deployment_response_altered",
    "deployment_preserved_strict",
    "deployment_preserved_tie_sensitive",
    "invalid_response_alignment",
]
TERMINAL_RESPONSE_CLASSIFICATIONS: Final[
    tuple[TerminalResponseClassification, ...]
] = (
    "no_training_response",
    "deployment_lost",
    "deployment_response_altered",
    "deployment_preserved_strict",
    "deployment_preserved_tie_sensitive",
    "invalid_response_alignment",
)
TERMINAL_DEPLOYMENT_RESPONSE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-deployment-response-v1"
)
TERMINAL_DEPLOYMENT_RESPONSE_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-deployment-response-bundle-v1"
)
TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-deployment-response-smoke-bundle-v1"
)
TERMINAL_RESPONSE_VARIANTS: Final[tuple[str, ...]] = (
    "action_spectral",
    "action_only_control",
)


class TerminalDeploymentResponseAuditError(ValueError):
    """Gate 6g 输入数组无法被可靠解释。"""


@dataclass(frozen=True)
class ActionPathEvidence:
    """一条静态模型路径的生成 token 与 clean-prefix teacher logits。"""

    generated_classes: IntArray
    teacher_logits: FloatArray


@dataclass(frozen=True)
class TerminalActionResponseDecision:
    """一个 ``(variant,state)`` 的首次因果分歧分类。"""

    classification: TerminalResponseClassification
    training_first_divergence_index: Optional[int]
    deployment_first_divergence_index: Optional[int]
    training_first_divergence_class: Optional[int]
    deployment_first_divergence_class: Optional[int]
    failures: tuple[str, ...] = ()


class TerminalActionResponseSummary(TypedDict):
    """一组同variant响应的无阈值精确计数。"""

    row_count: int
    classification_counts: dict[str, int]
    training_response_count: int
    preserved_first_response_count: int
    preserved_given_training_response: Optional[float]


@dataclass(frozen=True)
class ProcessorEquivalenceDecision:
    """training exact与official processor最终BF16输入的逐bit比较。"""

    bitwise_equal: bool
    mismatch_count: int
    total_value_count: int


@dataclass(frozen=True)
class VisibilityEquivalenceDecision:
    """Clean/B整数segmentation与逐实例hard alpha的严格比较。"""

    segmentation_bitwise_equal: bool
    alpha_bitwise_equal: bool
    segmentation_mismatch_count: int
    alpha_mismatch_count: int
    gate_pass: bool


@dataclass(frozen=True)
class TerminalBakePairingDecision:
    """两次canonical re-bake及bound PNG解码RGB的严格配对结果。"""

    repeat_bitwise_equal: bool
    bound_bitwise_equal: bool
    repeat_mismatch_count: int
    bound_mismatch_count: int
    gate_pass: bool


@dataclass(frozen=True)
class TerminalDeploymentResponseEvidence:
    """一个 ``(variant,state)`` 的权威C/A/B原始数组。

    路径轴固定为 ``[C,A,B]``。RGB为uint8 ``[3,H,W,3]``；processor数组为
    ``[3,1,num_vision_channels,Hm,Wm]``；模型响应为
    ``[3,action_dim,num_action_classes]``。segmentation保留source-view整数ID图，
    alpha保留source view中与整数segmentation同坐标系的逐实例0/1 hard mask。
    """

    variant: str
    state_id: int
    effective_rgb: NDArray[np.uint8]
    rgb_delta: NDArray[np.float32]
    clean_oriented_segmentation: NDArray[np.integer]
    deployment_oriented_segmentation: NDArray[np.integer]
    clean_instance_alpha: FloatArray
    deployment_instance_alpha: FloatArray
    training_exact_processor_bf16_bits: NDArray[np.uint16]
    official_processor_bf16_bits: NDArray[np.uint16]
    training_exact_processor_float32: NDArray[np.float32]
    official_processor_float32: NDArray[np.float32]
    teacher_logits: FloatArray
    generation_logits: FloatArray
    generated_classes: IntArray
    generated_token_ids: IntArray
    decoded_actions: FloatArray
    prompt_input_ids: IntArray
    teacher_input_ids: IntArray
    action_token_start: int
    action_token_end: int
    vocab_size: int
    bin_centers: FloatArray
    action_low: FloatArray
    action_high: FloatArray
    action_unnormalize_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class TerminalDeploymentResponseEvaluation:
    """CPU从单行权威NPZ独立复算的工程与动作响应结论。"""

    variant: str
    state_id: int
    evidence_valid: bool
    failures: tuple[str, ...]
    action_decision: TerminalActionResponseDecision
    processor_equivalence: ProcessorEquivalenceDecision
    visibility_equivalence: VisibilityEquivalenceDecision
    generation_alignment_pass: bool
    processor_float_views_exact: bool
    rgb_delta_exact: bool
    token_mapping_exact: bool
    decoded_actions_exact: bool


@dataclass(frozen=True)
class TerminalDeploymentResponseBundleDecision:
    """完整20-case bundle的独立CPU工程验收。"""

    audit_valid: bool
    failures: tuple[str, ...]
    case_count: int
    per_variant_summary: dict[str, TerminalActionResponseSummary]


@dataclass(frozen=True)
class _ValidatedPath:
    generated_classes: NDArray[np.int64]
    teacher_logits: NDArray[np.float64]


def _validate_path(
    name: str,
    evidence: ActionPathEvidence,
) -> _ValidatedPath:
    generated = np.asarray(evidence.generated_classes)
    logits = np.asarray(evidence.teacher_logits)
    if generated.ndim != 1 or generated.size == 0:
        raise TerminalDeploymentResponseAuditError(
            f"{name} generated_classes 必须为非空 [action_dim]"
        )
    if not np.issubdtype(generated.dtype, np.integer):
        raise TerminalDeploymentResponseAuditError(
            f"{name} generated_classes 必须为整数"
        )
    if logits.ndim != 2 or logits.shape[0] != generated.size:
        raise TerminalDeploymentResponseAuditError(
            f"{name} teacher_logits 必须为 [action_dim,num_classes]"
        )
    if logits.shape[1] < 2 or not np.issubdtype(logits.dtype, np.floating):
        raise TerminalDeploymentResponseAuditError(
            f"{name} teacher_logits 必须为至少两类的浮点数组"
        )
    if not bool(np.isfinite(logits).all()):
        raise TerminalDeploymentResponseAuditError(
            f"{name} teacher_logits 包含 NaN/Inf"
        )
    generated_int64 = generated.astype(np.int64, copy=False)
    if bool(np.any(generated_int64 < 0)) or bool(
        np.any(generated_int64 >= logits.shape[1])
    ):
        raise TerminalDeploymentResponseAuditError(
            f"{name} generated_classes 包含越界 class"
        )
    return _ValidatedPath(
        generated_classes=generated_int64,
        teacher_logits=logits.astype(np.float64, copy=False),
    )


def _first_divergence(
    clean: NDArray[np.int64],
    candidate: NDArray[np.int64],
) -> Optional[int]:
    indices = np.flatnonzero(candidate != clean)
    return int(indices[0]) if indices.size else None


def _argmax_classes(row: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.flatnonzero(row == np.max(row)).astype(np.int64, copy=False)


def _alignment_failures(
    *,
    name: str,
    clean_classes: NDArray[np.int64],
    path: _ValidatedPath,
    first_divergence_index: Optional[int],
) -> tuple[str, ...]:
    """验证 teacher logits 只在与生成共享 clean prefix 的位置上的含义。"""

    last_comparable_index = (
        clean_classes.size - 1
        if first_divergence_index is None
        else first_divergence_index
    )
    failures: list[str] = []
    for index in range(last_comparable_index + 1):
        expected_class = int(path.generated_classes[index])
        argmax_classes = _argmax_classes(path.teacher_logits[index])
        if expected_class not in argmax_classes:
            failures.append(
                f"{name} token {index} generation class {expected_class} "
                "不属于clean-prefix teacher argmax集合"
            )
    return tuple(failures)


def _divergence_class(
    path: _ValidatedPath,
    index: Optional[int],
) -> Optional[int]:
    return None if index is None else int(path.generated_classes[index])


def classify_terminal_action_response(
    *,
    clean: ActionPathEvidence,
    training: ActionPathEvidence,
    deployment: ActionPathEvidence,
) -> TerminalActionResponseDecision:
    """按首次因果分歧分类一组 ``C/A/B`` 静态动作响应。

    输入 schema 错误会抛出 :class:`TerminalDeploymentResponseAuditError`；模型
    generation 与 clean-prefix teacher argmax 不一致则返回
    ``invalid_response_alignment``，使上层 bundle evaluator 判整个审计无效。
    """

    clean_path = _validate_path("clean", clean)
    training_path = _validate_path("training", training)
    deployment_path = _validate_path("deployment", deployment)
    expected_shape = clean_path.teacher_logits.shape
    if (
        training_path.teacher_logits.shape != expected_shape
        or deployment_path.teacher_logits.shape != expected_shape
    ):
        raise TerminalDeploymentResponseAuditError(
            "C/A/B teacher_logits shape 必须完全一致"
        )

    clean_classes = clean_path.generated_classes
    training_index = _first_divergence(
        clean_classes,
        training_path.generated_classes,
    )
    deployment_index = _first_divergence(
        clean_classes,
        deployment_path.generated_classes,
    )
    failures = (
        *_alignment_failures(
            name="clean",
            clean_classes=clean_classes,
            path=clean_path,
            first_divergence_index=None,
        ),
        *_alignment_failures(
            name="training",
            clean_classes=clean_classes,
            path=training_path,
            first_divergence_index=training_index,
        ),
        *_alignment_failures(
            name="deployment",
            clean_classes=clean_classes,
            path=deployment_path,
            first_divergence_index=deployment_index,
        ),
    )
    training_class = _divergence_class(training_path, training_index)
    deployment_class = _divergence_class(deployment_path, deployment_index)
    if failures:
        classification: TerminalResponseClassification = (
            "invalid_response_alignment"
        )
    elif training_index is None:
        classification = "no_training_response"
    elif deployment_index is None:
        classification = "deployment_lost"
    elif (
        training_index != deployment_index
        or training_class != deployment_class
    ):
        classification = "deployment_response_altered"
    else:
        assert training_class is not None
        training_argmax = _argmax_classes(
            training_path.teacher_logits[training_index]
        )
        deployment_argmax = _argmax_classes(
            deployment_path.teacher_logits[deployment_index]
        )
        classification = (
            "deployment_preserved_strict"
            if training_argmax.size == 1 and deployment_argmax.size == 1
            else "deployment_preserved_tie_sensitive"
        )
    return TerminalActionResponseDecision(
        classification=classification,
        training_first_divergence_index=training_index,
        deployment_first_divergence_index=deployment_index,
        training_first_divergence_class=training_class,
        deployment_first_divergence_class=deployment_class,
        failures=tuple(failures),
    )


def summarize_terminal_action_responses(
    decisions: Sequence[TerminalActionResponseDecision],
) -> TerminalActionResponseSummary:
    """汇总精确分类计数；条件分母为零时返回 ``None``。"""

    counts: dict[str, int] = {
        classification: 0
        for classification in TERMINAL_RESPONSE_CLASSIFICATIONS
    }
    for decision in decisions:
        counts[decision.classification] += 1
    preserved_count = (
        counts["deployment_preserved_strict"]
        + counts["deployment_preserved_tie_sensitive"]
    )
    training_response_count = (
        counts["deployment_lost"]
        + counts["deployment_response_altered"]
        + preserved_count
    )
    return TerminalActionResponseSummary(
        row_count=len(decisions),
        classification_counts=counts,
        training_response_count=training_response_count,
        preserved_first_response_count=preserved_count,
        preserved_given_training_response=(
            None
            if training_response_count == 0
            else preserved_count / training_response_count
        ),
    )


def evaluate_final_processor_equivalence(
    *,
    training_exact_bf16_bits: NDArray[np.uint16],
    official_bf16_bits: NDArray[np.uint16],
) -> ProcessorEquivalenceDecision:
    """比较完成全部空间变换、归一化和BF16 cast后的模型输入。"""

    training_bits = np.asarray(training_exact_bf16_bits)
    official_bits = np.asarray(official_bf16_bits)
    if training_bits.dtype != np.uint16 or official_bits.dtype != np.uint16:
        raise TerminalDeploymentResponseAuditError(
            "processor BF16 bit pattern 必须保存为uint16"
        )
    if training_bits.shape != official_bits.shape or training_bits.size == 0:
        raise TerminalDeploymentResponseAuditError(
            "training/official processor bits必须为相同非空shape"
        )
    mismatch_count = int(np.count_nonzero(training_bits != official_bits))
    return ProcessorEquivalenceDecision(
        bitwise_equal=mismatch_count == 0,
        mismatch_count=mismatch_count,
        total_value_count=int(training_bits.size),
    )


def evaluate_visibility_equivalence(
    *,
    clean_oriented_segmentation: NDArray[np.integer],
    deployment_oriented_segmentation: NDArray[np.integer],
    clean_instance_alpha: FloatArray,
    deployment_instance_alpha: FloatArray,
) -> VisibilityEquivalenceDecision:
    """严格比较同一静态场景的整数segmentation和0/1 hard alpha。"""

    clean_segmentation = np.asarray(clean_oriented_segmentation)
    deployment_segmentation = np.asarray(deployment_oriented_segmentation)
    if (
        clean_segmentation.ndim != 3
        or clean_segmentation.shape[-1] != 2
        or clean_segmentation.size == 0
        or not np.issubdtype(clean_segmentation.dtype, np.integer)
        or not np.issubdtype(deployment_segmentation.dtype, np.integer)
        or clean_segmentation.shape != deployment_segmentation.shape
        or clean_segmentation.dtype != deployment_segmentation.dtype
    ):
        raise TerminalDeploymentResponseAuditError(
            "C/B oriented segmentation必须为相同dtype/shape的整数[H,W,2]"
        )
    clean_alpha = np.asarray(clean_instance_alpha)
    deployment_alpha = np.asarray(deployment_instance_alpha)
    if (
        clean_alpha.ndim != 4
        or clean_alpha.shape[0] <= 0
        or clean_alpha.shape[1] != 1
        or clean_alpha.shape[2] <= 0
        or clean_alpha.shape[3] <= 0
        or not np.issubdtype(clean_alpha.dtype, np.floating)
        or not np.issubdtype(deployment_alpha.dtype, np.floating)
        or clean_alpha.shape != deployment_alpha.shape
        or clean_alpha.dtype != deployment_alpha.dtype
    ):
        raise TerminalDeploymentResponseAuditError(
            "C/B instance alpha必须为相同dtype/shape的浮点[K,1,H,W]"
        )
    if not bool(np.isfinite(clean_alpha).all()) or not bool(
        np.isfinite(deployment_alpha).all()
    ):
        raise TerminalDeploymentResponseAuditError(
            "C/B instance alpha包含NaN/Inf"
        )
    if not bool(np.all((clean_alpha == 0.0) | (clean_alpha == 1.0))) or not bool(
        np.all((deployment_alpha == 0.0) | (deployment_alpha == 1.0))
    ):
        raise TerminalDeploymentResponseAuditError(
            "Gate 6g visibility必须保存严格0/1 hard alpha"
        )
    segmentation_mismatch_count = int(
        np.count_nonzero(clean_segmentation != deployment_segmentation)
    )
    alpha_mismatch_count = int(
        np.count_nonzero(clean_alpha != deployment_alpha)
    )
    return VisibilityEquivalenceDecision(
        segmentation_bitwise_equal=segmentation_mismatch_count == 0,
        alpha_bitwise_equal=alpha_mismatch_count == 0,
        segmentation_mismatch_count=segmentation_mismatch_count,
        alpha_mismatch_count=alpha_mismatch_count,
        gate_pass=(
            segmentation_mismatch_count == 0 and alpha_mismatch_count == 0
        ),
    )


def evaluate_terminal_bake_pairing(
    *,
    first_rebaked_rgb: NDArray[np.uint8],
    second_rebaked_rgb: NDArray[np.uint8],
    bound_png_rgb: NDArray[np.uint8],
) -> TerminalBakePairingDecision:
    """逐像素比较同一终态的重复re-bake与已有bound PNG。"""

    arrays = tuple(
        np.asarray(value)
        for value in (
            first_rebaked_rgb,
            second_rebaked_rgb,
            bound_png_rgb,
        )
    )
    expected_shape = arrays[0].shape
    if (
        arrays[0].dtype != np.uint8
        or arrays[0].ndim != 3
        or arrays[0].shape[-1] != 3
        or arrays[0].size == 0
        or any(
            value.dtype != np.uint8 or value.shape != expected_shape
            for value in arrays[1:]
        )
    ):
        raise TerminalDeploymentResponseAuditError(
            "重复re-bake与bound PNG必须为相同非空uint8 [H,W,3]"
        )
    repeat_mismatch_count = int(np.count_nonzero(arrays[0] != arrays[1]))
    first_bound_mismatch = int(np.count_nonzero(arrays[0] != arrays[2]))
    second_bound_mismatch = int(np.count_nonzero(arrays[1] != arrays[2]))
    bound_mismatch_count = max(first_bound_mismatch, second_bound_mismatch)
    return TerminalBakePairingDecision(
        repeat_bitwise_equal=repeat_mismatch_count == 0,
        bound_bitwise_equal=(
            first_bound_mismatch == 0 and second_bound_mismatch == 0
        ),
        repeat_mismatch_count=repeat_mismatch_count,
        bound_mismatch_count=bound_mismatch_count,
        gate_pass=(
            repeat_mismatch_count == 0
            and first_bound_mismatch == 0
            and second_bound_mismatch == 0
        ),
    )


def _bf16_bits_to_float32(bits: NDArray[np.uint16]) -> NDArray[np.float32]:
    """将BF16原始16 bit左移到IEEE float32高位，逐值精确恢复。"""

    unsigned = np.asarray(bits, dtype=np.uint16)
    float32_bits = unsigned.astype(np.uint32) << np.uint32(16)
    return float32_bits.view(np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_terminal_evidence_schema(
    evidence: TerminalDeploymentResponseEvidence,
) -> None:
    if evidence.variant not in TERMINAL_RESPONSE_VARIANTS:
        raise TerminalDeploymentResponseAuditError("Gate 6g variant无效")
    if not isinstance(evidence.state_id, int) or isinstance(
        evidence.state_id, bool
    ) or not 0 <= evidence.state_id <= 9:
        raise TerminalDeploymentResponseAuditError(
            "Gate 6g正式证据state_id必须位于0--9"
        )

    effective_rgb = np.asarray(evidence.effective_rgb)
    if (
        effective_rgb.dtype != np.uint8
        or effective_rgb.ndim != 4
        or effective_rgb.shape[0] != 3
        or effective_rgb.shape[-1] != 3
        or effective_rgb.size == 0
    ):
        raise TerminalDeploymentResponseAuditError(
            "effective_rgb必须为uint8 [3,H,W,3]"
        )
    rgb_delta = np.asarray(evidence.rgb_delta)
    if (
        rgb_delta.dtype != np.float32
        or rgb_delta.shape != (2, *effective_rgb.shape[1:])
        or not bool(np.isfinite(rgb_delta).all())
    ):
        raise TerminalDeploymentResponseAuditError(
            "rgb_delta必须为有限float32 [2,H,W,3]"
        )

    processor_shapes = {
        np.asarray(value).shape
        for value in (
            evidence.training_exact_processor_bf16_bits,
            evidence.official_processor_bf16_bits,
            evidence.training_exact_processor_float32,
            evidence.official_processor_float32,
        )
    }
    if len(processor_shapes) != 1:
        raise TerminalDeploymentResponseAuditError(
            "四组processor数组shape必须一致"
        )
    processor_shape = next(iter(processor_shapes))
    if (
        len(processor_shape) != 5
        or processor_shape[0] != 3
        or processor_shape[1] != 1
        or any(dimension <= 0 for dimension in processor_shape[2:])
    ):
        raise TerminalDeploymentResponseAuditError(
            "processor数组必须为[3,1,C,H,W]"
        )
    if (
        np.asarray(evidence.training_exact_processor_bf16_bits).dtype
        != np.uint16
        or np.asarray(evidence.official_processor_bf16_bits).dtype
        != np.uint16
        or np.asarray(evidence.training_exact_processor_float32).dtype
        != np.float32
        or np.asarray(evidence.official_processor_float32).dtype
        != np.float32
    ):
        raise TerminalDeploymentResponseAuditError(
            "processor bits/便利视图必须分别为uint16/float32"
        )

    teacher_logits = np.asarray(evidence.teacher_logits)
    generation_logits = np.asarray(evidence.generation_logits)
    if (
        teacher_logits.ndim != 3
        or teacher_logits.shape[0] != 3
        or teacher_logits.shape[1] <= 0
        or teacher_logits.shape[2] < 2
        or generation_logits.shape != teacher_logits.shape
        or not np.issubdtype(teacher_logits.dtype, np.floating)
        or not np.issubdtype(generation_logits.dtype, np.floating)
        or not bool(np.isfinite(teacher_logits).all())
        or not bool(np.isfinite(generation_logits).all())
    ):
        raise TerminalDeploymentResponseAuditError(
            "teacher/generation logits必须为相同有限浮点[3,A,C]"
        )
    action_dim = int(teacher_logits.shape[1])
    action_class_count = int(teacher_logits.shape[2])
    expected_action_shape = (3, action_dim)
    for name, value in (
        ("generated_classes", evidence.generated_classes),
        ("generated_token_ids", evidence.generated_token_ids),
    ):
        array = np.asarray(value)
        if array.shape != expected_action_shape or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise TerminalDeploymentResponseAuditError(
                f"{name}必须为整数[3,action_dim]"
            )
    decoded_actions = np.asarray(evidence.decoded_actions)
    if (
        decoded_actions.shape != expected_action_shape
        or not np.issubdtype(decoded_actions.dtype, np.floating)
        or not bool(np.isfinite(decoded_actions).all())
    ):
        raise TerminalDeploymentResponseAuditError(
            "decoded_actions必须为有限浮点[3,action_dim]"
        )
    if (
        not isinstance(evidence.action_token_start, int)
        or not isinstance(evidence.action_token_end, int)
        or evidence.action_token_end - evidence.action_token_start
        != action_class_count
    ):
        raise TerminalDeploymentResponseAuditError(
            "action-token slice与logits class数不一致"
        )
    if not isinstance(evidence.vocab_size, int) or evidence.vocab_size <= 0:
        raise TerminalDeploymentResponseAuditError("vocab_size必须为正整数")
    bin_centers = np.asarray(evidence.bin_centers)
    if (
        bin_centers.ndim != 1
        # OpenVLA以256个等距边界定义255个连续动作bin center。action
        # token子词表仍有256类；两个端点token经正式codec clip后共享最外侧
        # center，不能把token类别数误当成连续bin中心数。
        or bin_centers.size != action_class_count - 1
        or not np.issubdtype(bin_centers.dtype, np.floating)
        or not bool(np.isfinite(bin_centers).all())
    ):
        raise TerminalDeploymentResponseAuditError(
            "bin_centers必须为有限浮点一维数组，且数量等于"
            "action class数减一"
        )
    for name, value in (
        ("action_low", evidence.action_low),
        ("action_high", evidence.action_high),
    ):
        array = np.asarray(value)
        if (
            array.shape != (action_dim,)
            or not np.issubdtype(array.dtype, np.floating)
            or not bool(np.isfinite(array).all())
        ):
            raise TerminalDeploymentResponseAuditError(
                f"{name}必须为有限浮点[action_dim]"
            )
    action_mask = np.asarray(evidence.action_unnormalize_mask)
    if action_mask.dtype != np.bool_ or action_mask.shape != (action_dim,):
        raise TerminalDeploymentResponseAuditError(
            "action_unnormalize_mask必须为bool[action_dim]"
        )
    for name, value in (
        ("prompt_input_ids", evidence.prompt_input_ids),
        ("teacher_input_ids", evidence.teacher_input_ids),
    ):
        array = np.asarray(value)
        if array.ndim != 1 or array.size == 0 or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise TerminalDeploymentResponseAuditError(
                f"{name}必须为非空整数一维数组"
            )

    # 复用正式visibility schema校验；相等性结果由evaluator决定。
    evaluate_visibility_equivalence(
        clean_oriented_segmentation=evidence.clean_oriented_segmentation,
        deployment_oriented_segmentation=(
            evidence.deployment_oriented_segmentation
        ),
        clean_instance_alpha=evidence.clean_instance_alpha,
        deployment_instance_alpha=evidence.deployment_instance_alpha,
    )


def evaluate_terminal_response_evidence(
    evidence: TerminalDeploymentResponseEvidence,
) -> TerminalDeploymentResponseEvaluation:
    """从一个内存证据对象独立复算所有单行硬一致性结论。"""

    _validate_terminal_evidence_schema(evidence)
    teacher_logits = np.asarray(evidence.teacher_logits)
    generated_classes = np.asarray(evidence.generated_classes)
    action_decision = classify_terminal_action_response(
        clean=ActionPathEvidence(generated_classes[0], teacher_logits[0]),
        training=ActionPathEvidence(
            generated_classes[1], teacher_logits[1]
        ),
        deployment=ActionPathEvidence(
            generated_classes[2], teacher_logits[2]
        ),
    )
    processor_equivalence = evaluate_final_processor_equivalence(
        training_exact_bf16_bits=(
            evidence.training_exact_processor_bf16_bits
        ),
        official_bf16_bits=evidence.official_processor_bf16_bits,
    )
    visibility_equivalence = evaluate_visibility_equivalence(
        clean_oriented_segmentation=evidence.clean_oriented_segmentation,
        deployment_oriented_segmentation=(
            evidence.deployment_oriented_segmentation
        ),
        clean_instance_alpha=evidence.clean_instance_alpha,
        deployment_instance_alpha=evidence.deployment_instance_alpha,
    )

    failures: list[str] = list(action_decision.failures)
    generation_alignment_pass = True
    generation_logits = np.asarray(evidence.generation_logits).astype(
        np.float64, copy=False
    )
    for path_index, path_name in enumerate(("clean", "training", "deployment")):
        for token_index, generated_class in enumerate(
            generated_classes[path_index]
        ):
            argmax_classes = _argmax_classes(
                generation_logits[path_index, token_index]
            )
            if int(generated_class) not in argmax_classes:
                generation_alignment_pass = False
                failures.append(
                    f"{path_name} generation token {token_index}不属于score argmax集合"
                )

    expected_training_float = _bf16_bits_to_float32(
        evidence.training_exact_processor_bf16_bits
    )
    expected_official_float = _bf16_bits_to_float32(
        evidence.official_processor_bf16_bits
    )
    processor_float_views_exact = bool(
        np.array_equal(
            expected_training_float,
            evidence.training_exact_processor_float32,
        )
        and np.array_equal(
            expected_official_float,
            evidence.official_processor_float32,
        )
    )
    if not processor_float_views_exact:
        failures.append("processor float32便利视图不能由BF16 bits精确恢复")
    if not processor_equivalence.bitwise_equal:
        failures.append("training exact与official最终BF16 pixel_values不一致")
    if not visibility_equivalence.gate_pass:
        failures.append("Clean/Deployment segmentation或hard alpha不一致")

    effective_rgb = np.asarray(evidence.effective_rgb)
    expected_rgb_delta = (
        effective_rgb[1:].astype(np.float32)
        - effective_rgb[0:1].astype(np.float32)
    ) / np.float32(255.0)
    rgb_delta_exact = bool(np.array_equal(expected_rgb_delta, evidence.rgb_delta))
    if not rgb_delta_exact:
        failures.append("保存的RGB delta不能由C/A/B effective RGB逐值复算")

    expected_classes = (
        np.asarray(evidence.generated_token_ids, dtype=np.int64)
        - evidence.action_token_start
    )
    token_mapping_exact = bool(
        np.array_equal(expected_classes, generated_classes)
    )
    if not token_mapping_exact:
        failures.append("generated token ID与action class mapping不一致")
    if action_decision.classification == "invalid_response_alignment":
        failures.append("teacher/generation first-divergence alignment无效")

    token_ids = np.asarray(evidence.generated_token_ids, dtype=np.int64)
    bin_indices = np.clip(
        evidence.vocab_size - token_ids - 1,
        a_min=0,
        a_max=np.asarray(evidence.bin_centers).size - 1,
    )
    normalized_actions = np.asarray(evidence.bin_centers)[bin_indices]
    action_low = np.asarray(evidence.action_low)
    action_high = np.asarray(evidence.action_high)
    scaled_actions = (
        0.5 * (normalized_actions + 1.0) * (action_high - action_low)
        + action_low
    )
    expected_decoded_actions = np.where(
        np.asarray(evidence.action_unnormalize_mask),
        scaled_actions,
        normalized_actions,
    )
    decoded_actions_exact = bool(
        np.array_equal(expected_decoded_actions, evidence.decoded_actions)
    )
    if not decoded_actions_exact:
        failures.append("保存的decoded action不能由token与codec证据逐值复算")
    return TerminalDeploymentResponseEvaluation(
        variant=evidence.variant,
        state_id=evidence.state_id,
        evidence_valid=not failures,
        failures=tuple(failures),
        action_decision=action_decision,
        processor_equivalence=processor_equivalence,
        visibility_equivalence=visibility_equivalence,
        generation_alignment_pass=generation_alignment_pass,
        processor_float_views_exact=processor_float_views_exact,
        rgb_delta_exact=rgb_delta_exact,
        token_mapping_exact=token_mapping_exact,
        decoded_actions_exact=decoded_actions_exact,
    )


def write_terminal_response_npz(
    evidence: TerminalDeploymentResponseEvidence,
    *,
    output_path: str | Path,
) -> str:
    """以无pickle压缩NPZ保存一行权威原始证据并返回文件SHA-256。"""

    _validate_terminal_evidence_schema(evidence)
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(TERMINAL_DEPLOYMENT_RESPONSE_SCHEMA_VERSION),
        variant=np.asarray(evidence.variant),
        state_id=np.asarray(evidence.state_id, dtype=np.int64),
        effective_rgb=evidence.effective_rgb,
        rgb_delta=evidence.rgb_delta,
        clean_oriented_segmentation=evidence.clean_oriented_segmentation,
        deployment_oriented_segmentation=(
            evidence.deployment_oriented_segmentation
        ),
        clean_instance_alpha=evidence.clean_instance_alpha,
        deployment_instance_alpha=evidence.deployment_instance_alpha,
        training_exact_processor_bf16_bits=(
            evidence.training_exact_processor_bf16_bits
        ),
        official_processor_bf16_bits=evidence.official_processor_bf16_bits,
        training_exact_processor_float32=(
            evidence.training_exact_processor_float32
        ),
        official_processor_float32=evidence.official_processor_float32,
        teacher_logits=evidence.teacher_logits,
        generation_logits=evidence.generation_logits,
        generated_classes=evidence.generated_classes,
        generated_token_ids=evidence.generated_token_ids,
        decoded_actions=evidence.decoded_actions,
        prompt_input_ids=evidence.prompt_input_ids,
        teacher_input_ids=evidence.teacher_input_ids,
        action_token_start=np.asarray(evidence.action_token_start, dtype=np.int64),
        action_token_end=np.asarray(evidence.action_token_end, dtype=np.int64),
        vocab_size=np.asarray(evidence.vocab_size, dtype=np.int64),
        bin_centers=evidence.bin_centers,
        action_low=evidence.action_low,
        action_high=evidence.action_high,
        action_unnormalize_mask=evidence.action_unnormalize_mask,
    )
    return _file_sha256(path)


def _load_scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    if value.shape != ():
        raise TerminalDeploymentResponseAuditError(
            f"NPZ字段{name}必须为scalar"
        )
    return value.item()


def load_terminal_response_npz(
    path: str | Path,
) -> TerminalDeploymentResponseEvidence:
    """无pickle加载一行权威Gate 6g NPZ并验证schema version。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        schema_version = str(_load_scalar(archive, "schema_version"))
        if schema_version != TERMINAL_DEPLOYMENT_RESPONSE_SCHEMA_VERSION:
            raise TerminalDeploymentResponseAuditError(
                f"Gate 6g NPZ schema不匹配: {schema_version}"
            )
        evidence = TerminalDeploymentResponseEvidence(
            variant=str(_load_scalar(archive, "variant")),
            state_id=int(_load_scalar(archive, "state_id")),
            effective_rgb=archive["effective_rgb"].copy(),
            rgb_delta=archive["rgb_delta"].copy(),
            clean_oriented_segmentation=archive[
                "clean_oriented_segmentation"
            ].copy(),
            deployment_oriented_segmentation=archive[
                "deployment_oriented_segmentation"
            ].copy(),
            clean_instance_alpha=archive["clean_instance_alpha"].copy(),
            deployment_instance_alpha=archive[
                "deployment_instance_alpha"
            ].copy(),
            training_exact_processor_bf16_bits=archive[
                "training_exact_processor_bf16_bits"
            ].copy(),
            official_processor_bf16_bits=archive[
                "official_processor_bf16_bits"
            ].copy(),
            training_exact_processor_float32=archive[
                "training_exact_processor_float32"
            ].copy(),
            official_processor_float32=archive[
                "official_processor_float32"
            ].copy(),
            teacher_logits=archive["teacher_logits"].copy(),
            generation_logits=archive["generation_logits"].copy(),
            generated_classes=archive["generated_classes"].copy(),
            generated_token_ids=archive["generated_token_ids"].copy(),
            decoded_actions=archive["decoded_actions"].copy(),
            prompt_input_ids=archive["prompt_input_ids"].copy(),
            teacher_input_ids=archive["teacher_input_ids"].copy(),
            action_token_start=int(_load_scalar(archive, "action_token_start")),
            action_token_end=int(_load_scalar(archive, "action_token_end")),
            vocab_size=int(_load_scalar(archive, "vocab_size")),
            bin_centers=archive["bin_centers"].copy(),
            action_low=archive["action_low"].copy(),
            action_high=archive["action_high"].copy(),
            action_unnormalize_mask=archive[
                "action_unnormalize_mask"
            ].copy(),
        )
    _validate_terminal_evidence_schema(evidence)
    return evidence


def evaluate_terminal_response_npz(
    path: str | Path,
) -> TerminalDeploymentResponseEvaluation:
    """从服务器NPZ独立加载并复算一个Gate 6g case。"""

    return evaluate_terminal_response_evidence(
        load_terminal_response_npz(path)
    )


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_bundle_artifact(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TerminalDeploymentResponseAuditError("case NPZ相对路径无效")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise TerminalDeploymentResponseAuditError("case NPZ路径必须为相对路径")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise TerminalDeploymentResponseAuditError("case NPZ路径逃逸bundle目录")
    return resolved


def _shared_clean_arrays(
    evidence: TerminalDeploymentResponseEvidence,
) -> tuple[NDArray[Any], ...]:
    """返回同一state跨variant必须逐值共享的Clean与模型身份数组。"""

    return (
        np.asarray(evidence.effective_rgb)[0],
        np.asarray(evidence.clean_oriented_segmentation),
        np.asarray(evidence.clean_instance_alpha),
        np.asarray(evidence.training_exact_processor_bf16_bits)[0],
        np.asarray(evidence.official_processor_bf16_bits)[0],
        np.asarray(evidence.training_exact_processor_float32)[0],
        np.asarray(evidence.official_processor_float32)[0],
        np.asarray(evidence.teacher_logits)[0],
        np.asarray(evidence.generation_logits)[0],
        np.asarray(evidence.generated_classes)[0],
        np.asarray(evidence.generated_token_ids)[0],
        np.asarray(evidence.decoded_actions)[0],
        np.asarray(evidence.prompt_input_ids),
        np.asarray(evidence.teacher_input_ids),
        np.asarray(evidence.bin_centers),
        np.asarray(evidence.action_low),
        np.asarray(evidence.action_high),
        np.asarray(evidence.action_unnormalize_mask),
        np.asarray(
            (
                evidence.action_token_start,
                evidence.action_token_end,
                evidence.vocab_size,
            ),
            dtype=np.int64,
        ),
    )


def _shared_clean_is_exact(
    reference: TerminalDeploymentResponseEvidence,
    candidate: TerminalDeploymentResponseEvidence,
) -> bool:
    """检查两个终态case是否复用了同一份Clean事实与action codec。"""

    return all(
        np.array_equal(reference_array, candidate_array)
        for reference_array, candidate_array in zip(
            _shared_clean_arrays(reference),
            _shared_clean_arrays(candidate),
        )
    )


def _validate_bundle_provenance(
    manifest: Mapping[str, Any],
    failures: list[str],
) -> Mapping[str, Any]:
    """验证成功manifest中已冻结的输入、processor/view及恢复身份。"""

    raw_provenance = manifest.get("provenance")
    if not isinstance(raw_provenance, dict):
        failures.append("provenance缺失")
        return {}
    raw_inputs = raw_provenance.get("input_sha256")
    required_input_names = {
        "production_support",
        "rebake_preflight_manifest",
        *(
            f"{variant}_{suffix}"
            for variant in TERMINAL_RESPONSE_VARIANTS
            for suffix in ("formal_manifest", "parameter", "bake")
        ),
    }
    if (
        not isinstance(raw_inputs, dict)
        or set(raw_inputs) != required_input_names
        or not all(_is_lower_hex(value, 64) for value in raw_inputs.values())
    ):
        failures.append("provenance input_sha256 inventory缺失或无效")
    checkpoint_fingerprints = raw_provenance.get("checkpoint_fingerprints")
    if (
        not isinstance(checkpoint_fingerprints, dict)
        or not checkpoint_fingerprints
        or not all(
            isinstance(name, str)
            and bool(name)
            and _is_lower_hex(value, 64)
            for name, value in checkpoint_fingerprints.items()
        )
    ):
        failures.append("checkpoint_fingerprints缺失")
    processor_specification = raw_provenance.get("processor_specification")
    if not isinstance(processor_specification, dict) or not (
        processor_specification
    ):
        failures.append("processor_specification缺失")
    policy_view_specification = raw_provenance.get(
        "policy_view_specification"
    )
    if not isinstance(policy_view_specification, dict) or not (
        policy_view_specification
    ):
        failures.append("policy_view_specification缺失")
    if raw_provenance.get("asset_restore_status") != {
        "xml": True,
        "texture": True,
    }:
        failures.append("最终Runtime Asset恢复状态无效")
    if raw_provenance.get("runtime_asset_backup_paths_removed") is not True:
        failures.append("Runtime Asset backup未验证删除")
    return raw_provenance


def _evaluate_terminal_response_bundle(
    manifest_path: str | Path,
    *,
    expected_schema_version: str,
    expected_state_ids: Sequence[int],
) -> TerminalDeploymentResponseBundleDecision:
    """按调用方冻结的state inventory独立复算双终态全部NPZ。"""

    frozen_state_ids = tuple(expected_state_ids)
    if not frozen_state_ids or len(set(frozen_state_ids)) != len(
        frozen_state_ids
    ):
        raise ValueError("expected_state_ids必须为非空唯一序列")
    failures: list[str] = []
    per_variant_decisions: dict[str, list[TerminalActionResponseDecision]] = {
        variant: [] for variant in TERMINAL_RESPONSE_VARIANTS
    }
    try:
        resolved_manifest = Path(manifest_path).resolve()
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TerminalDeploymentResponseAuditError("manifest必须为JSON object")
    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        TerminalDeploymentResponseAuditError,
    ) as error:
        return TerminalDeploymentResponseBundleDecision(
            audit_valid=False,
            failures=(f"Gate 6g manifest无法加载: {error}",),
            case_count=0,
            per_variant_summary={},
        )

    if manifest.get("schema_version") != expected_schema_version:
        failures.append("bundle schema_version不匹配")
    if manifest.get("status") != "complete":
        failures.append("成功manifest status必须为complete")
    if not _is_lower_hex(manifest.get("code_commit"), 40):
        failures.append("code_commit不是40位小写Git SHA")
    if not _is_lower_hex(manifest.get("config_sha256"), 64):
        failures.append("config_sha256无效")
    if manifest.get("expected_variants") != list(TERMINAL_RESPONSE_VARIANTS):
        failures.append("expected_variants未冻结为两个正式终态")
    if manifest.get("expected_state_ids") != list(frozen_state_ids):
        failures.append(
            f"expected_state_ids未冻结为{list(frozen_state_ids)}"
        )
    raw_fingerprints = manifest.get("state_fingerprints")
    if (
        not isinstance(raw_fingerprints, list)
        or len(raw_fingerprints) != len(frozen_state_ids)
        or len(set(raw_fingerprints)) != len(frozen_state_ids)
        or not all(_is_lower_hex(value, 64) for value in raw_fingerprints)
    ):
        failures.append("state_fingerprints缺失、无效或不唯一")
        state_fingerprints: list[str] = []
    else:
        state_fingerprints = [str(value) for value in raw_fingerprints]
    fingerprint_by_state = dict(zip(frozen_state_ids, state_fingerprints))
    provenance = _validate_bundle_provenance(manifest, failures)
    input_sha256 = provenance.get("input_sha256", {})

    terminal_pairing = manifest.get("terminal_pairing")
    if not isinstance(terminal_pairing, dict):
        failures.append("terminal_pairing缺失")
    else:
        for variant in TERMINAL_RESPONSE_VARIANTS:
            pairing = terminal_pairing.get(variant)
            if not isinstance(pairing, dict) or pairing.get("gate_pass") is not True:
                failures.append(f"{variant} terminal parameter/bake配对未通过")
                continue
            if isinstance(input_sha256, dict) and (
                pairing.get("formal_manifest_sha256")
                != input_sha256.get(f"{variant}_formal_manifest")
                or pairing.get("parameter_sha256")
                != input_sha256.get(f"{variant}_parameter")
                or pairing.get("bound_png_sha256")
                != input_sha256.get(f"{variant}_bake")
            ):
                failures.append(f"{variant} terminal pairing与输入SHA不一致")

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        failures.append("cases必须为JSON list")
        cases: list[object] = []
    else:
        cases = raw_cases
    expected_keys = {
        (variant, state_id)
        for variant in TERMINAL_RESPONSE_VARIANTS
        for state_id in frozen_state_ids
    }
    observed_keys: list[tuple[str, int]] = []
    clean_evidence_by_state: dict[int, TerminalDeploymentResponseEvidence] = {}
    clean_static_sha256_by_state: dict[int, object] = {}
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            failures.append("case必须为JSON object")
            continue
        variant = raw_case.get("variant")
        state_id = raw_case.get("state_id")
        if (
            not isinstance(variant, str)
            or not isinstance(state_id, int)
            or isinstance(state_id, bool)
        ):
            failures.append("case variant/state_id无效")
            continue
        observed_keys.append((variant, state_id))
    observed_key_set = set(observed_keys)
    duplicate_keys = sorted(
        key for key in observed_key_set if observed_keys.count(key) > 1
    )
    missing_keys = sorted(expected_keys - observed_key_set)
    unexpected_keys = sorted(observed_key_set - expected_keys)
    if duplicate_keys:
        failures.append(f"case inventory包含重复key: {duplicate_keys}")
    if missing_keys:
        failures.append(f"case inventory缺失key: {missing_keys}")
    if unexpected_keys:
        failures.append(f"case inventory包含额外key: {unexpected_keys}")
    expected_case_count = len(TERMINAL_RESPONSE_VARIANTS) * len(
        frozen_state_ids
    )
    if len(cases) != expected_case_count:
        failures.append(
            f"case inventory必须恰有{expected_case_count}行，实际{len(cases)}"
        )

    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        variant = raw_case.get("variant")
        state_id = raw_case.get("state_id")
        if (variant, state_id) not in expected_keys:
            continue
        assert isinstance(variant, str) and isinstance(state_id, int)
        if state_fingerprints and raw_case.get(
            "initial_state_sha256"
        ) != fingerprint_by_state[state_id]:
            failures.append(f"{variant}/state{state_id} initial fingerprint漂移")
        if raw_case.get("clean_static_scene_sha256") != raw_case.get(
            "deployment_static_scene_sha256"
        ):
            failures.append(f"{variant}/state{state_id} static scene不一致")
        clean_static_sha256 = raw_case.get("clean_static_scene_sha256")
        if not _is_lower_hex(clean_static_sha256, 64):
            failures.append(f"{variant}/state{state_id} static scene SHA无效")
        if state_id in clean_static_sha256_by_state:
            if clean_static_sha256_by_state[state_id] != clean_static_sha256:
                failures.append(f"state{state_id}跨variant Clean static scene不一致")
        else:
            clean_static_sha256_by_state[state_id] = clean_static_sha256
        if raw_case.get("transaction_verified") is not True:
            failures.append(f"{variant}/state{state_id}静态事务未验证")
        if raw_case.get("asset_restore_verified") is not True:
            failures.append(f"{variant}/state{state_id}资产恢复未验证")
        expected_sha256 = raw_case.get("npz_sha256")
        if not _is_lower_hex(expected_sha256, 64):
            failures.append(f"{variant}/state{state_id} NPZ SHA无效")
            continue
        try:
            npz_path = _resolve_bundle_artifact(
                resolved_manifest.parent,
                raw_case.get("npz_relative_path"),
            )
            if _file_sha256(npz_path) != expected_sha256:
                failures.append(f"{variant}/state{state_id} NPZ SHA不匹配")
                continue
            evidence = load_terminal_response_npz(npz_path)
            evaluation = evaluate_terminal_response_evidence(evidence)
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{variant}/state{state_id} NPZ无法复算: {error}")
            continue
        if evaluation.variant != variant or evaluation.state_id != state_id:
            failures.append(f"{variant}/state{state_id} NPZ身份不匹配")
            continue
        if state_id in clean_evidence_by_state:
            if not _shared_clean_is_exact(
                clean_evidence_by_state[state_id],
                evidence,
            ):
                failures.append(f"state{state_id}跨variant Clean证据不一致")
        else:
            clean_evidence_by_state[state_id] = evidence
        if not evaluation.evidence_valid:
            failures.extend(
                f"{variant}/state{state_id}: {failure}"
                for failure in evaluation.failures
            )
        per_variant_decisions[variant].append(evaluation.action_decision)

    summaries = {
        variant: summarize_terminal_action_responses(decisions)
        for variant, decisions in per_variant_decisions.items()
    }
    return TerminalDeploymentResponseBundleDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        case_count=len(cases),
        per_variant_summary=summaries,
    )


def evaluate_terminal_response_bundle(
    manifest_path: str | Path,
) -> TerminalDeploymentResponseBundleDecision:
    """验证正式双终态×states 0--9 inventory并独立复算全部NPZ。"""

    return _evaluate_terminal_response_bundle(
        manifest_path,
        expected_schema_version=(
            TERMINAL_DEPLOYMENT_RESPONSE_BUNDLE_SCHEMA_VERSION
        ),
        expected_state_ids=tuple(range(10)),
    )


def evaluate_terminal_response_smoke_bundle(
    manifest_path: str | Path,
) -> TerminalDeploymentResponseBundleDecision:
    """验证state 0双终态smoke的严格2-case inventory与全部NPZ。"""

    return _evaluate_terminal_response_bundle(
        manifest_path,
        expected_schema_version=(
            TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION
        ),
        expected_state_ids=(0,),
    )


def write_json_atomically(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    """完整写入、fsync并原子发布JSON，拒绝覆盖已有成功或临时文件。"""

    path = Path(output_path)
    temporary_path = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(path)
    if temporary_path.exists():
        raise FileExistsError(temporary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return _file_sha256(path)


def publish_terminal_response_smoke_manifest(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    """先独立复算候选2-case bundle，再原子发布正式smoke manifest。"""

    path = Path(output_path)
    candidate_path = path.with_name(path.stem + "_candidate.json")
    if path.exists() or candidate_path.exists():
        raise FileExistsError(path if path.exists() else candidate_path)
    write_json_atomically(payload, output_path=candidate_path)
    decision = evaluate_terminal_response_smoke_bundle(candidate_path)
    if not decision.audit_valid:
        candidate_path.unlink()
        raise TerminalDeploymentResponseAuditError(
            "state 0 smoke候选manifest独立复核失败: "
            + "; ".join(decision.failures)
        )
    os.replace(candidate_path, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _file_sha256(path)


def write_audit_failure_record(
    *,
    output_directory: str | Path,
    failed_stage: str,
    error: BaseException,
    completed_keys: Sequence[tuple[str, int]],
    input_sha256: Mapping[str, str],
    asset_restore_status: Mapping[str, bool],
) -> Path:
    """best-effort保存失败上下文；该文件永远不是成功manifest。"""

    if not failed_stage:
        raise ValueError("failed_stage不得为空")
    output_path = Path(output_directory) / "audit_failed.json"
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_DEPLOYMENT_RESPONSE_BUNDLE_SCHEMA_VERSION,
        "status": "audit_invalid",
        "failed_stage": failed_stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "completed_keys": [
            {"variant": variant, "state_id": state_id}
            for variant, state_id in completed_keys
        ],
        "input_sha256": dict(input_sha256),
        "asset_restore_status": dict(asset_restore_status),
    }
    write_json_atomically(payload, output_path=output_path)
    return output_path
