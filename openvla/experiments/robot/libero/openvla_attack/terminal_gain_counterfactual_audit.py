"""Gate 6h scalar-gain counterfactual 的纯 CPU 审计核心。

本模块不加载 OpenVLA、CUDA、renderer 或 LIBERO。第一条公共接口只消费 Gate
6g 已保存的 uint8 Effective View ``[C,A,B]``，在完整模型输入视野与全部 RGB
通道上拟合无约束标量 ``alpha_star``，再按正式 uint8 round-to-even 语义构造
诊断性反事实 ``I_gain``。该图像不要求能够由真实 renderer 物理生成。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .terminal_deployment_response_audit import (
    ActionPathEvidence,
    TerminalResponseClassification,
    classify_terminal_action_response,
    evaluate_terminal_response_bundle,
    load_terminal_response_npz,
)


FloatArray: TypeAlias = NDArray[np.floating]
GainMechanismClassification: TypeAlias = Literal[
    "gain_sufficient",
    "residual_necessary_for_first_response",
    "ambiguous",
]
GAIN_COUNTERFACTUAL_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-gain-counterfactual-v1"
)
GAIN_COUNTERFACTUAL_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-gain-counterfactual-bundle-v1"
)
GAIN_COUNTERFACTUAL_VARIANTS: Final[tuple[str, ...]] = (
    "action_spectral",
    "action_only_control",
)
PRIMARY_SOURCE_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {"deployment_lost", "deployment_response_altered"}
)


class TerminalGainCounterfactualAuditError(ValueError):
    """Gate 6h 输入不能被可靠解释。"""


@dataclass(frozen=True)
class ScalarGainCounterfactual:
    """一个 ``(variant,state)`` 的 scalar-gain 图像分解。

    所有 RGB 数组均为 ``[height,width,3]``。连续数组采用 float32、值域沿用
    uint8 的 ``[0,255]`` 标度；量化图像为 uint8，量化残差为 int16。
    """

    alpha_star: float
    continuous_gain_rgb: NDArray[np.float32]
    quantized_gain_rgb: NDArray[np.uint8]
    continuous_residual: NDArray[np.float32]
    quantized_residual: NDArray[np.int16]
    clipped_value_count: int


@dataclass(frozen=True)
class FirstResponseSignature:
    """相对于同一Clean的首次cached-generation离散响应。"""

    token_index: int
    action_class: int


@dataclass(frozen=True)
class ScalarGainMechanismDecision:
    """一个Gate 6g ``lost/altered`` case的三类机制结论。"""

    classification: GainMechanismClassification
    training_signature: FirstResponseSignature
    deployment_signature: FirstResponseSignature | None
    gain_signature: FirstResponseSignature | None


@dataclass(frozen=True)
class GainCounterfactualModelEvidence:
    """一个case的Gate 6g重放与新增gain路径原始模型证据。

    source路径轴固定为 ``[C,A,B]``；logits为float32
    ``[3,action_dim,num_classes]``，gain logits为相同尾shape。processor bits
    是完成正式processor与BF16 cast后的uint16 ``[1,C,H,W]``。
    """

    variant: str
    state_id: int
    source_effective_rgb: NDArray[np.uint8]
    alpha_star: float
    continuous_gain_rgb: NDArray[np.float32]
    quantized_gain_rgb: NDArray[np.uint8]
    continuous_residual: NDArray[np.float32]
    quantized_residual: NDArray[np.int16]
    clipped_value_count: int
    action_token_start: int
    action_token_end: int
    source_generated_token_ids: NDArray[np.integer]
    source_generated_classes: NDArray[np.integer]
    source_generation_logits: FloatArray
    source_teacher_logits: FloatArray
    replay_generated_token_ids: NDArray[np.integer]
    replay_generated_classes: NDArray[np.integer]
    replay_generation_logits: FloatArray
    replay_teacher_logits: FloatArray
    gain_generated_token_ids: NDArray[np.integer]
    gain_generated_classes: NDArray[np.integer]
    gain_generation_logits: FloatArray
    gain_teacher_logits: FloatArray
    gain_processor_bf16_bits: NDArray[np.uint16]


@dataclass(frozen=True)
class GainCounterfactualModelDecision:
    """CPU从一个Gate 6h case独立复算的工程与机制结论。"""

    variant: str
    state_id: int
    evidence_valid: bool
    failures: tuple[str, ...]
    source_classification: TerminalResponseClassification
    primary_mechanism_case: bool
    mechanism_decision: ScalarGainMechanismDecision | None


@dataclass(frozen=True)
class GainCounterfactualBundleDecision:
    """正式20-case bundle的工程状态与无阈值机制分布。"""

    audit_valid: bool
    failures: tuple[str, ...]
    case_count: int
    primary_case_count: int
    primary_classification_counts: dict[str, int]
    per_variant_primary_counts: dict[str, dict[str, int]]
    interpretation_branch: str | None


def _classes(value: NDArray[np.integer], *, name: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TerminalGainCounterfactualAuditError(
            f"{name}必须为非空整数[action_dim]"
        )
    return array.astype(np.int64, copy=False)


def first_response_signature(
    clean_classes: NDArray[np.integer],
    candidate_classes: NDArray[np.integer],
) -> FirstResponseSignature | None:
    """返回candidate相对于Clean的首次分歧位置和真实generation class。"""

    clean = _classes(clean_classes, name="clean_classes")
    candidate = _classes(candidate_classes, name="candidate_classes")
    if candidate.shape != clean.shape:
        raise TerminalGainCounterfactualAuditError(
            "clean/candidate action class shape必须一致"
        )
    indices = np.flatnonzero(candidate != clean)
    if indices.size == 0:
        return None
    index = int(indices[0])
    return FirstResponseSignature(
        token_index=index,
        action_class=int(candidate[index]),
    )


def classify_scalar_gain_mechanism(
    *,
    clean_classes: NDArray[np.integer],
    training_classes: NDArray[np.integer],
    deployment_classes: NDArray[np.integer],
    gain_classes: NDArray[np.integer],
) -> ScalarGainMechanismDecision:
    """分类scalar gain能否复现A到B的首次离散响应变化。"""

    clean = _classes(clean_classes, name="clean_classes")
    training_signature = first_response_signature(clean, training_classes)
    deployment_signature = first_response_signature(clean, deployment_classes)
    gain_signature = first_response_signature(clean, gain_classes)
    if training_signature is None or training_signature == deployment_signature:
        raise TerminalGainCounterfactualAuditError(
            "主机制分类只接受Gate 6g lost/altered case"
        )
    if gain_signature == deployment_signature:
        classification: GainMechanismClassification = "gain_sufficient"
    elif gain_signature == training_signature:
        classification = "residual_necessary_for_first_response"
    else:
        classification = "ambiguous"
    return ScalarGainMechanismDecision(
        classification=classification,
        training_signature=training_signature,
        deployment_signature=deployment_signature,
        gain_signature=gain_signature,
    )


def construct_scalar_gain_counterfactual(
    effective_rgb: NDArray[np.uint8],
) -> ScalarGainCounterfactual:
    """从冻结的 ``[C,A,B]`` Effective RGB 构造 scalar-gain 反事实。"""

    rgb = np.asarray(effective_rgb)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or rgb.shape[0] != 3
        or rgb.shape[-1] != 3
        or rgb.size == 0
    ):
        raise TerminalGainCounterfactualAuditError(
            "effective_rgb必须为非空uint8 [3,H,W,3]"
        )
    clean = rgb[0].astype(np.float64)
    training_delta = rgb[1].astype(np.float64) - clean
    deployment_delta = rgb[2].astype(np.float64) - clean
    denominator = float(np.sum(training_delta * training_delta))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise TerminalGainCounterfactualAuditError(
            "training delta平方范数必须有限且非零"
        )
    alpha_star = float(
        np.sum(training_delta * deployment_delta) / denominator
    )
    if not np.isfinite(alpha_star):
        raise TerminalGainCounterfactualAuditError("alpha_star不是有限数")

    continuous_unclipped = clean + alpha_star * training_delta
    clipped_value_count = int(
        np.count_nonzero(
            (continuous_unclipped < 0.0) | (continuous_unclipped > 255.0)
        )
    )
    continuous_gain = np.clip(continuous_unclipped, 0.0, 255.0)
    # NumPy rint 与 torch.round 一样使用 round-to-nearest-even；这里保留完整
    # float64 计算后再量化，避免中间 float32 改变恰好位于半整数的判定。
    quantized_gain = np.rint(continuous_gain).astype(np.uint8)
    continuous_residual = deployment_delta - alpha_star * training_delta
    quantized_residual = (
        rgb[2].astype(np.int16) - quantized_gain.astype(np.int16)
    )
    return ScalarGainCounterfactual(
        alpha_star=alpha_star,
        continuous_gain_rgb=continuous_gain.astype(np.float32),
        quantized_gain_rgb=quantized_gain,
        continuous_residual=continuous_residual.astype(np.float32),
        quantized_residual=quantized_residual,
        clipped_value_count=clipped_value_count,
    )


def processor_bf16_bits_from_effective_rgb(
    effective_rgb: NDArray[np.uint8],
    processor_specification: Mapping[str, Any],
) -> NDArray[np.uint16]:
    """用冻结processor specification在CPU逐位导出BF16 fused input。

    Gate 6h输入已经是最终224×224 Effective View，因此这里只执行uint8转
    float32、逐分支Normalize、拼接与IEEE BF16 round-to-nearest-even，不再次
    resize或center crop。
    """

    image = np.asarray(effective_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise TerminalGainCounterfactualAuditError(
            "processor输入必须为uint8 [H,W,3]"
        )
    output_size = processor_specification.get("output_size")
    branches = processor_specification.get("branches")
    if output_size != [int(image.shape[0]), int(image.shape[1])]:
        raise TerminalGainCounterfactualAuditError(
            "Gate 6h Effective View必须已经等于processor output size"
        )
    if not isinstance(branches, list) or not branches:
        raise TerminalGainCounterfactualAuditError(
            "processor branches缺失"
        )
    rgb = (
        image.astype(np.float32)
        .transpose(2, 0, 1)[None]
        / np.float32(255.0)
    )
    normalized: list[NDArray[np.float32]] = []
    for branch in branches:
        if not isinstance(branch, dict):
            raise TerminalGainCounterfactualAuditError(
                "processor branch必须为object"
            )
        mean = np.asarray(branch.get("mean"), dtype=np.float32)
        std = np.asarray(branch.get("std"), dtype=np.float32)
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or not bool(np.isfinite(mean).all())
            or not bool(np.isfinite(std).all())
            or bool(np.any(std <= 0.0))
        ):
            raise TerminalGainCounterfactualAuditError(
                "processor branch mean/std无效"
            )
        normalized.append(
            ((rgb - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1))
            .astype(np.float32, copy=False)
        )
    fused = np.concatenate(normalized, axis=1).astype(np.float32, copy=False)
    float_bits = fused.view(np.uint32)
    least_significant_retained = (
        float_bits >> np.uint32(16)
    ) & np.uint32(1)
    rounded = (
        float_bits
        + np.uint32(0x7FFF)
        + least_significant_retained
    ) >> np.uint32(16)
    return rounded.astype(np.uint16)


def _path_evidence(
    classes: NDArray[np.integer],
    generation_logits: FloatArray,
    teacher_logits: FloatArray,
) -> ActionPathEvidence:
    return ActionPathEvidence(
        generated_classes=np.asarray(classes),
        generation_logits=np.asarray(generation_logits),
        teacher_logits=np.asarray(teacher_logits),
    )


def evaluate_gain_counterfactual_model_evidence(
    evidence: GainCounterfactualModelEvidence,
) -> GainCounterfactualModelDecision:
    """验证C/A/B重放与gain输入，并分类冻结的主机制case。"""

    failures: list[str] = []
    if evidence.variant not in GAIN_COUNTERFACTUAL_VARIANTS:
        raise TerminalGainCounterfactualAuditError("Gate 6h variant无效")
    if not isinstance(evidence.state_id, int) or not 0 <= evidence.state_id <= 9:
        raise TerminalGainCounterfactualAuditError("Gate 6h state_id必须为0--9")
    if (
        not isinstance(evidence.action_token_start, int)
        or not isinstance(evidence.action_token_end, int)
        or evidence.action_token_end <= evidence.action_token_start
    ):
        raise TerminalGainCounterfactualAuditError("action token slice无效")
    source_token_ids = np.asarray(evidence.source_generated_token_ids)
    source_classes = np.asarray(evidence.source_generated_classes)
    source_generation = np.asarray(evidence.source_generation_logits)
    source_teacher = np.asarray(evidence.source_teacher_logits)
    replay_token_ids = np.asarray(evidence.replay_generated_token_ids)
    replay_classes = np.asarray(evidence.replay_generated_classes)
    replay_generation = np.asarray(evidence.replay_generation_logits)
    replay_teacher = np.asarray(evidence.replay_teacher_logits)
    if (
        source_classes.ndim != 2
        or source_classes.shape[0] != 3
        or source_classes.shape[1] <= 0
        or source_classes.dtype != np.int64
        or source_token_ids.shape != source_classes.shape
        or source_token_ids.dtype != np.int64
        or source_generation.ndim != 3
        or source_generation.shape[:2] != source_classes.shape
        or source_generation.shape[2] < 2
        or source_generation.shape[2]
        != evidence.action_token_end - evidence.action_token_start
        or source_teacher.shape != source_generation.shape
        or source_generation.dtype != np.float32
        or source_teacher.dtype != np.float32
        or not bool(np.isfinite(source_generation).all())
        or not bool(np.isfinite(source_teacher).all())
    ):
        raise TerminalGainCounterfactualAuditError(
            "source C/A/B token或logits shape/dtype无效"
        )
    if (
        replay_classes.shape != source_classes.shape
        or replay_classes.dtype != np.int64
        or replay_token_ids.shape != source_token_ids.shape
        or replay_token_ids.dtype != np.int64
        or replay_generation.shape != source_generation.shape
        or replay_teacher.shape != source_teacher.shape
    ):
        raise TerminalGainCounterfactualAuditError(
            "C/A/B replay数组shape与source不一致"
        )
    if not np.array_equal(replay_classes, source_classes):
        failures.append("C/A/B replay generation classes未逐值复现Gate 6g")
    if not np.array_equal(replay_token_ids, source_token_ids):
        failures.append("C/A/B replay generation token IDs未逐值复现Gate 6g")
    if not np.array_equal(replay_generation, source_generation):
        failures.append("C/A/B replay generation logits未逐值复现Gate 6g")
    if not np.array_equal(replay_teacher, source_teacher):
        failures.append("C/A/B replay teacher logits未逐值复现Gate 6g")
    if not np.array_equal(
        source_token_ids - evidence.action_token_start,
        source_classes,
    ):
        failures.append("source token ID与action class映射无效")
    if not np.array_equal(
        replay_token_ids - evidence.action_token_start,
        replay_classes,
    ):
        failures.append("replay token ID与action class映射无效")

    reconstructed = construct_scalar_gain_counterfactual(
        evidence.source_effective_rgb
    )
    if evidence.alpha_star != reconstructed.alpha_star:
        failures.append("alpha_star不能由source RGB逐值复算")
    expected_dtypes = {
        "continuous_gain_rgb": np.dtype(np.float32),
        "quantized_gain_rgb": np.dtype(np.uint8),
        "continuous_residual": np.dtype(np.float32),
        "quantized_residual": np.dtype(np.int16),
    }
    for name in (
        "continuous_gain_rgb",
        "quantized_gain_rgb",
        "continuous_residual",
        "quantized_residual",
    ):
        if np.asarray(getattr(evidence, name)).dtype != expected_dtypes[name]:
            failures.append(f"{name} dtype不符合冻结合同")
        if not np.array_equal(
            np.asarray(getattr(evidence, name)),
            np.asarray(getattr(reconstructed, name)),
        ):
            failures.append(f"{name}不能由source RGB逐值复算")
    if evidence.clipped_value_count != reconstructed.clipped_value_count:
        failures.append("clipped_value_count不能由source RGB复算")

    gain_token_ids = np.asarray(evidence.gain_generated_token_ids)
    gain_classes = np.asarray(evidence.gain_generated_classes)
    gain_generation = np.asarray(evidence.gain_generation_logits)
    gain_teacher = np.asarray(evidence.gain_teacher_logits)
    if (
        gain_classes.shape != source_classes.shape[1:]
        or gain_classes.dtype != np.int64
        or gain_token_ids.shape != gain_classes.shape
        or gain_token_ids.dtype != np.int64
        or gain_generation.shape != source_generation.shape[1:]
        or gain_teacher.shape != source_teacher.shape[1:]
        or gain_generation.dtype != np.float32
        or gain_teacher.dtype != np.float32
        or not bool(np.isfinite(gain_generation).all())
        or not bool(np.isfinite(gain_teacher).all())
    ):
        raise TerminalGainCounterfactualAuditError(
            "gain token/logits shape/dtype无效"
        )
    if not np.array_equal(
        gain_token_ids - evidence.action_token_start,
        gain_classes,
    ):
        failures.append("gain token ID与action class映射无效")
    gain_bits = np.asarray(evidence.gain_processor_bf16_bits)
    if gain_bits.dtype != np.uint16 or gain_bits.ndim != 4 or gain_bits.size == 0:
        raise TerminalGainCounterfactualAuditError(
            "gain processor BF16 bits必须为非空uint16 [1,C,H,W]"
        )

    clean = _path_evidence(
        source_classes[0], source_generation[0], source_teacher[0]
    )
    training = _path_evidence(
        source_classes[1], source_generation[1], source_teacher[1]
    )
    deployment = _path_evidence(
        source_classes[2], source_generation[2], source_teacher[2]
    )
    source_decision = classify_terminal_action_response(
        clean=clean,
        training=training,
        deployment=deployment,
    )
    gain_path = _path_evidence(gain_classes, gain_generation, gain_teacher)
    gain_alignment = classify_terminal_action_response(
        clean=clean,
        training=training,
        deployment=gain_path,
    )
    if source_decision.classification == "invalid_response_alignment":
        failures.append("Gate 6g source generation自对齐无效")
    if gain_alignment.classification == "invalid_response_alignment":
        failures.append("gain generation token与score自对齐无效")
    primary = source_decision.classification in {
        "deployment_lost",
        "deployment_response_altered",
    }
    mechanism = (
        classify_scalar_gain_mechanism(
            clean_classes=source_classes[0],
            training_classes=source_classes[1],
            deployment_classes=source_classes[2],
            gain_classes=gain_classes,
        )
        if primary and not failures
        else None
    )
    return GainCounterfactualModelDecision(
        variant=evidence.variant,
        state_id=evidence.state_id,
        evidence_valid=not failures,
        failures=tuple(failures),
        source_classification=source_decision.classification,
        primary_mechanism_case=primary,
        mechanism_decision=mechanism,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_gain_counterfactual_npz(
    evidence: GainCounterfactualModelEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存一个无pickle Gate 6h case，并返回文件SHA-256。"""

    decision = evaluate_gain_counterfactual_model_evidence(evidence)
    if not decision.evidence_valid:
        raise TerminalGainCounterfactualAuditError(
            "不能保存无效Gate 6h case: " + "; ".join(decision.failures)
        )
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(GAIN_COUNTERFACTUAL_SCHEMA_VERSION),
        variant=np.asarray(evidence.variant),
        state_id=np.asarray(evidence.state_id, dtype=np.int64),
        source_effective_rgb=evidence.source_effective_rgb,
        alpha_star=np.asarray(evidence.alpha_star, dtype=np.float64),
        continuous_gain_rgb=evidence.continuous_gain_rgb,
        quantized_gain_rgb=evidence.quantized_gain_rgb,
        continuous_residual=evidence.continuous_residual,
        quantized_residual=evidence.quantized_residual,
        clipped_value_count=np.asarray(
            evidence.clipped_value_count, dtype=np.int64
        ),
        action_token_start=np.asarray(
            evidence.action_token_start, dtype=np.int64
        ),
        action_token_end=np.asarray(evidence.action_token_end, dtype=np.int64),
        source_generated_token_ids=evidence.source_generated_token_ids,
        source_generated_classes=evidence.source_generated_classes,
        source_generation_logits=evidence.source_generation_logits,
        source_teacher_logits=evidence.source_teacher_logits,
        replay_generated_token_ids=evidence.replay_generated_token_ids,
        replay_generated_classes=evidence.replay_generated_classes,
        replay_generation_logits=evidence.replay_generation_logits,
        replay_teacher_logits=evidence.replay_teacher_logits,
        gain_generated_token_ids=evidence.gain_generated_token_ids,
        gain_generated_classes=evidence.gain_generated_classes,
        gain_generation_logits=evidence.gain_generation_logits,
        gain_teacher_logits=evidence.gain_teacher_logits,
        gain_processor_bf16_bits=evidence.gain_processor_bf16_bits,
    )
    return _file_sha256(path)


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    if value.shape != ():
        raise TerminalGainCounterfactualAuditError(
            f"NPZ字段{name}必须为scalar"
        )
    return value.item()


def load_gain_counterfactual_npz(
    path: str | Path,
) -> GainCounterfactualModelEvidence:
    """无pickle加载一个Gate 6h case并执行完整CPU复核。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(_scalar(archive, "schema_version")) != (
            GAIN_COUNTERFACTUAL_SCHEMA_VERSION
        ):
            raise TerminalGainCounterfactualAuditError(
                "Gate 6h NPZ schema不匹配"
            )
        evidence = GainCounterfactualModelEvidence(
            variant=str(_scalar(archive, "variant")),
            state_id=int(_scalar(archive, "state_id")),
            source_effective_rgb=archive["source_effective_rgb"].copy(),
            alpha_star=float(_scalar(archive, "alpha_star")),
            continuous_gain_rgb=archive["continuous_gain_rgb"].copy(),
            quantized_gain_rgb=archive["quantized_gain_rgb"].copy(),
            continuous_residual=archive["continuous_residual"].copy(),
            quantized_residual=archive["quantized_residual"].copy(),
            clipped_value_count=int(_scalar(archive, "clipped_value_count")),
            action_token_start=int(_scalar(archive, "action_token_start")),
            action_token_end=int(_scalar(archive, "action_token_end")),
            source_generated_token_ids=archive[
                "source_generated_token_ids"
            ].copy(),
            source_generated_classes=archive[
                "source_generated_classes"
            ].copy(),
            source_generation_logits=archive[
                "source_generation_logits"
            ].copy(),
            source_teacher_logits=archive["source_teacher_logits"].copy(),
            replay_generated_token_ids=archive[
                "replay_generated_token_ids"
            ].copy(),
            replay_generated_classes=archive[
                "replay_generated_classes"
            ].copy(),
            replay_generation_logits=archive[
                "replay_generation_logits"
            ].copy(),
            replay_teacher_logits=archive["replay_teacher_logits"].copy(),
            gain_generated_token_ids=archive[
                "gain_generated_token_ids"
            ].copy(),
            gain_generated_classes=archive["gain_generated_classes"].copy(),
            gain_generation_logits=archive["gain_generation_logits"].copy(),
            gain_teacher_logits=archive["gain_teacher_logits"].copy(),
            gain_processor_bf16_bits=archive[
                "gain_processor_bf16_bits"
            ].copy(),
        )
    evaluate_gain_counterfactual_model_evidence(evidence)
    return evidence


def gain_counterfactual_decision_record(
    decision: GainCounterfactualModelDecision,
) -> dict[str, Any]:
    """生成可由独立CPU evaluator逐值复算的JSON记录。"""

    result = json.loads(json.dumps(asdict(decision), sort_keys=True))
    if not isinstance(result, dict):
        raise AssertionError("Gate 6h decision必须编码为object")
    return result


def evaluate_gain_counterfactual_npz(
    path: str | Path,
) -> GainCounterfactualModelDecision:
    """从单个无pickleNPZ复算Gate 6h工程与机制结论。"""

    return evaluate_gain_counterfactual_model_evidence(
        load_gain_counterfactual_npz(path)
    )


def _is_sha(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise TerminalGainCounterfactualAuditError("artifact相对路径无效")
    candidate = Path(value)
    if candidate.is_absolute():
        raise TerminalGainCounterfactualAuditError("artifact路径必须为相对路径")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise TerminalGainCounterfactualAuditError("artifact路径逃逸bundle")
    return resolved


def _source_case_map(
    source_manifest_path: Path,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], Mapping[str, Any]]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise TerminalGainCounterfactualAuditError("Gate 6g manifest不是object")
    cases = source_manifest.get("cases")
    if not isinstance(cases, list):
        raise TerminalGainCounterfactualAuditError("Gate 6g cases无效")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise TerminalGainCounterfactualAuditError("Gate 6g case无效")
        key = (case.get("variant"), case.get("state_id"))
        if not isinstance(key[0], str) or not isinstance(key[1], int):
            raise TerminalGainCounterfactualAuditError("Gate 6g case身份无效")
        result[(key[0], key[1])] = case
    return result, source_manifest


def _empty_counts() -> dict[str, int]:
    return {
        "gain_sufficient": 0,
        "residual_necessary_for_first_response": 0,
        "ambiguous": 0,
    }


def interpret_gain_counterfactual_results(
    per_variant: Mapping[str, Mapping[str, int]],
) -> str:
    """应用预注册解释树；返回研究分支而非统计pass/fail。"""

    if set(per_variant) != set(GAIN_COUNTERFACTUAL_VARIANTS):
        raise TerminalGainCounterfactualAuditError("解释树variant inventory无效")
    required = set(_empty_counts())
    if any(
        set(counts) != required
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
        for counts in per_variant.values()
    ):
        raise TerminalGainCounterfactualAuditError("解释树分类计数无效")
    residual_variants = {
        variant
        for variant, counts in per_variant.items()
        if counts["residual_necessary_for_first_response"] > 0
    }
    if residual_variants == set(GAIN_COUNTERFACTUAL_VARIANTS):
        return "shared_non_scalar_residual_priority"
    if residual_variants:
        return "endpoint_or_trajectory_specific"
    total_gain = sum(counts["gain_sufficient"] for counts in per_variant.values())
    total_ambiguous = sum(counts["ambiguous"] for counts in per_variant.values())
    if total_gain > total_ambiguous:
        return "scalar_gain_calibration_priority"
    return "counterfactual_inconclusive_stop"


def evaluate_gain_counterfactual_bundle(
    manifest_path: str | Path,
    *,
    source_gate6g_manifest_path: str | Path | None = None,
) -> GainCounterfactualBundleDecision:
    """独立复核Gate 6h正式20-case bundle和冻结的6-case主分母。"""

    failures: list[str] = []
    try:
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TerminalGainCounterfactualAuditError("manifest不是object")
        source_manifest_raw = (
            source_gate6g_manifest_path
            if source_gate6g_manifest_path is not None
            else manifest.get("source_gate6g_manifest_path")
        )
        if not isinstance(source_manifest_raw, (str, Path)):
            raise TerminalGainCounterfactualAuditError("source manifest路径缺失")
        source_manifest_path = Path(source_manifest_raw).resolve()
        source_cases, source_manifest = _source_case_map(source_manifest_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return GainCounterfactualBundleDecision(
            False, (f"Gate 6h manifest无法加载: {error}",), 0, 0, {}, {}, None
        )
    if manifest.get("schema_version") != GAIN_COUNTERFACTUAL_BUNDLE_SCHEMA_VERSION:
        failures.append("bundle schema_version不匹配")
    if manifest.get("status") != "complete":
        failures.append("bundle status必须为complete")
    if not _is_sha(manifest.get("code_commit"), 40):
        failures.append("code_commit无效")
    if _file_sha256(source_manifest_path) != manifest.get(
        "source_gate6g_manifest_sha256"
    ):
        failures.append("source Gate 6g manifest SHA不匹配")
    source_decision = evaluate_terminal_response_bundle(source_manifest_path)
    if not source_decision.audit_valid:
        failures.append("source Gate 6g bundle独立复核失败")
    if manifest.get("expected_variants") != list(
        GAIN_COUNTERFACTUAL_VARIANTS
    ):
        failures.append("expected_variants未冻结为双终态")
    if manifest.get("expected_state_ids") != list(range(10)):
        failures.append("expected_state_ids未冻结为states0--9")
    if manifest.get("primary_analysis_source_classifications") != [
        "deployment_lost",
        "deployment_response_altered",
    ]:
        failures.append("主机制source分类未冻结为lost/altered")
    if manifest.get("response_authority") != source_manifest.get(
        "response_authority"
    ):
        failures.append("response authority与source Gate 6g不一致")
    provenance = manifest.get("provenance")
    source_provenance = source_manifest.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(
        source_provenance, dict
    ):
        failures.append("provenance缺失")
    else:
        for name in (
            "checkpoint_fingerprints",
            "processor_specification",
            "policy_view_specification",
        ):
            if provenance.get(name) != source_provenance.get(name):
                failures.append(f"{name}与source Gate 6g不一致")
        if provenance.get("source_gate6g_state_fingerprints") != (
            source_manifest.get("state_fingerprints")
        ):
            failures.append("state fingerprints与source Gate 6g不一致")
        if provenance.get("legacy_optimizer_modules") != []:
            failures.append("Gate 6h加载了legacy optimizer")
        if provenance.get("training_or_backward_run") is not False:
            failures.append("Gate 6h training/backward provenance无效")
        if provenance.get("libero_or_renderer_loaded") is not False:
            failures.append("Gate 6h加载了LIBERO或renderer")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        cases = []
        failures.append("cases必须为list")
    expected_keys = {
        (variant, state_id)
        for variant in GAIN_COUNTERFACTUAL_VARIANTS
        for state_id in range(10)
    }
    observed: list[tuple[str, int]] = []
    primary_counts = _empty_counts()
    per_variant = {
        variant: _empty_counts() for variant in GAIN_COUNTERFACTUAL_VARIANTS
    }
    primary_keys: list[tuple[str, int]] = []
    for raw in cases:
        if not isinstance(raw, dict):
            failures.append("case必须为object")
            continue
        variant, state_id = raw.get("variant"), raw.get("state_id")
        if not isinstance(variant, str) or not isinstance(state_id, int):
            failures.append("case身份无效")
            continue
        key = (variant, state_id)
        observed.append(key)
        source = source_cases.get(key)
        if source is None:
            failures.append(f"{key}不属于source Gate 6g")
            continue
        if raw.get("source_npz_sha256") != source.get("npz_sha256"):
            failures.append(f"{key} source NPZ SHA漂移")
        try:
            artifact = _resolve(path.parent, raw.get("npz_relative_path"))
            if _file_sha256(artifact) != raw.get("npz_sha256"):
                failures.append(f"{key} Gate 6h NPZ SHA不匹配")
                continue
            evidence = load_gain_counterfactual_npz(artifact)
            decision = evaluate_gain_counterfactual_model_evidence(evidence)
            source_artifact = _resolve(
                source_manifest_path.parent,
                source.get("npz_relative_path"),
            )
            source_evidence = load_terminal_response_npz(source_artifact)
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{key} NPZ无法复算: {error}")
            continue
        if decision.variant != variant or decision.state_id != state_id:
            failures.append(f"{key} NPZ身份漂移")
        source_bindings = (
            (
                "source_effective_rgb",
                evidence.source_effective_rgb,
                source_evidence.effective_rgb,
            ),
            (
                "source_generated_token_ids",
                evidence.source_generated_token_ids,
                source_evidence.generated_token_ids,
            ),
            (
                "source_generated_classes",
                evidence.source_generated_classes,
                source_evidence.generated_classes,
            ),
            (
                "source_generation_logits",
                evidence.source_generation_logits,
                source_evidence.generation_logits,
            ),
            (
                "source_teacher_logits",
                evidence.source_teacher_logits,
                source_evidence.teacher_logits,
            ),
        )
        for name, observed_array, expected_array in source_bindings:
            if not np.array_equal(observed_array, expected_array):
                failures.append(
                    f"{key} {name}未逐值绑定source Gate 6g NPZ"
                )
        if (
            evidence.action_token_start != source_evidence.action_token_start
            or evidence.action_token_end != source_evidence.action_token_end
        ):
            failures.append(
                f"{key} action token slice未绑定source Gate 6g NPZ"
            )
        if isinstance(provenance, dict):
            try:
                expected_gain_bits = processor_bf16_bits_from_effective_rgb(
                    evidence.quantized_gain_rgb,
                    provenance.get("processor_specification", {}),
                )
            except (ValueError, TypeError) as error:
                failures.append(f"{key} gain processor bits无法复算: {error}")
            else:
                if not np.array_equal(
                    evidence.gain_processor_bf16_bits,
                    expected_gain_bits,
                ):
                    failures.append(
                        f"{key} gain processor BF16 bits不能由I_gain逐位复算"
                    )
        if not decision.evidence_valid:
            failures.extend(
                f"{key}: {failure}" for failure in decision.failures
            )
        if raw.get("evaluation") != gain_counterfactual_decision_record(decision):
            failures.append(f"{key} evaluation不能由NPZ复算")
        if decision.primary_mechanism_case:
            primary_keys.append(key)
            if decision.mechanism_decision is None:
                failures.append(f"{key}缺少主机制分类")
            else:
                classification = decision.mechanism_decision.classification
                primary_counts[classification] += 1
                per_variant[variant][classification] += 1
    if (
        len(cases) != 20
        or set(observed) != expected_keys
        or len(observed) != len(set(observed))
    ):
        failures.append("case inventory必须为双终态×states0--9的20个唯一case")
    expected_primary = sorted(
        (str(case["variant"]), int(case["state_id"]))
        for case in source_cases.values()
        if case.get("response_evaluation", {})
        .get("action_decision", {})
        .get("classification") in PRIMARY_SOURCE_CLASSIFICATIONS
    )
    if sorted(primary_keys) != expected_primary or len(primary_keys) != 6:
        failures.append("主机制分母未严格复用Gate 6g冻结的6个lost/altered case")
    return GainCounterfactualBundleDecision(
        audit_valid=not failures,
        failures=tuple(failures),
        case_count=len(cases),
        primary_case_count=len(primary_keys),
        primary_classification_counts=primary_counts,
        per_variant_primary_counts=per_variant,
        interpretation_branch=(
            None
            if failures
            else interpret_gain_counterfactual_results(per_variant)
        ),
    )


def write_json_atomically(
    payload: Mapping[str, Any], *, output_path: str | Path
) -> str:
    """原子写JSON并拒绝覆盖。"""

    path = Path(output_path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _file_sha256(path)


def publish_gain_counterfactual_manifest(
    payload: Mapping[str, Any], *, output_path: str | Path
) -> str:
    """独立复核候选bundle后原子发布Gate 6h成功manifest。"""

    path = Path(output_path)
    candidate = path.with_name(path.stem + "_candidate.json")
    write_json_atomically(payload, output_path=candidate)
    decision = evaluate_gain_counterfactual_bundle(candidate)
    if not decision.audit_valid:
        candidate.unlink()
        raise TerminalGainCounterfactualAuditError(
            "Gate 6h候选manifest复核失败: " + "; ".join(decision.failures)
        )
    os.replace(candidate, path)
    return _file_sha256(path)
