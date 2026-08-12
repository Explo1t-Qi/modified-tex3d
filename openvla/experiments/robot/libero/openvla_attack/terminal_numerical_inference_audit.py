"""Gate 6g numerical inference replay 的纯 CPU evidence contract。

本模块只解释已经保存的 NumPy 数组，不加载 OpenVLA、CUDA、renderer 或
LIBERO。它把三个不能混淆的状态分开：失败 bundle 中原始跨路径观察是否存在、
当前环境能否逐位重放原始模型输出，以及重放通过后是否允许对 execution path
差异作因果归因。

路径轴固定为 ``[clean, training, deployment]``，variant 轴固定为
``[action_spectral, action_only_control]``。每个模型执行数组都有显式 repeat 轴；
任何比较均使用完整 ``[action_dim, 256]`` action-subvocabulary logits。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]
FidelityStatus: TypeAlias = Literal[
    "pass",
    "input_reconstruction_invalid",
    "runtime_nondeterministic",
    "replay_provenance_unresolved",
    "mixed_replay_failure",
]

NUMERICAL_FIDELITY_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-numerical-fidelity-v1"
)
NUMERICAL_ATTRIBUTION_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-numerical-attribution-v1"
)
NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "openvla-terminal-numerical-inference-bundle-v1"
)
NUMERICAL_VARIANTS: Final[tuple[str, ...]] = (
    "action_spectral",
    "action_only_control",
)
NUMERICAL_PATHS: Final[tuple[str, ...]] = (
    "clean",
    "training",
    "deployment",
)
ATTRIBUTION_COMPARISON_NAMES: Final[tuple[str, ...]] = (
    "cached_vs_no_cache_generation",
    "cached_vs_generation_prefix_no_cache",
    "original_teacher_vs_full_teacher_no_cache",
    "full_teacher_no_cache_vs_clean_prefix_no_cache",
)


class TerminalNumericalInferenceAuditError(ValueError):
    """Numerical inference evidence 无法被严格解释。"""


@dataclass(frozen=True)
class NumericalFidelityEvidence:
    """原 NPZ 与三次模型 replay 的完整逐输入数组。"""

    variant_names: tuple[str, ...]
    path_names: tuple[str, ...]
    repeat_count: int
    action_token_start: int
    source_prompt_input_ids: IntArray
    source_teacher_input_ids: IntArray
    reconstructed_prompt_input_ids: IntArray
    reconstructed_teacher_input_ids: IntArray
    reconstructed_attention_mask: IntArray
    source_pixel_bf16_bits: NDArray[np.uint16]
    reconstructed_pixel_bf16_bits: NDArray[np.uint16]
    source_generated_token_ids: IntArray
    source_generated_classes: IntArray
    source_generation_logits: FloatArray
    source_teacher_logits: FloatArray
    replay_generated_token_ids: IntArray
    replay_generated_classes: IntArray
    replay_generation_logits: FloatArray
    replay_teacher_logits: FloatArray


@dataclass(frozen=True)
class OriginalMismatchRecord:
    """失败 bundle 中一个共享 causal prefix 上的原始跨路径分歧。"""

    variant: str
    path: str
    token_index: int
    generation_class: int
    teacher_argmax_classes: tuple[int, ...]
    evidence_scope: str = "invalid_bundle_diagnostic_observation"


@dataclass(frozen=True)
class FidelityInputDecision:
    """一个 ``(variant,C/A/B)`` 输入的逐位重放判定。"""

    variant: str
    path: str
    fidelity_pass: bool
    status: FidelityStatus
    input_reconstruction_exact: bool
    generation_repeat_exact: bool
    teacher_repeat_exact: bool
    generation_tokens_match_source: bool
    generation_logits_match_source: bool
    teacher_matches_source: bool


@dataclass(frozen=True)
class NumericalFidelityDecision:
    """Replay Fidelity 的 per-input 与 run-level 状态。"""

    fidelity_pass: bool
    causal_attribution_allowed: bool
    per_input: tuple[FidelityInputDecision, ...]
    original_mismatches: tuple[OriginalMismatchRecord, ...]


@dataclass(frozen=True)
class NumericalAttributionEvidence:
    """Fidelity 通过后四条补充 execution path 的重复原始数组。"""

    variant_names: tuple[str, ...]
    path_names: tuple[str, ...]
    repeat_count: int
    no_cache_generated_token_ids: IntArray
    no_cache_generation_logits: FloatArray
    generation_prefix_no_cache_logits: FloatArray
    clean_prefix_no_cache_logits: FloatArray
    full_teacher_no_cache_logits: FloatArray


@dataclass(frozen=True)
class TokenLogitComparison:
    """相同 token 位置的两条 action-subvocabulary logits 对照。"""

    token_index: int
    linf: float
    max_abs_difference_classes: tuple[int, ...]
    left_argmax_classes: tuple[int, ...]
    right_argmax_classes: tuple[int, ...]
    argmax_equal: bool
    causal_prefix_comparable: bool
    left_top1_class: int
    left_top1_logit: float
    left_top2_logit: float
    left_top2_margin: float
    right_top1_class: int
    right_top1_logit: float
    right_top2_logit: float
    right_top2_margin: float


@dataclass(frozen=True)
class LogitComparisonDecision:
    """一个命名 execution-path 对照的全部 token 摘要。"""

    name: str
    tokens: tuple[TokenLogitComparison, ...]

    @property
    def linf(self) -> tuple[float, ...]:
        return tuple(token.linf for token in self.tokens)

    @property
    def argmax_equal(self) -> tuple[bool, ...]:
        return tuple(token.argmax_equal for token in self.tokens)


@dataclass(frozen=True)
class AttributionInputDecision:
    """一个输入的 alternate-path repeat 与对照结论。"""

    variant: str
    path: str
    paths_stable: bool
    no_cache_generation_repeat_exact: bool
    generation_prefix_repeat_exact: bool
    clean_prefix_repeat_exact: bool
    full_teacher_no_cache_repeat_exact: bool
    cached_vs_no_cache_first_divergence_index: int | None
    comparisons: tuple[LogitComparisonDecision, ...]


@dataclass(frozen=True)
class NumericalAttributionDecision:
    """Execution-path attribution 的 run-level 判定。"""

    attribution_valid: bool
    causal_attribution_allowed: bool
    per_input: tuple[AttributionInputDecision, ...]


@dataclass(frozen=True)
class NumericalInferenceBundleDecision:
    """独立CPU复核一个 numerical inference bundle 的工程结论。"""

    bundle_valid: bool
    failures: tuple[str, ...]
    fidelity_decision: NumericalFidelityDecision | None
    attribution_decision: NumericalAttributionDecision | None


def _is_integer_array(value: NDArray[object]) -> bool:
    return np.issubdtype(np.asarray(value).dtype, np.integer)


def _is_finite_float_array(value: NDArray[object]) -> bool:
    array = np.asarray(value)
    return np.issubdtype(array.dtype, np.floating) and bool(
        np.isfinite(array).all()
    )


def _validate_fidelity_evidence(evidence: NumericalFidelityEvidence) -> None:
    variants = len(evidence.variant_names)
    paths = len(evidence.path_names)
    if evidence.variant_names != NUMERICAL_VARIANTS:
        raise TerminalNumericalInferenceAuditError("numerical variant顺序无效")
    if evidence.path_names != NUMERICAL_PATHS:
        raise TerminalNumericalInferenceAuditError("numerical C/A/B顺序无效")
    if evidence.repeat_count != 3:
        raise TerminalNumericalInferenceAuditError("Replay Fidelity固定重复3次")
    prompt = np.asarray(evidence.source_prompt_input_ids)
    teacher = np.asarray(evidence.source_teacher_input_ids)
    if (
        prompt.ndim != 2
        or prompt.shape[0] != variants
        or prompt.shape[1] <= 0
        or teacher.ndim != 2
        or teacher.shape[0] != variants
        or teacher.shape[1] <= prompt.shape[1]
        or not _is_integer_array(prompt)
        or not _is_integer_array(teacher)
    ):
        raise TerminalNumericalInferenceAuditError("source prompt/teacher IDs无效")
    action_dim = teacher.shape[1] - prompt.shape[1]
    if not np.array_equal(teacher[:, : prompt.shape[1]], prompt):
        raise TerminalNumericalInferenceAuditError(
            "source teacher IDs必须以保存的prompt IDs开头"
        )
    clean_action_classes = (
        teacher[:, -action_dim:] - int(evidence.action_token_start)
    )
    expected_id_shape = (variants, paths, action_dim)
    expected_logit_prefix = (variants, paths, action_dim)
    source_tokens = np.asarray(evidence.source_generated_token_ids)
    source_classes = np.asarray(evidence.source_generated_classes)
    source_generation = np.asarray(evidence.source_generation_logits)
    source_teacher = np.asarray(evidence.source_teacher_logits)
    if (
        source_tokens.shape != expected_id_shape
        or source_classes.shape != expected_id_shape
        or not _is_integer_array(source_tokens)
        or not _is_integer_array(source_classes)
        or source_generation.ndim != 4
        or source_generation.shape[:3] != expected_logit_prefix
        or source_generation.shape[3] < 2
        or source_teacher.shape != source_generation.shape
        or not _is_finite_float_array(source_generation)
        or not _is_finite_float_array(source_teacher)
    ):
        raise TerminalNumericalInferenceAuditError("source token/logits shape无效")
    if not np.array_equal(
        source_tokens - int(evidence.action_token_start), source_classes
    ):
        raise TerminalNumericalInferenceAuditError("source token/class mapping无效")
    num_classes = int(source_generation.shape[3])
    if (
        bool(np.any(source_classes < 0))
        or bool(np.any(source_classes >= num_classes))
        or bool(np.any(clean_action_classes < 0))
        or bool(np.any(clean_action_classes >= num_classes))
    ):
        raise TerminalNumericalInferenceAuditError(
            "source generated/clean action class越界"
        )
    replay_id_shape = (variants, paths, evidence.repeat_count, action_dim)
    replay_logit_shape = (
        variants,
        paths,
        evidence.repeat_count,
        action_dim,
        source_generation.shape[3],
    )
    if (
        np.asarray(evidence.replay_generated_token_ids).shape != replay_id_shape
        or np.asarray(evidence.replay_generated_classes).shape != replay_id_shape
        or np.asarray(evidence.replay_generation_logits).shape
        != replay_logit_shape
        or np.asarray(evidence.replay_teacher_logits).shape != replay_logit_shape
        or not _is_integer_array(evidence.replay_generated_token_ids)
        or not _is_integer_array(evidence.replay_generated_classes)
        or not _is_finite_float_array(evidence.replay_generation_logits)
        or not _is_finite_float_array(evidence.replay_teacher_logits)
    ):
        raise TerminalNumericalInferenceAuditError("replay token/logits shape无效")
    if not np.array_equal(
        np.asarray(evidence.replay_generated_token_ids)
        - int(evidence.action_token_start),
        np.asarray(evidence.replay_generated_classes),
    ):
        raise TerminalNumericalInferenceAuditError("replay token/class mapping无效")
    for name, value, expected_shape in (
        (
            "reconstructed_prompt_input_ids",
            evidence.reconstructed_prompt_input_ids,
            prompt.shape,
        ),
        (
            "reconstructed_teacher_input_ids",
            evidence.reconstructed_teacher_input_ids,
            teacher.shape,
        ),
        (
            "reconstructed_attention_mask",
            evidence.reconstructed_attention_mask,
            prompt.shape,
        ),
    ):
        array = np.asarray(value)
        if array.shape != expected_shape or not _is_integer_array(array):
            raise TerminalNumericalInferenceAuditError(f"{name} shape/dtype无效")
    source_bits = np.asarray(evidence.source_pixel_bf16_bits)
    reconstructed_bits = np.asarray(evidence.reconstructed_pixel_bf16_bits)
    if (
        source_bits.dtype != np.uint16
        or reconstructed_bits.dtype != np.uint16
        or source_bits.shape != reconstructed_bits.shape
        or source_bits.ndim < 4
        or source_bits.shape[:2] != (variants, paths)
        or source_bits.size == 0
    ):
        raise TerminalNumericalInferenceAuditError("BF16 bit evidence无效")


def _all_repeats_equal(array: NDArray[object]) -> bool:
    values = np.asarray(array)
    return all(
        np.array_equal(values[0], values[index])
        for index in range(1, values.shape[0])
    )


def _all_repeats_match_source(
    source: NDArray[object], replay: NDArray[object]
) -> bool:
    values = np.asarray(replay)
    return all(
        np.array_equal(source, values[index])
        for index in range(values.shape[0])
    )


def _original_mismatches(
    evidence: NumericalFidelityEvidence,
) -> tuple[OriginalMismatchRecord, ...]:
    action_dim = np.asarray(evidence.source_generated_token_ids).shape[2]
    records: list[OriginalMismatchRecord] = []
    for variant_index, variant in enumerate(evidence.variant_names):
        clean_tokens = np.asarray(evidence.source_teacher_input_ids)[
            variant_index, -action_dim:
        ]
        for path_index, path in enumerate(evidence.path_names):
            generated_tokens = np.asarray(evidence.source_generated_token_ids)[
                variant_index, path_index
            ]
            generated_classes = np.asarray(evidence.source_generated_classes)[
                variant_index, path_index
            ]
            teacher_logits = np.asarray(evidence.source_teacher_logits)[
                variant_index, path_index
            ]
            for token_index in range(action_dim):
                if not np.array_equal(
                    generated_tokens[:token_index],
                    clean_tokens[:token_index],
                ):
                    break
                row = teacher_logits[token_index]
                argmax_classes = tuple(
                    int(index) for index in np.flatnonzero(row == np.max(row))
                )
                generation_class = int(generated_classes[token_index])
                if generation_class not in argmax_classes:
                    records.append(
                        OriginalMismatchRecord(
                            variant=variant,
                            path=path,
                            token_index=token_index,
                            generation_class=generation_class,
                            teacher_argmax_classes=argmax_classes,
                        )
                    )
    return tuple(records)


def evaluate_numerical_fidelity(
    evidence: NumericalFidelityEvidence,
) -> NumericalFidelityDecision:
    """逐输入检查bit reconstruction、三次repeat和原NPZ逐位重放。"""

    _validate_fidelity_evidence(evidence)
    per_input: list[FidelityInputDecision] = []
    for variant_index, variant in enumerate(evidence.variant_names):
        prompt_exact = np.array_equal(
            evidence.source_prompt_input_ids[variant_index],
            evidence.reconstructed_prompt_input_ids[variant_index],
        )
        teacher_exact = np.array_equal(
            evidence.source_teacher_input_ids[variant_index],
            evidence.reconstructed_teacher_input_ids[variant_index],
        )
        attention_is_ones = bool(
            np.all(
                np.asarray(evidence.reconstructed_attention_mask)[
                    variant_index
                ]
                == 1
            )
        )
        for path_index, path in enumerate(evidence.path_names):
            input_exact = bool(
                prompt_exact
                and teacher_exact
                and attention_is_ones
                and np.array_equal(
                    evidence.source_pixel_bf16_bits[variant_index, path_index],
                    evidence.reconstructed_pixel_bf16_bits[
                        variant_index, path_index
                    ],
                )
            )
            replay_tokens = evidence.replay_generated_token_ids[
                variant_index, path_index
            ]
            replay_generation = evidence.replay_generation_logits[
                variant_index, path_index
            ]
            replay_teacher = evidence.replay_teacher_logits[
                variant_index, path_index
            ]
            generation_repeat_exact = bool(
                _all_repeats_equal(replay_tokens)
                and _all_repeats_equal(replay_generation)
            )
            teacher_repeat_exact = _all_repeats_equal(replay_teacher)
            generation_tokens_match = _all_repeats_match_source(
                evidence.source_generated_token_ids[variant_index, path_index],
                replay_tokens,
            )
            generation_logits_match = _all_repeats_match_source(
                evidence.source_generation_logits[variant_index, path_index],
                replay_generation,
            )
            teacher_matches = _all_repeats_match_source(
                evidence.source_teacher_logits[variant_index, path_index],
                replay_teacher,
            )
            reconstruction_failure = not input_exact
            nondeterministic = not (
                generation_repeat_exact and teacher_repeat_exact
            )
            provenance_failure = bool(
                not nondeterministic
                and not (
                    generation_tokens_match
                    and generation_logits_match
                    and teacher_matches
                )
            )
            failure_kinds = sum(
                (reconstruction_failure, nondeterministic, provenance_failure)
            )
            if failure_kinds == 0:
                status: FidelityStatus = "pass"
            elif failure_kinds > 1:
                status = "mixed_replay_failure"
            elif reconstruction_failure:
                status = "input_reconstruction_invalid"
            elif nondeterministic:
                status = "runtime_nondeterministic"
            else:
                status = "replay_provenance_unresolved"
            per_input.append(
                FidelityInputDecision(
                    variant=variant,
                    path=path,
                    fidelity_pass=status == "pass",
                    status=status,
                    input_reconstruction_exact=input_exact,
                    generation_repeat_exact=generation_repeat_exact,
                    teacher_repeat_exact=teacher_repeat_exact,
                    generation_tokens_match_source=generation_tokens_match,
                    generation_logits_match_source=generation_logits_match,
                    teacher_matches_source=teacher_matches,
                )
            )
    fidelity_pass = all(item.fidelity_pass for item in per_input)
    return NumericalFidelityDecision(
        fidelity_pass=fidelity_pass,
        causal_attribution_allowed=fidelity_pass,
        per_input=tuple(per_input),
        original_mismatches=_original_mismatches(evidence),
    )


def _validate_attribution_evidence(
    fidelity: NumericalFidelityEvidence,
    evidence: NumericalAttributionEvidence,
) -> None:
    if evidence.variant_names != fidelity.variant_names:
        raise TerminalNumericalInferenceAuditError("attribution variant漂移")
    if evidence.path_names != fidelity.path_names:
        raise TerminalNumericalInferenceAuditError("attribution path漂移")
    if evidence.repeat_count != fidelity.repeat_count:
        raise TerminalNumericalInferenceAuditError("attribution repeat漂移")
    expected_ids = np.asarray(fidelity.replay_generated_token_ids).shape
    expected_logits = np.asarray(fidelity.replay_generation_logits).shape
    if (
        np.asarray(evidence.no_cache_generated_token_ids).shape != expected_ids
        or not _is_integer_array(evidence.no_cache_generated_token_ids)
    ):
        raise TerminalNumericalInferenceAuditError("no-cache generation IDs无效")
    for name, value in (
        ("no_cache_generation_logits", evidence.no_cache_generation_logits),
        (
            "generation_prefix_no_cache_logits",
            evidence.generation_prefix_no_cache_logits,
        ),
        ("clean_prefix_no_cache_logits", evidence.clean_prefix_no_cache_logits),
        ("full_teacher_no_cache_logits", evidence.full_teacher_no_cache_logits),
    ):
        if np.asarray(value).shape != expected_logits or not (
            _is_finite_float_array(value)
        ):
            raise TerminalNumericalInferenceAuditError(f"{name} shape/dtype无效")


def _top_two(row: NDArray[np.floating]) -> tuple[int, float, float, float]:
    values = np.asarray(row, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    first = int(order[0])
    top1 = float(values[first])
    top2 = float(values[int(order[1])])
    return first, top1, top2, top1 - top2


def _compare_logits(
    name: str,
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    causal_prefix_comparable: tuple[bool, ...] | None = None,
) -> LogitComparisonDecision:
    comparable = (
        (True,) * int(np.asarray(left).shape[0])
        if causal_prefix_comparable is None
        else causal_prefix_comparable
    )
    if len(comparable) != int(np.asarray(left).shape[0]):
        raise TerminalNumericalInferenceAuditError(
            "causal prefix comparable数量与action_dim不一致"
        )
    tokens: list[TokenLogitComparison] = []
    for token_index, (left_row, right_row) in enumerate(zip(left, right)):
        difference = np.abs(
            np.asarray(left_row, dtype=np.float64)
            - np.asarray(right_row, dtype=np.float64)
        )
        linf = float(np.max(difference))
        left_argmax = tuple(
            int(index)
            for index in np.flatnonzero(left_row == np.max(left_row))
        )
        right_argmax = tuple(
            int(index)
            for index in np.flatnonzero(right_row == np.max(right_row))
        )
        left_top = _top_two(left_row)
        right_top = _top_two(right_row)
        tokens.append(
            TokenLogitComparison(
                token_index=token_index,
                linf=linf,
                max_abs_difference_classes=tuple(
                    int(index)
                    for index in np.flatnonzero(difference == linf)
                ),
                left_argmax_classes=left_argmax,
                right_argmax_classes=right_argmax,
                argmax_equal=left_argmax == right_argmax,
                causal_prefix_comparable=bool(comparable[token_index]),
                left_top1_class=left_top[0],
                left_top1_logit=left_top[1],
                left_top2_logit=left_top[2],
                left_top2_margin=left_top[3],
                right_top1_class=right_top[0],
                right_top1_logit=right_top[1],
                right_top2_logit=right_top[2],
                right_top2_margin=right_top[3],
            )
        )
    return LogitComparisonDecision(name=name, tokens=tuple(tokens))


def _first_divergence(left: IntArray, right: IntArray) -> int | None:
    indices = np.flatnonzero(np.asarray(left) != np.asarray(right))
    return None if indices.size == 0 else int(indices[0])


def evaluate_numerical_attribution(
    *,
    fidelity: NumericalFidelityEvidence,
    evidence: NumericalAttributionEvidence,
) -> NumericalAttributionDecision:
    """在 Fidelity 通过后复算双 prefix、teacher和完整生成对照。"""

    fidelity_decision = evaluate_numerical_fidelity(fidelity)
    _validate_attribution_evidence(fidelity, evidence)
    per_input: list[AttributionInputDecision] = []
    for variant_index, variant in enumerate(evidence.variant_names):
        for path_index, path in enumerate(evidence.path_names):
            no_cache_ids = evidence.no_cache_generated_token_ids[
                variant_index, path_index
            ]
            no_cache_logits = evidence.no_cache_generation_logits[
                variant_index, path_index
            ]
            generation_prefix = evidence.generation_prefix_no_cache_logits[
                variant_index, path_index
            ]
            clean_prefix = evidence.clean_prefix_no_cache_logits[
                variant_index, path_index
            ]
            full_teacher_no_cache = evidence.full_teacher_no_cache_logits[
                variant_index, path_index
            ]
            stability = (
                _all_repeats_equal(no_cache_ids)
                and _all_repeats_equal(no_cache_logits),
                _all_repeats_equal(generation_prefix),
                _all_repeats_equal(clean_prefix),
                _all_repeats_equal(full_teacher_no_cache),
            )
            cached_ids = fidelity.replay_generated_token_ids[
                variant_index, path_index, 0
            ]
            cached_logits = fidelity.replay_generation_logits[
                variant_index, path_index, 0
            ]
            original_teacher = fidelity.replay_teacher_logits[
                variant_index, path_index, 0
            ]
            comparisons = (
                _compare_logits(
                    "cached_vs_no_cache_generation",
                    cached_logits,
                    no_cache_logits[0],
                    causal_prefix_comparable=tuple(
                        bool(
                            np.array_equal(
                                cached_ids[:token_index],
                                no_cache_ids[0, :token_index],
                            )
                        )
                        for token_index in range(cached_ids.size)
                    ),
                ),
                _compare_logits(
                    "cached_vs_generation_prefix_no_cache",
                    cached_logits,
                    generation_prefix[0],
                ),
                _compare_logits(
                    "original_teacher_vs_full_teacher_no_cache",
                    original_teacher,
                    full_teacher_no_cache[0],
                ),
                _compare_logits(
                    "full_teacher_no_cache_vs_clean_prefix_no_cache",
                    full_teacher_no_cache[0],
                    clean_prefix[0],
                ),
            )
            per_input.append(
                AttributionInputDecision(
                    variant=variant,
                    path=path,
                    paths_stable=all(stability),
                    no_cache_generation_repeat_exact=stability[0],
                    generation_prefix_repeat_exact=stability[1],
                    clean_prefix_repeat_exact=stability[2],
                    full_teacher_no_cache_repeat_exact=stability[3],
                    cached_vs_no_cache_first_divergence_index=(
                        _first_divergence(cached_ids, no_cache_ids[0])
                    ),
                    comparisons=comparisons,
                )
            )
    attribution_valid = bool(
        fidelity_decision.fidelity_pass
        and all(item.paths_stable for item in per_input)
    )
    return NumericalAttributionDecision(
        attribution_valid=attribution_valid,
        causal_attribution_allowed=attribution_valid,
        per_input=tuple(per_input),
    )


def _write_npz(
    path: Path,
    *,
    schema_version: str,
    fields: dict[str, object],
) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(schema_version),
        **fields,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def write_numerical_fidelity_npz(
    evidence: NumericalFidelityEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存 pickle-free Fidelity artifact 并返回文件 SHA-256。"""

    _validate_fidelity_evidence(evidence)
    return _write_npz(
        Path(output_path),
        schema_version=NUMERICAL_FIDELITY_SCHEMA_VERSION,
        fields={
            name: np.asarray(value)
            for name, value in evidence.__dict__.items()
        },
    )


def write_numerical_attribution_npz(
    evidence: NumericalAttributionEvidence,
    *,
    output_path: str | Path,
) -> str:
    """保存 pickle-free attribution artifact 并返回文件 SHA-256。"""

    dummy_fidelity_shape = np.asarray(evidence.no_cache_generation_logits).shape
    if dummy_fidelity_shape[2] != evidence.repeat_count:
        raise TerminalNumericalInferenceAuditError("attribution repeat shape无效")
    return _write_npz(
        Path(output_path),
        schema_version=NUMERICAL_ATTRIBUTION_SCHEMA_VERSION,
        fields={
            name: np.asarray(value)
            for name, value in evidence.__dict__.items()
        },
    )


def _load_scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    if value.shape != ():
        raise TerminalNumericalInferenceAuditError(f"{name}必须为scalar")
    return value.item()


def load_numerical_fidelity_npz(
    path: str | Path,
) -> NumericalFidelityEvidence:
    """无 pickle 加载并验证 Fidelity artifact。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(_load_scalar(archive, "schema_version")) != (
            NUMERICAL_FIDELITY_SCHEMA_VERSION
        ):
            raise TerminalNumericalInferenceAuditError("Fidelity schema不匹配")
        evidence = NumericalFidelityEvidence(
            variant_names=tuple(str(value) for value in archive["variant_names"]),
            path_names=tuple(str(value) for value in archive["path_names"]),
            repeat_count=int(_load_scalar(archive, "repeat_count")),
            action_token_start=int(_load_scalar(archive, "action_token_start")),
            source_prompt_input_ids=archive["source_prompt_input_ids"].copy(),
            source_teacher_input_ids=archive["source_teacher_input_ids"].copy(),
            reconstructed_prompt_input_ids=archive[
                "reconstructed_prompt_input_ids"
            ].copy(),
            reconstructed_teacher_input_ids=archive[
                "reconstructed_teacher_input_ids"
            ].copy(),
            reconstructed_attention_mask=archive[
                "reconstructed_attention_mask"
            ].copy(),
            source_pixel_bf16_bits=archive["source_pixel_bf16_bits"].copy(),
            reconstructed_pixel_bf16_bits=archive[
                "reconstructed_pixel_bf16_bits"
            ].copy(),
            source_generated_token_ids=archive[
                "source_generated_token_ids"
            ].copy(),
            source_generated_classes=archive[
                "source_generated_classes"
            ].copy(),
            source_generation_logits=archive["source_generation_logits"].copy(),
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
        )
    _validate_fidelity_evidence(evidence)
    return evidence


def load_numerical_attribution_npz(
    path: str | Path,
) -> NumericalAttributionEvidence:
    """无 pickle 加载 attribution artifact。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(_load_scalar(archive, "schema_version")) != (
            NUMERICAL_ATTRIBUTION_SCHEMA_VERSION
        ):
            raise TerminalNumericalInferenceAuditError(
                "Attribution schema不匹配"
            )
        evidence = NumericalAttributionEvidence(
            variant_names=tuple(str(value) for value in archive["variant_names"]),
            path_names=tuple(str(value) for value in archive["path_names"]),
            repeat_count=int(_load_scalar(archive, "repeat_count")),
            no_cache_generated_token_ids=archive[
                "no_cache_generated_token_ids"
            ].copy(),
            no_cache_generation_logits=archive[
                "no_cache_generation_logits"
            ].copy(),
            generation_prefix_no_cache_logits=archive[
                "generation_prefix_no_cache_logits"
            ].copy(),
            clean_prefix_no_cache_logits=archive[
                "clean_prefix_no_cache_logits"
            ].copy(),
            full_teacher_no_cache_logits=archive[
                "full_teacher_no_cache_logits"
            ].copy(),
        )
    return evidence


def numerical_decision_payload(value: object) -> object:
    """把decision dataclass规范化成可逐值比较的JSON对象。"""

    return json.loads(
        json.dumps(asdict(value), sort_keys=True, allow_nan=False)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TerminalNumericalInferenceAuditError("artifact相对路径无效")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise TerminalNumericalInferenceAuditError("artifact路径必须为相对路径")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise TerminalNumericalInferenceAuditError("artifact路径逃逸bundle")
    return resolved


def _artifact_path_and_hash(
    root: Path,
    raw: object,
    *,
    name: str,
) -> tuple[Path, str]:
    if not isinstance(raw, Mapping):
        raise TerminalNumericalInferenceAuditError(f"{name} artifact记录无效")
    path = _resolve_artifact(root, raw.get("relative_path"))
    expected_sha = raw.get("sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise TerminalNumericalInferenceAuditError(f"{name} SHA-256无效")
    if not path.is_file():
        raise TerminalNumericalInferenceAuditError(f"{name} artifact不存在")
    if _file_sha256(path) != expected_sha:
        raise TerminalNumericalInferenceAuditError(f"{name} artifact SHA漂移")
    return path, expected_sha


def evaluate_numerical_inference_bundle(
    manifest_path: str | Path,
) -> NumericalInferenceBundleDecision:
    """按SHA重载NPZ并独立复算 Fidelity 与 attribution decision。"""

    path = Path(manifest_path)
    failures: list[str] = []
    fidelity_decision: NumericalFidelityDecision | None = None
    attribution_decision: NumericalAttributionDecision | None = None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TerminalNumericalInferenceAuditError("manifest根必须为对象")
        if manifest.get("schema_version") != (
            NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION
        ):
            raise TerminalNumericalInferenceAuditError("bundle schema不匹配")
        if manifest.get("status") != "diagnostic_complete":
            raise TerminalNumericalInferenceAuditError("bundle状态不是complete")
        if manifest.get("gate6g_contract_status") != "invalid":
            raise TerminalNumericalInferenceAuditError(
                "numerical audit必须保留原Gate 6g invalid状态"
            )
        if manifest.get("repeat_count") != 3:
            raise TerminalNumericalInferenceAuditError("bundle repeat_count漂移")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TerminalNumericalInferenceAuditError("bundle artifacts缺失")
        fidelity_path, _ = _artifact_path_and_hash(
            path.parent,
            artifacts.get("fidelity"),
            name="fidelity",
        )
        fidelity = load_numerical_fidelity_npz(fidelity_path)
        fidelity_decision = evaluate_numerical_fidelity(fidelity)
        if manifest.get("fidelity_decision") != numerical_decision_payload(
            fidelity_decision
        ):
            raise TerminalNumericalInferenceAuditError(
                "manifest Fidelity decision不能从NPZ逐值复算"
            )
        attribution_record = artifacts.get("attribution")
        if fidelity_decision.fidelity_pass:
            attribution_path, _ = _artifact_path_and_hash(
                path.parent,
                attribution_record,
                name="attribution",
            )
            attribution = load_numerical_attribution_npz(attribution_path)
            attribution_decision = evaluate_numerical_attribution(
                fidelity=fidelity,
                evidence=attribution,
            )
            if manifest.get("attribution_decision") != (
                numerical_decision_payload(attribution_decision)
            ):
                raise TerminalNumericalInferenceAuditError(
                    "manifest attribution decision不能从NPZ逐值复算"
                )
        else:
            if attribution_record is not None:
                raise TerminalNumericalInferenceAuditError(
                    "Fidelity失败时不得发布attribution artifact"
                )
            if manifest.get("attribution_decision") is not None:
                raise TerminalNumericalInferenceAuditError(
                    "Fidelity失败时attribution decision必须为null"
                )
        expected_allowed = bool(
            attribution_decision is not None
            and attribution_decision.causal_attribution_allowed
        )
        if manifest.get("causal_attribution_allowed") is not expected_allowed:
            raise TerminalNumericalInferenceAuditError(
                "run-level causal attribution状态不能独立复算"
            )
        source = manifest.get("source_invalid_bundle")
        if not isinstance(source, dict):
            raise TerminalNumericalInferenceAuditError("source bundle绑定缺失")
        for name in ("audit_failed_sha256", *NUMERICAL_VARIANTS):
            value = source.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TerminalNumericalInferenceAuditError(
                    f"source bundle {name} SHA-256无效"
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(str(error))
    return NumericalInferenceBundleDecision(
        bundle_valid=not failures,
        failures=tuple(failures),
        fidelity_decision=fidelity_decision,
        attribution_decision=attribution_decision,
    )


def publish_numerical_inference_manifest(
    manifest: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    """候选manifest通过独立CPU复核后原子发布，并返回SHA-256。"""

    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.candidate")
    if candidate.exists():
        raise FileExistsError(candidate)
    candidate.write_text(
        json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        decision = evaluate_numerical_inference_bundle(candidate)
        if not decision.bundle_valid:
            raise TerminalNumericalInferenceAuditError(
                "numerical bundle独立复核失败: " + "; ".join(decision.failures)
            )
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)
    return _file_sha256(path)
