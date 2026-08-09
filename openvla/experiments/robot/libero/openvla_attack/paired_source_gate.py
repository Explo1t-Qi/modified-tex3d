"""正式 source rollout 的成对 Clean/Adversarial 科学判定。

本模块只处理已经完成的 rollout 结果，不导入模型、LIBERO、renderer 或 CUDA。
每个 held-out initial state 必须在 clean asset 与最终 bake texture 下各参与一次；
Gate 只统计 ``clean success -> adversarial failure``，因此 clean 原有失败不会被
错误解释为纹理造成的失败。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

from .seed_score_audit import file_sha256


PAIRED_SOURCE_GATE_SCHEMA_VERSION: Final[str] = (
    "openvla-fixed-support-paired-source-gate-v1"
)
EXPECTED_EVAL_STATE_IDS: Final[tuple[int, ...]] = tuple(range(10, 20))
MIN_ATTACK_INDUCED_FAILURES: Final[int] = 3


class PairedSourceGateError(ValueError):
    """成对 rollout 输入不满足冻结状态或唯一性契约。"""


@dataclass(frozen=True)
class StateRolloutOutcome:
    """一个 asset 条件下某个 held-out state 的 rollout 结果。"""

    state_id: int
    initial_state_sha256: str
    success: bool


@dataclass(frozen=True)
class PairedStateOutcome:
    """同一 initial state 的 clean/adversarial 配对结果。"""

    state_id: int
    initial_state_sha256: str
    clean_success: bool
    adversarial_success: bool
    attack_induced_failure: bool
    preexisting_clean_failure: bool
    adversarial_recovery: bool


@dataclass(frozen=True)
class PairedSourceGateDecision:
    """Gate 6e 的可复核汇总。"""

    pairs: tuple[PairedStateOutcome, ...]
    clean_successes: int
    adversarial_successes: int
    adversarial_failures: int
    attack_induced_failures: int
    preexisting_clean_failures: int
    adversarial_recoveries: int
    clean_control_perfect: bool
    gate_pass: bool


@dataclass(frozen=True)
class PairedSourceArtifactDecision:
    """落盘成对Gate artifact的独立复核结果。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PairedSourceGateError(f"{field_name}必须是64位小写SHA-256")


def _validate_git_sha(value: str, *, field_name: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PairedSourceGateError(f"{field_name}必须是40位小写Git SHA")


def _index_outcomes(
    outcomes: Sequence[StateRolloutOutcome],
    *,
    condition: str,
) -> dict[int, StateRolloutOutcome]:
    if tuple(outcome.state_id for outcome in outcomes) != EXPECTED_EVAL_STATE_IDS:
        raise PairedSourceGateError(
            f"{condition} rollout必须按10-19完整、唯一且各参与一次"
        )
    indexed = {outcome.state_id: outcome for outcome in outcomes}
    if len(indexed) != len(EXPECTED_EVAL_STATE_IDS):
        raise PairedSourceGateError(f"{condition} rollout包含重复state ID")
    for outcome in outcomes:
        _validate_sha256(
            outcome.initial_state_sha256,
            field_name=f"{condition} state {outcome.state_id} fingerprint",
        )
    if len({outcome.initial_state_sha256 for outcome in outcomes}) != len(
        EXPECTED_EVAL_STATE_IDS
    ):
        raise PairedSourceGateError(f"{condition} state fingerprint必须唯一")
    return indexed


def evaluate_paired_source_gate(
    clean: Sequence[StateRolloutOutcome],
    adversarial: Sequence[StateRolloutOutcome],
) -> PairedSourceGateDecision:
    """配对同 states 10--19，并只用新增失败判断 source gate。"""

    clean_by_state = _index_outcomes(clean, condition="clean")
    adversarial_by_state = _index_outcomes(adversarial, condition="adversarial")
    pairs: list[PairedStateOutcome] = []
    for state_id in EXPECTED_EVAL_STATE_IDS:
        clean_outcome = clean_by_state[state_id]
        adversarial_outcome = adversarial_by_state[state_id]
        if (
            clean_outcome.initial_state_sha256
            != adversarial_outcome.initial_state_sha256
        ):
            raise PairedSourceGateError(
                f"state {state_id} clean/adversarial fingerprint不一致"
            )
        attack_induced_failure = (
            clean_outcome.success and not adversarial_outcome.success
        )
        preexisting_clean_failure = not clean_outcome.success
        adversarial_recovery = (
            not clean_outcome.success and adversarial_outcome.success
        )
        pairs.append(
            PairedStateOutcome(
                state_id=state_id,
                initial_state_sha256=clean_outcome.initial_state_sha256,
                clean_success=clean_outcome.success,
                adversarial_success=adversarial_outcome.success,
                attack_induced_failure=attack_induced_failure,
                preexisting_clean_failure=preexisting_clean_failure,
                adversarial_recovery=adversarial_recovery,
            )
        )

    clean_successes = sum(pair.clean_success for pair in pairs)
    adversarial_successes = sum(pair.adversarial_success for pair in pairs)
    attack_induced_failures = sum(
        pair.attack_induced_failure for pair in pairs
    )
    preexisting_clean_failures = sum(
        pair.preexisting_clean_failure for pair in pairs
    )
    adversarial_recoveries = sum(pair.adversarial_recovery for pair in pairs)
    return PairedSourceGateDecision(
        pairs=tuple(pairs),
        clean_successes=clean_successes,
        adversarial_successes=adversarial_successes,
        adversarial_failures=len(pairs) - adversarial_successes,
        attack_induced_failures=attack_induced_failures,
        preexisting_clean_failures=preexisting_clean_failures,
        adversarial_recoveries=adversarial_recoveries,
        clean_control_perfect=clean_successes == len(pairs),
        gate_pass=attack_induced_failures >= MIN_ATTACK_INDUCED_FAILURES,
    )


def write_paired_source_gate_artifact(
    path: str | Path,
    *,
    decision: PairedSourceGateDecision,
    training_manifest_sha256: str,
    baked_texture_sha256: str,
    evaluation_code_commit: str,
) -> Path:
    """以拒绝覆盖方式保存成对结果及其正式训练 provenance。"""

    _validate_sha256(
        training_manifest_sha256,
        field_name="training_manifest_sha256",
    )
    _validate_sha256(
        baked_texture_sha256,
        field_name="baked_texture_sha256",
    )
    _validate_git_sha(
        evaluation_code_commit,
        field_name="evaluation_code_commit",
    )
    resolved = Path(path)
    payload = {
        "schema_version": PAIRED_SOURCE_GATE_SCHEMA_VERSION,
        "training_manifest_sha256": training_manifest_sha256,
        "baked_texture_sha256": baked_texture_sha256,
        "evaluation_code_commit": evaluation_code_commit,
        "eval_state_ids": list(EXPECTED_EVAL_STATE_IDS),
        "minimum_attack_induced_failures": MIN_ATTACK_INDUCED_FAILURES,
        **{
            key: value
            for key, value in asdict(decision).items()
            if key != "pairs"
        },
        "pairs": [asdict(pair) for pair in decision.pairs],
    }
    with resolved.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def evaluate_paired_source_gate_artifact(
    path: str | Path,
) -> PairedSourceArtifactDecision:
    """从pairs重算所有计数与Gate，拒绝信任manifest自报结果。"""

    failures: list[str] = []
    try:
        resolved_path = Path(path).resolve()
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        raw_pairs = payload["pairs"]
        if not isinstance(raw_pairs, list) or len(raw_pairs) != 10:
            raise PairedSourceGateError("pairs必须恰好包含10行")
        if any(
            type(row.get(name)) is not bool
            for row in raw_pairs
            for name in ("clean_success", "adversarial_success")
        ):
            raise PairedSourceGateError("paired success字段必须是JSON boolean")
        clean = tuple(
            StateRolloutOutcome(
                state_id=int(row["state_id"]),
                initial_state_sha256=str(row["initial_state_sha256"]),
                success=bool(row["clean_success"]),
            )
            for row in raw_pairs
        )
        adversarial = tuple(
            StateRolloutOutcome(
                state_id=int(row["state_id"]),
                initial_state_sha256=str(row["initial_state_sha256"]),
                success=bool(row["adversarial_success"]),
            )
            for row in raw_pairs
        )
        recomputed = evaluate_paired_source_gate(clean, adversarial)
        training_manifest_path = (
            resolved_path.parent / "formal_source_training_manifest.json"
        )
        training_manifest = json.loads(
            training_manifest_path.read_text(encoding="utf-8")
        )
        baked_path = resolved_path.parent / str(
            training_manifest["baked_texture_relative_path"]
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return PairedSourceArtifactDecision(
            False,
            (f"paired source gate artifact无法加载: {error}",),
        )
    if payload.get("schema_version") != PAIRED_SOURCE_GATE_SCHEMA_VERSION:
        failures.append("paired source gate schema不匹配")
    if payload.get("eval_state_ids") != list(EXPECTED_EVAL_STATE_IDS):
        failures.append("paired artifact未固定states 10-19")
    if payload.get("minimum_attack_induced_failures") != (
        MIN_ATTACK_INDUCED_FAILURES
    ):
        failures.append("paired artifact门槛不是3个新增失败")
    for name in ("training_manifest_sha256", "baked_texture_sha256"):
        try:
            _validate_sha256(str(payload.get(name)), field_name=name)
        except PairedSourceGateError as error:
            failures.append(str(error))
    try:
        _validate_git_sha(
            str(payload.get("evaluation_code_commit")),
            field_name="evaluation_code_commit",
        )
    except PairedSourceGateError as error:
        failures.append(str(error))
    if file_sha256(training_manifest_path) != payload.get(
        "training_manifest_sha256"
    ):
        failures.append("paired artifact未绑定同目录正式training manifest")
    if file_sha256(baked_path) != payload.get("baked_texture_sha256"):
        failures.append("paired artifact未绑定正式训练bake texture")
    recomputed_payload = asdict(recomputed)
    for name, expected in recomputed_payload.items():
        if name == "pairs":
            expected_pairs = [asdict(pair) for pair in recomputed.pairs]
            if raw_pairs != expected_pairs:
                failures.append("逐state派生标签不可由clean/adversarial重算")
        elif payload.get(name) != expected:
            failures.append(f"{name}不可由逐state pairs重算")
    return PairedSourceArtifactDecision(not failures, tuple(failures))


def state_fingerprint(value: object) -> str:
    """从 initial-state array 的 dtype、shape 和原始字节生成稳定 fingerprint。"""

    # 延迟导入使成对判定模块在没有 NumPy 的轻量文档工具中仍可被解析。
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()
