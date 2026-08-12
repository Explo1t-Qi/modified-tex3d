"""重放 Gate 6g 失败 bundle 的 OpenVLA numerical inference 路径。

命令只加载 source OpenVLA checkpoint 和已有 state 0 NPZ，不导入 LIBERO、
renderer，不修改 Active Texture，也不执行训练、反向传播或 rollout。第一阶段从
保存的 ``uint16`` bits 恢复所有 C/A/B 的最终 BF16 processor tensor，并各重复
三次原始 default generation 与 teacher forward。只有全部输入都逐位复现原 NPZ
时，才执行完整 no-cache generation、generation-prefix no-cache、clean-prefix
no-cache 和 full-teacher no-cache 对照。

无论 Fidelity 是否通过，成功结束都表示 numerical diagnostic artifact 完整，
不表示原 Gate 6g bundle 有效。原 Gate 状态始终保留为 ``invalid``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
OPENVLA_ROOT = THIS_FILE.parents[4]
ROBOT_EXPERIMENT_DIR = THIS_FILE.parents[2]
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
for import_path in (
    REPOSITORY_ROOT,
    OPENVLA_ROOT,
    ROBOT_EXPERIMENT_DIR,
    LIBERO_EXPERIMENT_DIR,
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.fixed_support_source_training import (  # noqa: E402
    loaded_legacy_optimizer_modules,
    verify_executing_commit,
)
from openvla_attack.terminal_deployment_response_audit import (  # noqa: E402
    TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION,
    TerminalDeploymentResponseEvidence,
    load_terminal_response_npz,
)
from openvla_attack.terminal_numerical_inference import (  # noqa: E402
    AttributionReplay,
    FidelityReplay,
    FidelitySample,
    capture_attribution_repeats,
    capture_fidelity_sample,
    tensor_from_bfloat16_bits,
)
from openvla_attack.terminal_numerical_inference_audit import (  # noqa: E402
    NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION,
    NUMERICAL_PATHS,
    NUMERICAL_VARIANTS,
    NumericalAttributionEvidence,
    NumericalAttributionDecision,
    NumericalFidelityEvidence,
    evaluate_numerical_attribution,
    evaluate_numerical_fidelity,
    numerical_decision_payload,
    publish_numerical_inference_manifest,
    write_numerical_attribution_npz,
    write_numerical_fidelity_npz,
)
from openvla_attack.terminal_openvla_response import (  # noqa: E402
    bfloat16_tensor_to_uint16_bits,
)


REPEAT_COUNT = 3


@dataclass(frozen=True)
class TerminalNumericalInferenceConfig:
    """失败 state 0 bundle 的只读 numerical replay CLI 配置。"""

    pretrained_checkpoint: str
    source_bundle_dir: str
    output_dir: str
    code_commit: str
    seed: int = 7
    unnorm_key: Optional[str] = "libero_spatial_no_noops"
    model_family: str = "openvla"
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> TerminalNumericalInferenceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--source_bundle_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    return TerminalNumericalInferenceConfig(**vars(parser.parse_args(argv)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fingerprints(checkpoint_path: Path) -> dict[str, str]:
    configuration_names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
    )
    result = {
        name: _file_sha256(checkpoint_path / name)
        for name in configuration_names
        if (checkpoint_path / name).is_file()
    }
    weights = sorted(
        tuple(checkpoint_path.glob("*.safetensors"))
        + tuple(checkpoint_path.glob("*.bin"))
    )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weights
    ).encode("utf-8")
    result["__weight_name_size_inventory_sha256__"] = hashlib.sha256(
        inventory
    ).hexdigest()
    return result


def _validate_config(cfg: TerminalNumericalInferenceConfig) -> None:
    if (
        cfg.task_suite_name != "libero_spatial"
        or cfg.task_id != 0
        or cfg.object_name != "akita_black_bowl"
        or not cfg.center_crop
    ):
        raise ValueError("numerical replay固定为LIBERO Spatial task 0/Akita")
    if cfg.load_in_8bit or cfg.load_in_4bit:
        raise ValueError("numerical replay必须使用原BF16 checkpoint")


def _load_source_bundle(
    bundle_dir: Path,
) -> tuple[
    dict[str, TerminalDeploymentResponseEvidence],
    dict[str, str],
]:
    failure_path = bundle_dir / "audit_failed.json"
    if not failure_path.is_file():
        raise FileNotFoundError(failure_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        failure.get("schema_version")
        != TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION
        or failure.get("status") != "audit_invalid"
        or failure.get("failed_stage") != "manifest_publication"
    ):
        raise RuntimeError("source目录不是目标Gate 6g manifest失败bundle")
    completed = {
        (item.get("variant"), item.get("state_id"))
        for item in failure.get("completed_keys", [])
        if isinstance(item, dict)
    }
    expected = {(variant, 0) for variant in NUMERICAL_VARIANTS}
    if completed != expected:
        raise RuntimeError("source失败bundle未完整保存两个state 0 NPZ")
    evidence: dict[str, TerminalDeploymentResponseEvidence] = {}
    hashes = {"audit_failed_sha256": _file_sha256(failure_path)}
    for variant in NUMERICAL_VARIANTS:
        npz_path = bundle_dir / "arrays" / variant / "state_00.npz"
        loaded = load_terminal_response_npz(npz_path)
        if loaded.variant != variant or loaded.state_id != 0:
            raise RuntimeError(f"source NPZ身份漂移: {variant}")
        evidence[variant] = loaded
        hashes[variant] = _file_sha256(npz_path)
    reference = evidence[NUMERICAL_VARIANTS[0]]
    candidate = evidence[NUMERICAL_VARIANTS[1]]
    for name in (
        "prompt_input_ids",
        "teacher_input_ids",
        "action_token_start",
        "action_token_end",
        "vocab_size",
    ):
        if not np.array_equal(
            np.asarray(getattr(reference, name)),
            np.asarray(getattr(candidate, name)),
        ):
            raise RuntimeError(f"跨variant共享模型输入漂移: {name}")
    for name in (
        "official_processor_bf16_bits",
        "generation_logits",
        "teacher_logits",
        "generated_token_ids",
        "generated_classes",
    ):
        if not np.array_equal(
            np.asarray(getattr(reference, name))[0],
            np.asarray(getattr(candidate, name))[0],
        ):
            raise RuntimeError(f"跨variant共享Clean漂移: {name}")
    return evidence, hashes


def _selected_pixel_bits(
    evidence: TerminalDeploymentResponseEvidence,
) -> np.ndarray:
    """按原采集语义选择C=official、A=training、B=official的BF16 bits。"""

    official = np.asarray(evidence.official_processor_bf16_bits)
    training = np.asarray(evidence.training_exact_processor_bf16_bits)
    selected = official.copy()
    selected[1] = training[1]
    return selected


def _runtime_model_config(model: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for prefix, owner in (
        ("model", model),
        ("language_model", getattr(model, "language_model", None)),
    ):
        config = getattr(owner, "config", None)
        if config is None:
            continue
        for name in ("use_cache", "_attn_implementation", "torch_dtype"):
            value = getattr(config, name, None)
            result[f"{prefix}.{name}"] = None if value is None else str(value)
    return result


def _stack_fidelity(
    source: dict[str, TerminalDeploymentResponseEvidence],
    replay: list[list[FidelityReplay]],
    reconstructed_bits: np.ndarray,
    reconstructed_prompt_ids: np.ndarray,
    reconstructed_teacher_ids: np.ndarray,
    reconstructed_attention_mask: np.ndarray,
) -> NumericalFidelityEvidence:
    source_evidence = [source[variant] for variant in NUMERICAL_VARIANTS]
    source_bits = np.stack(
        tuple(_selected_pixel_bits(item) for item in source_evidence), axis=0
    )
    return NumericalFidelityEvidence(
        variant_names=NUMERICAL_VARIANTS,
        path_names=NUMERICAL_PATHS,
        repeat_count=REPEAT_COUNT,
        action_token_start=int(source_evidence[0].action_token_start),
        source_prompt_input_ids=np.stack(
            tuple(item.prompt_input_ids for item in source_evidence), axis=0
        ),
        source_teacher_input_ids=np.stack(
            tuple(item.teacher_input_ids for item in source_evidence), axis=0
        ),
        reconstructed_prompt_input_ids=reconstructed_prompt_ids,
        reconstructed_teacher_input_ids=reconstructed_teacher_ids,
        reconstructed_attention_mask=reconstructed_attention_mask,
        source_pixel_bf16_bits=source_bits,
        reconstructed_pixel_bf16_bits=reconstructed_bits,
        source_generated_token_ids=np.stack(
            tuple(item.generated_token_ids for item in source_evidence), axis=0
        ),
        source_generated_classes=np.stack(
            tuple(item.generated_classes for item in source_evidence), axis=0
        ),
        source_generation_logits=np.stack(
            tuple(item.generation_logits for item in source_evidence), axis=0
        ),
        source_teacher_logits=np.stack(
            tuple(item.teacher_logits for item in source_evidence), axis=0
        ),
        replay_generated_token_ids=np.stack(
            tuple(
                np.stack(
                    tuple(item.generated_token_ids for item in variant), axis=0
                )
                for variant in replay
            ),
            axis=0,
        ),
        replay_generated_classes=np.stack(
            tuple(
                np.stack(
                    tuple(item.generated_classes for item in variant), axis=0
                )
                for variant in replay
            ),
            axis=0,
        ),
        replay_generation_logits=np.stack(
            tuple(
                np.stack(
                    tuple(item.generation_logits for item in variant), axis=0
                )
                for variant in replay
            ),
            axis=0,
        ),
        replay_teacher_logits=np.stack(
            tuple(
                np.stack(
                    tuple(item.teacher_logits for item in variant), axis=0
                )
                for variant in replay
            ),
            axis=0,
        ),
    )


def _stack_attribution(
    replay: list[list[AttributionReplay]],
) -> NumericalAttributionEvidence:
    def stack_field(name: str) -> np.ndarray:
        return np.stack(
            tuple(
                np.stack(
                    tuple(np.asarray(getattr(item, name)) for item in variant),
                    axis=0,
                )
                for variant in replay
            ),
            axis=0,
        )

    return NumericalAttributionEvidence(
        variant_names=NUMERICAL_VARIANTS,
        path_names=NUMERICAL_PATHS,
        repeat_count=REPEAT_COUNT,
        no_cache_generated_token_ids=stack_field(
            "no_cache_generated_token_ids"
        ),
        no_cache_generation_logits=stack_field(
            "no_cache_generation_logits"
        ),
        generation_prefix_no_cache_logits=stack_field(
            "generation_prefix_no_cache_logits"
        ),
        clean_prefix_no_cache_logits=stack_field(
            "clean_prefix_no_cache_logits"
        ),
        full_teacher_no_cache_logits=stack_field(
            "full_teacher_no_cache_logits"
        ),
    )


def _fidelity_samples_to_replay(
    samples: list[FidelitySample],
) -> FidelityReplay:
    if len(samples) != REPEAT_COUNT:
        raise RuntimeError("每个numerical输入必须恰好采集3次Fidelity")
    return FidelityReplay(
        generated_token_ids=np.stack(
            tuple(item.generated_token_ids for item in samples), axis=0
        ),
        generated_classes=np.stack(
            tuple(item.generated_classes for item in samples), axis=0
        ),
        generation_logits=np.stack(
            tuple(item.generation_logits for item in samples), axis=0
        ),
        teacher_logits=np.stack(
            tuple(item.teacher_logits for item in samples), axis=0
        ),
    )


def _run_terminal_numerical_inference(
    cfg: TerminalNumericalInferenceConfig,
) -> Path:
    """执行分层numerical replay并返回独立CPU复核后的manifest。"""

    _validate_config(cfg)
    verify_executing_commit(cfg.code_commit)
    if loaded_legacy_optimizer_modules():
        raise RuntimeError("numerical replay进程加载了legacy optimizer")
    if not torch.cuda.is_available():
        raise RuntimeError("numerical replay需要真实CUDA device")
    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"numerical replay输出目录必须为空: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.source_bundle_dir).resolve()
    checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    source, source_hashes = _load_source_bundle(source_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    model_cfg = GenerateConfig(
        pretrained_checkpoint=cfg.pretrained_checkpoint,
        model_family=cfg.model_family,
        center_crop=True,
        object_name=cfg.object_name,
        task_suite_name=cfg.task_suite_name,
        task_id=cfg.task_id,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=cfg.unnorm_key,
    )
    # 延迟导入可保证本命令的CPU evaluator和--help不依赖LIBERO环境。
    from openvla_utils import get_processor
    from robot_utils import get_model

    model: Any = get_model(model_cfg)
    model.eval()
    model.requires_grad_(False)
    processor: Any = get_processor(model_cfg)
    pad_token_id = int(processor.tokenizer.pad_token_id)
    fidelity_samples: dict[tuple[str, str], list[FidelitySample]] = {
        (variant, path): []
        for variant in NUMERICAL_VARIANTS
        for path in NUMERICAL_PATHS
    }
    reconstructed_bits: list[np.ndarray] = []
    reconstructed_prompt_ids: list[np.ndarray] = []
    reconstructed_teacher_ids: list[np.ndarray] = []
    reconstructed_attention_masks: list[np.ndarray] = []
    prompt_tensors: dict[str, torch.Tensor] = {}
    teacher_tensors: dict[str, torch.Tensor] = {}
    pixel_tensors: dict[tuple[str, str], torch.Tensor] = {}
    for variant in NUMERICAL_VARIANTS:
        evidence = source[variant]
        prompt_ids = torch.from_numpy(
            np.asarray(evidence.prompt_input_ids, dtype=np.int64)
        ).unsqueeze(0).to(model.device)
        teacher_ids = torch.from_numpy(
            np.asarray(evidence.teacher_input_ids, dtype=np.int64)
        ).unsqueeze(0).to(model.device)
        reconstructed_prompt_ids.append(
            prompt_ids[0].detach().cpu().numpy().copy()
        )
        reconstructed_teacher_ids.append(
            teacher_ids[0].detach().cpu().numpy().copy()
        )
        reconstructed_attention_masks.append(
            torch.ones_like(prompt_ids)[0].detach().cpu().numpy().copy()
        )
        prompt_tensors[variant] = prompt_ids
        teacher_tensors[variant] = teacher_ids
        selected_bits = _selected_pixel_bits(evidence)
        variant_reconstructed: list[np.ndarray] = []
        for path_index, path_name in enumerate(NUMERICAL_PATHS):
            pixels = tensor_from_bfloat16_bits(
                selected_bits[path_index], device=model.device
            )
            pixel_tensors[(variant, path_name)] = pixels
            variant_reconstructed.append(
                bfloat16_tensor_to_uint16_bits(pixels)
            )
        reconstructed_bits.append(np.stack(variant_reconstructed, axis=0))
    # 第一轮严格复刻原 smoke 的唯一模型调用顺序：共享Clean一次、两个A、两个B。
    # 后两轮按同一inventory顺序重复，既减少共享Clean重复歧义，也检查全局调用
    # 顺序下的稳定性。evidence中仍为两个variant各保留完整C/A/B行。
    inventory = (
        (NUMERICAL_VARIANTS[0], "clean"),
        (NUMERICAL_VARIANTS[0], "training"),
        (NUMERICAL_VARIANTS[1], "training"),
        (NUMERICAL_VARIANTS[0], "deployment"),
        (NUMERICAL_VARIANTS[1], "deployment"),
    )
    for repeat_index in range(REPEAT_COUNT):
        for variant, path_name in inventory:
            print(
                "[GATE-6G-NUMERICAL] fidelity "
                f"repeat={repeat_index} {variant}/{path_name}"
            )
            sample = capture_fidelity_sample(
                model,
                prompt_input_ids=prompt_tensors[variant],
                teacher_input_ids=teacher_tensors[variant],
                pixel_values=pixel_tensors[(variant, path_name)],
                pad_token_id=pad_token_id,
                unnorm_key=cfg.unnorm_key,
            )
            fidelity_samples[(variant, path_name)].append(sample)
            if path_name == "clean":
                fidelity_samples[(NUMERICAL_VARIANTS[1], "clean")].append(
                    sample
                )
    fidelity_replays = [
        [
            _fidelity_samples_to_replay(fidelity_samples[(variant, path)])
            for path in NUMERICAL_PATHS
        ]
        for variant in NUMERICAL_VARIANTS
    ]
    fidelity = _stack_fidelity(
        source,
        fidelity_replays,
        np.stack(reconstructed_bits, axis=0),
        np.stack(reconstructed_prompt_ids, axis=0),
        np.stack(reconstructed_teacher_ids, axis=0),
        np.stack(reconstructed_attention_masks, axis=0),
    )
    fidelity_path = output_dir / "arrays" / "fidelity.npz"
    fidelity_sha = write_numerical_fidelity_npz(
        fidelity,
        output_path=fidelity_path,
    )
    fidelity_decision = evaluate_numerical_fidelity(fidelity)

    attribution: NumericalAttributionEvidence | None = None
    attribution_decision: NumericalAttributionDecision | None = None
    attribution_record: dict[str, str] | None = None
    if fidelity_decision.fidelity_pass:
        attribution_replays: list[list[AttributionReplay]] = []
        for variant_index, variant in enumerate(NUMERICAL_VARIANTS):
            evidence = source[variant]
            prompt_ids = torch.from_numpy(
                np.asarray(evidence.prompt_input_ids, dtype=np.int64)
            ).unsqueeze(0).to(model.device)
            teacher_ids = torch.from_numpy(
                np.asarray(evidence.teacher_input_ids, dtype=np.int64)
            ).unsqueeze(0).to(model.device)
            selected_bits = _selected_pixel_bits(evidence)
            variant_replays: list[AttributionReplay] = []
            for path_index, path_name in enumerate(NUMERICAL_PATHS):
                print(
                    f"[GATE-6G-NUMERICAL] attribution {variant}/{path_name}"
                )
                pixels = tensor_from_bfloat16_bits(
                    selected_bits[path_index], device=model.device
                )
                variant_replays.append(
                    capture_attribution_repeats(
                        model,
                        prompt_input_ids=prompt_ids,
                        teacher_input_ids=teacher_ids,
                        pixel_values=pixels,
                        cached_generated_token_ids=(
                            fidelity_replays[variant_index][path_index]
                            .generated_token_ids[0]
                        ),
                        pad_token_id=pad_token_id,
                        unnorm_key=cfg.unnorm_key,
                        repeat_count=REPEAT_COUNT,
                    )
                )
            attribution_replays.append(variant_replays)
        attribution = _stack_attribution(attribution_replays)
        attribution_path = output_dir / "arrays" / "attribution.npz"
        attribution_sha = write_numerical_attribution_npz(
            attribution,
            output_path=attribution_path,
        )
        attribution_decision = evaluate_numerical_attribution(
            fidelity=fidelity,
            evidence=attribution,
        )
        attribution_record = {
            "relative_path": str(attribution_path.relative_to(output_dir)),
            "sha256": attribution_sha,
        }

    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except ImportError:
        transformers_version = None
    manifest = {
        "schema_version": NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "gate6g_contract_status": "invalid",
        "code_commit": cfg.code_commit,
        "repeat_count": REPEAT_COUNT,
        "source_invalid_bundle": {
            **source_hashes,
            "path": str(source_dir),
        },
        "artifacts": {
            "fidelity": {
                "relative_path": str(fidelity_path.relative_to(output_dir)),
                "sha256": fidelity_sha,
            },
            "attribution": attribution_record,
        },
        "fidelity_decision": numerical_decision_payload(fidelity_decision),
        "attribution_decision": (
            None
            if attribution_decision is None
            else numerical_decision_payload(attribution_decision)
        ),
        "causal_attribution_allowed": bool(
            attribution_decision is not None
            and attribution_decision.causal_attribution_allowed
        ),
        "provenance": {
            "configuration": asdict(cfg),
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
            "pad_token_id": pad_token_id,
            "attention_mask_reconstruction": "single_sample_all_ones",
            "original_generation_use_cache_argument": "omitted",
            "original_teacher_use_cache_argument": "omitted",
            "alternate_paths_use_cache": False,
            "runtime_model_config": _runtime_model_config(model),
            "framework_versions": {
                "numpy": np.__version__,
                "torch": torch.__version__,
                "transformers": transformers_version,
                "cuda_runtime": torch.version.cuda,
            },
            "cuda": {
                "device_name": torch.cuda.get_device_name(model.device),
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": (
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            },
        },
    }
    manifest_path = output_dir / "terminal_numerical_inference_manifest.json"
    manifest_sha = publish_numerical_inference_manifest(
        manifest,
        output_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "status": "diagnostic_complete",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "fidelity_pass": fidelity_decision.fidelity_pass,
                "causal_attribution_allowed": manifest[
                    "causal_attribution_allowed"
                ],
            },
            sort_keys=True,
        )
    )
    return manifest_path


def run_terminal_numerical_inference(
    cfg: TerminalNumericalInferenceConfig,
) -> Path:
    """运行诊断；异常时best-effort保存不冒充成功manifest的失败记录。"""

    try:
        return _run_terminal_numerical_inference(cfg)
    except BaseException as error:
        output_dir = Path(cfg.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "numerical_inference_failed.json"
            if not failure_path.exists():
                candidate = failure_path.with_name(
                    f".{failure_path.name}.candidate"
                )
                candidate.write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                NUMERICAL_INFERENCE_BUNDLE_SCHEMA_VERSION
                            ),
                            "status": "diagnostic_failed",
                            "gate6g_contract_status": "invalid",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "code_commit": cfg.code_commit,
                            "source_bundle_dir": cfg.source_bundle_dir,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(candidate, failure_path)
        except BaseException as record_error:
            print(
                "[GATE-6G-NUMERICAL] 无法保存best-effort失败记录: "
                f"{record_error}",
                file=sys.stderr,
            )
        raise


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_terminal_numerical_inference(_parse_args(argv))


if __name__ == "__main__":
    main()
