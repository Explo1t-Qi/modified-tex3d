"""运行 Gate 6h scalar-gain counterfactual OpenVLA GPU audit。

命令只加载 source OpenVLA checkpoint 和已通过的 Gate 6g formal bundle；不加载
LIBERO、renderer、纹理参数、Feature、wrist或OFT，也不训练、反传或 rollout。
每个 ``(variant,state)`` 先从原NPZ重建C/A/B最终BF16输入并逐值重放原始模型
输出，再由完整224×224 Effective RGB构造唯一新增的 ``I_gain`` 并采集默认
cached generation与clean-prefix teacher诊断。只有20个case全部通过独立CPU
复核时才原子发布成功manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.terminal_deployment_response_audit import (  # noqa: E402
    TERMINAL_RESPONSE_VARIANTS,
    evaluate_terminal_response_bundle,
    load_terminal_response_npz,
)
from openvla_attack.terminal_gain_counterfactual_audit import (  # noqa: E402
    GAIN_COUNTERFACTUAL_BUNDLE_SCHEMA_VERSION,
    GainCounterfactualModelEvidence,
    construct_scalar_gain_counterfactual,
    evaluate_gain_counterfactual_model_evidence,
    gain_counterfactual_decision_record,
    publish_gain_counterfactual_manifest,
    write_gain_counterfactual_npz,
)
from openvla_attack.terminal_numerical_inference import (  # noqa: E402
    FidelitySample,
    capture_fidelity_sample,
    tensor_from_bfloat16_bits,
)
from openvla_attack.terminal_openvla_response import (  # noqa: E402
    bfloat16_tensor_to_uint16_bits,
)


@dataclass(frozen=True)
class TerminalGainCounterfactualConfig:
    """Gate 6h只读GPU replay配置。"""

    pretrained_checkpoint: str
    source_gate6g_manifest_path: str
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
) -> TerminalGainCounterfactualConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--source_gate6g_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    return TerminalGainCounterfactualConfig(**vars(parser.parse_args(argv)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _loaded_forbidden_modules() -> tuple[str, ...]:
    """返回Gate 6h只读NPZ replay不应加载的环境/renderer模块。"""

    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "libero_utils"
            or name == "libero"
            or name.startswith("libero.")
            or name.startswith("nvdiffrast")
            or name.endswith("openvla_attack.renderer")
        )
    )


def _validate_config(cfg: TerminalGainCounterfactualConfig) -> None:
    if (
        cfg.task_suite_name != "libero_spatial"
        or cfg.task_id != 0
        or cfg.object_name != "akita_black_bowl"
        or not cfg.center_crop
    ):
        raise ValueError("Gate 6h固定为LIBERO Spatial task 0/Akita")
    if cfg.load_in_8bit or cfg.load_in_4bit:
        raise ValueError("Gate 6h必须使用原BF16 checkpoint")


def _source_cases(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decision = evaluate_terminal_response_bundle(manifest_path)
    if not decision.audit_valid:
        raise RuntimeError(
            "source Gate 6g bundle无效: " + "; ".join(decision.failures)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("source Gate 6g cases无效")
    by_key = {
        (str(case["variant"]), int(case["state_id"])): case for case in cases
    }
    ordered = [
        by_key[(variant, state_id)]
        for state_id in range(10)
        for variant in TERMINAL_RESPONSE_VARIANTS
    ]
    return ordered, manifest


def _selected_source_bits(evidence: Any) -> np.ndarray:
    """复用Gate 6g路径语义：C/B official，A training exact。"""

    official = np.asarray(evidence.official_processor_bf16_bits)
    training = np.asarray(evidence.training_exact_processor_bf16_bits)
    selected = official.copy()
    selected[1] = training[1]
    return selected


def _sample_arrays(samples: Sequence[FidelitySample], name: str) -> np.ndarray:
    return np.stack(tuple(np.asarray(getattr(sample, name)) for sample in samples))


def _checkpoint_fingerprints(checkpoint: Path) -> dict[str, str]:
    names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
    )
    result = {
        name: _file_sha256(checkpoint / name)
        for name in names
        if (checkpoint / name).is_file()
    }
    weights = sorted(
        (*checkpoint.glob("*.safetensors"), *checkpoint.glob("*.bin"))
    )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weights
    ).encode("utf-8")
    result["__weight_name_size_inventory_sha256__"] = hashlib.sha256(
        inventory
    ).hexdigest()
    return result


def _run(cfg: TerminalGainCounterfactualConfig) -> Path:
    _validate_config(cfg)
    verify_executing_commit(cfg.code_commit)
    if loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6h进程加载了legacy optimizer")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 6h需要真实CUDA device")
    output = Path(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Gate 6h输出目录必须为空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = Path(cfg.source_gate6g_manifest_path).resolve()
    source_cases, source_manifest = _source_cases(source_manifest_path)
    checkpoint = Path(cfg.pretrained_checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    source_checkpoint_fingerprints = source_manifest.get(
        "provenance", {}
    ).get("checkpoint_fingerprints")
    if not isinstance(source_checkpoint_fingerprints, dict):
        raise RuntimeError("source Gate 6g checkpoint fingerprints缺失")
    normalized_source_fingerprints = {
        (
            name
            if name == "__weight_name_size_inventory_sha256__"
            else Path(name).name
        ): value
        for name, value in source_checkpoint_fingerprints.items()
    }
    if _checkpoint_fingerprints(checkpoint) != normalized_source_fingerprints:
        raise RuntimeError("runtime checkpoint fingerprints与source Gate 6g不一致")

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
    from openvla_utils import get_processor
    from robot_utils import get_model

    model: Any = get_model(model_cfg)
    model.eval()
    model.requires_grad_(False)
    processor: Any = get_processor(model_cfg)
    image_processor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model,
        processor=processor,
    )
    pad_token_id = int(processor.tokenizer.pad_token_id)
    source_by_key: dict[tuple[str, int], Any] = {}
    prompt_by_state: dict[int, torch.Tensor] = {}
    teacher_by_state: dict[int, torch.Tensor] = {}
    pixel_by_key_path: dict[tuple[str, int, int], torch.Tensor] = {}
    counterfactual_by_key: dict[tuple[str, int], Any] = {}
    gain_pixels_by_key: dict[tuple[str, int], torch.Tensor] = {}
    for case in source_cases:
        variant, state_id = str(case["variant"]), int(case["state_id"])
        key = (variant, state_id)
        source_npz = source_manifest_path.parent / str(
            case["npz_relative_path"]
        )
        if _file_sha256(source_npz) != case["npz_sha256"]:
            raise RuntimeError(f"{variant}/state{state_id} source NPZ SHA漂移")
        source = load_terminal_response_npz(source_npz)
        source_by_key[key] = source
        if state_id not in prompt_by_state:
            prompt_by_state[state_id] = (
                torch.from_numpy(source.prompt_input_ids.astype(np.int64))
                .unsqueeze(0)
                .to(model.device)
            )
            teacher_by_state[state_id] = (
                torch.from_numpy(source.teacher_input_ids.astype(np.int64))
                .unsqueeze(0)
                .to(model.device)
            )
        for path_index, bits in enumerate(_selected_source_bits(source)):
            pixel_by_key_path[(variant, state_id, path_index)] = (
                tensor_from_bfloat16_bits(bits, device=model.device)
            )
        counterfactual = construct_scalar_gain_counterfactual(
            source.effective_rgb
        )
        counterfactual_by_key[key] = counterfactual
        gain_rgb = torch.from_numpy(counterfactual.quantized_gain_rgb).to(
            device=model.device, dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0).div(255.0)
        gain_pixels_by_key[key] = (
            image_processor.build_fused_pixel_values(gain_rgb).to(
                torch.bfloat16
            )
        )

    source_sample_by_key_path: dict[
        tuple[str, int, int], FidelitySample
    ] = {}
    gain_sample_by_key: dict[tuple[str, int], FidelitySample] = {}
    for state_id in range(10):
        prompt = prompt_by_state[state_id]
        teacher = teacher_by_state[state_id]
        reference_variant = TERMINAL_RESPONSE_VARIANTS[0]
        print(f"[GATE-6H] replay state{state_id} shared Clean")
        shared_clean = capture_fidelity_sample(
            model,
            prompt_input_ids=prompt,
            teacher_input_ids=teacher,
            pixel_values=pixel_by_key_path[
                (reference_variant, state_id, 0)
            ],
            pad_token_id=pad_token_id,
            unnorm_key=cfg.unnorm_key,
        )
        for variant in TERMINAL_RESPONSE_VARIANTS:
            source_sample_by_key_path[(variant, state_id, 0)] = shared_clean
        # 严格复用Gate 6g模型调用顺序：共享Clean一次、两个A、两个B。
        for path_index, path_name in ((1, "A"), (2, "B")):
            for variant in TERMINAL_RESPONSE_VARIANTS:
                print(
                    f"[GATE-6H] replay state{state_id} "
                    f"{variant}/{path_name}"
                )
                source_sample_by_key_path[(variant, state_id, path_index)] = (
                    capture_fidelity_sample(
                        model,
                        prompt_input_ids=prompt,
                        teacher_input_ids=teacher,
                        pixel_values=pixel_by_key_path[
                            (variant, state_id, path_index)
                        ],
                        pad_token_id=pad_token_id,
                        unnorm_key=cfg.unnorm_key,
                    )
                )
        for variant in TERMINAL_RESPONSE_VARIANTS:
            print(f"[GATE-6H] replay state{state_id} {variant}/gain")
            gain_sample_by_key[(variant, state_id)] = capture_fidelity_sample(
                model,
                prompt_input_ids=prompt,
                teacher_input_ids=teacher,
                pixel_values=gain_pixels_by_key[(variant, state_id)],
                pad_token_id=pad_token_id,
                unnorm_key=cfg.unnorm_key,
            )

    case_records: list[dict[str, Any]] = []
    for case in source_cases:
        variant, state_id = str(case["variant"]), int(case["state_id"])
        key = (variant, state_id)
        source = source_by_key[key]
        counterfactual = counterfactual_by_key[key]
        source_samples = [
            source_sample_by_key_path[(variant, state_id, path_index)]
            for path_index in range(3)
        ]
        gain_sample = gain_sample_by_key[key]
        gain_pixels = gain_pixels_by_key[key]
        evidence = GainCounterfactualModelEvidence(
            variant=variant,
            state_id=state_id,
            source_effective_rgb=source.effective_rgb,
            alpha_star=counterfactual.alpha_star,
            continuous_gain_rgb=counterfactual.continuous_gain_rgb,
            quantized_gain_rgb=counterfactual.quantized_gain_rgb,
            continuous_residual=counterfactual.continuous_residual,
            quantized_residual=counterfactual.quantized_residual,
            clipped_value_count=counterfactual.clipped_value_count,
            action_token_start=source.action_token_start,
            action_token_end=source.action_token_end,
            source_generated_token_ids=source.generated_token_ids,
            source_generated_classes=source.generated_classes,
            source_generation_logits=source.generation_logits,
            source_teacher_logits=source.teacher_logits,
            replay_generated_token_ids=_sample_arrays(
                source_samples, "generated_token_ids"
            ),
            replay_generated_classes=_sample_arrays(
                source_samples, "generated_classes"
            ),
            replay_generation_logits=_sample_arrays(
                source_samples, "generation_logits"
            ),
            replay_teacher_logits=_sample_arrays(
                source_samples, "teacher_logits"
            ),
            gain_generated_token_ids=gain_sample.generated_token_ids,
            gain_generated_classes=gain_sample.generated_classes,
            gain_generation_logits=gain_sample.generation_logits,
            gain_teacher_logits=gain_sample.teacher_logits,
            gain_processor_bf16_bits=bfloat16_tensor_to_uint16_bits(
                gain_pixels
            ),
        )
        decision = evaluate_gain_counterfactual_model_evidence(evidence)
        if not decision.evidence_valid:
            raise RuntimeError(
                f"{variant}/state{state_id} Gate 6h case无效: "
                + "; ".join(decision.failures)
            )
        relative = Path("arrays") / variant / f"state_{state_id:02d}.npz"
        digest = write_gain_counterfactual_npz(
            evidence, output_path=output / relative
        )
        case_records.append(
            {
                "variant": variant,
                "state_id": state_id,
                "source_npz_relative_path": case["npz_relative_path"],
                "source_npz_sha256": case["npz_sha256"],
                "npz_relative_path": str(relative),
                "npz_sha256": digest,
                "evaluation": gain_counterfactual_decision_record(decision),
            }
        )
    forbidden_modules = _loaded_forbidden_modules()
    if forbidden_modules:
        raise RuntimeError(
            "Gate 6h进程加载了LIBERO/renderer模块: "
            + ", ".join(forbidden_modules)
        )
    configuration = asdict(cfg)
    payload = {
        "schema_version": GAIN_COUNTERFACTUAL_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "code_commit": cfg.code_commit,
        "config_sha256": _json_sha256(configuration),
        "source_gate6g_manifest_path": str(source_manifest_path),
        "source_gate6g_manifest_sha256": _file_sha256(source_manifest_path),
        "source_gate6g_code_commit": source_manifest.get("code_commit"),
        "expected_variants": list(TERMINAL_RESPONSE_VARIANTS),
        "expected_state_ids": list(range(10)),
        "primary_analysis_source_classifications": [
            "deployment_lost",
            "deployment_response_altered",
        ],
        "response_authority": source_manifest.get("response_authority"),
        "configuration": configuration,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "provenance": {
            "checkpoint_fingerprints": source_checkpoint_fingerprints,
            "source_gate6g_state_fingerprints": source_manifest.get(
                "state_fingerprints"
            ),
            "processor_specification": source_manifest.get(
                "provenance", {}
            ).get("processor_specification"),
            "policy_view_specification": source_manifest.get(
                "provenance", {}
            ).get("policy_view_specification"),
            "legacy_optimizer_modules": list(loaded_legacy_optimizer_modules()),
            "training_or_backward_run": False,
            "libero_or_renderer_modules": list(forbidden_modules),
        },
        "cases": case_records,
    }
    manifest = output / "terminal_gain_counterfactual_manifest.json"
    publish_gain_counterfactual_manifest(payload, output_path=manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    try:
        manifest = _run(cfg)
    except BaseException as error:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        failure = output / "audit_failed.json"
        if not failure.exists():
            failure.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            GAIN_COUNTERFACTUAL_BUNDLE_SCHEMA_VERSION
                        ),
                        "status": "audit_invalid",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "code_commit": cfg.code_commit,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    else:
        print(f"[GATE-6H] complete manifest={manifest}")


if __name__ == "__main__":
    main()
