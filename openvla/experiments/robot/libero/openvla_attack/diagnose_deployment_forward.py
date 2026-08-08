"""运行 Gate 1D：states 0–9 完整 deployment-path forward equivalence。

该命令只读取 clean LIBERO state、checkpoint 和物体资产，不修改纹理或 XML，
也不运行攻击优化。每个 state 对比正式 rollout 路径与完整 BPDA candidate 的
Pre-Crop RGB、Effective View、fused pixel values、action token 和连续 action，
并把机器可读证据交给 ``openvla-gate-1d-v1`` 严格判定层。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping, Optional, Sequence

import numpy as np
import PIL
import tensorflow as tf
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.cuda.amp import autocast


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

from experiments.robot.openvla_utils import (  # noqa: E402
    ensure_trailing_empty_token,
    get_vla_action,
)
from openvla_attack.action_codec import (  # noqa: E402
    decode_action_from_generated_ids,
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
)
from openvla_attack.deployment_forward_audit import (  # noqa: E402
    DeploymentForwardEvidence,
    DeploymentForwardRow,
    evaluate_deployment_forward_evidence,
    summarize_deployment_forward_rows,
    write_deployment_forward_jsonl,
    write_deployment_forward_manifest,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.policy_view import (  # noqa: E402
    DeploymentViewSpecification,
    DifferentiablePolicyViewTransform,
    ExactDeploymentViewStages,
    build_exact_deployment_view_stages,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402
from openvla_utils import get_processor  # noqa: E402


@dataclass
class DeploymentForwardConfig:
    pretrained_checkpoint: str
    output_dir: str
    code_commit: str
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    state_ids: str = "0-9"
    num_steps_wait: int = 10
    seed: int = 7
    unnorm_key: Optional[str] = "libero_spatial_no_noops"
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


class _RecordingProcessor:
    """透明代理 processor，并记录 rollout 真正收到的 PIL RGB 与 inputs。"""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.last_rgb: Optional[NDArray[np.uint8]] = None
        self.last_inputs: Optional[MutableMapping[str, torch.Tensor]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        image: Any = args[1] if len(args) > 1 else kwargs.get("images")
        if not isinstance(image, Image.Image):
            raise RuntimeError("Gate 1D rollout processor 未收到 PIL Image")
        self.last_rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        inputs: Any = self._processor(*args, **kwargs)
        self.last_inputs = inputs
        return inputs


def _parse_args(argv: Optional[Sequence[str]] = None) -> DeploymentForwardConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--state_ids", default="0-9")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    return DeploymentForwardConfig(**vars(parser.parse_args(argv)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_files(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _checkpoint_fingerprints(checkpoint_path: Path) -> dict[str, str]:
    """记录关键配置内容 hash 与权重分片 name/size inventory hash。"""

    configuration_files = tuple(
        checkpoint_path / name
        for name in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "dataset_statistics.json",
            "model.safetensors.index.json",
        )
    )
    fingerprints = _fingerprint_files(configuration_files)
    weight_files = sorted(
        tuple(checkpoint_path.glob("*.safetensors"))
        + tuple(checkpoint_path.glob("*.bin"))
    )
    inventory_payload = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weight_files
    ).encode("utf-8")
    fingerprints["__weight_name_size_inventory_sha256__"] = hashlib.sha256(
        inventory_payload
    ).hexdigest()
    return fingerprints


def _state_fingerprint(state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(state))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rgb_tensor(image: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _uint8_hwc(tensor: torch.Tensor) -> NDArray[np.uint8]:
    return (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
        .copy()
    )


def _clone_inputs(
    inputs: MutableMapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def run_deployment_forward_audit(cfg: DeploymentForwardConfig) -> Path:
    """采集真实 states、保存 Gate 1D bundle，并返回 manifest path。"""

    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("Gate 1D state_ids 不得为空")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    set_seed_everywhere(cfg.seed)
    model: Any = get_model(cfg)
    model.eval()
    processor: Any = get_processor(cfg)
    image_preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model,
        processor=processor,
    )
    view_specification = DeploymentViewSpecification()
    view_transform = DifferentiablePolicyViewTransform(view_specification)
    output_dir = Path(cfg.output_dir)
    image_dir = output_dir / "stages"
    array_dir = output_dir / "arrays"
    image_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if max(selected_state_ids) >= len(initial_states):
        raise ValueError("Gate 1D state_id 超出 LIBERO 初始状态范围")

    rows: list[DeploymentForwardRow] = []
    state_fingerprints: dict[str, str] = {}
    task_description: Optional[str] = None
    for state_id in selected_state_ids:
        print(f"[GATE-1D] state {state_id}")
        initial_state = initial_states[state_id]
        state_fingerprints[str(state_id)] = _state_fingerprint(initial_state)
        env, current_task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=512,
        )
        task_description = current_task_description
        try:
            env.reset()
            observation = env.set_init_state(initial_state)
            env.env.sim.forward()
            for _ in range(cfg.num_steps_wait):
                observation, _, _, _ = env.step(
                    get_libero_dummy_action(cfg.model_family)
                )
            source_rgb = get_libero_image(observation, 512)
        finally:
            env.close()

        exact_stages: ExactDeploymentViewStages = (
            build_exact_deployment_view_stages(
                source_rgb,
                specification=view_specification,
            )
        )
        source_tensor = _rgb_tensor(source_rgb, model.device)
        candidate_pre_crop = view_transform.build_pre_crop_canvas(source_tensor)
        candidate_effective = view_transform.build_effective_view(source_tensor)
        candidate_pixels = image_preprocessor.build_fused_pixel_values(
            candidate_effective
        ).to(torch.bfloat16)

        recording_processor = _RecordingProcessor(processor)
        rollout_action = get_vla_action(
            model,
            recording_processor,
            cfg.pretrained_checkpoint,
            {"full_image": exact_stages.pre_crop_rgb},
            current_task_description,
            cfg.unnorm_key,
            center_crop=True,
        )
        if recording_processor.last_rgb is None or (
            recording_processor.last_inputs is None
        ):
            raise RuntimeError("Gate 1D 未捕获 rollout processor 输入")
        rollout_inputs = recording_processor.last_inputs
        if "pixel_values" not in rollout_inputs:
            raise RuntimeError("rollout processor inputs 缺少 pixel_values")
        rollout_pixels = rollout_inputs["pixel_values"]
        candidate_inputs = _clone_inputs(rollout_inputs)
        candidate_inputs["pixel_values"] = candidate_pixels
        ensure_trailing_empty_token(candidate_inputs)
        action_dim = int(model.get_action_dim(cfg.unnorm_key))
        generation_arguments = {
            "max_new_tokens": action_dim,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            rollout_generated = model.generate(
                **rollout_inputs,
                **generation_arguments,
            )
            candidate_generated = model.generate(
                **candidate_inputs,
                **generation_arguments,
            )
        rollout_tokens = rollout_generated[0, -action_dim:]
        candidate_tokens = candidate_generated[0, -action_dim:]
        candidate_action = decode_action_from_generated_ids(
            model,
            candidate_generated,
            cfg.unnorm_key,
        )

        evidence = DeploymentForwardEvidence(
            state_id=state_id,
            source_rgb=exact_stages.source_rgb,
            exact_pre_crop_rgb=exact_stages.pre_crop_rgb,
            candidate_pre_crop_rgb=_uint8_hwc(candidate_pre_crop),
            rollout_effective_rgb=recording_processor.last_rgb,
            candidate_effective_rgb=_uint8_hwc(candidate_effective),
            rollout_pixel_values=rollout_pixels.float().cpu().numpy(),
            candidate_pixel_values=candidate_pixels.float().cpu().numpy(),
            rollout_action_token_ids=rollout_tokens.cpu().numpy(),
            candidate_action_token_ids=candidate_tokens.cpu().numpy(),
            rollout_action=np.asarray(rollout_action, dtype=np.float64),
            candidate_action=np.asarray(candidate_action, dtype=np.float64),
        )
        row = evaluate_deployment_forward_evidence(
            evidence,
            code_commit=cfg.code_commit,
        )
        rows.append(row)
        prefix = f"state_{state_id:02d}"
        Image.fromarray(exact_stages.source_rgb).save(image_dir / f"{prefix}_source.png")
        Image.fromarray(exact_stages.pre_crop_rgb).save(image_dir / f"{prefix}_pre_crop.png")
        Image.fromarray(recording_processor.last_rgb).save(image_dir / f"{prefix}_effective.png")
        Image.fromarray(evidence.candidate_pre_crop_rgb).save(
            image_dir / f"{prefix}_candidate_pre_crop.png"
        )
        Image.fromarray(evidence.candidate_effective_rgb).save(
            image_dir / f"{prefix}_candidate_effective.png"
        )
        np.savez_compressed(
            array_dir / f"{prefix}.npz",
            rollout_pixel_values=evidence.rollout_pixel_values,
            candidate_pixel_values=evidence.candidate_pixel_values,
            rollout_action_token_ids=evidence.rollout_action_token_ids,
            candidate_action_token_ids=evidence.candidate_action_token_ids,
            rollout_action=evidence.rollout_action,
            candidate_action=evidence.candidate_action,
        )

    summary = summarize_deployment_forward_rows(
        rows,
        expected_state_ids=selected_state_ids,
    )
    metrics_path = output_dir / "deployment_forward_metrics.jsonl"
    metrics_sha256 = write_deployment_forward_jsonl(rows, output_path=metrics_path)
    checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
    asset = OBJECT_ASSETS[cfg.object_name]
    asset_paths = tuple(Path(asset[key]).resolve() for key in ("xml", "mesh", "texture"))
    metadata = {
        "code_commit": cfg.code_commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "task_name": str(getattr(task, "name", task_description)),
        "object_name": cfg.object_name,
        "object_asset_fingerprints": _fingerprint_files(asset_paths),
        "state_ids": list(selected_state_ids),
        "state_fingerprints": state_fingerprints,
        "seed": cfg.seed,
        "deployment_configuration": {
            "policy_source_resolution": 512,
            "pre_crop_resolution": 224,
            "crop_area": 0.9,
            "num_steps_wait": cfg.num_steps_wait,
            "center_crop": True,
            "unnorm_key": cfg.unnorm_key,
        },
        "framework_versions": {
            "pillow": PIL.__version__,
            "tensorflow": tf.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
    }
    manifest_path = output_dir / "deployment_forward_manifest.json"
    manifest_sha256 = write_deployment_forward_manifest(
        metadata=metadata,
        summary=summary,
        metrics_jsonl_sha256=metrics_sha256,
        output_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "gate_pass": summary["gate_pass"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "metrics_sha256": metrics_sha256,
                **summary,
            },
            sort_keys=True,
        )
    )
    if not summary["gate_pass"]:
        raise RuntimeError("Gate 1D 未通过，详见 manifest/JSONL")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_deployment_forward_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
