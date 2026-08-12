"""运行 Gate 6g state 0 双终态完整 C/A/B GPU smoke。

本命令只读使用已经完成的 Action+Spectral 与 Action-only formal bundle：

* ``C``：clean MuJoCo observation，经 official processor 与 source OpenVLA；
* ``A``：同一静止 clean state 上的 Renderer Delta Composition，经训练 exact
  BPDA forward 与 source OpenVLA；
* ``B``：激活 hash-bound bake PNG 后重新恢复同一 MuJoCo state，经 official
  processor 与 source OpenVLA。

GPU runner只保存原始RGB、segmentation/alpha、BF16 bits、logits、token与codec
数组。成功manifest只有在纯CPU smoke evaluator重新加载两个NPZ并返回
``audit_valid`` 后才原子发布。命令不训练、不反向传播、不运行rollout，也不加载
Feature、wrist或OFT。
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
from typing import Any, Mapping, MutableMapping, Optional, Sequence

# 必须在导入LIBERO/Robosuite/nvdiffrast前固定headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import PIL
import torch
from numpy.typing import NDArray
from PIL import Image


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
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.fixed_support_source_training import (  # noqa: E402
    loaded_legacy_optimizer_modules,
    verify_executing_commit,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    render_shared_texture_instances,
)
from openvla_attack.objective import (  # noqa: E402
    ACTION_TOKEN_END,
    ACTION_TOKEN_START,
)
from openvla_attack.paired_source_gate import state_fingerprint  # noqa: E402
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiablePolicyViewTransform,
    build_exact_deployment_view_stages,
    build_policy_view_transform,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.runtime_assets import RuntimeAssetTransaction  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.terminal_deployment_response_audit import (  # noqa: E402
    TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION,
    TERMINAL_RESPONSE_AUTHORITY_CONTRACT,
    TERMINAL_RESPONSE_VARIANTS,
    TerminalDeploymentResponseEvidence,
    evaluate_terminal_response_evidence,
    publish_terminal_response_smoke_manifest,
    terminal_response_evaluation_record,
    write_json_atomically,
    write_terminal_response_npz,
)
from openvla_attack.terminal_deployment_response_inputs import (  # noqa: E402
    EXPECTED_SURFACE_EPSILON,
    TerminalDeploymentInputs,
    resolve_terminal_deployment_inputs,
)
from openvla_attack.terminal_openvla_response import (  # noqa: E402
    OpenVLAActionResponse,
    bfloat16_tensor_to_uint16_bits,
    capture_openvla_action_response,
)
from openvla_attack.terminal_rebake_preflight import (  # noqa: E402
    canonical_json_sha256,
    evaluate_terminal_rebake_preflight_bundle,
)
from openvla_attack.visibility_capture import (  # noqa: E402
    CapturedInstanceSegmentation,
    StaticSceneEvidenceTransaction,
    static_scene_evidence_transaction,
    capture_instance_segmentation,
)
from openvla_attack.visibility_compositing import (  # noqa: E402
    compose_visibility_masked_renderer_delta,
)
from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
)
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass(frozen=True)
class TerminalDeploymentResponseConfig:
    """Gate 6g state 0 smoke CLI配置；方法与state inventory不可改写。"""

    pretrained_checkpoint: str
    action_spectral_manifest_path: str
    action_only_manifest_path: str
    production_support_path: str
    rebake_preflight_manifest_path: str
    output_dir: str
    code_commit: str
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    state_id: int = 0
    num_steps_wait: int = 10
    seed: int = 7
    unnorm_key: Optional[str] = "libero_spatial_no_noops"
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass(frozen=True)
class _ProcessorPair:
    """同一路径的training-exact与official最终processor证据。"""

    effective_rgb: NDArray[np.uint8]
    prompt_inputs: Mapping[str, torch.Tensor]
    training_exact_pixels: torch.Tensor
    official_pixels: torch.Tensor


@dataclass(frozen=True)
class _AdversarialTrainingPath:
    """一个终态在clean静止场景中的A路径响应。"""

    processor: _ProcessorPair
    response: OpenVLAActionResponse
    saturated_pixel_fraction: float
    saturated_channel_fraction: float


@dataclass(frozen=True)
class _DeploymentPath:
    """一个终态激活bake后的B路径环境与模型证据。"""

    processor: _ProcessorPair
    response: OpenVLAActionResponse
    segmentation: CapturedInstanceSegmentation
    static_transaction: StaticSceneEvidenceTransaction
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> TerminalDeploymentResponseConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--action_spectral_manifest_path", required=True)
    parser.add_argument("--action_only_manifest_path", required=True)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--rebake_preflight_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--state_id", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    return TerminalDeploymentResponseConfig(**vars(parser.parse_args(argv)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fingerprints(checkpoint_path: Path) -> dict[str, str]:
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
    fingerprints = {
        str(path): _file_sha256(path)
        for path in configuration_files
        if path.is_file()
    }
    weight_files = sorted(
        tuple(checkpoint_path.glob("*.safetensors"))
        + tuple(checkpoint_path.glob("*.bin"))
    )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weight_files
    ).encode("utf-8")
    fingerprints["__weight_name_size_inventory_sha256__"] = hashlib.sha256(
        inventory
    ).hexdigest()
    return fingerprints


def _prompt(task_description: str) -> str:
    return (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )


def _rgb_tensor(
    image: NDArray[np.uint8],
    *,
    device: torch.device,
) -> torch.Tensor:
    """uint8 HWC RGB转float32 device NCHW ``[1,3,H,W]``。"""

    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _effective_rgb_from_tensor(
    value: torch.Tensor,
) -> NDArray[np.uint8]:
    """量化exact-forward Effective View为独立uint8 HWC RGB。"""

    if (
        value.dtype != torch.float32
        or value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] != 3
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("Effective View必须为有限float32 [1,3,H,W]")
    return (
        value.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
        .copy()
    )


def _processor_pair(
    *,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    prompt: str,
    effective_view: torch.Tensor,
    device: torch.device,
) -> _ProcessorPair:
    effective_rgb = _effective_rgb_from_tensor(effective_view)
    official_inputs: MutableMapping[str, torch.Tensor] = processor(
        prompt,
        images=Image.fromarray(effective_rgb, mode="RGB"),
    ).to(device)
    ensure_trailing_empty_token(official_inputs)
    if "pixel_values" not in official_inputs:
        raise RuntimeError("official processor inputs缺少pixel_values")
    official_pixels = official_inputs["pixel_values"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    training_pixels = image_preprocessor.build_fused_pixel_values(
        effective_view
    ).to(torch.bfloat16)
    if official_pixels.shape != training_pixels.shape:
        raise RuntimeError("training/official processor最终shape不一致")
    prompt_inputs = {
        name: tensor.detach().clone()
        for name, tensor in official_inputs.items()
        if name != "pixel_values"
    }
    return _ProcessorPair(
        effective_rgb=effective_rgb,
        prompt_inputs=prompt_inputs,
        training_exact_pixels=training_pixels.detach(),
        official_pixels=official_pixels.detach(),
    )


def _build_instances(
    env: Any,
    target_poses: tuple[TargetBodyPose, ...],
    *,
    device: torch.device,
) -> tuple[TextureRenderInstance, ...]:
    return tuple(
        {
            "mvp": compute_render_mvp(
                env,
                pose.model_matrix.to(device),
                resolution=(POLICY_SOURCE_RESOLUTION, POLICY_SOURCE_RESOLUTION),
                camera_name="agentview",
            ).detach(),
            "model_rot": pose.model_matrix[:3, :3].to(device).detach(),
        }
        for pose in target_poses
    )


def _settle_environment(
    env: Any,
    initial_state: Any,
    *,
    num_steps_wait: int,
    model_family: str,
    get_dummy_action: Any,
) -> Any:
    env.reset()
    observation = env.set_init_state(initial_state)
    env.env.sim.forward()
    for _ in range(num_steps_wait):
        observation, _, _, _ = env.step(get_dummy_action(model_family))
    return observation


def _validate_config(cfg: TerminalDeploymentResponseConfig) -> None:
    if cfg.task_suite_name != "libero_spatial" or cfg.task_id != 0:
        raise ValueError("Gate 6g smoke只接受LIBERO Spatial task 0")
    if cfg.object_name != "akita_black_bowl" or cfg.state_id != 0:
        raise ValueError("Gate 6g smoke只接受Akita state 0")
    if cfg.num_steps_wait < 0 or not cfg.center_crop:
        raise ValueError("Gate 6g smoke要求非负wait和center_crop=True")
    if cfg.load_in_8bit or cfg.load_in_4bit:
        raise ValueError("Gate 6g smoke固定使用正式BF16 checkpoint，不允许量化加载")


def _validate_preflight_binding(
    preflight_manifest: Mapping[str, Any],
    inputs: TerminalDeploymentInputs,
) -> None:
    if preflight_manifest.get("production_support_sha256") != (
        inputs.production_support_sha256
    ):
        raise RuntimeError("re-bake preflight未绑定当前Production Support")
    raw_pairing = preflight_manifest.get("terminal_pairing")
    if not isinstance(raw_pairing, dict):
        raise RuntimeError("re-bake preflight缺少terminal_pairing")
    for variant in TERMINAL_RESPONSE_VARIANTS:
        pairing = raw_pairing.get(variant)
        terminal_input = inputs.variants[variant]
        if not isinstance(pairing, dict) or pairing.get("gate_pass") is not True:
            raise RuntimeError(f"{variant} re-bake pairing未通过")
        if pairing.get("parameter_sha256") != terminal_input.parameter_sha256:
            raise RuntimeError(f"{variant} preflight parameter SHA漂移")
        if pairing.get("bound_png_sha256") != (
            terminal_input.baked_texture_sha256
        ):
            raise RuntimeError(f"{variant} preflight bake SHA漂移")


def _same_prompt_and_teacher(
    reference: OpenVLAActionResponse,
    candidate: OpenVLAActionResponse,
) -> None:
    if not np.array_equal(
        reference.prompt_input_ids,
        candidate.prompt_input_ids,
    ):
        raise RuntimeError("C/A/B prompt input IDs不一致")
    if not np.array_equal(
        reference.teacher_input_ids,
        candidate.teacher_input_ids,
    ):
        raise RuntimeError("C/A/B clean teacher input IDs不一致")


def _codec_evidence(
    model: Any,
    *,
    unnorm_key: Optional[str],
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.bool_],
]:
    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"]).copy()
    high = np.asarray(stats["q99"]).copy()
    mask = np.asarray(
        stats.get("mask", np.ones_like(low, dtype=np.bool_)),
        dtype=np.bool_,
    ).copy()
    return np.asarray(model.bin_centers).copy(), low, high, mask


def _build_case_evidence(
    *,
    variant: str,
    state_id: int,
    clean_processor: _ProcessorPair,
    training_path: _AdversarialTrainingPath,
    deployment_path: _DeploymentPath,
    clean_response: OpenVLAActionResponse,
    clean_segmentation: CapturedInstanceSegmentation,
    model: Any,
    unnorm_key: Optional[str],
) -> TerminalDeploymentResponseEvidence:
    _same_prompt_and_teacher(clean_response, training_path.response)
    _same_prompt_and_teacher(clean_response, deployment_path.response)
    effective_rgb = np.stack(
        (
            clean_processor.effective_rgb,
            training_path.processor.effective_rgb,
            deployment_path.processor.effective_rgb,
        ),
        axis=0,
    )
    rgb_delta = (
        effective_rgb[1:].astype(np.float32)
        - effective_rgb[0:1].astype(np.float32)
    ) / np.float32(255.0)
    processor_pairs = (
        clean_processor,
        training_path.processor,
        deployment_path.processor,
    )
    training_pixels = torch.stack(
        tuple(pair.training_exact_pixels for pair in processor_pairs),
        dim=0,
    )
    official_pixels = torch.stack(
        tuple(pair.official_pixels for pair in processor_pairs),
        dim=0,
    )
    responses = (
        clean_response,
        training_path.response,
        deployment_path.response,
    )
    bin_centers, action_low, action_high, action_mask = _codec_evidence(
        model,
        unnorm_key=unnorm_key,
    )
    return TerminalDeploymentResponseEvidence(
        variant=variant,
        state_id=state_id,
        effective_rgb=effective_rgb,
        rgb_delta=rgb_delta,
        clean_oriented_segmentation=(
            clean_segmentation.oriented_segmentation.copy()
        ),
        deployment_oriented_segmentation=(
            deployment_path.segmentation.oriented_segmentation.copy()
        ),
        clean_instance_alpha=(
            clean_segmentation.parsed.instance_alpha.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        ),
        deployment_instance_alpha=(
            deployment_path.segmentation.parsed.instance_alpha.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        ),
        training_exact_processor_bf16_bits=(
            bfloat16_tensor_to_uint16_bits(training_pixels)
        ),
        official_processor_bf16_bits=(
            bfloat16_tensor_to_uint16_bits(official_pixels)
        ),
        training_exact_processor_float32=(
            training_pixels.float().cpu().numpy().copy()
        ),
        official_processor_float32=(
            official_pixels.float().cpu().numpy().copy()
        ),
        teacher_logits=np.stack(
            tuple(response.teacher_logits for response in responses),
            axis=0,
        ),
        generation_logits=np.stack(
            tuple(response.generation_logits for response in responses),
            axis=0,
        ),
        generated_classes=np.stack(
            tuple(response.generated_classes for response in responses),
            axis=0,
        ),
        generated_token_ids=np.stack(
            tuple(response.generated_token_ids for response in responses),
            axis=0,
        ),
        decoded_actions=np.stack(
            tuple(response.decoded_action for response in responses),
            axis=0,
        ),
        prompt_input_ids=clean_response.prompt_input_ids.copy(),
        teacher_input_ids=clean_response.teacher_input_ids.copy(),
        action_token_start=ACTION_TOKEN_START,
        action_token_end=ACTION_TOKEN_END,
        vocab_size=int(model.vocab_size),
        bin_centers=bin_centers,
        action_low=action_low,
        action_high=action_high,
        action_unnormalize_mask=action_mask,
    )


def _failure_record(
    *,
    output_dir: Path,
    stage: str,
    error: BaseException,
    completed_keys: Sequence[tuple[str, int]],
    input_sha256: Mapping[str, str],
    asset_restore_status: Mapping[str, bool],
) -> None:
    try:
        write_json_atomically(
            {
                "schema_version": (
                    TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION
                ),
                "status": "audit_invalid",
                "failed_stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "completed_keys": [
                    {"variant": variant, "state_id": state_id}
                    for variant, state_id in completed_keys
                ],
                "input_sha256": dict(input_sha256),
                "asset_restore_status": dict(asset_restore_status),
            },
            output_path=output_dir / "audit_failed.json",
        )
    except BaseException as record_error:
        print(
            f"[GATE-6G-SMOKE] 无法保存best-effort失败记录: {record_error}",
            file=sys.stderr,
        )


def run_terminal_deployment_response_smoke(
    cfg: TerminalDeploymentResponseConfig,
) -> Path:
    """采集state 0双终态C/A/B事实并返回独立复核后的成功manifest。"""

    _validate_config(cfg)
    verify_executing_commit(cfg.code_commit)
    if loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6g smoke进程加载了legacy optimizer")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 6g state 0 smoke需要真实CUDA device")
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Gate 6g smoke输出目录必须为空: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = "input_resolution"
    completed_keys: list[tuple[str, int]] = []
    asset_restore_status: dict[str, bool] = {"xml": False, "texture": False}
    transaction: Optional[RuntimeAssetTransaction] = None
    backup_paths: tuple[Path, ...] = ()
    input_sha256: dict[str, str] = {}
    clean_env: Any = None
    deployment_env: Any = None
    try:
        inputs = resolve_terminal_deployment_inputs(
            action_spectral_manifest_path=cfg.action_spectral_manifest_path,
            action_only_manifest_path=cfg.action_only_manifest_path,
            production_support_path=cfg.production_support_path,
        )
        preflight_path = Path(cfg.rebake_preflight_manifest_path).resolve()
        preflight_decision = evaluate_terminal_rebake_preflight_bundle(
            preflight_path
        )
        if not preflight_decision.gate_pass:
            raise RuntimeError(
                "re-bake preflight未通过独立复核: "
                + "; ".join(preflight_decision.failures)
            )
        preflight_manifest = json.loads(
            preflight_path.read_text(encoding="utf-8")
        )
        _validate_preflight_binding(preflight_manifest, inputs)
        input_sha256 = {
            "production_support": inputs.production_support_sha256,
            "rebake_preflight_manifest": _file_sha256(preflight_path),
            **{
                f"{variant}_formal_manifest": terminal.manifest_sha256
                for variant, terminal in inputs.variants.items()
            },
            **{
                f"{variant}_parameter": terminal.parameter_sha256
                for variant, terminal in inputs.variants.items()
            },
            **{
                f"{variant}_bake": terminal.baked_texture_sha256
                for variant, terminal in inputs.variants.items()
            },
        }

        asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
        xml_path = Path(asset["xml"]).resolve()
        mesh_path = Path(asset["mesh"]).resolve()
        texture_path = Path(asset["texture"]).resolve()
        checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
        for path in (xml_path, mesh_path, texture_path, checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(path)
        clean_xml_sha256 = _file_sha256(xml_path)
        clean_texture_sha256 = _file_sha256(texture_path)

        stage = "model_renderer_initialization"
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
        model: Any = get_model(model_cfg)
        model.eval()
        model.requires_grad_(False)
        processor: Any = get_processor(model_cfg)
        image_preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
            model=model,
            processor=processor,
        )
        model_height, model_width = image_preprocessor.output_size
        if model_height != model_width:
            raise RuntimeError("Gate 6g只支持正方形checkpoint输入")
        policy_view_transform: DifferentiablePolicyViewTransform = (
            build_policy_view_transform(
                source_resolution=POLICY_SOURCE_RESOLUTION,
                model_input_resolution=model_height,
            )
        )
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=parse_mesh_scale(xml_path),
            epsilon=EXPECTED_SURFACE_EPSILON,
            texture_parameterization="fixed_support",
            fixed_support_path=inputs.production_support_path,
        ).to(model.device)

        benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
        task_suite = benchmark_class()
        task = task_suite.get_task(cfg.task_id)
        initial_states = task_suite.get_task_init_states(cfg.task_id)
        initial_state = initial_states[cfg.state_id]
        initial_sha256 = state_fingerprint(initial_state)
        if initial_sha256 != inputs.state_fingerprints[cfg.state_id]:
            raise RuntimeError("state 0 initial fingerprint与formal训练不一致")
        prompt: str
        task_description: str

        set_seed_everywhere(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        transaction = RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=texture_path,
            object_name=cfg.object_name,
            backup_tag=f"gate6g_smoke_{cfg.code_commit[:12]}",
            install_process_handlers=True,
        )
        backup_paths = tuple(
            path
            for path in (
                transaction.xml_backup_path,
                transaction.real_texture_backup_path,
            )
            if path is not None
        )

        stage = "clean_and_training_paths"
        transaction.restore(context="Gate 6g smoke clean", remove_backups=False)
        clean_env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=POLICY_SOURCE_RESOLUTION,
        )
        prompt = _prompt(task_description)
        clean_observation = _settle_environment(
            clean_env,
            initial_state,
            num_steps_wait=cfg.num_steps_wait,
            model_family=cfg.model_family,
            get_dummy_action=get_libero_dummy_action,
        )
        clean_source_rgb = get_libero_image(
            clean_observation,
            POLICY_SOURCE_RESOLUTION,
        )
        clean_source_tensor = _rgb_tensor(clean_source_rgb, device=model.device)
        clean_exact = build_exact_deployment_view_stages(
            clean_source_rgb,
            specification=policy_view_transform.specification,
        )
        clean_effective = policy_view_transform.build_effective_view(
            clean_source_tensor
        )
        clean_processor = _processor_pair(
            processor=processor,
            image_preprocessor=image_preprocessor,
            prompt=prompt,
            effective_view=clean_effective,
            device=model.device,
        )
        if not np.array_equal(
            clean_processor.effective_rgb,
            clean_exact.effective_view_rgb,
        ):
            raise RuntimeError("Clean BPDA Effective View与exact deployment不一致")
        clean_response = capture_openvla_action_response(
            model,
            prompt_inputs=clean_processor.prompt_inputs,
            pixel_values=clean_processor.official_pixels,
            clean_teacher_input_ids=None,
            pad_token_id=int(processor.tokenizer.pad_token_id),
            unnorm_key=cfg.unnorm_key,
        )
        clean_teacher_ids = torch.from_numpy(
            clean_response.teacher_input_ids
        ).unsqueeze(0).to(model.device)

        clean_poses = find_target_body_poses(
            clean_env,
            asset["search"],
            model.device,
        )
        if not clean_poses or any(pose.body_name is None for pose in clean_poses):
            raise RuntimeError("Clean state共享纹理实例不完整")
        clean_body_ids = tuple(pose.body_id for pose in clean_poses)
        clean_body_names = tuple(str(pose.body_name) for pose in clean_poses)
        clean_roots = tuple(
            TargetInstanceRoot(body_id=pose.body_id, body_name=str(pose.body_name))
            for pose in clean_poses
        )
        training_paths: dict[str, _AdversarialTrainingPath] = {}
        with static_scene_evidence_transaction(
            clean_env,
            clean_body_ids,
        ) as clean_static_transaction:
            transaction_poses = find_target_body_poses(
                clean_env,
                asset["search"],
                model.device,
            )
            if tuple(pose.body_id for pose in transaction_poses) != clean_body_ids:
                raise RuntimeError("Clean static transaction实例集合改变")
            clean_segmentation = capture_instance_segmentation(
                clean_env,
                clean_roots,
                camera_name="agentview",
                resolution=POLICY_SOURCE_RESOLUTION,
            )
            mujoco_alpha = clean_segmentation.parsed.instance_alpha.to(
                device=model.device,
                dtype=torch.float32,
            )
            instances = _build_instances(
                clean_env,
                transaction_poses,
                device=model.device,
            )
            for variant in TERMINAL_RESPONSE_VARIANTS:
                terminal_input = inputs.variants[variant]
                renderer.load_adversarial_texture(terminal_input.parameter_path)
                with torch.no_grad():
                    rendered = render_shared_texture_instances(
                        renderer,
                        instances,
                        resolution=(
                            POLICY_SOURCE_RESOLUTION,
                            POLICY_SOURCE_RESOLUTION,
                        ),
                    )
                    composition = compose_visibility_masked_renderer_delta(
                        clean_source_tensor,
                        mujoco_alpha,
                        rendered.adversarial_rgb,
                        rendered.clean_rgb,
                        rendered.visibility_mask,
                    )
                    training_effective = (
                        policy_view_transform.build_effective_view(
                            composition.composited_rgb
                        )
                    )
                training_processor = _processor_pair(
                    processor=processor,
                    image_preprocessor=image_preprocessor,
                    prompt=prompt,
                    effective_view=training_effective,
                    device=model.device,
                )
                training_response = capture_openvla_action_response(
                    model,
                    prompt_inputs=training_processor.prompt_inputs,
                    pixel_values=training_processor.training_exact_pixels,
                    clean_teacher_input_ids=clean_teacher_ids,
                    pad_token_id=int(processor.tokenizer.pad_token_id),
                    unnorm_key=cfg.unnorm_key,
                )
                training_paths[variant] = _AdversarialTrainingPath(
                    processor=training_processor,
                    response=training_response,
                    saturated_pixel_fraction=(
                        composition.saturated_pixel_fraction
                    ),
                    saturated_channel_fraction=(
                        composition.saturated_channel_fraction
                    ),
                )
        if not clean_static_transaction.verified:
            raise RuntimeError("Clean static scene transaction未通过")
        clean_static_sha256 = (
            clean_static_transaction.before.fingerprint_sha256
        )
        clean_env.close()
        clean_env = None

        stage = "deployment_paths"
        cases: list[dict[str, Any]] = []
        deployment_diagnostics: dict[str, Any] = {}
        for variant in TERMINAL_RESPONSE_VARIANTS:
            terminal_input = inputs.variants[variant]
            transaction.restore(
                context=f"Gate 6g smoke {variant} pre-activate",
                remove_backups=False,
            )
            mirrored = transaction.activate_texture(
                terminal_input.baked_texture_path,
                mirror_real_texture=True,
            )
            active_texture_sha256 = _file_sha256(texture_path)
            if not mirrored or active_texture_sha256 != (
                terminal_input.baked_texture_sha256
            ):
                raise RuntimeError(f"{variant} Active Texture未绑定formal bake")
            deployment_env, deployment_description = get_libero_env(
                task,
                cfg.model_family,
                resolution=POLICY_SOURCE_RESOLUTION,
            )
            if deployment_description != task_description:
                raise RuntimeError("C/B task description不一致")
            deployment_observation = _settle_environment(
                deployment_env,
                initial_state,
                num_steps_wait=cfg.num_steps_wait,
                model_family=cfg.model_family,
                get_dummy_action=get_libero_dummy_action,
            )
            deployment_source_rgb = get_libero_image(
                deployment_observation,
                POLICY_SOURCE_RESOLUTION,
            )
            deployment_source_tensor = _rgb_tensor(
                deployment_source_rgb,
                device=model.device,
            )
            deployment_exact = build_exact_deployment_view_stages(
                deployment_source_rgb,
                specification=policy_view_transform.specification,
            )
            deployment_effective = policy_view_transform.build_effective_view(
                deployment_source_tensor
            )
            deployment_processor = _processor_pair(
                processor=processor,
                image_preprocessor=image_preprocessor,
                prompt=prompt,
                effective_view=deployment_effective,
                device=model.device,
            )
            if not np.array_equal(
                deployment_processor.effective_rgb,
                deployment_exact.effective_view_rgb,
            ):
                raise RuntimeError(
                    f"{variant} B路径BPDA Effective View与exact deployment不一致"
                )
            deployment_response = capture_openvla_action_response(
                model,
                prompt_inputs=deployment_processor.prompt_inputs,
                pixel_values=deployment_processor.official_pixels,
                clean_teacher_input_ids=clean_teacher_ids,
                pad_token_id=int(processor.tokenizer.pad_token_id),
                unnorm_key=cfg.unnorm_key,
            )
            deployment_poses = find_target_body_poses(
                deployment_env,
                asset["search"],
                model.device,
            )
            deployment_body_ids = tuple(pose.body_id for pose in deployment_poses)
            deployment_body_names = tuple(
                str(pose.body_name) for pose in deployment_poses
            )
            if (
                deployment_body_ids != clean_body_ids
                or deployment_body_names != clean_body_names
            ):
                raise RuntimeError(f"{variant} C/B共享实例身份不一致")
            deployment_roots = tuple(
                TargetInstanceRoot(
                    body_id=pose.body_id,
                    body_name=str(pose.body_name),
                )
                for pose in deployment_poses
            )
            with static_scene_evidence_transaction(
                deployment_env,
                deployment_body_ids,
            ) as deployment_static_transaction:
                deployment_segmentation = capture_instance_segmentation(
                    deployment_env,
                    deployment_roots,
                    camera_name="agentview",
                    resolution=POLICY_SOURCE_RESOLUTION,
                )
            if not deployment_static_transaction.verified:
                raise RuntimeError(f"{variant} B路径static transaction未通过")
            deployment_path = _DeploymentPath(
                processor=deployment_processor,
                response=deployment_response,
                segmentation=deployment_segmentation,
                static_transaction=deployment_static_transaction,
                body_ids=deployment_body_ids,
                body_names=deployment_body_names,
            )
            deployment_env.close()
            deployment_env = None
            transaction.restore(
                context=f"Gate 6g smoke {variant} post-deployment",
                remove_backups=False,
            )
            xml_restored = _file_sha256(xml_path) == clean_xml_sha256
            texture_restored = (
                _file_sha256(texture_path) == clean_texture_sha256
            )
            asset_restore_status = {
                "xml": xml_restored,
                "texture": texture_restored,
            }
            if not xml_restored or not texture_restored:
                raise RuntimeError(f"{variant} Runtime Asset恢复失败")

            evidence = _build_case_evidence(
                variant=variant,
                state_id=cfg.state_id,
                clean_processor=clean_processor,
                training_path=training_paths[variant],
                deployment_path=deployment_path,
                clean_response=clean_response,
                clean_segmentation=clean_segmentation,
                model=model,
                unnorm_key=cfg.unnorm_key,
            )
            npz_path = (
                output_dir
                / "arrays"
                / variant
                / f"state_{cfg.state_id:02d}.npz"
            )
            npz_sha256 = write_terminal_response_npz(
                evidence,
                output_path=npz_path,
            )
            response_evaluation = evaluate_terminal_response_evidence(evidence)
            cases.append(
                {
                    "variant": variant,
                    "state_id": cfg.state_id,
                    "npz_relative_path": str(npz_path.relative_to(output_dir)),
                    "npz_sha256": npz_sha256,
                    "initial_state_sha256": initial_sha256,
                    "clean_static_scene_sha256": clean_static_sha256,
                    "deployment_static_scene_sha256": (
                        deployment_static_transaction.before.fingerprint_sha256
                    ),
                    "transaction_verified": bool(
                        clean_static_transaction.verified
                        and deployment_static_transaction.verified
                    ),
                    "asset_restore_verified": bool(
                        xml_restored and texture_restored
                    ),
                    "response_evaluation": (
                        terminal_response_evaluation_record(
                            response_evaluation
                        )
                    ),
                }
            )
            deployment_diagnostics[variant] = {
                "body_ids": list(deployment_body_ids),
                "body_names": list(deployment_body_names),
                "active_texture_sha256": active_texture_sha256,
                "saturated_pixel_fraction": (
                    training_paths[variant].saturated_pixel_fraction
                ),
                "saturated_channel_fraction": (
                    training_paths[variant].saturated_channel_fraction
                ),
                "segmentation_backend": asdict(
                    deployment_segmentation.backend
                ),
            }
            completed_keys.append((variant, cfg.state_id))

        stage = "asset_transaction_close"
        transaction.close(context="Gate 6g smoke final")
        transaction = None
        backup_paths_removed = bool(backup_paths) and not any(
            path.exists() for path in backup_paths
        )
        asset_restore_status = {
            "xml": _file_sha256(xml_path) == clean_xml_sha256,
            "texture": _file_sha256(texture_path) == clean_texture_sha256,
        }
        if not all(asset_restore_status.values()):
            raise RuntimeError("Gate 6g smoke最终资产恢复失败")
        if not backup_paths_removed:
            raise RuntimeError("Gate 6g smoke未验证Runtime Asset backup删除")
        if loaded_legacy_optimizer_modules():
            raise RuntimeError("Gate 6g smoke运行期间加载了legacy optimizer")

        stage = "manifest_publication"
        config_payload = asdict(cfg)
        action_stats = model.get_action_stats(cfg.unnorm_key)
        manifest = {
            "schema_version": (
                TERMINAL_DEPLOYMENT_RESPONSE_SMOKE_BUNDLE_SCHEMA_VERSION
            ),
            "status": "complete",
            "code_commit": cfg.code_commit,
            "config_sha256": canonical_json_sha256(config_payload),
            "expected_variants": list(TERMINAL_RESPONSE_VARIANTS),
            "expected_state_ids": [cfg.state_id],
            "state_fingerprints": [initial_sha256],
            "response_authority": dict(
                TERMINAL_RESPONSE_AUTHORITY_CONTRACT
            ),
            "terminal_pairing": preflight_manifest["terminal_pairing"],
            "cases": cases,
            "provenance": {
                "configuration": config_payload,
                "command": " ".join(
                    shlex.quote(argument) for argument in sys.argv
                ),
                "input_sha256": input_sha256,
                "pretrained_checkpoint": str(checkpoint_path),
                "checkpoint_fingerprints": _checkpoint_fingerprints(
                    checkpoint_path
                ),
                "processor_specification": asdict(image_preprocessor),
                "processor_classes": {
                    "processor": (
                        f"{type(processor).__module__}."
                        f"{type(processor).__qualname__}"
                    ),
                    "image_processor": (
                        f"{type(processor.image_processor).__module__}."
                        f"{type(processor.image_processor).__qualname__}"
                    ),
                    "tokenizer": (
                        f"{type(processor.tokenizer).__module__}."
                        f"{type(processor.tokenizer).__qualname__}"
                    ),
                },
                "policy_view_specification": asdict(
                    policy_view_transform.specification
                ),
                "task_description": task_description,
                "prompt": prompt,
                "generation": {
                    "max_new_tokens": int(
                        model.get_action_dim(cfg.unnorm_key)
                    ),
                    "do_sample": False,
                    "pad_token_id": int(processor.tokenizer.pad_token_id),
                },
                "action_codec": {
                    "vocab_size": int(model.vocab_size),
                    "bin_centers": np.asarray(model.bin_centers).tolist(),
                    "q01": np.asarray(action_stats["q01"]).tolist(),
                    "q99": np.asarray(action_stats["q99"]).tolist(),
                    "mask": np.asarray(
                        action_stats.get(
                            "mask",
                            np.ones_like(action_stats["q01"], dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    ).tolist(),
                },
                "object_asset_paths": {
                    "xml": str(xml_path),
                    "mesh": str(mesh_path),
                    "texture": str(texture_path),
                },
                "object_asset_clean_sha256": {
                    "xml": clean_xml_sha256,
                    "texture": clean_texture_sha256,
                    "mesh": _file_sha256(mesh_path),
                },
                "clean_body_ids": list(clean_body_ids),
                "clean_body_names": list(clean_body_names),
                "clean_segmentation_backend": asdict(
                    clean_segmentation.backend
                ),
                "deployment_diagnostics": deployment_diagnostics,
                "asset_restore_status": asset_restore_status,
                "runtime_asset_backup_paths": [
                    str(path) for path in backup_paths
                ],
                "runtime_asset_backup_paths_removed": backup_paths_removed,
                "framework_versions": {
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "torch": torch.__version__,
                },
            },
        }
        manifest_path = output_dir / "terminal_response_smoke_manifest.json"
        manifest_sha256 = publish_terminal_response_smoke_manifest(
            manifest,
            output_path=manifest_path,
        )
    except BaseException as error:
        for environment_name, environment in (
            ("clean", clean_env),
            ("deployment", deployment_env),
        ):
            if environment is None:
                continue
            try:
                environment.close()
            except BaseException as close_error:
                print(
                    f"[GATE-6G-SMOKE] {environment_name} env关闭失败: "
                    f"{close_error}",
                    file=sys.stderr,
                )
        if transaction is not None:
            try:
                transaction.close(context="Gate 6g smoke exception")
            except BaseException as restore_error:
                print(
                    f"[GATE-6G-SMOKE] 异常恢复资产失败: {restore_error}",
                    file=sys.stderr,
                )
        _failure_record(
            output_dir=output_dir,
            stage=stage,
            error=error,
            completed_keys=completed_keys,
            input_sha256=input_sha256,
            asset_restore_status=asset_restore_status,
        )
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "case_count": len(cases),
            },
            sort_keys=True,
        )
    )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_terminal_deployment_response_smoke(_parse_args(argv))


if __name__ == "__main__":
    main()
