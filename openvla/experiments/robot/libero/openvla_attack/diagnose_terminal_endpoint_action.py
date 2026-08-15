"""采集 Gate 6i 双正式终态的 Action gradient/response 权威证据。

本 runner 只读恢复 Action+Spectral 与严格匹配的 Action-only compact 终态，
把二者 scatter 到同一个全顶点 ``GeometryVertexTextureParameterization``。对每个
终态和 states 0--9，它沿已经通过 Objective GPU Audit 的训练 Renderer Delta
Composition，仅计算 teacher-forced Untargeted Clean-Action Margin Hinge。

关键边界：raw ``G_s = dL/dDelta_surface`` 不读取 parameter ``.grad``。终态可能
位于 Surface-Linf 边界，functional projection 的 Jacobian 会改变参数梯度；本
命令因此捕获每个共享纹理实例实际进入 renderer 的 Surface Delta 梯度，先按
实例求和，再用严格 face-corner seam mapping scatter-add 回 OBJ geometry 顶点。

20 份梯度完成后，每个终态分别从同一 realized endpoint 出发，复用正式 trainer
的 ``surface_normalized_step_`` 产生 Dense/Support 单步。随后采集 baseline、
Dense、Support 共 60 份静态动作响应。Feature、wrist、OFT、legacy optimizer、
rollout 和重新训练均不得进入本进程。成功 manifest 最后才由纯 CPU evaluator
原子发布。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
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
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.dense_seed_audit import (  # noqa: E402
    load_dense_seed_gradient_evidence,
    summarize_dense_seed_gradient_evidence,
)
from openvla_attack.fixed_support_source_training import (  # noqa: E402
    verify_executing_commit,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    render_shared_texture_instances,
)
from openvla_attack.objective import (  # noqa: E402
    ACTION_TOKEN_START,
    UntargetedCleanActionMarginHinge,
    untargeted_clean_action_margin_hinge,
)
from openvla_attack.paired_source_gate import state_fingerprint  # noqa: E402
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiablePolicyViewTransform,
    build_policy_view_transform,
)
from openvla_attack.production_support import (  # noqa: E402
    FrozenProductionSupport,
    load_production_support_artifact,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.terminal_deployment_response_inputs import (  # noqa: E402
    TerminalDeploymentInputs,
    resolve_terminal_deployment_inputs,
)
from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    ENDPOINT_NAMES,
    EXPECTED_STATE_IDS,
    RESPONSE_ARMS,
    EndpointGradientEvidence,
    EndpointResponseEvidence,
    SurfaceCounterfactualStep,
    SurfaceCounterfactualStepStats,
    aggregate_state_gradients,
    compute_action_hinge,
    write_endpoint_gradient_npz,
    write_endpoint_response_npz,
)
from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    ENDPOINT_BUNDLE_SCHEMA_VERSION,
    FORMAL_SURFACE_EPSILON,
    FORMAL_SURFACE_STEP,
    EndpointStepEvidence,
    array_sha256,
    evaluate_endpoint_step_evidence,
    file_sha256,
    frozen_bundle_configuration,
    json_sha256,
    publish_terminal_endpoint_bundle,
    write_endpoint_step_npz,
)
from openvla_attack.terminal_openvla_response import (  # noqa: E402
    OpenVLAActionResponse,
    bfloat16_tensor_to_uint16_bits,
    capture_openvla_action_response,
)
from openvla_attack.texture_parameterization import (  # noqa: E402
    GeometryVertexTextureParameterization,
    SurfaceStepStats,
    surface_normalized_step_,
)
from openvla_attack.visibility_capture import (  # noqa: E402
    capture_instance_segmentation,
    static_scene_evidence_transaction,
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
class TerminalEndpointActionConfig:
    """Gate 6i 正式 GPU runner CLI；inventory 固定为双终态和 states 0--9。"""

    pretrained_checkpoint: str
    action_spectral_manifest_path: str
    action_only_manifest_path: str
    production_support_path: str
    dense_seed_metrics_path: str
    output_dir: str
    code_commit: str
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    num_steps_wait: int = 10
    seed: int = 7
    unnorm_key: Optional[str] = "libero_spatial_no_noops"
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass(frozen=True)
class _CleanTarget:
    """Dense Seed 固定的 clean Action target 与 state 身份。"""

    fingerprint: str
    token_ids: NDArray[np.int64]
    classes: NDArray[np.int64]


@dataclass(frozen=True)
class _ProcessorInput:
    """同一 exact uint8 forward 的可微 BF16 与 prompt/teacher 输入。"""

    effective_rgb: NDArray[np.uint8]
    prompt_inputs: Mapping[str, torch.Tensor]
    pixel_values: torch.Tensor
    teacher_input_ids: torch.Tensor


@dataclass(frozen=True)
class _EndpointSurfaces:
    """一个终态的 realized、Dense-step、Support-step surface arrays。"""

    realized: NDArray[np.float32]
    dense: NDArray[np.float32]
    support: NDArray[np.float32]
    step_evidence: EndpointStepEvidence


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> TerminalEndpointActionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--action_spectral_manifest_path", required=True)
    parser.add_argument("--action_only_manifest_path", required=True)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--dense_seed_metrics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    return TerminalEndpointActionConfig(**vars(parser.parse_args(argv)))


def _validate_config(cfg: TerminalEndpointActionConfig) -> None:
    if len(cfg.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in cfg.code_commit
    ):
        raise ValueError("code_commit 必须是40位小写Git SHA")
    if cfg.task_suite_name != "libero_spatial" or cfg.task_id != 0:
        raise ValueError("Gate 6i只接受LIBERO Spatial task 0")
    if cfg.object_name != "akita_black_bowl":
        raise ValueError("Gate 6i只接受akita_black_bowl")
    if cfg.num_steps_wait < 0 or not cfg.center_crop:
        raise ValueError("Gate 6i要求非负wait且center_crop=True")
    if cfg.load_in_8bit or cfg.load_in_4bit:
        raise ValueError("Gate 6i固定正式BF16 checkpoint，禁止量化加载")


def _loaded_legacy_optimizer_modules() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "openvla_attack.optimization"
            or name.endswith(".openvla_attack.optimization")
        )
    )


def _checkpoint_fingerprints(checkpoint_path: Path) -> dict[str, str]:
    """绑定配置内容及大权重文件的名称/大小inventory。"""

    fingerprints = {
        name: file_sha256(checkpoint_path / name)
        for name in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "dataset_statistics.json",
            "model.safetensors.index.json",
        )
        if (checkpoint_path / name).is_file()
    }
    weights = sorted(
        tuple(checkpoint_path.glob("*.safetensors"))
        + tuple(checkpoint_path.glob("*.bin"))
    )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weights
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
    """uint8 HWC RGB -> float32 device NCHW ``[1,3,H,W]``。"""

    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _effective_rgb(value: torch.Tensor) -> NDArray[np.uint8]:
    """把 final Effective View 按 exact forward 规则量化为 uint8 HWC。"""

    if value.shape != (1, 3, 224, 224) or value.dtype != torch.float32:
        raise ValueError("Effective View必须为float32 [1,3,224,224]")
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
    cfg: TerminalEndpointActionConfig,
    get_dummy_action: Any,
) -> Any:
    env.reset()
    observation = env.set_init_state(initial_state)
    env.env.sim.forward()
    for _ in range(cfg.num_steps_wait):
        observation, _, _, _ = env.step(get_dummy_action(cfg.model_family))
    return observation


def _load_clean_targets(
    metrics_path: Path,
) -> dict[int, _CleanTarget]:
    rows = load_dense_seed_gradient_evidence(metrics_path)
    summary = summarize_dense_seed_gradient_evidence(
        rows,
        artifact_root=metrics_path.parent,
    )
    if not summary.gate_pass:
        raise RuntimeError(
            "Dense Seed source evidence复核失败: " + "; ".join(summary.failures)
        )
    targets: dict[int, _CleanTarget] = {}
    for row in rows:
        evidence = row.objective_evidence
        state_id = int(evidence.state_id)
        targets[state_id] = _CleanTarget(
            fingerprint=evidence.initial_state_sha256,
            token_ids=np.asarray(
                evidence.clean_action_token_ids,
                dtype=np.int64,
            ),
            classes=np.asarray(evidence.clean_classes, dtype=np.int64),
        )
    if tuple(sorted(targets)) != EXPECTED_STATE_IDS:
        raise RuntimeError("Dense Seed source必须恰好包含states 0--9")
    return targets


def _processor_input(
    *,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    prompt: str,
    effective_view: torch.Tensor,
    clean_target: _CleanTarget,
    device: torch.device,
) -> _ProcessorInput:
    """构造 forward 精确、backward 连续的 BF16 teacher 输入。"""

    effective_rgb = _effective_rgb(effective_view)
    official_inputs: MutableMapping[str, torch.Tensor] = processor(
        prompt,
        images=Image.fromarray(effective_rgb, mode="RGB"),
    ).to(device)
    ensure_trailing_empty_token(official_inputs)
    if "pixel_values" not in official_inputs:
        raise RuntimeError("checkpoint processor缺少pixel_values")
    official_pixels = official_inputs["pixel_values"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    bpda_pixels = image_preprocessor.build_fused_pixel_values(
        effective_view
    ).to(torch.bfloat16)
    if not torch.equal(official_pixels, bpda_pixels):
        difference = float(
            (official_pixels.float() - bpda_pixels.float()).abs().amax().item()
        )
        raise RuntimeError(f"official/BPDA BF16 forward不一致: Linf={difference}")
    prompt_inputs = {
        name: tensor.detach().clone()
        for name, tensor in official_inputs.items()
        if name != "pixel_values"
    }
    clean_tokens = torch.as_tensor(
        clean_target.token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    teacher_ids = torch.cat((prompt_inputs["input_ids"], clean_tokens), dim=1)
    return _ProcessorInput(
        effective_rgb=effective_rgb,
        prompt_inputs=prompt_inputs,
        pixel_values=bpda_pixels,
        teacher_input_ids=teacher_ids,
    )


def _copy_endpoint_sources(
    inputs: TerminalDeploymentInputs,
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_dir = output_dir / "inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINT_NAMES:
        terminal = inputs.variants[endpoint]
        manifest_copy = source_dir / f"{endpoint}_formal_manifest.json"
        parameter_copy = source_dir / f"{endpoint}_compact_parameter.pt"
        for source, destination in (
            (terminal.manifest_path, manifest_copy),
            (terminal.parameter_path, parameter_copy),
        ):
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(source, destination)
        if file_sha256(manifest_copy) != terminal.manifest_sha256:
            raise RuntimeError(f"{endpoint} formal manifest复制后SHA漂移")
        if file_sha256(parameter_copy) != terminal.parameter_sha256:
            raise RuntimeError(f"{endpoint} compact parameter复制后SHA漂移")
        records.append(
            {
                "endpoint": endpoint,
                "training_code_commit": terminal.training_code_commit,
                "formal_manifest_relative_path": str(
                    manifest_copy.relative_to(output_dir)
                ),
                "formal_manifest_sha256": terminal.manifest_sha256,
                "compact_parameter_relative_path": str(
                    parameter_copy.relative_to(output_dir)
                ),
                "compact_parameter_sha256": terminal.parameter_sha256,
            }
        )
    return records


def _realize_endpoints(
    inputs: TerminalDeploymentInputs,
    support: FrozenProductionSupport,
    renderer: DifferentiableRenderer,
) -> dict[str, NDArray[np.float32]]:
    """把 compact Support 坐标无损 scatter 到全 geometry 顶点并固化投影。"""

    parameter = renderer.get_texture_param()
    parameterization = renderer.surface_parameterization
    if not isinstance(parameterization, GeometryVertexTextureParameterization):
        raise RuntimeError("Gate 6i renderer必须为GeometryVertex参数化")
    realized: dict[str, NDArray[np.float32]] = {}
    support_indices = torch.as_tensor(
        support.support_vertex_indices,
        dtype=torch.long,
        device=parameter.device,
    )
    for endpoint in ENDPOINT_NAMES:
        terminal = inputs.variants[endpoint]
        compact = torch.load(
            terminal.parameter_path,
            map_location=parameter.device,
            weights_only=True,
        )
        if not isinstance(compact, torch.Tensor) or compact.shape != (
            len(support.support_vertex_indices),
            3,
        ):
            raise RuntimeError(f"{endpoint} compact parameter shape/type无效")
        full = torch.zeros_like(parameter)
        full.index_copy_(0, support_indices, compact.to(full.dtype))
        with torch.no_grad():
            parameter.copy_(full)
            parameterization.project_coefficients_()
            value = parameterization.geometry_delta().detach().float().cpu()
        array = np.ascontiguousarray(value.numpy(), dtype=np.float32)
        if array.shape != (support.num_geometry_vertices, 3):
            raise RuntimeError(f"{endpoint} realized endpoint geometry shape无效")
        realized[endpoint] = array
    return realized


def _render_training_composition(
    *,
    env: Any,
    observation: Any,
    asset: ObjectAssetSpec,
    renderer: DifferentiableRenderer,
    model_device: torch.device,
    get_libero_image: Any,
) -> tuple[torch.Tensor, int]:
    """在冻结场景中渲染并组合全部共享纹理实例。"""

    clean_rgb = get_libero_image(observation, POLICY_SOURCE_RESOLUTION)
    clean_tensor = _rgb_tensor(clean_rgb, device=model_device)
    target_poses = find_target_body_poses(env, asset["search"], model_device)
    if not target_poses or any(pose.body_name is None for pose in target_poses):
        raise RuntimeError("未找到完整共享纹理实例")
    body_ids = tuple(pose.body_id for pose in target_poses)
    roots = tuple(
        TargetInstanceRoot(pose.body_id, str(pose.body_name))
        for pose in target_poses
    )
    with static_scene_evidence_transaction(env, body_ids):
        current_poses = find_target_body_poses(env, asset["search"], model_device)
        if tuple(pose.body_id for pose in current_poses) != body_ids:
            raise RuntimeError("static transaction内共享实例集合改变")
        segmentation = capture_instance_segmentation(
            env,
            roots,
            camera_name="agentview",
            resolution=POLICY_SOURCE_RESOLUTION,
        )
        rendered = render_shared_texture_instances(
            renderer,
            _build_instances(env, current_poses, device=model_device),
            resolution=(POLICY_SOURCE_RESOLUTION, POLICY_SOURCE_RESOLUTION),
        )
        composition = compose_visibility_masked_renderer_delta(
            clean_tensor,
            segmentation.parsed.instance_alpha.to(
                device=model_device,
                dtype=torch.float32,
            ),
            rendered.adversarial_rgb,
            rendered.clean_rgb,
            rendered.visibility_mask,
        )
    return composition.composited_rgb, len(body_ids)


def _capture_surface_gradient(
    *,
    cfg: TerminalEndpointActionConfig,
    endpoint: str,
    state_id: int,
    initial_state: Any,
    clean_target: _CleanTarget,
    endpoint_surface: NDArray[np.float32],
    task: Any,
    task_description: str,
    asset: ObjectAssetSpec,
    model: Any,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    view_transform: DifferentiablePolicyViewTransform,
    renderer: DifferentiableRenderer,
    get_dummy_action: Any,
    get_libero_env: Any,
    get_libero_image: Any,
) -> NDArray[np.float32]:
    """采集一个 endpoint/state 的 raw ``dL/dDelta_surface``。"""

    if state_fingerprint(initial_state) != clean_target.fingerprint:
        raise RuntimeError(f"state {state_id} fingerprint与Dense Seed不一致")
    parameter = renderer.get_texture_param()
    with torch.no_grad():
        parameter.copy_(torch.from_numpy(endpoint_surface).to(parameter.device))
    parameter.grad = None
    env, observed_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    if observed_description != task_description:
        env.close()
        raise RuntimeError("LIBERO task description改变")
    try:
        observation = _settle_environment(
            env,
            initial_state,
            cfg=cfg,
            get_dummy_action=get_dummy_action,
        )
        with renderer.capture_surface_delta_gradients() as capture:
            composed, instance_count = _render_training_composition(
                env=env,
                observation=observation,
                asset=asset,
                renderer=renderer,
                model_device=model.device,
                get_libero_image=get_libero_image,
            )
            effective = view_transform.build_stages(composed).effective_view
            model_input = _processor_input(
                processor=processor,
                image_preprocessor=image_preprocessor,
                prompt=_prompt(task_description),
                effective_view=effective,
                clean_target=clean_target,
                device=model.device,
            )
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=model_input.teacher_input_ids,
                    attention_mask=torch.ones_like(model_input.teacher_input_ids),
                    pixel_values=model_input.pixel_values,
                    output_hidden_states=False,
                )
            objective: UntargetedCleanActionMarginHinge = (
                untargeted_clean_action_margin_hinge(
                    outputs.logits,
                    model_input.teacher_input_ids,
                )
            )
            observed_classes = objective.clean_classes.detach().cpu().numpy()
            if not np.array_equal(observed_classes, clean_target.classes):
                raise RuntimeError("teacher clean classes与Dense Seed target漂移")
            objective.loss.backward()
        if len(capture.tensors) != instance_count:
            raise RuntimeError("Surface gradient捕获数量与共享实例数不一致")
        render_gradient = capture.summed_gradient().to(torch.float32)
        mapping = renderer.get_render_to_geometry_mapping().to(torch.long)
        if render_gradient.shape != (mapping.numel(), 3):
            raise RuntimeError("renderer Surface gradient与seam mapping shape不一致")
        geometry_gradient = torch.zeros_like(parameter, dtype=torch.float32)
        geometry_gradient.index_add_(0, mapping, render_gradient)
        if not bool(torch.isfinite(geometry_gradient).all()) or not bool(
            torch.any(geometry_gradient != 0.0)
        ):
            raise RuntimeError("raw geometry Surface gradient非finite或全零")
        return np.ascontiguousarray(
            geometry_gradient.detach().cpu().numpy(),
            dtype=np.float32,
        )
    finally:
        env.close()
        parameter.grad = None


def _authoritative_arm_step(
    parameterization: GeometryVertexTextureParameterization,
    *,
    realized: NDArray[np.float32],
    masked_gradient: NDArray[np.float32],
) -> SurfaceCounterfactualStep:
    """复用正式 trainer 更新，并导出 CPU 可逐数组复算的 step。"""

    parameter = parameterization.coefficients
    gradient = torch.from_numpy(masked_gradient).to(parameter.device)
    with torch.no_grad():
        parameter.copy_(torch.from_numpy(realized).to(parameter.device))
        before = parameterization.geometry_delta().detach().clone()
        stats: SurfaceStepStats = surface_normalized_step_(
            parameterization,
            gradient,
            FORMAL_SURFACE_STEP,
        )
        after = parameterization.geometry_delta().detach().clone()
    normalized = -gradient * np.float32(stats.parameter_scale)
    executed = after - before
    if stats.step_cap_scale <= 0.0:
        raise RuntimeError("正式 SurfaceStepStats.step_cap_scale 无效")
    projected = executed / np.float32(stats.step_cap_scale)
    residual = executed - normalized

    def array(value: torch.Tensor) -> NDArray[np.float32]:
        return np.ascontiguousarray(value.float().cpu().numpy(), dtype=np.float32)

    return SurfaceCounterfactualStep(
        realized_endpoint=array(before),
        masked_gradient=array(gradient),
        unconstrained_normalized_step=array(normalized),
        projected_step=array(projected),
        executed_step=array(executed),
        projection_residual=array(residual),
        stats=SurfaceCounterfactualStepStats(
            endpoint_projection_scale=1.0,
            direction_surface_max=stats.direction_surface_max,
            parameter_scale=stats.parameter_scale,
            projection_scale=stats.projection_scale,
            step_cap_scale=stats.step_cap_scale,
            actual_surface_step=stats.actual_surface_step,
            max_abs_delta=stats.max_abs_delta,
        ),
    )


def _build_endpoint_surfaces(
    *,
    endpoint: str,
    realized: NDArray[np.float32],
    gradients: Sequence[NDArray[np.float32]],
    support: FrozenProductionSupport,
    renderer: DifferentiableRenderer,
) -> _EndpointSurfaces:
    aggregate = aggregate_state_gradients(
        np.stack(tuple(gradients), axis=0).astype(np.float32, copy=False)
    )
    parameterization = renderer.surface_parameterization
    if not isinstance(parameterization, GeometryVertexTextureParameterization):
        raise RuntimeError("Gate 6i step要求GeometryVertex参数化")
    support_gradient = aggregate.copy()
    support_gradient[~support.support_mask] = np.float32(0.0)
    dense_step = _authoritative_arm_step(
        parameterization,
        realized=realized,
        masked_gradient=aggregate,
    )
    support_step = _authoritative_arm_step(
        parameterization,
        realized=realized,
        masked_gradient=support_gradient,
    )
    dense_surface = np.ascontiguousarray(
        dense_step.realized_endpoint + dense_step.executed_step,
        dtype=np.float32,
    )
    support_surface = np.ascontiguousarray(
        support_step.realized_endpoint + support_step.executed_step,
        dtype=np.float32,
    )
    evidence = EndpointStepEvidence(
        endpoint=endpoint,
        epsilon=FORMAL_SURFACE_EPSILON,
        surface_step=FORMAL_SURFACE_STEP,
        support_mask_sha256=array_sha256(support.support_mask),
        realized_endpoint_sha256=array_sha256(realized),
        dense_surface_delta_sha256=array_sha256(dense_surface),
        support_surface_delta_sha256=array_sha256(support_surface),
        dense_step=dense_step,
        support_step=support_step,
    )
    decision = evaluate_endpoint_step_evidence(
        evidence,
        aggregate_gradient=aggregate,
        support_mask=support.support_mask,
    )
    if not decision.audit_valid:
        raise RuntimeError(
            "GPU authoritative step CPU复核失败: "
            + "; ".join(decision.failures)
        )
    return _EndpointSurfaces(realized, dense_surface, support_surface, evidence)


def _capture_response(
    *,
    cfg: TerminalEndpointActionConfig,
    endpoint: str,
    state_id: int,
    arm: str,
    surface: NDArray[np.float32],
    initial_state: Any,
    clean_target: _CleanTarget,
    task: Any,
    task_description: str,
    asset: ObjectAssetSpec,
    model: Any,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    view_transform: DifferentiablePolicyViewTransform,
    renderer: DifferentiableRenderer,
    get_dummy_action: Any,
    get_libero_env: Any,
    get_libero_image: Any,
) -> EndpointResponseEvidence:
    """采集一个 endpoint/state/arm 的静态动作响应。"""

    if state_fingerprint(initial_state) != clean_target.fingerprint:
        raise RuntimeError(f"state {state_id} fingerprint与Dense Seed不一致")
    parameter = renderer.get_texture_param()
    with torch.no_grad():
        parameter.copy_(torch.from_numpy(surface).to(parameter.device))
    env, observed_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    if observed_description != task_description:
        env.close()
        raise RuntimeError("LIBERO task description改变")
    try:
        observation = _settle_environment(
            env,
            initial_state,
            cfg=cfg,
            get_dummy_action=get_dummy_action,
        )
        with torch.no_grad():
            composed, _ = _render_training_composition(
                env=env,
                observation=observation,
                asset=asset,
                renderer=renderer,
                model_device=model.device,
                get_libero_image=get_libero_image,
            )
            effective = view_transform.build_stages(composed).effective_view
            model_input = _processor_input(
                processor=processor,
                image_preprocessor=image_preprocessor,
                prompt=_prompt(task_description),
                effective_view=effective,
                clean_target=clean_target,
                device=model.device,
            )
            response: OpenVLAActionResponse = capture_openvla_action_response(
                model,
                prompt_inputs=model_input.prompt_inputs,
                pixel_values=model_input.pixel_values.detach(),
                clean_teacher_input_ids=model_input.teacher_input_ids,
                pad_token_id=processor.tokenizer.pad_token_id,
                unnorm_key=cfg.unnorm_key,
            )
        if not np.array_equal(
            response.teacher_input_ids,
            model_input.teacher_input_ids.cpu().numpy(),
        ):
            raise RuntimeError("response teacher IDs漂移")
        action = compute_action_hinge(response.teacher_logits, clean_target.classes)
        return EndpointResponseEvidence(
            endpoint=endpoint,
            state_id=state_id,
            arm=arm,
            state_fingerprint=clean_target.fingerprint,
            surface_delta_sha256=array_sha256(surface),
            clean_action_token_ids=clean_target.token_ids.copy(),
            clean_classes=clean_target.classes.copy(),
            action_loss=action.action_loss,
            margins=action.margins,
            hinge_values=action.hinge_values,
            generated_token_ids=response.generated_token_ids,
            generated_classes=response.generated_classes,
            generation_logits=response.generation_logits,
            teacher_logits=response.teacher_logits,
            decoded_action=response.decoded_action,
            effective_view_rgb=model_input.effective_rgb,
            processor_bf16_bits=bfloat16_tensor_to_uint16_bits(
                model_input.pixel_values
            ),
        )
    finally:
        env.close()


def run_terminal_endpoint_action_audit(
    cfg: TerminalEndpointActionConfig,
) -> Path:
    """执行完整 Gate 6i GPU 采集并返回原子发布的成功 manifest。"""

    _validate_config(cfg)
    verify_executing_commit(cfg.code_commit)
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    set_seed_everywhere(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 6i需要真实CUDA nvdiffrast device")
    if _loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6i禁止加载legacy optimization.py")

    output_dir = Path(cfg.output_dir).resolve()
    manifest_path = output_dir / "terminal_endpoint_action_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_metrics_path = Path(cfg.dense_seed_metrics_path).resolve()
    clean_targets = _load_clean_targets(dense_metrics_path)
    inputs = resolve_terminal_deployment_inputs(
        action_spectral_manifest_path=cfg.action_spectral_manifest_path,
        action_only_manifest_path=cfg.action_only_manifest_path,
        production_support_path=cfg.production_support_path,
    )
    support = load_production_support_artifact(inputs.production_support_path)
    if not support.production_support_constructed or not support.fixed_support_frozen:
        raise RuntimeError("Production Support未constructed/frozen")
    dense_fingerprints = tuple(
        clean_targets[state_id].fingerprint for state_id in EXPECTED_STATE_IDS
    )
    if dense_fingerprints != inputs.state_fingerprints:
        raise RuntimeError("正式终态与Dense Seed state fingerprints不一致")

    asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"]).resolve()
    mesh_path = Path(asset["mesh"]).resolve()
    texture_path = Path(asset["texture"]).resolve()
    checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
    for required in (
        xml_path,
        mesh_path,
        texture_path,
        checkpoint_path,
        dense_metrics_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

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
    if (model_height, model_width) != (224, 224):
        raise RuntimeError("Gate 6i evidence schema固定224x224 checkpoint输入")
    view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_height,
    )
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(model.device),
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=FORMAL_SURFACE_EPSILON,
        texture_parameterization="geometry_vertex",
    ).to(model.device)
    if renderer.get_texture_param().shape != (support.num_geometry_vertices, 3):
        raise RuntimeError("Production Support与renderer geometry数量不一致")
    endpoint_surfaces = _realize_endpoints(inputs, support, renderer)

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    suite = benchmark_class()
    task = suite.get_task(cfg.task_id)
    initial_states = suite.get_task_init_states(cfg.task_id)
    description_env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()

    endpoint_source_records = _copy_endpoint_sources(inputs, output_dir=output_dir)
    gradient_records: list[dict[str, Any]] = []
    gradients: dict[str, list[NDArray[np.float32]]] = {
        endpoint: [] for endpoint in ENDPOINT_NAMES
    }
    for endpoint in ENDPOINT_NAMES:
        for state_id in EXPECTED_STATE_IDS:
            print(f"[GATE-6I] gradient endpoint={endpoint} state={state_id}")
            gradient = _capture_surface_gradient(
                cfg=cfg,
                endpoint=endpoint,
                state_id=state_id,
                initial_state=initial_states[state_id],
                clean_target=clean_targets[state_id],
                endpoint_surface=endpoint_surfaces[endpoint],
                task=task,
                task_description=task_description,
                asset=asset,
                model=model,
                processor=processor,
                image_preprocessor=image_preprocessor,
                view_transform=view_transform,
                renderer=renderer,
                get_dummy_action=get_libero_dummy_action,
                get_libero_env=get_libero_env,
                get_libero_image=get_libero_image,
            )
            gradients[endpoint].append(gradient)
            evidence = EndpointGradientEvidence(
                endpoint=endpoint,
                state_id=state_id,
                state_fingerprint=clean_targets[state_id].fingerprint,
                realized_endpoint_sha256=array_sha256(endpoint_surfaces[endpoint]),
                dense_surface_gradient=gradient,
            )
            relative = Path("gradients") / f"{endpoint}_state_{state_id:02d}.npz"
            digest = write_endpoint_gradient_npz(
                evidence,
                output_path=output_dir / relative,
            )
            gradient_records.append(
                {
                    "endpoint": endpoint,
                    "state_id": state_id,
                    "state_fingerprint": clean_targets[state_id].fingerprint,
                    "npz_relative_path": str(relative),
                    "npz_sha256": digest,
                }
            )

    surfaces: dict[str, _EndpointSurfaces] = {}
    step_records: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_NAMES:
        result = _build_endpoint_surfaces(
            endpoint=endpoint,
            realized=endpoint_surfaces[endpoint],
            gradients=gradients[endpoint],
            support=support,
            renderer=renderer,
        )
        surfaces[endpoint] = result
        relative = Path("steps") / f"{endpoint}_step.npz"
        digest = write_endpoint_step_npz(
            result.step_evidence,
            output_path=output_dir / relative,
        )
        step_records.append(
            {
                "endpoint": endpoint,
                "npz_relative_path": str(relative),
                "npz_sha256": digest,
            }
        )

    response_records: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_NAMES:
        arm_surfaces = {
            "baseline": surfaces[endpoint].realized,
            "dense": surfaces[endpoint].dense,
            "support": surfaces[endpoint].support,
        }
        for state_id in EXPECTED_STATE_IDS:
            for arm in RESPONSE_ARMS:
                print(
                    f"[GATE-6I] response endpoint={endpoint} "
                    f"state={state_id} arm={arm}"
                )
                evidence = _capture_response(
                    cfg=cfg,
                    endpoint=endpoint,
                    state_id=state_id,
                    arm=arm,
                    surface=arm_surfaces[arm],
                    initial_state=initial_states[state_id],
                    clean_target=clean_targets[state_id],
                    task=task,
                    task_description=task_description,
                    asset=asset,
                    model=model,
                    processor=processor,
                    image_preprocessor=image_preprocessor,
                    view_transform=view_transform,
                    renderer=renderer,
                    get_dummy_action=get_libero_dummy_action,
                    get_libero_env=get_libero_env,
                    get_libero_image=get_libero_image,
                )
                relative = (
                    Path("responses")
                    / f"{endpoint}_state_{state_id:02d}_{arm}.npz"
                )
                digest = write_endpoint_response_npz(
                    evidence,
                    output_path=output_dir / relative,
                )
                response_records.append(
                    {
                        "endpoint": endpoint,
                        "state_id": state_id,
                        "arm": arm,
                        "npz_relative_path": str(relative),
                        "npz_sha256": digest,
                    }
                )

    if _loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6i运行中加载了legacy optimization.py")
    processor_specification = asdict(image_preprocessor)
    configuration = frozen_bundle_configuration()
    payload = {
        "schema_version": ENDPOINT_BUNDLE_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "configuration": configuration,
        "config_sha256": json_sha256(configuration),
        "provenance": {
            "source_dense_seed_metrics_path": str(dense_metrics_path),
            "source_dense_seed_metrics_sha256": file_sha256(dense_metrics_path),
            "production_support_path": str(inputs.production_support_path),
            "production_support_sha256": inputs.production_support_sha256,
            "endpoint_sources": endpoint_source_records,
            "processor_specification": processor_specification,
            "processor_specification_sha256": json_sha256(processor_specification),
            "gradient_modalities": ["source_openvla_primary_action"],
            "teacher_forced_clean_prefix": True,
            "shared_texture_instances_aggregated": True,
            "render_to_geometry_mapping": "strict_face_corner_scatter_add",
            "feature_gradient": False,
            "wrist_gradient": False,
            "oft_gradient": False,
            "training_or_rollout_run": False,
            "legacy_optimizer_modules": [],
            "runtime_config": asdict(cfg),
            "checkpoint_fingerprints": _checkpoint_fingerprints(
                checkpoint_path
            ),
            "asset_sha256": {
                "xml": file_sha256(xml_path),
                "mesh": file_sha256(mesh_path),
                "texture": file_sha256(texture_path),
            },
        },
        "gradient_cases": gradient_records,
        "endpoint_steps": step_records,
        "response_records": response_records,
    }
    publish_terminal_endpoint_bundle(payload, output_path=manifest_path)
    print(f"[GATE-6I] manifest={manifest_path}")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    print(
        "[GATE-6I] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_terminal_endpoint_action_audit(cfg)


if __name__ == "__main__":
    main()
