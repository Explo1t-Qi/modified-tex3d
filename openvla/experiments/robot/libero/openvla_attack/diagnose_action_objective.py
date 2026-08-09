"""运行 states 0--9 Untargeted Clean-Action Objective GPU Audit。

该命令在零 Surface Delta 的全 OBJ 几何顶点 RGB 空间中验证新 Action
objective 的真实可微链路：

``dense geometry delta -> all shared texture instances -> MuJoCo visibility``
``-> Policy Source -> center crop -> checkpoint BPDA -> teacher-forced logits``
``-> clean-action margin hinge``。

每个 state 必须保存逐 token margin/hinge，并证明 Policy Source、Pre-Crop、
Effective View、renderer Surface Delta 和 dense geometry 参数梯度均有限非零。
它不计算 Feature loss，不读取 wrist/OFT，不更新纹理，也不生成或冻结 Support；
梯度只保存摘要和 SHA-256，不能冒充后续 Dense Seed Audit 产物。
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

# 必须在导入 LIBERO/Robosuite 前固定 headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
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
from openvla_attack.action_objective_audit import (  # noqa: E402
    ACTION_OBJECTIVE_NAME,
    ACTION_OBJECTIVE_SCHEMA_VERSION,
    ActionObjectiveAuditEvidence,
    evaluate_action_objective_evidence,
    summarize_action_objective_evidence,
    write_action_objective_evidence,
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.deployment_backward_audit import (  # noqa: E402
    GradientEvidence,
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
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiableDeploymentViewStages,
    DifferentiablePolicyViewTransform,
    build_exact_deployment_view_stages,
    build_policy_view_transform,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from openvla_attack.visibility_capture import (  # noqa: E402
    capture_instance_segmentation,
    static_scene_evidence_transaction,
)
from openvla_attack.visibility_compositing import (  # noqa: E402
    VisibilityMaskedComposition,
    compose_visibility_masked_renderer_delta,
)
from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
)
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass(frozen=True)
class ActionObjectiveAuditConfig:
    """真实 Objective GPU Audit 的 CLI schema。"""

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


@dataclass(frozen=True)
class ActionObjectiveStateCapture:
    """一次共享 Action-only runtime capture 的完整内存结果。

    ``dense_geometry_gradient`` 是拥有数据的 CPU float32
    ``[num_geometry_vertices,3]`` NumPy 数组。Objective GPU Audit 只消费其
    hash/统计；Dense Seed Audit 复用同一次捕获并把完整数组写入独立 artifact。
    其余 hash 与实例字段绑定产生梯度的几何 correspondence、有效输入和
    visibility evidence。
    """

    evidence: ActionObjectiveAuditEvidence
    dense_geometry_gradient: np.ndarray
    mesh_sha256: str
    render_to_geometry_sha256: str
    policy_source_rgb_sha256: str
    effective_view_rgb_sha256: str
    mujoco_instance_alpha_sha256: str
    renderer_visibility_sha256: str
    shared_instance_body_ids: tuple[int, ...]
    shared_instance_body_names: tuple[str, ...]


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> ActionObjectiveAuditConfig:
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
    return ActionObjectiveAuditConfig(**vars(parser.parse_args(argv)))


def _validate_code_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit 必须是40位小写十六进制 Git SHA")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_fingerprint(state: Any) -> str:
    return _array_sha256(np.asarray(state))


def _array_sha256(value: np.ndarray) -> str:
    """按 contiguous dtype/shape/bytes 绑定 NumPy 数组。"""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """按 CPU contiguous dtype/shape/bytes 绑定任意 tensor。"""

    return _array_sha256(tensor.detach().cpu().numpy())


def _checkpoint_fingerprints(checkpoint_path: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
    ):
        candidate = checkpoint_path / name
        if candidate.is_file():
            fingerprints[name] = _sha256_file(candidate)
    weight_files = sorted(
        tuple(checkpoint_path.glob("*.safetensors"))
        + tuple(checkpoint_path.glob("*.bin"))
    )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in weight_files
    ).encode("utf-8")
    fingerprints["__weight_name_size_inventory_sha256__"] = (
        hashlib.sha256(inventory).hexdigest()
    )
    return fingerprints


def _rgb_tensor(image: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """uint8 HWC RGB 转 float32 device NCHW ``[1,3,H,W]``。"""

    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _build_instances(
    env: Any,
    target_poses: tuple[TargetBodyPose, ...],
    *,
    device: torch.device,
) -> tuple[TextureRenderInstance, ...]:
    """构造同一 Active Texture 的全部 Primary renderer 实例。"""

    return tuple(
        {
            "mvp": compute_render_mvp(
                env,
                pose.model_matrix.to(device),
                resolution=(
                    POLICY_SOURCE_RESOLUTION,
                    POLICY_SOURCE_RESOLUTION,
                ),
                camera_name="agentview",
            ),
            "model_rot": pose.model_matrix[:3, :3].to(device),
        }
        for pose in target_poses
    )


def _prompt(task_description: str) -> str:
    return (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )


def _require_gradient(
    name: str,
    gradient: Optional[torch.Tensor],
) -> torch.Tensor:
    if gradient is None:
        raise RuntimeError(f"Objective GPU Audit 缺少 {name} 梯度")
    return gradient.detach()


def collect_action_objective_state_capture(
    *,
    cfg: ActionObjectiveAuditConfig,
    state_id: int,
    initial_state: Any,
    task: Any,
    task_description: str,
    asset: ObjectAssetSpec,
    model: Any,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    policy_view_transform: DifferentiablePolicyViewTransform,
    renderer: DifferentiableRenderer,
    get_libero_dummy_action: Any,
    get_libero_env: Any,
    get_libero_image: Any,
) -> ActionObjectiveStateCapture:
    """执行一个 state 的零扰动 teacher-forward/backward 公共链路。"""

    renderer.reset_texture()
    dense_parameter = renderer.get_texture_param()
    zero_surface_delta_linf = float(
        renderer.get_surface_delta().detach().abs().amax().item()
    )
    if zero_surface_delta_linf != 0.0:
        raise RuntimeError("Objective GPU Audit 未从精确零 Surface Delta 开始")

    env, current_task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    if current_task_description != task_description:
        raise RuntimeError("LIBERO task description 在 states 间改变")
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        env.env.sim.forward()
        for _ in range(cfg.num_steps_wait):
            observation, _, _, _ = env.step(
                get_libero_dummy_action(cfg.model_family)
            )
        clean_source_rgb = get_libero_image(
            observation,
            POLICY_SOURCE_RESOLUTION,
        )
        clean_source_tensor = _rgb_tensor(
            clean_source_rgb,
            device=model.device,
        )
        exact_clean = build_exact_deployment_view_stages(
            clean_source_rgb,
            specification=policy_view_transform.specification,
        )
        target_poses = find_target_body_poses(
            env,
            asset["search"],
            model.device,
        )
        if not target_poses or any(
            pose.body_name is None for pose in target_poses
        ):
            raise RuntimeError(
                f"state {state_id} 未找到完整共享纹理实例"
            )
        body_ids = tuple(pose.body_id for pose in target_poses)
        instance_roots = tuple(
            TargetInstanceRoot(
                body_id=pose.body_id,
                body_name=str(pose.body_name),
            )
            for pose in target_poses
        )

        dense_parameter.grad = None
        with renderer.capture_surface_delta_gradients() as surface_capture:
            with static_scene_evidence_transaction(
                env,
                body_ids,
            ):
                transaction_poses = find_target_body_poses(
                    env,
                    asset["search"],
                    model.device,
                )
                if tuple(
                    pose.body_id for pose in transaction_poses
                ) != body_ids:
                    raise RuntimeError(
                        "static transaction 内目标实例集合改变"
                    )
                segmentation = capture_instance_segmentation(
                    env,
                    instance_roots,
                    camera_name="agentview",
                    resolution=POLICY_SOURCE_RESOLUTION,
                )
                renderer_evidence = render_shared_texture_instances(
                    renderer,
                    _build_instances(
                        env,
                        transaction_poses,
                        device=model.device,
                    ),
                    resolution=(
                        POLICY_SOURCE_RESOLUTION,
                        POLICY_SOURCE_RESOLUTION,
                    ),
                )
                mujoco_alpha = segmentation.parsed.instance_alpha.to(
                    device=model.device,
                    dtype=torch.float32,
                )
                composition: VisibilityMaskedComposition = (
                    compose_visibility_masked_renderer_delta(
                        clean_source_tensor,
                        mujoco_alpha,
                        renderer_evidence.adversarial_rgb,
                        renderer_evidence.clean_rgb,
                        renderer_evidence.visibility_mask,
                    )
                )

            if float(composition.total_delta.detach().abs().amax().item()) != 0.0:
                raise RuntimeError("零 Surface Delta compositor 不再严格恒等")
            source_rgb = composition.composited_rgb
            source_rgb.retain_grad()
            deployment_stages: DifferentiableDeploymentViewStages = (
                policy_view_transform.build_stages(source_rgb)
            )
            deployment_stages.pre_crop_canvas.retain_grad()
            deployment_stages.effective_view.retain_grad()
            bpda_pixel_values = image_preprocessor.build_fused_pixel_values(
                deployment_stages.effective_view
            ).to(torch.bfloat16)

            clean_image = Image.fromarray(
                exact_clean.effective_view_rgb,
                mode="RGB",
            )
            clean_inputs: Any = processor(
                _prompt(task_description),
                images=clean_image,
            ).to(model.device)
            ensure_trailing_empty_token(clean_inputs)
            if "pixel_values" not in clean_inputs:
                raise RuntimeError("checkpoint processor 缺少 pixel_values")
            processor_pixel_values = clean_inputs["pixel_values"].to(
                device=model.device,
                dtype=torch.bfloat16,
            )
            if not torch.equal(processor_pixel_values, bpda_pixel_values):
                maximum_difference = float(
                    (
                        processor_pixel_values.float()
                        - bpda_pixel_values.float()
                    )
                    .abs()
                    .amax()
                    .item()
                )
                raise RuntimeError(
                    "Objective Audit processor/BPDA forward 不一致，"
                    f"Linf={maximum_difference}"
                )
            action_dim = int(model.get_action_dim(cfg.unnorm_key))
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                clean_output_ids = model.generate(
                    **clean_inputs,
                    max_new_tokens=action_dim,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=clean_output_ids,
                    attention_mask=torch.ones_like(clean_output_ids),
                    pixel_values=bpda_pixel_values,
                    output_hidden_states=False,
                )
            objective_result: UntargetedCleanActionMarginHinge = (
                untargeted_clean_action_margin_hinge(
                    outputs.logits,
                    clean_output_ids,
                )
            )
            if int(objective_result.margins.numel()) != action_dim:
                raise RuntimeError(
                    "Action Objective token 数与 checkpoint action_dim 不一致"
                )
            objective_result.loss.backward()

        source_gradient = _require_gradient("Policy Source", source_rgb.grad)
        pre_crop_gradient = _require_gradient(
            "Pre-Crop",
            deployment_stages.pre_crop_canvas.grad,
        )
        effective_gradient = _require_gradient(
            "Effective View",
            deployment_stages.effective_view.grad,
        )
        dense_gradient = _require_gradient(
            "Dense Geometry",
            dense_parameter.grad,
        )
        render_surface_gradient = surface_capture.summed_gradient()
        clean_classes = objective_result.clean_classes.detach().to(
            torch.int64
        )
        clean_token_ids = clean_classes + ACTION_TOKEN_START
        dense_gradient_array = np.ascontiguousarray(
            dense_gradient.to(dtype=torch.float32).cpu().numpy()
        )
        evidence = ActionObjectiveAuditEvidence(
            schema_version=ACTION_OBJECTIVE_SCHEMA_VERSION,
            code_commit=cfg.code_commit,
            state_id=state_id,
            initial_state_sha256=_state_fingerprint(initial_state),
            objective=ACTION_OBJECTIVE_NAME,
            parameterization="geometry_vertex_dense",
            zero_surface_delta_linf=zero_surface_delta_linf,
            teacher_forced_clean_prefix=True,
            clean_action_token_ids=[
                int(value) for value in clean_token_ids.cpu().tolist()
            ],
            clean_classes=[
                int(value) for value in clean_classes.cpu().tolist()
            ],
            margins=[
                float(value)
                for value in objective_result.margins.detach()
                .float()
                .cpu()
                .tolist()
            ],
            hinge_values=[
                float(value)
                for value in objective_result.hinge_values.detach()
                .float()
                .cpu()
                .tolist()
            ],
            action_loss=float(objective_result.loss.detach().float().item()),
            parameter_shape=tuple(int(size) for size in dense_parameter.shape),
            source_rgb_gradient=GradientEvidence.from_tensor(source_gradient),
            pre_crop_gradient=GradientEvidence.from_tensor(pre_crop_gradient),
            effective_view_gradient=GradientEvidence.from_tensor(
                effective_gradient
            ),
            render_surface_delta_gradient=GradientEvidence.from_tensor(
                render_surface_gradient
            ),
            dense_geometry_gradient=GradientEvidence.from_tensor(
                dense_gradient
            ),
            dense_geometry_gradient_sha256=_array_sha256(
                dense_gradient_array
            ),
        )
        return ActionObjectiveStateCapture(
            evidence=evidence,
            dense_geometry_gradient=dense_gradient_array,
            mesh_sha256=_sha256_file(Path(asset["mesh"]).resolve()),
            render_to_geometry_sha256=_tensor_sha256(
                renderer.get_render_to_geometry_mapping()
            ),
            policy_source_rgb_sha256=_array_sha256(clean_source_rgb),
            effective_view_rgb_sha256=_array_sha256(
                exact_clean.effective_view_rgb
            ),
            mujoco_instance_alpha_sha256=_tensor_sha256(mujoco_alpha),
            renderer_visibility_sha256=_tensor_sha256(
                renderer_evidence.visibility_mask
            ),
            shared_instance_body_ids=body_ids,
            shared_instance_body_names=tuple(
                str(pose.body_name) for pose in target_poses
            ),
        )
    finally:
        env.close()


def run_action_objective_audit(
    cfg: ActionObjectiveAuditConfig,
) -> Path:
    """运行真实 states audit，并返回 manifest 路径。"""

    _validate_code_commit(cfg.code_commit)
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("Objective GPU Audit state_ids 不得为空")
    state_ids = tuple(selected_state_ids)
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
        raise RuntimeError("Objective GPU Audit 需要真实 CUDA nvdiffrast device")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "action_objective_metrics.jsonl"
    manifest_path = output_dir / "action_objective_manifest.json"
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有 Objective GPU Audit 权威文件")

    asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"]).resolve()
    mesh_path = Path(asset["mesh"]).resolve()
    texture_path = Path(asset["texture"]).resolve()
    checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
    for required_path in (xml_path, mesh_path, texture_path, checkpoint_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    model_cfg = GenerateConfig(
        pretrained_checkpoint=cfg.pretrained_checkpoint,
        model_family=cfg.model_family,
        center_crop=True,
        object_name=cfg.object_name,
        task_suite_name=cfg.task_suite_name,
        task_id=cfg.task_id,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
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
    model_input_height, model_input_width = image_preprocessor.output_size
    if model_input_height != model_input_width:
        raise RuntimeError("Objective GPU Audit 只支持正方形 checkpoint 输入")
    policy_view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_input_height,
    )
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(model.device),
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=128.0 / 255.0,
        texture_parameterization="geometry_vertex",
    ).to(model.device)
    renderer.reset_texture()

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if min(state_ids) < 0 or max(state_ids) >= len(initial_states):
        raise ValueError("Objective GPU Audit state_id 超出初始状态范围")
    description_env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()

    rows: list[ActionObjectiveAuditEvidence] = []
    for state_id in state_ids:
        print(f"[ACTION-OBJECTIVE-AUDIT] state {state_id}")
        capture = collect_action_objective_state_capture(
            cfg=cfg,
            state_id=state_id,
            initial_state=initial_states[state_id],
            task=task,
            task_description=task_description,
            asset=asset,
            model=model,
            processor=processor,
            image_preprocessor=image_preprocessor,
            policy_view_transform=policy_view_transform,
            renderer=renderer,
            get_libero_dummy_action=get_libero_dummy_action,
            get_libero_env=get_libero_env,
            get_libero_image=get_libero_image,
        )
        row = capture.evidence
        decision = evaluate_action_objective_evidence(row)
        rows.append(row)
        print(
            "[ACTION-OBJECTIVE-AUDIT] "
            f"state={state_id} loss={row.action_loss:.6f} "
            f"margin_min={min(row.margins):.6f} "
            f"ties={list(decision.tie_indices)} "
            f"dense_grad_l2={row.dense_geometry_gradient.l2_norm:.6e} "
            f"pass={decision.gate_pass}"
        )
        if not decision.gate_pass:
            print(
                "[ACTION-OBJECTIVE-AUDIT] 零扰动契约失败，停止后续 states: "
                + "; ".join(decision.failures)
            )
            break

    write_action_objective_evidence(metrics_path, rows)
    summary = summarize_action_objective_evidence(rows)
    manifest = {
        "schema_version": ACTION_OBJECTIVE_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "summary": asdict(summary),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": _sha256_file(metrics_path),
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "asset_sha256": {
            "xml": _sha256_file(xml_path),
            "mesh": _sha256_file(mesh_path),
            "texture": _sha256_file(texture_path),
        },
        "objective_components": [ACTION_OBJECTIVE_NAME],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "production_support_constructed": False,
        "dense_gradient_payload_saved": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"[ACTION-OBJECTIVE-AUDIT] manifest={manifest_path}")
    if not summary.gate_pass:
        raise RuntimeError(
            "Objective GPU Audit 未通过: " + "; ".join(summary.failures)
        )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    print(
        "[ACTION-OBJECTIVE-AUDIT] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_action_objective_audit(cfg)


if __name__ == "__main__":
    main()
