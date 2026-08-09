"""运行states 0--9真实Action-only Spectral Guard GPU Calibration。

该runner只沿已通过Objective Audit的新链路计算固定clean prefix的Untargeted
Clean-Action Margin hinge。每轮按state 0--9顺序逐帧求紧凑Fixed-Support梯度并
立即释放计算图，再取严格算术均值；Feature、wrist、OFT和legacy optimizer均
不导入。Action-only update由正式Fixed-Support trainer共享核心执行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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

from experiments.robot.openvla_utils import ensure_trailing_empty_token  # noqa: E402
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.fixed_support_training import (  # noqa: E402
    FixedSupportTrainerCore,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    render_shared_texture_instances,
)
from openvla_attack.objective import (  # noqa: E402
    ACTION_TOKEN_START,
    untargeted_clean_action_margin_hinge,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiablePolicyViewTransform,
    build_exact_deployment_view_stages,
    build_policy_view_transform,
)
from openvla_attack.production_support import (  # noqa: E402
    load_production_support_artifact,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.seed_score_audit import file_sha256  # noqa: E402
from openvla_attack.spectral_guard import (  # noqa: E402
    MeanActionGradient,
    SpectralGuardCalibrationResult,
    SpectralNaturalnessRegularizer,
    calibrate_spectral_guard,
)
from openvla_attack.spectral_guard_evidence import (  # noqa: E402
    SPECTRAL_GUARD_SCHEMA_VERSION,
    evaluate_spectral_guard_bundle,
)
from openvla_attack.spectral_naturalness import (  # noqa: E402
    load_rho_nat_calibration_artifact,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
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
class SpectralGuardGpuConfig:
    pretrained_checkpoint: str
    production_support_path: str
    rho_nat_calibration_path: str
    spectral_basis_path: str
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
    attack_epsilon: float = 128.0 / 255.0
    attack_surface_step: float = 2.0 / 255.0


@dataclass(frozen=True)
class FrozenGuardActionFrame:
    """一次静止state的只读Action梯度输入。"""

    state_id: int
    initial_state_sha256: str
    static_scene_sha256: str
    clean_source_rgb: torch.Tensor
    mujoco_instance_alpha: torch.Tensor
    instances: tuple[TextureRenderInstance, ...]
    clean_output_ids: torch.Tensor
    clean_action_token_ids: tuple[int, ...]
    shared_instance_body_ids: tuple[int, ...]
    shared_instance_body_names: tuple[str, ...]


def _parse_args(argv: Optional[Sequence[str]] = None) -> SpectralGuardGpuConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--rho_nat_calibration_path", required=True)
    parser.add_argument("--spectral_basis_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--state_ids", default="0-9")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    parser.add_argument("--attack_epsilon", type=float, default=128.0 / 255.0)
    parser.add_argument("--attack_surface_step", type=float, default=2.0 / 255.0)
    return SpectralGuardGpuConfig(**vars(parser.parse_args(argv)))


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_array(value.detach().cpu().numpy())


def _rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _prompt(task_description: str) -> str:
    return (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )


def loaded_legacy_optimizer_modules() -> tuple[str, ...]:
    """拒绝当前进程加载legacy Action+Feature训练实现。"""

    suffixes = (
        "openvla_attack.optimization",
        "openvla_attack.training",
    )
    return tuple(
        sorted(
            name
            for name in sys.modules
            if any(name.endswith(suffix) for suffix in suffixes)
        )
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


def capture_action_frame(
    *,
    cfg: SpectralGuardGpuConfig,
    state_id: int,
    initial_state: Any,
    task: Any,
    task_description: str,
    asset: ObjectAssetSpec,
    model: Any,
    processor: Any,
    image_preprocessor: DifferentiableOpenVLAImageProcessor,
    policy_view_transform: DifferentiablePolicyViewTransform,
    get_libero_dummy_action: Any,
    get_libero_env: Any,
    get_libero_image: Any,
) -> FrozenGuardActionFrame:
    """冻结一个state的clean输入、可见性与全部共享纹理实例。"""

    env, current_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    if current_description != task_description:
        raise RuntimeError("LIBERO task description在states间改变")
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
        clean_source_tensor = _rgb_tensor(clean_source_rgb, model.device)
        exact_stages = build_exact_deployment_view_stages(
            clean_source_rgb,
            specification=policy_view_transform.specification,
        )
        clean_image = Image.fromarray(exact_stages.effective_view_rgb, mode="RGB")
        clean_inputs = processor(
            _prompt(task_description),
            images=clean_image,
        ).to(model.device)
        ensure_trailing_empty_token(clean_inputs)
        clean_bpda_pixels = image_preprocessor.build_fused_pixel_values(
            policy_view_transform.build_effective_view(clean_source_tensor)
        ).to(torch.bfloat16)
        processor_pixels = clean_inputs["pixel_values"].to(
            device=model.device,
            dtype=torch.bfloat16,
        )
        if not torch.equal(clean_bpda_pixels, processor_pixels):
            raise RuntimeError(f"state {state_id} clean processor/BPDA不一致")
        action_dim = int(model.get_action_dim(cfg.unnorm_key))
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            clean_output_ids = model.generate(
                **clean_inputs,
                max_new_tokens=action_dim,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            clean_outputs = model(
                input_ids=clean_output_ids,
                attention_mask=torch.ones_like(clean_output_ids),
                pixel_values=processor_pixels,
                output_hidden_states=False,
            )
            clean_objective = untargeted_clean_action_margin_hinge(
                clean_outputs.logits,
                clean_output_ids,
            )
        if bool((clean_objective.margins < 0.0).any()):
            raise RuntimeError(f"state {state_id} clean token不属于argmax集合")

        target_poses = find_target_body_poses(env, asset["search"], model.device)
        if not target_poses or any(pose.body_name is None for pose in target_poses):
            raise RuntimeError(f"state {state_id}共享纹理实例不完整")
        body_ids = tuple(pose.body_id for pose in target_poses)
        roots = tuple(
            TargetInstanceRoot(
                body_id=pose.body_id,
                body_name=str(pose.body_name),
            )
            for pose in target_poses
        )
        with static_scene_evidence_transaction(env, body_ids) as transaction:
            transaction_poses = find_target_body_poses(
                env, asset["search"], model.device
            )
            if tuple(pose.body_id for pose in transaction_poses) != body_ids:
                raise RuntimeError("static transaction内实例集合改变")
            segmentation = capture_instance_segmentation(
                env,
                roots,
                camera_name="agentview",
                resolution=POLICY_SOURCE_RESOLUTION,
            )
            instances = _build_instances(
                env,
                transaction_poses,
                device=model.device,
            )
        if not transaction.verified:
            raise RuntimeError("state capture静止事务未通过")
        alpha = segmentation.parsed.instance_alpha.to(
            device=model.device,
            dtype=torch.float32,
        ).detach()
        if alpha.shape[0] != len(instances):
            raise RuntimeError("MuJoCo alpha与renderer实例数不一致")
        clean_classes = clean_objective.clean_classes.detach().to(torch.int64)
        return FrozenGuardActionFrame(
            state_id=state_id,
            initial_state_sha256=sha256_array(np.asarray(initial_state)),
            static_scene_sha256=transaction.before.fingerprint_sha256,
            clean_source_rgb=clean_source_tensor.detach(),
            mujoco_instance_alpha=alpha,
            instances=instances,
            clean_output_ids=clean_output_ids.detach(),
            clean_action_token_ids=tuple(
                int(value)
                for value in (clean_classes + ACTION_TOKEN_START).cpu().tolist()
            ),
            shared_instance_body_ids=body_ids,
            shared_instance_body_names=tuple(
                str(pose.body_name) for pose in target_poses
            ),
        )
    finally:
        env.close()


class AllStateActionGradientProvider:
    """按0--9固定顺序逐state计算并平均Action梯度。"""

    def __init__(
        self,
        *,
        frames: Sequence[FrozenGuardActionFrame],
        renderer: DifferentiableRenderer,
        model: Any,
        image_preprocessor: DifferentiableOpenVLAImageProcessor,
        policy_view_transform: DifferentiablePolicyViewTransform,
    ) -> None:
        self.frames = tuple(frames)
        if tuple(frame.state_id for frame in self.frames) != tuple(range(10)):
            raise ValueError("Action provider必须恰好按0-9持有10个state")
        if len({frame.initial_state_sha256 for frame in self.frames}) != 10:
            raise ValueError("Action provider的state fingerprint必须唯一")
        self.renderer = renderer
        self.model = model
        self.image_preprocessor = image_preprocessor
        self.policy_view_transform = policy_view_transform
        self.call_count = 0
        self.frame_evidence_history: list[dict[str, Any]] = []

    def state_dict(self) -> dict[str, Any]:
        return {"call_count": self.call_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.call_count = int(state_dict["call_count"])

    def drain_frame_evidence(self) -> tuple[dict[str, Any], ...]:
        """移交并清空逐state证据，供5000轮正式训练增量落盘。

        校准和两步 smoke 仍可直接读取 ``frame_evidence_history``；正式训练每轮
        调用本方法，避免在内存中长期保留50000行 margins 与 gradient hash。
        """

        rows = tuple(self.frame_evidence_history)
        self.frame_evidence_history.clear()
        return rows

    def __call__(self) -> MeanActionGradient:
        parameter = self.renderer.get_texture_param()
        losses: list[float] = []
        gradients: list[torch.Tensor] = []
        iteration = self.call_count
        for frame in self.frames:
            renderer_evidence = render_shared_texture_instances(
                self.renderer,
                frame.instances,
                resolution=(POLICY_SOURCE_RESOLUTION, POLICY_SOURCE_RESOLUTION),
            )
            composition = compose_visibility_masked_renderer_delta(
                frame.clean_source_rgb,
                frame.mujoco_instance_alpha,
                renderer_evidence.adversarial_rgb,
                renderer_evidence.clean_rgb,
                renderer_evidence.visibility_mask,
            )
            effective_view = self.policy_view_transform.build_effective_view(
                composition.composited_rgb
            )
            pixel_values = self.image_preprocessor.build_fused_pixel_values(
                effective_view
            )
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=frame.clean_output_ids,
                    attention_mask=torch.ones_like(frame.clean_output_ids),
                    pixel_values=pixel_values.to(torch.bfloat16),
                    output_hidden_states=False,
                )
                objective = untargeted_clean_action_margin_hinge(
                    outputs.logits,
                    frame.clean_output_ids,
                )
            gradient = torch.autograd.grad(objective.loss, parameter)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(
                    f"iteration {iteration} state {frame.state_id} Action梯度非有限"
                )
            gradient = gradient.detach()
            loss_value = float(objective.loss.detach().float().item())
            losses.append(loss_value)
            gradients.append(gradient)
            margins = objective.margins.detach().float().cpu()
            hinges = objective.hinge_values.detach().float().cpu()
            self.frame_evidence_history.append(
                {
                    "iteration": iteration,
                    "state_id": frame.state_id,
                    "initial_state_sha256": frame.initial_state_sha256,
                    "static_scene_sha256": frame.static_scene_sha256,
                    "clean_action_token_ids": list(frame.clean_action_token_ids),
                    "margins": margins.tolist(),
                    "hinge_values": hinges.tolist(),
                    "nonpositive_margin_count": int((margins <= 0.0).sum()),
                    "action_loss": loss_value,
                    "action_gradient_l2": float(
                        torch.linalg.vector_norm(gradient).item()
                    ),
                    "action_gradient_sha256": tensor_sha256(gradient),
                    "saturated_pixel_fraction": (
                        composition.saturated_pixel_fraction
                    ),
                    "saturated_channel_fraction": (
                        composition.saturated_channel_fraction
                    ),
                }
            )
        self.call_count += 1
        mean_gradient = torch.stack(gradients, dim=0).mean(dim=0)
        return MeanActionGradient(
            loss=float(np.mean(np.asarray(losses, dtype=np.float64))),
            gradient=mean_gradient,
            num_frames=len(self.frames),
            state_ids=tuple(frame.state_id for frame in self.frames),
            state_fingerprints=tuple(
                frame.initial_state_sha256 for frame in self.frames
            ),
        )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_spectral_guard_gpu_calibration(cfg: SpectralGuardGpuConfig) -> Path:
    """运行正式GPU校准并写入逐iteration/state证据与manifest。"""

    if len(cfg.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in cfg.code_commit
    ):
        raise ValueError("code_commit必须是40位小写Git SHA")
    selected = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if tuple(selected or ()) != tuple(range(10)):
        raise ValueError("Spectral Guard正式校准必须精确使用states 0-9")
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait不得为负数")
    expected_iterations = math.ceil(
        cfg.attack_epsilon / cfg.attack_surface_step
    )
    if expected_iterations != 64:
        raise ValueError("第一版epsilon/surface_step必须对应恰好64轮上限")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知object_name: {cfg.object_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("Spectral Guard GPU Calibration需要真实CUDA")
    legacy_modules = loaded_legacy_optimizer_modules()
    if legacy_modules:
        raise RuntimeError(f"Spectral Guard进程加载了legacy optimizer: {legacy_modules}")
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration_path = output_dir / "spectral_guard_iterations.jsonl"
    frame_path = output_dir / "spectral_guard_action_frames.jsonl"
    manifest_path = output_dir / "spectral_guard_manifest.json"
    if any(path.exists() for path in (iteration_path, frame_path, manifest_path)):
        raise FileExistsError("拒绝覆盖已有Spectral Guard校准证据")

    support_path = Path(cfg.production_support_path)
    rho_path = Path(cfg.rho_nat_calibration_path)
    basis_path = Path(cfg.spectral_basis_path)
    support = load_production_support_artifact(support_path)
    rho_calibration = load_rho_nat_calibration_artifact(rho_path)
    if file_sha256(support_path) != (
        rho_calibration.production_support_artifact_sha256
    ):
        raise RuntimeError("rho_nat校准未绑定当前Production Support")
    if support.support_mask_sha256 != rho_calibration.support_mask_sha256:
        raise RuntimeError("rho_nat校准Support mask不匹配")

    set_seed_everywhere(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    asset = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"]).resolve()
    mesh_path = Path(asset["mesh"]).resolve()
    texture_path = Path(asset["texture"]).resolve()
    checkpoint_path = Path(cfg.pretrained_checkpoint).resolve()
    for path in (
        support_path,
        rho_path,
        basis_path,
        xml_path,
        mesh_path,
        texture_path,
        checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

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
    model = get_model(model_cfg)
    model.eval()
    model.requires_grad_(False)
    processor = get_processor(model_cfg)
    image_preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model,
        processor=processor,
    )
    model_height, model_width = image_preprocessor.output_size
    if model_height != model_width:
        raise RuntimeError("Spectral Guard只支持正方形checkpoint输入")
    policy_view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_height,
    )
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(model.device),
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=cfg.attack_epsilon,
        texture_parameterization="fixed_support",
        fixed_support_path=support_path,
    ).to(model.device)
    renderer.reset_texture()
    if renderer.surface_parameterization is None:
        raise RuntimeError("Fixed-Support renderer缺少surface parameterization")

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    description_env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()
    frames = tuple(
        capture_action_frame(
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
            get_libero_dummy_action=get_libero_dummy_action,
            get_libero_env=get_libero_env,
            get_libero_image=get_libero_image,
        )
        for state_id in range(10)
    )
    state_fingerprints = {
        frame.state_id: frame.initial_state_sha256 for frame in frames
    }
    provider = AllStateActionGradientProvider(
        frames=frames,
        renderer=renderer,
        model=model,
        image_preprocessor=image_preprocessor,
        policy_view_transform=policy_view_transform,
    )
    trainer_core = FixedSupportTrainerCore(
        renderer,
        surface_step=cfg.attack_surface_step,
    )
    regularizer = SpectralNaturalnessRegularizer.from_artifacts(
        rho_path,
        basis_path,
        device=model.device,
        dtype=torch.float32,
    )
    result: SpectralGuardCalibrationResult = calibrate_spectral_guard(
        mutable_module=renderer.surface_parameterization,
        texture_parameter=trainer_core.texture_parameter,
        geometry_delta_provider=trainer_core.geometry_delta,
        mean_action_gradient_provider=provider,
        action_only_update=trainer_core.apply_action_gradient,
        regularizer=regularizer,
        stateful_components={
            "fixed_support_trainer_core": trainer_core,
            "all_state_action_provider": provider,
        },
        expected_state_fingerprints=state_fingerprints,
        surface_step=cfg.attack_surface_step,
        max_iterations=expected_iterations,
    )
    legacy_modules = loaded_legacy_optimizer_modules()
    if legacy_modules:
        raise RuntimeError(f"校准期间加载了legacy optimizer: {legacy_modules}")
    if len(provider.frame_evidence_history) != 10 * len(result.iterations):
        raise RuntimeError("逐state证据数量与校准iterations不一致")

    iteration_rows = [asdict(row) for row in result.iterations]
    write_jsonl(iteration_path, iteration_rows)
    write_jsonl(frame_path, provider.frame_evidence_history)
    manifest = {
        "schema_version": SPECTRAL_GUARD_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "input_sha256": {
            "production_support": file_sha256(support_path),
            "rho_nat_calibration": file_sha256(rho_path),
            "spectral_basis": file_sha256(basis_path),
            "mesh": file_sha256(mesh_path),
            "texture": file_sha256(texture_path),
        },
        "state_ids": list(range(10)),
        "state_fingerprints": state_fingerprints,
        "static_scene_fingerprints": {
            frame.state_id: frame.static_scene_sha256 for frame in frames
        },
        "shared_instances": {
            frame.state_id: {
                "body_ids": list(frame.shared_instance_body_ids),
                "body_names": list(frame.shared_instance_body_names),
            }
            for frame in frames
        },
        "objective_components": ["untargeted_clean_action_margin_hinge"],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "num_iterations": len(result.iterations),
        "num_action_frames_per_iteration": 10,
        "iterations_relative_path": iteration_path.name,
        "iterations_sha256": file_sha256(iteration_path),
        "action_frames_relative_path": frame_path.name,
        "action_frames_sha256": file_sha256(frame_path),
        "selected_window_start": result.selected_window_start,
        "selected_window_end": result.selected_window_end,
        "q_median": result.q_median,
        "lambda_spec": result.lambda_spec,
        "calibration_status": result.calibration_status,
        "restore_evidence": asdict(result.restore_evidence),
        "rho_nat_calibrated": True,
        "lambda_spec_calibrated": True,
        "formal_training_allowed": False,
        "next_required_gate": "fixed_support_action_spectral_training_smoke",
        "gate_pass": True,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    decision = evaluate_spectral_guard_bundle(manifest_path)
    if not decision.gate_pass:
        raise RuntimeError(
            "Spectral Guard bundle独立复核失败: "
            + "; ".join(decision.failures)
        )
    print(
        json.dumps(
            {
                "gate_pass": True,
                "num_iterations": len(result.iterations),
                "selected_window": [
                    result.selected_window_start,
                    result.selected_window_end,
                ],
                "q_median": result.q_median,
                "lambda_spec": result.lambda_spec,
                "calibration_status": result.calibration_status,
                "manifest_sha256": file_sha256(manifest_path),
                "formal_training_allowed": False,
            },
            sort_keys=True,
        )
    )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(
        "[SPECTRAL-GUARD] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_spectral_guard_gpu_calibration(_parse_args(argv))


if __name__ == "__main__":
    main()
