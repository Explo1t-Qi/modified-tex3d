"""运行 states 0–9 零 Surface Delta Compositor Gate。

本命令只在 source OpenVLA Primary 视角验证 Visibility-Masked Renderer Delta
Composition 的严格恒等性和局部可微性；不构造 Support、不读取 Action 梯度、
不更新或 bake 纹理，也不运行 rollout。每个 state 在同一个静止事务中采集
MuJoCo front-most instance alpha、全部共享纹理实例的同次 renderer evidence，
然后比较 clean 与零 delta compositor 的：

* raw 512 Policy Source RGB；
* 224 Policy Pre-Crop Canvas 与 center-crop Effective View；
* checkpoint fused pixel values；
* source OpenVLA action token 和连续 action。

权威输出为 ``compositor_zero_delta_metrics.jsonl``。每个 state 另保存一个禁止
pickle 的 NPZ 和可视化 PNG，使 WSL 能独立复算，而不是只相信服务器 stdout。
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
from typing import Any, MutableMapping, Optional, Sequence

# 必须在导入 LIBERO/Robosuite 前固定 headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import PIL
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
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.compositor_zero_delta_audit import (  # noqa: E402
    COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION,
    ZeroDeltaCompositorEvidence,
    ZeroDeltaCompositorRow,
    evaluate_zero_delta_compositor_evidence,
    summarize_zero_delta_compositor_rows,
    write_zero_delta_compositor_jsonl,
    write_zero_delta_compositor_manifest,
)
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
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
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


@dataclass
class CompositorZeroDeltaConfig:
    """真实 Gate runner 的 CLI schema。"""

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
    """透明代理 processor，并记录 clean rollout 的实际 PIL/input。"""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.last_rgb: Optional[NDArray[np.uint8]] = None
        self.last_inputs: Optional[MutableMapping[str, torch.Tensor]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        image: Any = args[1] if len(args) > 1 else kwargs.get("images")
        if not isinstance(image, Image.Image):
            raise RuntimeError("零 delta Gate processor 未收到 PIL Image")
        self.last_rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        inputs: Any = self._processor(*args, **kwargs)
        self.last_inputs = inputs
        return inputs


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> CompositorZeroDeltaConfig:
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
    return CompositorZeroDeltaConfig(**vars(parser.parse_args(argv)))


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
    array = np.ascontiguousarray(np.asarray(state))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fingerprint_files(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in paths if path.is_file()}


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
    fingerprints = _fingerprint_files(configuration_files)
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


def _rgb_tensor(
    image: NDArray[np.uint8],
    *,
    device: torch.device,
) -> torch.Tensor:
    """uint8 HWC RGB → float32 device NCHW ``[1,3,H,W]``。"""

    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _uint8_hwc(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """float NCHW ``[1,3,H,W]`` → 拥有数据的 uint8 HWC RGB。"""

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


def _save_rgb(path: Path, value: NDArray[np.uint8]) -> str:
    Image.fromarray(value, mode="RGB").save(path)
    return _sha256_file(path)


def _save_alpha(path: Path, alpha: torch.Tensor) -> str:
    union = alpha.sum(dim=0)[0]
    value = (
        union.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(value, mode="L").save(path)
    return _sha256_file(path)


def _persist_state_artifacts(
    *,
    output_dir: Path,
    state_id: int,
    clean_source_rgb: NDArray[np.uint8],
    composited_source_rgb: NDArray[np.uint8],
    clean_pre_crop_rgb: NDArray[np.uint8],
    composited_pre_crop_rgb: NDArray[np.uint8],
    clean_effective_rgb: NDArray[np.uint8],
    composited_effective_rgb: NDArray[np.uint8],
    clean_pixel_values: NDArray[np.floating],
    composited_pixel_values: NDArray[np.floating],
    clean_action_token_ids: NDArray[np.integer],
    composited_action_token_ids: NDArray[np.integer],
    clean_action: NDArray[np.floating],
    composited_action: NDArray[np.floating],
    mujoco_instance_alpha: torch.Tensor,
    renderer_valid_mask: torch.Tensor,
    renderer_delta: torch.Tensor,
    per_instance_delta: torch.Tensor,
    total_delta: torch.Tensor,
    surface_parameter_gradient: torch.Tensor,
) -> tuple[str, dict[str, str]]:
    """保存禁止 pickle 的精确数组与便读图，并返回 hash inventory。"""

    arrays_dir = output_dir / "arrays"
    images_dir = output_dir / "images"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"state_{state_id:02d}"
    arrays_path = arrays_dir / f"{prefix}.npz"
    if arrays_path.exists():
        raise FileExistsError(arrays_path)
    np.savez_compressed(
        arrays_path,
        schema_version=np.asarray(COMPOSITOR_ZERO_DELTA_SCHEMA_VERSION),
        state_id=np.asarray(state_id, dtype=np.int64),
        clean_source_rgb=clean_source_rgb,
        composited_source_rgb=composited_source_rgb,
        clean_pre_crop_rgb=clean_pre_crop_rgb,
        composited_pre_crop_rgb=composited_pre_crop_rgb,
        clean_effective_rgb=clean_effective_rgb,
        composited_effective_rgb=composited_effective_rgb,
        clean_pixel_values=clean_pixel_values,
        composited_pixel_values=composited_pixel_values,
        clean_action_token_ids=clean_action_token_ids,
        composited_action_token_ids=composited_action_token_ids,
        clean_action=clean_action,
        composited_action=composited_action,
        mujoco_instance_alpha=(
            mujoco_instance_alpha.detach().cpu().numpy()
        ),
        renderer_valid_mask=renderer_valid_mask.detach().cpu().numpy(),
        renderer_delta=renderer_delta.detach().cpu().numpy(),
        per_instance_delta=per_instance_delta.detach().cpu().numpy(),
        total_delta=total_delta.detach().cpu().numpy(),
        surface_parameter_gradient=(
            surface_parameter_gradient.detach().cpu().numpy()
        ),
    )
    artifact_sha256: dict[str, str] = {
        str(arrays_path.relative_to(output_dir)): _sha256_file(arrays_path)
    }
    rgb_images = {
        "clean_source": clean_source_rgb,
        "composited_source": composited_source_rgb,
        "clean_effective": clean_effective_rgb,
        "composited_effective": composited_effective_rgb,
    }
    for name, value in rgb_images.items():
        path = images_dir / f"{prefix}_{name}.png"
        artifact_sha256[str(path.relative_to(output_dir))] = _save_rgb(
            path,
            value,
        )
    alpha_path = images_dir / f"{prefix}_mujoco_alpha_union.png"
    artifact_sha256[str(alpha_path.relative_to(output_dir))] = _save_alpha(
        alpha_path,
        mujoco_instance_alpha,
    )
    return artifact_sha256[str(arrays_path.relative_to(output_dir))], (
        artifact_sha256
    )


def run_compositor_zero_delta_audit(
    cfg: CompositorZeroDeltaConfig,
) -> Path:
    """运行真实 states audit，并返回 manifest path。"""

    _validate_code_commit(cfg.code_commit)
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("零 delta Gate state_ids 不得为空")
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
        raise RuntimeError("零 delta Gate 需要真实 CUDA nvdiffrast device")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "compositor_zero_delta_metrics.jsonl"
    manifest_path = output_dir / "compositor_zero_delta_manifest.json"
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有零 delta Gate 权威文件")

    asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"])
    mesh_path = Path(asset["mesh"])
    texture_path = Path(asset["texture"])
    checkpoint_path = Path(cfg.pretrained_checkpoint)
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
        raise RuntimeError("零 delta Gate 只支持正方形 checkpoint 输入")
    policy_view_transform: DifferentiablePolicyViewTransform = (
        build_policy_view_transform(
            source_resolution=POLICY_SOURCE_RESOLUTION,
            model_input_resolution=model_input_height,
        )
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
    surface_parameter = renderer.get_texture_param()

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if min(state_ids) < 0 or max(state_ids) >= len(initial_states):
        raise ValueError("零 delta Gate state_id 超出初始状态范围")

    rows: list[ZeroDeltaCompositorRow] = []
    state_fingerprints: dict[str, str] = {}
    backend_identity: Optional[dict[str, Any]] = None
    task_description: Optional[str] = None
    for state_id in state_ids:
        print(f"[COMPOSITOR-ZERO-DELTA] state {state_id}")
        initial_state = initial_states[state_id]
        state_fingerprints[str(state_id)] = _state_fingerprint(initial_state)
        env, current_task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=POLICY_SOURCE_RESOLUTION,
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
            clean_source_rgb = get_libero_image(
                observation,
                POLICY_SOURCE_RESOLUTION,
            )
            clean_source_tensor = _rgb_tensor(
                clean_source_rgb,
                device=model.device,
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
            surface_parameter.grad = None
            with static_scene_evidence_transaction(
                env,
                body_ids,
            ) as transaction:
                transaction_poses = find_target_body_poses(
                    env,
                    asset["search"],
                    model.device,
                )
                if tuple(pose.body_id for pose in transaction_poses) != body_ids:
                    raise RuntimeError(
                        "static transaction 内目标实例集合改变"
                    )
                segmentation = capture_instance_segmentation(
                    env,
                    instance_roots,
                    camera_name="agentview",
                    resolution=POLICY_SOURCE_RESOLUTION,
                )
                backend_identity = asdict(segmentation.backend)
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
                renderer_delta = (
                    renderer_evidence.adversarial_rgb
                    - renderer_evidence.clean_rgb
                )
                # 直接探测真实 compositor 的 joint-visible 响应，不能只证明
                # nvdiffrast 前景在 MuJoCo 遮挡之外仍有梯度。
                gradient_probe = composition.total_delta.sum()
                surface_gradient = torch.autograd.grad(
                    gradient_probe,
                    surface_parameter,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            if transaction.after is None:
                raise RuntimeError("static transaction 未生成 after snapshot")

            exact_clean = build_exact_deployment_view_stages(
                clean_source_rgb,
                specification=policy_view_transform.specification,
            )
            # Gate 的 renderer 梯度已经单独取证；后续模型只做严格 forward
            # 等价比较，显式断图以免保留无用的 512² raster graph。
            composited_source_tensor = composition.composited_rgb.detach()
            composited_stages = policy_view_transform.build_stages(
                composited_source_tensor
            )
            composited_source_rgb = _uint8_hwc(composited_source_tensor)
            composited_pre_crop_rgb = _uint8_hwc(
                composited_stages.pre_crop_canvas
            )
            composited_effective_rgb = _uint8_hwc(
                composited_stages.effective_view
            )
            composited_pixel_values_tensor = (
                image_preprocessor.build_fused_pixel_values(
                    composited_stages.effective_view
                ).to(torch.bfloat16)
            )

            recording_processor = _RecordingProcessor(processor)
            clean_action = get_vla_action(
                model,
                recording_processor,
                cfg.pretrained_checkpoint,
                {"full_image": exact_clean.pre_crop_rgb},
                current_task_description,
                cfg.unnorm_key,
                center_crop=True,
            )
            if recording_processor.last_inputs is None or (
                recording_processor.last_rgb is None
            ):
                raise RuntimeError("零 delta Gate 未捕获 clean processor 输入")
            clean_effective_rgb = recording_processor.last_rgb
            clean_inputs = recording_processor.last_inputs
            if "pixel_values" not in clean_inputs:
                raise RuntimeError("clean processor inputs 缺少 pixel_values")
            clean_pixel_values_tensor = clean_inputs["pixel_values"]
            composited_inputs = _clone_inputs(clean_inputs)
            composited_inputs["pixel_values"] = composited_pixel_values_tensor
            ensure_trailing_empty_token(composited_inputs)
            action_dim = int(model.get_action_dim(cfg.unnorm_key))
            generation_arguments = {
                "max_new_tokens": action_dim,
                "do_sample": False,
                "pad_token_id": processor.tokenizer.pad_token_id,
            }
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                clean_generated = model.generate(
                    **clean_inputs,
                    **generation_arguments,
                )
                composited_generated = model.generate(
                    **composited_inputs,
                    **generation_arguments,
                )
            clean_tokens = clean_generated[0, -action_dim:]
            composited_tokens = composited_generated[0, -action_dim:]
            composited_action = decode_action_from_generated_ids(
                model,
                composited_generated,
                cfg.unnorm_key,
            )

            clean_pixel_values = (
                clean_pixel_values_tensor.float().detach().cpu().numpy()
            )
            composited_pixel_values = (
                composited_pixel_values_tensor.float().detach().cpu().numpy()
            )
            clean_token_array = clean_tokens.detach().cpu().numpy()
            composited_token_array = composited_tokens.detach().cpu().numpy()
            clean_action_array = np.asarray(clean_action, dtype=np.float64)
            composited_action_array = np.asarray(
                composited_action,
                dtype=np.float64,
            )
            arrays_sha256, artifact_sha256 = _persist_state_artifacts(
                output_dir=output_dir,
                state_id=state_id,
                clean_source_rgb=exact_clean.source_rgb,
                composited_source_rgb=composited_source_rgb,
                clean_pre_crop_rgb=exact_clean.pre_crop_rgb,
                composited_pre_crop_rgb=composited_pre_crop_rgb,
                clean_effective_rgb=clean_effective_rgb,
                composited_effective_rgb=composited_effective_rgb,
                clean_pixel_values=clean_pixel_values,
                composited_pixel_values=composited_pixel_values,
                clean_action_token_ids=clean_token_array,
                composited_action_token_ids=composited_token_array,
                clean_action=clean_action_array,
                composited_action=composited_action_array,
                mujoco_instance_alpha=mujoco_alpha,
                renderer_valid_mask=renderer_evidence.visibility_mask,
                renderer_delta=renderer_delta,
                per_instance_delta=composition.per_instance_delta,
                total_delta=composition.total_delta,
                surface_parameter_gradient=surface_gradient,
            )
            per_instance_delta_linf = tuple(
                float(value)
                for value in composition.per_instance_delta.detach()
                .abs()
                .flatten(start_dim=1)
                .amax(dim=1)
                .cpu()
                .tolist()
            )
            instance_visible_pixel_counts = tuple(
                int(value)
                for value in (
                    mujoco_alpha * renderer_evidence.visibility_mask
                ).detach()
                .sum(dim=(1, 2, 3))
                .to(torch.int64)
                .cpu()
                .tolist()
            )
            evidence = ZeroDeltaCompositorEvidence(
                state_id=state_id,
                clean_source_rgb=exact_clean.source_rgb,
                composited_source_rgb=composited_source_rgb,
                clean_pre_crop_rgb=exact_clean.pre_crop_rgb,
                composited_pre_crop_rgb=composited_pre_crop_rgb,
                clean_effective_rgb=clean_effective_rgb,
                composited_effective_rgb=composited_effective_rgb,
                clean_pixel_values=clean_pixel_values,
                composited_pixel_values=composited_pixel_values,
                clean_action_token_ids=clean_token_array,
                composited_action_token_ids=composited_token_array,
                clean_action=clean_action_array,
                composited_action=composited_action_array,
                renderer_delta_linf=float(
                    renderer_delta.detach().abs().amax().item()
                ),
                per_instance_delta_linf=per_instance_delta_linf,
                total_delta_linf=float(
                    composition.total_delta.detach().abs().amax().item()
                ),
                saturated_pixel_fraction=(
                    composition.saturated_pixel_fraction
                ),
                saturated_channel_fraction=(
                    composition.saturated_channel_fraction
                ),
                clean_renderer_requires_grad=bool(
                    renderer_evidence.clean_rgb.requires_grad
                ),
                valid_renderer_pixel_count=sum(
                    instance_visible_pixel_counts
                ),
                instance_visible_pixel_counts=(
                    instance_visible_pixel_counts
                ),
                surface_parameter_gradient=GradientEvidence.from_tensor(
                    surface_gradient
                ),
                transaction_verified=bool(transaction.verified),
                transaction_before_sha256=(
                    transaction.before.fingerprint_sha256
                ),
                transaction_after_sha256=(
                    transaction.after.fingerprint_sha256
                ),
                arrays_npz_sha256=arrays_sha256,
                artifact_sha256=artifact_sha256,
            )
            row = evaluate_zero_delta_compositor_evidence(
                evidence,
                code_commit=cfg.code_commit,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "state_id": state_id,
                        "gate_pass": row["gate_pass"],
                        "failures": row["failures"],
                        "valid_renderer_pixel_count": row[
                            "valid_renderer_pixel_count"
                        ],
                        "surface_gradient_l2": row[
                            "surface_parameter_gradient"
                        ]["l2_norm"],
                    },
                    sort_keys=True,
                )
            )
        finally:
            env.close()

    summary = summarize_zero_delta_compositor_rows(
        rows,
        expected_state_ids=state_ids,
    )
    metrics_sha256 = write_zero_delta_compositor_jsonl(
        rows,
        output_path=metrics_path,
    )
    metadata = {
        "code_commit": cfg.code_commit,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "task_name": str(getattr(task, "name", task_description)),
        "task_description": task_description,
        "object_name": cfg.object_name,
        "state_ids": list(state_ids),
        "state_fingerprints": state_fingerprints,
        "seed": cfg.seed,
        "num_steps_wait": cfg.num_steps_wait,
        "backend": backend_identity,
        "deployment": {
            "policy_source_resolution": POLICY_SOURCE_RESOLUTION,
            "policy_pre_crop_resolution": model_input_height,
            "center_crop_area": 0.9,
            "view": "primary",
            "texture_parameterization": "geometry_vertex",
        },
        "asset_sha256": _fingerprint_files(
            (xml_path, mesh_path, texture_path)
        ),
        "versions": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
        },
        "command": " ".join(
            shlex.quote(argument) for argument in sys.argv
        ),
    }
    manifest_sha256 = write_zero_delta_compositor_manifest(
        summary=summary,
        metadata=metadata,
        metrics_jsonl_sha256=metrics_sha256,
        output_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "gate_pass": summary["gate_pass"],
                "metrics_sha256": metrics_sha256,
                "manifest_sha256": manifest_sha256,
                **summary,
            },
            sort_keys=True,
        )
    )
    if not summary["gate_pass"]:
        raise RuntimeError("零 Surface Delta Compositor Gate 未通过")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_compositor_zero_delta_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
