"""运行 Gate 2R renderer-to-bake response 真实审计。

本命令固定使用 source OpenVLA、Primary effective view、states 0–9 和原始 OBJ
全部几何顶点上的 R/G/B 三个 ``2/255`` 常量 Surface Delta。它不读取 Action
梯度、Fixed Support、OFT 或 rollout 结果。每个 ``(state,probe)`` 比较：

* Visibility-Masked Renderer Delta Composition 的 effective-view RGB 响应；
* 同一 probe bake PNG 激活后 MuJoCo observation 的 effective-view RGB 响应。

权威输出为 ``response_metrics.jsonl``。每行绑定无 pickle NPZ、五类可视化、
Runtime Asset Transaction、clean/bake 静止状态 fingerprint、纹理/配置/代码 hash
和三条路径的 Untargeted Clean-Action Margin。Action margin 只作诊断，不进入
第一版 Gate。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
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
from PIL import Image, ImageDraw
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
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    render_shared_texture_instances,
)
from openvla_attack.objective import (  # noqa: E402
    ACTION_TOKEN_START,
    ActionTokenLogits,
    extract_action_token_logits,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiablePolicyViewTransform,
    ExactDeploymentViewStages,
    build_exact_deployment_view_stages,
    build_policy_view_transform,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.renderer_bake_response_audit import (  # noqa: E402
    PROBE_CHANNELS,
    PROBE_SURFACE_DELTA,
    RendererBakeResponseEvidence,
    RendererBakeResponseProvenance,
    RendererBakeResponseRow,
    VisibilityResponseStatus,
    compute_untargeted_clean_action_margins,
    evaluate_renderer_bake_response_evidence,
    summarize_renderer_bake_response_rows,
    write_renderer_bake_response_csv,
    write_renderer_bake_response_jsonl,
    write_renderer_bake_response_manifest,
    write_renderer_bake_response_npz,
)
from openvla_attack.runtime_assets import (  # noqa: E402
    RuntimeAssetTransaction,
)
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from openvla_attack.visibility_capture import (  # noqa: E402
    capture_instance_segmentation,
    capture_static_scene_snapshot,
    static_scene_evidence_transaction,
)
from openvla_attack.visibility_compositing import (  # noqa: E402
    compose_visibility_masked_renderer_delta,
)
from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
)
from openvla_attack.visibility_view_evidence import (  # noqa: E402
    build_view_visibility_evidence,
)
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass
class RendererBakeResponseConfig:
    """Gate 2R CLI schema；center crop 与 probe 设计不可由命令行改写。"""

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
class _CleanStateProbe:
    """clean env 中一个 probe 的 surrogate 与模型诊断缓存。"""

    channel: str
    visibility_status: VisibilityResponseStatus
    alpha: NDArray[np.floating[Any]]
    d_sur: NDArray[np.floating[Any]]
    surrogate_pixel_values: torch.Tensor
    surrogate_action_margins: NDArray[np.float64]


class _RecordingProcessor:
    """透明代理 processor，并记录部署路径实际送入的 PIL/input。"""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.last_rgb: Optional[NDArray[np.uint8]] = None
        self.last_inputs: Optional[MutableMapping[str, torch.Tensor]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        image: Any = args[1] if len(args) > 1 else kwargs.get("images")
        if not isinstance(image, Image.Image):
            raise RuntimeError("Gate 2R processor 未收到 PIL Image")
        self.last_rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        inputs: Any = self._processor(*args, **kwargs)
        self.last_inputs = inputs
        return inputs


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> RendererBakeResponseConfig:
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
    return RendererBakeResponseConfig(**vars(parser.parse_args(argv)))


def _validate_code_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit 必须是40位小写十六进制 Git SHA")


def _validate_repository_commit(code_commit: str) -> None:
    """拒绝错误 HEAD 或 tracked dirty tree，保护实验代码 provenance。"""

    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = completed.stdout.strip()
    if actual_commit != code_commit:
        raise RuntimeError(
            "--code_commit 与仓库 HEAD 不一致："
            f"{code_commit} != {actual_commit}"
        )
    tracked_status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(
            "Gate 2R 要求 tracked worktree clean，发现未提交修改：\n"
            f"{tracked_status}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _combined_evidence_sha256(
    evidence: RendererBakeResponseEvidence,
) -> str:
    return _canonical_sha256(
        {
            "state_id": evidence.state_id,
            "probe_channel": evidence.probe_channel,
            "visibility_status": evidence.visibility_status,
            "alpha": _array_sha256(evidence.alpha),
            "d_sur": _array_sha256(evidence.d_sur),
            "d_bake": _array_sha256(evidence.d_bake),
            "clean_action_margins": _array_sha256(
                evidence.clean_action_margins
            ),
            "surrogate_action_margins": _array_sha256(
                evidence.surrogate_action_margins
            ),
            "bake_action_margins": _array_sha256(
                evidence.bake_action_margins
            ),
        }
    )


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


def _float_hwc(tensor: torch.Tensor) -> NDArray[np.float32]:
    """float NCHW ``[1,3,H,W]`` → 独立 float32 HWC。"""

    return (
        tensor.detach()
        .to(dtype=torch.float32)
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


def _set_probe(renderer: DifferentiableRenderer, channel: str) -> None:
    parameter = renderer.get_texture_param()
    channel_index = PROBE_CHANNELS.index(channel)
    with torch.no_grad():
        parameter.zero_()
        parameter[:, channel_index] = PROBE_SURFACE_DELTA
    surface_delta = renderer.get_surface_delta().detach()
    expected = torch.zeros_like(surface_delta)
    expected[:, channel_index] = PROBE_SURFACE_DELTA
    if not torch.equal(surface_delta, expected):
        raise RuntimeError(
            f"probe {channel} 未产生严格全顶点 {PROBE_SURFACE_DELTA} Surface Delta"
        )


def _save_baked_texture(
    renderer: DifferentiableRenderer,
    *,
    output_path: Path,
) -> str:
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baked = renderer.get_baked_adv_texture().squeeze(0).detach().cpu().numpy()
    pixels = np.rint(baked * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(output_path)
    return _sha256_file(output_path)


def _clone_inputs(
    inputs: MutableMapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def _teacher_margins(
    model: Any,
    *,
    pixel_values: torch.Tensor,
    clean_generated_ids: torch.Tensor,
    expected_clean_classes: Optional[torch.Tensor] = None,
) -> tuple[NDArray[np.float64], torch.Tensor]:
    """以固定 clean sequence 前缀计算一条图像路径的逐 token margin。"""

    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        outputs: Any = model(
            input_ids=clean_generated_ids,
            attention_mask=torch.ones_like(clean_generated_ids),
            pixel_values=pixel_values,
            output_hidden_states=False,
        )
    action_tokens: ActionTokenLogits = extract_action_token_logits(
        outputs.logits,
        clean_generated_ids,
    )
    if expected_clean_classes is not None and not torch.equal(
        action_tokens.clean_classes,
        expected_clean_classes,
    ):
        raise RuntimeError("三条路径的 teacher-forced clean classes 不一致")
    margins = compute_untargeted_clean_action_margins(
        action_tokens.logits.float().detach().cpu().numpy(),
        action_tokens.clean_classes.detach().cpu().numpy(),
    )
    return margins, action_tokens.clean_classes.detach()


def _capture_clean_model_context(
    *,
    model: Any,
    processor: Any,
    checkpoint: str,
    unnorm_key: Optional[str],
    task_description: str,
    exact_clean: ExactDeploymentViewStages,
) -> tuple[
    torch.Tensor,
    NDArray[np.float64],
    torch.Tensor,
]:
    """获得固定 clean labels、clean margin/classes 和 clean pixel values。"""

    recording_processor = _RecordingProcessor(processor)
    get_vla_action(
        model,
        recording_processor,
        checkpoint,
        {"full_image": exact_clean.pre_crop_rgb},
        task_description,
        unnorm_key,
        center_crop=True,
    )
    if recording_processor.last_inputs is None or recording_processor.last_rgb is None:
        raise RuntimeError("Gate 2R 未捕获 clean processor 输入")
    if not np.array_equal(
        recording_processor.last_rgb,
        exact_clean.effective_view_rgb,
    ):
        raise RuntimeError("Gate 2R clean effective view 与 processor PIL 输入不一致")
    clean_inputs = _clone_inputs(recording_processor.last_inputs)
    ensure_trailing_empty_token(clean_inputs)
    if "pixel_values" not in clean_inputs:
        raise RuntimeError("Gate 2R clean inputs 缺少 pixel_values")
    clean_pixel_values = clean_inputs["pixel_values"]
    action_dim = int(model.get_action_dim(unnorm_key))
    generation_arguments: dict[str, Any] = {
        **clean_inputs,
        "max_new_tokens": action_dim,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        clean_generated_ids: torch.Tensor = model.generate(
            **generation_arguments
        )
    clean_margins, clean_classes = _teacher_margins(
        model,
        pixel_values=clean_pixel_values,
        clean_generated_ids=clean_generated_ids,
    )
    if clean_classes.shape != (action_dim,):
        raise RuntimeError(
            "Gate 2R teacher-forced action 数与 checkpoint action_dim 不一致"
        )
    generated_classes = (
        clean_generated_ids[0, -action_dim:] - ACTION_TOKEN_START
    ).to(clean_classes.device)
    if not torch.equal(clean_classes, generated_classes):
        raise RuntimeError("Gate 2R clean generated token 与 teacher labels 不一致")

    # 冻结 objective 还要求零 delta 时 clean token 是同一 teacher forward argmax。
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        clean_outputs: Any = model(
            input_ids=clean_generated_ids,
            attention_mask=torch.ones_like(clean_generated_ids),
            pixel_values=clean_pixel_values,
            output_hidden_states=False,
        )
    clean_action_tokens = extract_action_token_logits(
        clean_outputs.logits,
        clean_generated_ids,
    )
    if not torch.equal(
        clean_action_tokens.logits.argmax(dim=1),
        clean_classes,
    ):
        raise RuntimeError("Gate 2R clean token 与 teacher-forced argmax 不一致")
    return clean_generated_ids, clean_margins, clean_classes


def _settle_environment(
    env: Any,
    initial_state: Any,
    *,
    num_steps_wait: int,
    model_family: str,
    dummy_action: Any,
) -> Any:
    """重置到指定初始 state，并执行固定空动作等待。"""

    env.reset()
    observation = env.set_init_state(initial_state)
    env.env.sim.forward()
    for _ in range(num_steps_wait):
        observation, _, _, _ = env.step(dummy_action(model_family))
    return observation


def _save_alpha_visualization(path: Path, alpha: np.ndarray) -> str:
    pixels = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    return _sha256_file(path)


def _save_delta_visualization(path: Path, delta: np.ndarray) -> str:
    """以固定 32×放大保存有符号 RGB delta；机器数值仍以 NPZ 为准。"""

    pixels = np.rint(
        (0.5 + 0.5 * np.clip(delta * 32.0, -1.0, 1.0)) * 255.0
    ).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
    return _sha256_file(path)


def _save_weighted_scatter(
    path: Path,
    *,
    alpha: np.ndarray,
    d_sur: np.ndarray,
    d_bake: np.ndarray,
) -> str:
    """保存 alpha 加权的 D_sur-vs-D_bake RGB 通道散点图。"""

    size = 512
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    plot_margin = 12
    plot_maximum = size - 1 - plot_margin
    plot_span = plot_maximum - plot_margin
    plot_center = plot_margin + plot_span // 2
    # y=x 是理想响应参考；水平/垂直轴帮助识别单侧零响应。
    draw.line(
        (plot_margin, plot_maximum, plot_maximum, plot_margin),
        fill=(220, 220, 220),
    )
    draw.line(
        (plot_center, plot_margin, plot_center, plot_maximum),
        fill=(170, 170, 170),
    )
    draw.line(
        (plot_margin, plot_center, plot_maximum, plot_center),
        fill=(170, 170, 170),
    )
    visible = alpha > 0.0
    scale = float(
        max(
            np.max(np.abs(d_sur[visible]), initial=0.0),
            np.max(np.abs(d_bake[visible]), initial=0.0),
            1e-12,
        )
    )
    colors = ((220, 40, 40), (40, 170, 40), (40, 80, 220))
    for channel_index, base_color in enumerate(colors):
        # 与 sign-consistency 分母一致：双零通道分量不提供响应方向证据。
        channel_evidence = visible & (
            (d_sur[..., channel_index] != 0.0)
            | (d_bake[..., channel_index] != 0.0)
        )
        x_values = d_sur[..., channel_index][channel_evidence]
        y_values = d_bake[..., channel_index][channel_evidence]
        weights = alpha[channel_evidence]
        stride = max(1, int(np.ceil(x_values.size / 50_000)))
        for x_value, y_value, weight in zip(
            x_values[::stride],
            y_values[::stride],
            weights[::stride],
            strict=True,
        ):
            x = int(
                np.clip(
                    plot_margin
                    + (float(x_value) / scale + 1.0) * 0.5 * plot_span,
                    plot_margin,
                    plot_maximum,
                )
            )
            y = int(
                np.clip(
                    plot_margin
                    + (1.0 - (float(y_value) / scale + 1.0) * 0.5)
                    * plot_span,
                    plot_margin,
                    plot_maximum,
                )
            )
            # alpha 仍控制颜色深浅，但设置足够的最低对比度并使用半径2的点，
            # 使 uint8 导致的少量离散坐标在便读 PNG 中可见。
            intensity = 0.70 + 0.30 * float(weight)
            color = tuple(
                int(255.0 - (255.0 - value) * intensity)
                for value in base_color
            )
            draw.ellipse(
                (x - 2, y - 2, x + 2, y + 2),
                fill=color,
            )
    for legend_index, (label, color) in enumerate(
        zip(("R", "G", "B"), colors, strict=True)
    ):
        legend_x = 18 + legend_index * 42
        draw.rectangle((legend_x, 18, legend_x + 9, 27), fill=color)
        draw.text((legend_x + 13, 15), label, fill=(30, 30, 30))
    image.save(path)
    return _sha256_file(path)


def _persist_visualizations(
    *,
    output_dir: Path,
    state_id: int,
    channel: str,
    alpha: np.ndarray,
    d_sur: np.ndarray,
    d_bake: np.ndarray,
) -> dict[str, str]:
    images_dir = output_dir / "visualizations"
    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"state_{state_id:02d}_probe_{channel}"
    paths = {
        "alpha": images_dir / f"{prefix}_alpha.png",
        "d_sur": images_dir / f"{prefix}_d_sur.png",
        "d_bake": images_dir / f"{prefix}_d_bake.png",
        "difference": images_dir / f"{prefix}_difference.png",
        "weighted_scatter": images_dir / f"{prefix}_weighted_scatter.png",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"拒绝覆盖已有 Gate 2R visualization: {prefix}")
    fingerprints = {
        str(paths["alpha"].relative_to(output_dir)): _save_alpha_visualization(
            paths["alpha"], alpha
        ),
        str(paths["d_sur"].relative_to(output_dir)): _save_delta_visualization(
            paths["d_sur"], d_sur
        ),
        str(paths["d_bake"].relative_to(output_dir)): _save_delta_visualization(
            paths["d_bake"], d_bake
        ),
        str(paths["difference"].relative_to(output_dir)): _save_delta_visualization(
            paths["difference"], d_sur - d_bake
        ),
    }
    fingerprints[str(paths["weighted_scatter"].relative_to(output_dir))] = (
        _save_weighted_scatter(
            paths["weighted_scatter"],
            alpha=alpha,
            d_sur=d_sur,
            d_bake=d_bake,
        )
    )
    return fingerprints


def run_renderer_bake_response_audit(
    cfg: RendererBakeResponseConfig,
) -> Path:
    """运行完整 10×3 Gate 2R，并返回 manifest path。"""

    _validate_code_commit(cfg.code_commit)
    _validate_repository_commit(cfg.code_commit)
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    if not cfg.center_crop:
        raise ValueError("Gate 2R 固定要求 center_crop=True")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("Gate 2R state_ids 不得为空")
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
        raise RuntimeError("Gate 2R 需要真实 CUDA nvdiffrast device")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "response_metrics.jsonl"
    csv_path = output_dir / "response_metrics.csv"
    manifest_path = output_dir / "response_manifest.json"
    if any(path.exists() for path in (metrics_path, csv_path, manifest_path)):
        raise FileExistsError("拒绝覆盖已有 Gate 2R 权威文件")

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
        raise RuntimeError("Gate 2R 只支持正方形 checkpoint 输入")
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

    texture_dir = output_dir / "probe_textures"
    baked_texture_paths: dict[str, Path] = {}
    baked_texture_hashes: dict[str, str] = {}
    for channel in PROBE_CHANNELS:
        _set_probe(renderer, channel)
        baked_path = texture_dir / f"all_vertices_probe_{channel}.png"
        baked_texture_paths[channel] = baked_path
        baked_texture_hashes[channel] = _save_baked_texture(
            renderer,
            output_path=baked_path,
        )
    renderer.reset_texture()

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if min(state_ids) < 0 or max(state_ids) >= len(initial_states):
        raise ValueError("Gate 2R state_id 超出初始状态范围")

    config_payload = asdict(cfg)
    config_sha256 = _canonical_sha256(config_payload)
    clean_xml_sha256 = _sha256_file(xml_path)
    clean_texture_sha256 = _sha256_file(texture_path)
    transaction: Optional[RuntimeAssetTransaction] = None
    rows: list[RendererBakeResponseRow] = []
    task_description: Optional[str] = None
    backend_identity: Optional[dict[str, Any]] = None
    backup_paths: tuple[Path, ...] = ()
    try:
        transaction = RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=texture_path,
            object_name=cfg.object_name,
            backup_tag=f"gate2r_{cfg.code_commit[:12]}",
            install_process_handlers=False,
        )
        backup_paths = tuple(
            path
            for path in (
                transaction.xml_backup_path,
                transaction.real_texture_backup_path,
            )
            if path is not None
        )
        for state_id in state_ids:
            print(f"[GATE-2R] clean/surrogate state {state_id}")
            transaction.restore(
                context=f"Gate 2R state {state_id} clean",
                remove_backups=False,
            )
            initial_state = initial_states[state_id]
            initial_state_sha256 = _array_sha256(np.asarray(initial_state))
            clean_env, current_task_description = get_libero_env(
                task,
                cfg.model_family,
                resolution=POLICY_SOURCE_RESOLUTION,
            )
            task_description = current_task_description
            clean_generated_ids: torch.Tensor
            clean_classes: torch.Tensor
            clean_action_margins: NDArray[np.float64]
            clean_exact: ExactDeploymentViewStages
            clean_sim_state_sha256: str
            clean_probes: dict[str, _CleanStateProbe] = {}
            body_ids: tuple[int, ...]
            try:
                observation = _settle_environment(
                    clean_env,
                    initial_state,
                    num_steps_wait=cfg.num_steps_wait,
                    model_family=cfg.model_family,
                    dummy_action=get_libero_dummy_action,
                )
                clean_source_rgb = get_libero_image(
                    observation,
                    POLICY_SOURCE_RESOLUTION,
                )
                clean_source_tensor = _rgb_tensor(
                    clean_source_rgb,
                    device=model.device,
                )
                clean_exact = build_exact_deployment_view_stages(
                    clean_source_rgb,
                    specification=policy_view_transform.specification,
                )
                clean_effective_tensor = _rgb_tensor(
                    clean_exact.effective_view_rgb,
                    device=model.device,
                )
                target_poses = find_target_body_poses(
                    clean_env,
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
                with static_scene_evidence_transaction(
                    clean_env,
                    body_ids,
                ) as static_transaction:
                    transaction_poses = find_target_body_poses(
                        clean_env,
                        asset["search"],
                        model.device,
                    )
                    if tuple(pose.body_id for pose in transaction_poses) != body_ids:
                        raise RuntimeError("Gate 2R static transaction 实例集合改变")
                    segmentation = capture_instance_segmentation(
                        clean_env,
                        instance_roots,
                        camera_name="agentview",
                        resolution=POLICY_SOURCE_RESOLUTION,
                    )
                    backend_identity = asdict(segmentation.backend)
                    mujoco_alpha = segmentation.parsed.instance_alpha.to(
                        device=model.device,
                        dtype=torch.float32,
                    )
                    for channel in PROBE_CHANNELS:
                        _set_probe(renderer, channel)
                        renderer_evidence = render_shared_texture_instances(
                            renderer,
                            _build_instances(
                                clean_env,
                                transaction_poses,
                                device=model.device,
                            ),
                            resolution=(
                                POLICY_SOURCE_RESOLUTION,
                                POLICY_SOURCE_RESOLUTION,
                            ),
                        )
                        view_evidence = build_view_visibility_evidence(
                            mujoco_alpha,
                            renderer_evidence.visibility_mask,
                        )
                        effective_alpha = (
                            view_evidence.mujoco_stages.effective.alpha.sum(
                                dim=0
                            )[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=True)
                        )
                        composition = compose_visibility_masked_renderer_delta(
                            clean_source_tensor,
                            mujoco_alpha,
                            renderer_evidence.adversarial_rgb,
                            renderer_evidence.clean_rgb,
                            renderer_evidence.visibility_mask,
                        )
                        surrogate_effective = policy_view_transform.build_stages(
                            composition.composited_rgb
                        ).effective_view
                        d_sur = _float_hwc(
                            surrogate_effective - clean_effective_tensor
                        )
                        surrogate_pixels = (
                            image_preprocessor.build_fused_pixel_values(
                                surrogate_effective
                            ).to(torch.bfloat16)
                        )
                        # Action labels 在 static transaction 后构造；这里只暂存
                        # surrogate pixel，避免持有全部 512² raster graph。
                        clean_probes[channel] = _CleanStateProbe(
                            channel=channel,
                            visibility_status=(
                                view_evidence.evaluation.status
                            ),
                            alpha=effective_alpha.copy(),
                            d_sur=d_sur,
                            surrogate_pixel_values=(
                                surrogate_pixels.detach()
                            ),
                            surrogate_action_margins=np.empty(
                                (0,), dtype=np.float64
                            ),
                        )
                    renderer.reset_texture()
                if static_transaction.after is None or not static_transaction.verified:
                    raise RuntimeError("Gate 2R static transaction 未通过")
                clean_sim_state_sha256 = (
                    static_transaction.before.fingerprint_sha256
                )
                (
                    clean_generated_ids,
                    clean_action_margins,
                    clean_classes,
                ) = _capture_clean_model_context(
                    model=model,
                    processor=processor,
                    checkpoint=cfg.pretrained_checkpoint,
                    unnorm_key=cfg.unnorm_key,
                    task_description=current_task_description,
                    exact_clean=clean_exact,
                )
                for channel, cached_probe in tuple(clean_probes.items()):
                    surrogate_margins, surrogate_classes = _teacher_margins(
                        model,
                        pixel_values=cached_probe.surrogate_pixel_values,
                        clean_generated_ids=clean_generated_ids,
                        expected_clean_classes=clean_classes,
                    )
                    if not torch.equal(surrogate_classes, clean_classes):
                        raise RuntimeError("surrogate clean classes 不一致")
                    clean_probes[channel] = _CleanStateProbe(
                        channel=channel,
                        visibility_status=cached_probe.visibility_status,
                        alpha=cached_probe.alpha,
                        d_sur=cached_probe.d_sur,
                        surrogate_pixel_values=(
                            cached_probe.surrogate_pixel_values
                        ),
                        surrogate_action_margins=surrogate_margins,
                    )
            finally:
                clean_env.close()

            for channel in PROBE_CHANNELS:
                print(f"[GATE-2R] baked state {state_id} probe {channel}")
                baked_path = baked_texture_paths[channel]
                mirrored = transaction.activate_texture(
                    baked_path,
                    mirror_real_texture=True,
                )
                active_texture_sha256 = _sha256_file(texture_path)
                baked_env: Optional[Any] = None
                bake_sim_state_sha256 = ""
                d_bake: NDArray[np.float32]
                bake_action_margins: NDArray[np.float64]
                try:
                    if not mirrored:
                        raise RuntimeError("Gate 2R 未同步激活真实 MuJoCo texture")
                    baked_env, baked_task_description = get_libero_env(
                        task,
                        cfg.model_family,
                        resolution=POLICY_SOURCE_RESOLUTION,
                    )
                    if baked_task_description != current_task_description:
                        raise RuntimeError("clean/bake task description 不一致")
                    baked_observation = _settle_environment(
                        baked_env,
                        initial_state,
                        num_steps_wait=cfg.num_steps_wait,
                        model_family=cfg.model_family,
                        dummy_action=get_libero_dummy_action,
                    )
                    baked_poses = find_target_body_poses(
                        baked_env,
                        asset["search"],
                        model.device,
                    )
                    if tuple(pose.body_id for pose in baked_poses) != body_ids:
                        raise RuntimeError("clean/bake 目标实例集合不一致")
                    bake_sim_state_sha256 = capture_static_scene_snapshot(
                        baked_env,
                        body_ids,
                    ).fingerprint_sha256
                    baked_source_rgb = get_libero_image(
                        baked_observation,
                        POLICY_SOURCE_RESOLUTION,
                    )
                    baked_exact = build_exact_deployment_view_stages(
                        baked_source_rgb,
                        specification=policy_view_transform.specification,
                    )
                    baked_effective_tensor = _rgb_tensor(
                        baked_exact.effective_view_rgb,
                        device=model.device,
                    )
                    clean_effective_tensor = _rgb_tensor(
                        clean_exact.effective_view_rgb,
                        device=model.device,
                    )
                    d_bake = _float_hwc(
                        baked_effective_tensor - clean_effective_tensor
                    )
                    baked_pixels = image_preprocessor.build_fused_pixel_values(
                        baked_effective_tensor
                    ).to(torch.bfloat16)
                    bake_action_margins, bake_classes = _teacher_margins(
                        model,
                        pixel_values=baked_pixels,
                        clean_generated_ids=clean_generated_ids,
                        expected_clean_classes=clean_classes,
                    )
                    if not torch.equal(bake_classes, clean_classes):
                        raise RuntimeError("bake clean classes 不一致")
                finally:
                    if baked_env is not None:
                        baked_env.close()
                    transaction.restore(
                        context=f"Gate 2R state {state_id} probe {channel}",
                        remove_backups=False,
                    )

                restored_xml_sha256 = _sha256_file(xml_path)
                restored_texture_sha256 = _sha256_file(texture_path)
                clean_probe = clean_probes[channel]
                evidence = RendererBakeResponseEvidence(
                    state_id=state_id,
                    probe_channel=channel,  # type: ignore[arg-type]
                    alpha=clean_probe.alpha,
                    d_sur=clean_probe.d_sur,
                    d_bake=d_bake,
                    clean_action_margins=clean_action_margins,
                    surrogate_action_margins=(
                        clean_probe.surrogate_action_margins
                    ),
                    bake_action_margins=bake_action_margins,
                    visibility_status=clean_probe.visibility_status,
                )
                arrays_path = (
                    output_dir
                    / "arrays"
                    / f"state_{state_id:02d}_probe_{channel}.npz"
                )
                arrays_sha256 = write_renderer_bake_response_npz(
                    evidence,
                    output_path=arrays_path,
                )
                artifact_sha256 = _persist_visualizations(
                    output_dir=output_dir,
                    state_id=state_id,
                    channel=channel,
                    alpha=evidence.alpha,
                    d_sur=evidence.d_sur,
                    d_bake=evidence.d_bake,
                )
                arrays_relative = str(arrays_path.relative_to(output_dir))
                texture_relative = str(baked_path.relative_to(output_dir))
                artifact_sha256[arrays_relative] = arrays_sha256
                artifact_sha256[texture_relative] = baked_texture_hashes[channel]
                provenance = RendererBakeResponseProvenance(
                    code_commit=cfg.code_commit,
                    config_sha256=config_sha256,
                    evidence_sha256=_combined_evidence_sha256(evidence),
                    initial_state_sha256=initial_state_sha256,
                    clean_sim_state_sha256=clean_sim_state_sha256,
                    bake_sim_state_sha256=bake_sim_state_sha256,
                    asset_xml_path=str(xml_path),
                    real_texture_path=str(texture_path),
                    transaction_verified=bool(
                        mirrored
                        and clean_sim_state_sha256 == bake_sim_state_sha256
                        and active_texture_sha256
                        == baked_texture_hashes[channel]
                        and restored_xml_sha256 == clean_xml_sha256
                        and restored_texture_sha256 == clean_texture_sha256
                    ),
                    xml_sha256_before=clean_xml_sha256,
                    xml_sha256_after_restore=restored_xml_sha256,
                    clean_texture_sha256=clean_texture_sha256,
                    baked_texture_sha256=baked_texture_hashes[channel],
                    active_texture_sha256=active_texture_sha256,
                    restored_texture_sha256=restored_texture_sha256,
                    arrays_npz_path=arrays_relative,
                    arrays_npz_sha256=arrays_sha256,
                    artifact_sha256=artifact_sha256,
                )
                row = evaluate_renderer_bake_response_evidence(
                    evidence,
                    provenance,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "state_id": state_id,
                            "probe_channel": channel,
                            "status": row["status"],
                            "surrogate_rms": row["surrogate_rms"],
                            "bake_rms": row["bake_rms"],
                            "cos_alpha": row["cos_alpha"],
                            "gate_pass": row["gate_pass"],
                            "failures": row["failures"],
                        },
                        sort_keys=True,
                    )
                )
    finally:
        if transaction is not None:
            transaction.close(context="Gate 2R final cleanup")

    backup_paths_removed = bool(backup_paths) and not any(
        path.exists() for path in backup_paths
    )
    summary = summarize_renderer_bake_response_rows(
        rows,
        expected_state_ids=state_ids,
    )
    metrics_sha256 = write_renderer_bake_response_jsonl(
        rows,
        output_path=metrics_path,
    )
    csv_sha256 = write_renderer_bake_response_csv(
        rows,
        output_path=csv_path,
    )
    metadata = {
        "code_commit": cfg.code_commit,
        "config": config_payload,
        "config_sha256": config_sha256,
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "checkpoint": str(checkpoint_path),
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "task_name": str(getattr(task, "name", task_description)),
        "task_description": task_description,
        "object_name": cfg.object_name,
        "state_ids": list(state_ids),
        "probe_channels": list(PROBE_CHANNELS),
        "probe_surface_delta": PROBE_SURFACE_DELTA,
        "backend": backend_identity,
        "asset_sha256_after_restore": _fingerprint_files(
            (xml_path, mesh_path, texture_path)
        ),
        "runtime_asset_backup_paths_removed": backup_paths_removed,
        "visualization": {
            "signed_delta_magnification": 32.0,
            "scatter_scale": "per_state_probe_joint_abs_max",
        },
        "deployment": {
            "policy_source_resolution": POLICY_SOURCE_RESOLUTION,
            "policy_pre_crop_resolution": model_input_height,
            "center_crop_area": 0.9,
            "view": "primary",
            "texture_parameterization": "geometry_vertex",
        },
        "versions": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
        },
    }
    manifest_sha256 = write_renderer_bake_response_manifest(
        summary=summary,
        metadata=metadata,
        metrics_jsonl_sha256=metrics_sha256,
        derived_csv_sha256=csv_sha256,
        output_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "gate_pass": summary["gate_pass"],
                "metrics_sha256": metrics_sha256,
                "csv_sha256": csv_sha256,
                "manifest_sha256": manifest_sha256,
                **summary,
            },
            sort_keys=True,
        )
    )
    if not backup_paths_removed:
        raise RuntimeError("Gate 2R Runtime Asset backup 未完成清理")
    if not summary["gate_pass"]:
        raise RuntimeError("Gate 2R 未通过严格判定")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_renderer_bake_response_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
