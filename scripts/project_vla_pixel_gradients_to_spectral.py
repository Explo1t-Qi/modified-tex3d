"""把 OpenVLA/OFT 像素梯度经同一 renderer VJP 投影到 K 维谱空间。

本入口不加载任何 VLA 模型。它复用已经保存的 ``dL/dRGB``，在相同 LIBERO
固定状态上重建主视角和腕部相机矩阵，然后计算：

``dL/dC = J(renderer(C))^T @ dL/dRGB``，其中 ``C`` 为 ``[K,3]`` 谱系数。

OFT 梯度只写入诊断产物，绝不参与 source-only 纹理训练或选基。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image


# 必须在导入 LIBERO/Robosuite 前固定无窗口渲染后端。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# 当前 robosuite 安装位于只读环境；Numba 无法在该位置创建函数 cache。响应
# 诊断也使用同一设置，因此禁用 JIT 同时避免权限错误并保持状态重建路径一致。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

THIS_FILE: Path = Path(__file__).resolve()
REPOSITORY_ROOT: Path = THIS_FILE.parents[1]
LIBERO_EXPERIMENT_DIR: Path = (
    REPOSITORY_ROOT / "openvla/experiments/robot/libero"
)
for import_path in (REPOSITORY_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)

if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

from libero.libero import benchmark  # noqa: E402
from libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_pose,
)
from scripts.vla_pixel_gradient_audit import (  # noqa: E402
    PixelGradientArtifact,
    visible_perturbation_mask,
)
from scripts.vla_spectral_gradient_projection import (  # noqa: E402
    SpectralGradientProjectionArtifact,
    summarize_spectral_projection,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ProjectionConfig:
    """谱投影所需的文件、相机和 source objective 配置。"""

    source_path: str
    target_path: str
    response_report: str
    spectral_basis_path: str
    coefficients_path: str
    output_npz: str
    output_json: str
    source_action_weight: float = 0.1
    source_feature_weight: float = 4.0
    attack_epsilon: float = 128.0 / 255.0
    primary_camera_name: str = "agentview"
    wrist_camera_name: str = "robot0_eye_in_hand"
    device: str = "cuda"


def _parse_args(argv: Optional[Sequence[str]] = None) -> ProjectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--response_report", required=True)
    parser.add_argument("--spectral_basis_path", required=True)
    parser.add_argument("--coefficients_path", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--source_action_weight", type=float, default=0.1)
    parser.add_argument("--source_feature_weight", type=float, default=4.0)
    parser.add_argument("--attack_epsilon", type=float, default=128.0 / 255.0)
    parser.add_argument("--primary_camera_name", default="agentview")
    parser.add_argument("--wrist_camera_name", default="robot0_eye_in_hand")
    parser.add_argument("--device", default="cuda")
    return ProjectionConfig(**vars(parser.parse_args(argv)))


def _load_coefficients(path: str | Path) -> torch.Tensor:
    """安全读取 float ``[K,3]`` 系数；产物必须只包含 tensor。"""
    resolved_path: Path = Path(path).resolve()
    try:
        raw_value: Any = torch.load(
            resolved_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # 兼容尚不支持 weights_only 的旧 PyTorch；该路径是本实验自产 tensor。
        raw_value = torch.load(resolved_path, map_location="cpu")
    if not isinstance(raw_value, torch.Tensor):
        raise TypeError("谱系数 .pt 必须直接保存 torch.Tensor")
    coefficients: torch.Tensor = raw_value.detach().float().cpu()
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError(
            f"谱系数应为 [K,3]，实际 shape={tuple(coefficients.shape)}"
        )
    if not torch.isfinite(coefficients).all():
        raise ValueError("谱系数包含 NaN/Inf")
    return coefficients


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _observed_wrist_masks(
    *,
    input_directory: Path,
    state_ids: Sequence[int],
) -> BoolArray:
    """从 OFT 成对 policy 输入提取真实腕部纹理变化区域 ``[S,H,W]``。"""
    image_directory: Path = input_directory / "paired_inputs"
    masks: list[BoolArray] = []
    for state_id in state_ids:
        clean: np.ndarray = _load_rgb(
            image_directory / f"state_{state_id:02d}_clean_wrist.png"
        )
        adversarial: np.ndarray = _load_rgb(
            image_directory / f"state_{state_id:02d}_adversarial_wrist.png"
        )
        masks.append(visible_perturbation_mask(clean, adversarial))
    return np.stack(masks, axis=0)


def _simulation(env: Any) -> Any:
    return env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim


def _require_camera(env: Any, camera_name: str) -> None:
    """诊断禁止把不存在的腕部相机静默回退为 camera 0。"""
    try:
        camera_id: int = int(_simulation(env).model.camera_name2id(camera_name))
    except Exception as error:
        raise RuntimeError(f"MuJoCo 中不存在 camera {camera_name!r}") from error
    if camera_id < 0:
        raise RuntimeError(f"MuJoCo camera ID 非法: {camera_name!r} -> {camera_id}")


def _pixel_gradient_tensor(
    gradient: FloatArray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """float64 HWC ``[H,W,3]`` → renderer 同 device 的 float32 NHWC。"""
    array: np.ndarray = np.asarray(gradient, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"像素梯度应为 [H,W,3]，实际 {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).unsqueeze(0)


def _project_view_gradients(
    *,
    renderer: DifferentiableRenderer,
    mvp: torch.Tensor,
    model_rotation: torch.Tensor,
    pixel_gradients: Sequence[FloatArray],
) -> tuple[list[FloatArray], BoolArray]:
    """一次 rasterization 上计算多组 ``J^T @ pixel_gradient``。

    ``pixel_gradients`` 每项为 ``[H,W,3]``；返回每项对应的 ``[K,3]`` 与
    renderer visibility ``[H,W]``。各 VJP 共享完全相同的相机 Jacobian。
    """
    if not pixel_gradients:
        raise ValueError("pixel_gradients 不能为空")
    height, width, channels = pixel_gradients[0].shape
    if channels != 3 or any(
        gradient.shape != (height, width, 3) for gradient in pixel_gradients
    ):
        raise ValueError("同一视角的像素梯度必须共享 [H,W,3] shape")

    rendered_rgb: torch.Tensor
    visibility: torch.Tensor
    rendered_rgb, visibility = renderer.render(
        mvp,
        resolution=(height, width),
        model_rot=model_rotation,
    )
    parameter: torch.Tensor = renderer.get_texture_param()
    coefficient_gradients: list[FloatArray] = []
    for gradient_index, pixel_gradient in enumerate(pixel_gradients):
        coefficient_gradient: torch.Tensor = torch.autograd.grad(
            outputs=rendered_rgb,
            inputs=parameter,
            grad_outputs=_pixel_gradient_tensor(
                pixel_gradient,
                device=rendered_rgb.device,
            ),
            retain_graph=gradient_index < len(pixel_gradients) - 1,
            create_graph=False,
        )[0]
        coefficient_gradients.append(
            np.asarray(
                coefficient_gradient.detach().double().cpu().numpy(),
                dtype=np.float64,
            )
        )
    visibility_mask: BoolArray = np.asarray(
        visibility.detach().cpu().numpy()[0, ..., 0] > 0.5,
        dtype=np.bool_,
    )
    return coefficient_gradients, visibility_mask


def run_projection(cfg: ProjectionConfig) -> tuple[Path, Path]:
    """重建固定状态相机，运行两视角 VJP 并写入 NPZ/JSON。"""
    source: PixelGradientArtifact = PixelGradientArtifact.load(cfg.source_path)
    target: PixelGradientArtifact = PixelGradientArtifact.load(cfg.target_path)
    if not np.array_equal(source.state_ids, target.state_ids):
        raise ValueError("source/target state IDs 不一致")
    if (
        source.primary_feature_gradients.shape
        != target.primary_feature_gradients.shape
    ):
        raise ValueError("source/target 像素梯度 shape 不一致")
    if target.wrist_feature_gradients is None or target.wrist_action_gradients is None:
        raise ValueError("目标像素梯度缺少 OFT 腕部 Feature/Action")

    report_path: Path = Path(cfg.response_report).resolve()
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    report_state_ids: tuple[int, ...] = tuple(int(v) for v in report["state_ids"])
    state_ids: tuple[int, ...] = tuple(int(v) for v in source.state_ids.tolist())
    if report_state_ids != state_ids:
        raise ValueError(
            f"响应报告 states {report_state_ids} 与梯度 states {state_ids} 不一致"
        )
    report_config: dict[str, Any] = report["config"]
    object_name: str = str(report_config["object_name"])
    if object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {object_name!r}")
    object_spec: ObjectAssetSpec = OBJECT_ASSETS[object_name]
    task_suite_name: str = str(report_config["task_suite_name"])
    task_id: int = int(report_config["task_id"])
    num_steps_wait: int = int(report_config["num_steps_wait"])
    if task_suite_name != object_spec["task_suite"]:
        raise ValueError("响应报告 task suite 与物体注册表不一致")

    coefficients_cpu: torch.Tensor = _load_coefficients(cfg.coefficients_path)
    num_basis: int = int(coefficients_cpu.shape[0])
    device: torch.device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 当前不可用；nvdiffrast 谱投影必须在 GPU 上运行")
    renderer = DifferentiableRenderer(
        mesh_path=object_spec["mesh"],
        orig_texture_path=object_spec["texture"],
        device=str(device),
        scale_xyz=parse_mesh_scale(object_spec["xml"]),
        epsilon=cfg.attack_epsilon,
        texture_parameterization="spectral",
        spectral_basis_path=cfg.spectral_basis_path,
        spectral_basis_count=num_basis,
    ).to(device)
    parameter: torch.Tensor = renderer.get_texture_param()
    if parameter.shape != coefficients_cpu.shape:
        raise ValueError(
            f"renderer 参数 {tuple(parameter.shape)} 与系数 "
            f"{tuple(coefficients_cpu.shape)} 不一致"
        )
    with torch.no_grad():
        parameter.copy_(coefficients_cpu.to(device))

    benchmark_types: dict[str, Any] = benchmark.get_benchmark_dict()
    task_suite: Any = benchmark_types[task_suite_name]()
    task: Any = task_suite.get_task(task_id)
    initial_states: np.ndarray = task_suite.get_task_init_states(task_id)
    input_height: int = int(source.primary_feature_gradients.shape[1])
    input_width: int = int(source.primary_feature_gradients.shape[2])
    if input_height != input_width:
        raise ValueError("当前诊断只支持正方形 policy 输入")

    source_feature_coefficients: list[FloatArray] = []
    source_action_coefficients: list[FloatArray] = []
    target_primary_feature_coefficients: list[FloatArray] = []
    target_primary_action_coefficients: list[FloatArray] = []
    target_wrist_feature_coefficients: list[FloatArray] = []
    target_wrist_action_coefficients: list[FloatArray] = []
    primary_renderer_masks: list[BoolArray] = []
    wrist_renderer_masks: list[BoolArray] = []

    for sample_index, state_id in enumerate(state_ids):
        print(f"[INFO] Projecting state {state_id}")
        env, _ = get_libero_env(task, "openvla", resolution=input_height)
        try:
            env.reset()
            observation: Any = env.set_init_state(initial_states[state_id])
            _simulation(env).forward()
            for _ in range(num_steps_wait):
                observation, _, _, _ = env.step(
                    get_libero_dummy_action("openvla")
                )
            del observation
            _require_camera(env, cfg.primary_camera_name)
            _require_camera(env, cfg.wrist_camera_name)
            target_pose: TargetBodyPose = find_target_body_pose(
                env,
                object_spec["search"],
                device,
            )
            if target_pose.body_id < 0:
                raise RuntimeError(f"state {state_id} 无法定位目标物体 body")
            primary_mvp: torch.Tensor = compute_render_mvp(
                env,
                target_pose.model_matrix,
                resolution=(input_width, input_height),
                camera_name=cfg.primary_camera_name,
            )
            wrist_mvp: torch.Tensor = compute_render_mvp(
                env,
                target_pose.model_matrix,
                resolution=(input_width, input_height),
                camera_name=cfg.wrist_camera_name,
            )
            primary_projected, primary_mask = _project_view_gradients(
                renderer=renderer,
                mvp=primary_mvp,
                model_rotation=target_pose.model_matrix[:3, :3],
                pixel_gradients=(
                    source.primary_feature_gradients[sample_index],
                    source.primary_action_gradients[sample_index],
                    target.primary_feature_gradients[sample_index],
                    target.primary_action_gradients[sample_index],
                ),
            )
            wrist_projected, wrist_mask = _project_view_gradients(
                renderer=renderer,
                mvp=wrist_mvp,
                model_rotation=target_pose.model_matrix[:3, :3],
                pixel_gradients=(
                    target.wrist_feature_gradients[sample_index],
                    target.wrist_action_gradients[sample_index],
                ),
            )
        finally:
            env.close()

        source_feature_coefficients.append(primary_projected[0])
        source_action_coefficients.append(primary_projected[1])
        target_primary_feature_coefficients.append(primary_projected[2])
        target_primary_action_coefficients.append(primary_projected[3])
        target_wrist_feature_coefficients.append(wrist_projected[0])
        target_wrist_action_coefficients.append(wrist_projected[1])
        primary_renderer_masks.append(primary_mask)
        wrist_renderer_masks.append(wrist_mask)

    input_directory_value: Any = (target.metadata or {}).get("input_directory")
    if not isinstance(input_directory_value, str):
        raise ValueError("target 像素梯度 metadata 缺少 input_directory")
    wrist_observed_masks: BoolArray = _observed_wrist_masks(
        input_directory=Path(input_directory_value).resolve(),
        state_ids=state_ids,
    )
    artifact = SpectralGradientProjectionArtifact(
        state_ids=np.asarray(state_ids, dtype=np.int64),
        source_primary_feature=np.stack(source_feature_coefficients, axis=0),
        source_primary_action=np.stack(source_action_coefficients, axis=0),
        target_primary_feature=np.stack(
            target_primary_feature_coefficients, axis=0
        ),
        target_primary_action=np.stack(
            target_primary_action_coefficients, axis=0
        ),
        target_wrist_feature=np.stack(target_wrist_feature_coefficients, axis=0),
        target_wrist_action=np.stack(target_wrist_action_coefficients, axis=0),
        primary_renderer_masks=np.stack(primary_renderer_masks, axis=0),
        wrist_renderer_masks=np.stack(wrist_renderer_masks, axis=0),
        primary_observed_masks=source.perturbation_masks,
        wrist_observed_masks=wrist_observed_masks,
        metadata={
            "source_pixel_gradient_path": str(Path(cfg.source_path).resolve()),
            "target_pixel_gradient_path": str(Path(cfg.target_path).resolve()),
            "response_report": str(report_path),
            "spectral_basis_path": str(Path(cfg.spectral_basis_path).resolve()),
            "coefficients_path": str(Path(cfg.coefficients_path).resolve()),
            "reference_point": "trained_k256_spectral_coefficients",
            "renderer_photometry": "default_uncalibrated_shared_jacobian",
            "primary_camera_name": cfg.primary_camera_name,
            "wrist_camera_name": cfg.wrist_camera_name,
            "target_gradient_role": "diagnostic_only_not_training_or_selection",
        },
    )
    output_npz: Path = artifact.save(cfg.output_npz)
    summary: dict[str, Any] = summarize_spectral_projection(
        artifact,
        source_action_weight=cfg.source_action_weight,
        source_feature_weight=cfg.source_feature_weight,
    )
    output_json: Path = Path(cfg.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Spectral projection artifact: {output_npz}")
    print(f"[DONE] Spectral projection summary: {output_json}")
    return output_npz, output_json


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_projection(_parse_args(argv))


if __name__ == "__main__":
    main()
