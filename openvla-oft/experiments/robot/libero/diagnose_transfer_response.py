"""诊断攻击纹理在 OpenVLA-OFT 上的视觉特征与动作响应。

该入口刻意不执行闭环 rollout。它在指定的固定 LIBERO 初始状态上分别采集
clean 与 adversarial observation，并对完全相同的模型输入计算：

- 主视角和腕部图像差异；
- OFT SigLIP patch feature 差异；
- 原始连续动作 chunk 与夹爪符号差异。

输出用于快速判断迁移失败发生在视觉编码、动作决策还是更下游的闭环轨迹层。
攻击 PNG 通过 MuJoCo XML 直接激活，不经过 PNG→顶点→PNG 重采样。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image


THIS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_EXTERNAL_OFT_ROOT: Path = Path(
    "/home/xiaomengqi/src/github/paper_code/openvla-oft"
)
DEFAULT_LIBERO_ROOT: Path = Path(
    "/home/xiaomengqi/src/github/paper_code/LIBERO"
)

# 本仓库只维护实验入口；OFT 的 prismatic 包仍由原项目提供。先注册原项目根
# 目录，再把当前目录放到最前，确保 transfer_* 模块使用本次受版本控制的实现。
external_oft_root: Path = Path(
    os.environ.get("OPENVLA_OFT_ROOT", str(DEFAULT_EXTERNAL_OFT_ROOT))
).resolve()
libero_root: Path = Path(
    os.environ.get("LIBERO_ROOT", str(DEFAULT_LIBERO_ROOT))
).resolve()
sys.path.insert(0, str(external_oft_root))
sys.path.insert(0, str(libero_root))
sys.path.insert(0, str(THIS_DIR))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
    prepare_images_for_vla,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from transfer_evaluation import (  # noqa: E402
    TemporaryTextureActivation,
    parse_eval_state_ids,
    validate_active_texture,
)
from transfer_response import (  # noqa: E402
    compute_action_diagnostics,
    compute_distance,
    extract_primary_siglip_features,
    metrics_to_dict,
)


@dataclass
class DiagnosticConfig:
    """OFT helper 与本诊断共同需要的强类型配置。"""

    pretrained_checkpoint: str
    texture_path: str
    output_dir: str
    task_suite_name: str = "libero_spatial"
    object_name: str = "akita_black_bowl"
    task_id: int = 0
    state_ids: str = "10,11,13,15,16"
    object_xml_path: Optional[str] = None
    num_steps_wait: int = 10
    seed: int = 7

    # 以下字段由 OFT 的 model/action helper 读取。第一版诊断只支持当前实验所用
    # 的 L1 regression OFT checkpoint，避免引入无关分支。
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 1
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: Optional[str] = None


@dataclass(frozen=True)
class CapturedObservation:
    """一个固定状态上送入 policy 前的数据。

    Attributes:
        primary_image: RGB uint8，shape ``[224, 224, 3]``。
        wrist_image: RGB uint8，shape ``[224, 224, 3]``。
        robot_state: float32，shape ``[8]``，依次为位置、轴角和夹爪 qpos。
    """

    primary_image: np.ndarray
    wrist_image: np.ndarray
    robot_state: np.ndarray


@dataclass(frozen=True)
class ModelResponse:
    """OFT 对单个 observation 的原始响应（尚未二值化夹爪）。"""

    primary_siglip: np.ndarray
    wrist_siglip: np.ndarray
    actions: np.ndarray


def _parse_args(argv: Optional[Sequence[str]] = None) -> DiagnosticConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--texture_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--state_ids", default="10,11,13,15,16")
    parser.add_argument("--object_xml_path", default=None)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args(argv)
    return DiagnosticConfig(**vars(arguments))


def _resolve_object_xml(cfg: DiagnosticConfig) -> Path:
    """从 LIBERO 实际 assets 根目录定位物体 XML。"""
    if cfg.object_xml_path is not None:
        xml_path: Path = Path(cfg.object_xml_path).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        return xml_path

    assets_root: Path = Path(get_libero_path("assets")).resolve()
    candidates: tuple[Path, ...] = (
        assets_root
        / "stable_scanned_objects"
        / cfg.object_name
        / f"{cfg.object_name}.xml",
        assets_root
        / "stable_hope_objects"
        / cfg.object_name
        / f"{cfg.object_name}.xml",
    )
    existing: list[Path] = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"无法唯一定位 {cfg.object_name!r} 的 XML，候选={candidates}"
        )
    return existing[0]


def _check_unnorm_key(cfg: DiagnosticConfig, model: Any) -> None:
    """使用与 OFT rollout 相同的 action/proprio 反归一化统计。"""
    preferred_key: str = cfg.unnorm_key or cfg.task_suite_name
    norm_stats: Any = getattr(model, "norm_stats", None)
    if not isinstance(norm_stats, dict):
        raise RuntimeError("OFT model 缺少 norm_stats")
    if (
        preferred_key not in norm_stats
        and f"{preferred_key}_no_noops" in norm_stats
    ):
        preferred_key = f"{preferred_key}_no_noops"
    if preferred_key not in norm_stats:
        raise ValueError(
            f"Action unnorm key {preferred_key!r} 不在 model.norm_stats 中"
        )
    cfg.unnorm_key = preferred_key


def _collect_observation(
    cfg: DiagnosticConfig,
    task: Any,
    initial_state: np.ndarray,
) -> tuple[CapturedObservation, str]:
    """重建环境并在固定初始状态等待后采集一次 observation。"""
    render_resolution: int = 512
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=render_resolution,
    )
    try:
        env.reset()
        raw_observation: dict[str, Any] = env.set_init_state(initial_state)
        env.env.sim.forward()
        for _ in range(cfg.num_steps_wait):
            raw_observation, _, _, _ = env.step(
                get_libero_dummy_action(cfg.model_family)
            )

        model_input_size_value: int | tuple[int, int] = get_image_resize_size(cfg)
        if not isinstance(model_input_size_value, int):
            raise RuntimeError("当前 OFT 诊断只支持正方形模型输入")
        model_input_size: int = model_input_size_value
        primary_high_resolution: np.ndarray = np.ascontiguousarray(
            get_libero_image(raw_observation)
        )
        wrist_high_resolution: np.ndarray = np.ascontiguousarray(
            get_libero_wrist_image(raw_observation)
        )
        primary_image: np.ndarray = np.asarray(
            Image.fromarray(primary_high_resolution).resize(
                (model_input_size, model_input_size)
            ),
            dtype=np.uint8,
        )
        wrist_image: np.ndarray = np.asarray(
            Image.fromarray(wrist_high_resolution).resize(
                (model_input_size, model_input_size)
            ),
            dtype=np.uint8,
        )
        quaternion: np.ndarray = np.asarray(
            raw_observation["robot0_eef_quat"],
            dtype=np.float64,
        ).copy()
        robot_state: np.ndarray = np.concatenate(
            (
                np.asarray(raw_observation["robot0_eef_pos"]),
                quat2axisangle(quaternion),
                np.asarray(raw_observation["robot0_gripper_qpos"]),
            )
        ).astype(np.float32, copy=False)
        if robot_state.shape != (8,):
            raise RuntimeError(
                f"OFT proprio 应为 [8]，实际 shape={robot_state.shape}"
            )
        return (
            CapturedObservation(
                primary_image=primary_image,
                wrist_image=wrist_image,
                robot_state=robot_state,
            ),
            str(task_description),
        )
    finally:
        env.close()


def _query_model_response(
    cfg: DiagnosticConfig,
    model: Any,
    processor: Any,
    action_head: torch.nn.Module,
    proprio_projector: torch.nn.Module,
    observation: CapturedObservation,
    task_description: str,
) -> ModelResponse:
    """以 rollout 的同一路径预处理图像并读取 feature/action。"""
    prompt: str = (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )
    prepared_images: list[Image.Image] = prepare_images_for_vla(
        [observation.primary_image, observation.wrist_image],
        cfg,
    )
    device: torch.device = model.device
    with torch.inference_mode():
        primary_inputs: Any = processor(prompt, prepared_images[0]).to(
            device,
            dtype=torch.bfloat16,
        )
        wrist_inputs: Any = processor(prompt, prepared_images[1]).to(
            device,
            dtype=torch.bfloat16,
        )
        # 单张图像的 processor 输出为 [1, 6, 224, 224]，通道依次对应
        # timm_model_ids。保留两个视角的独立 feature，便于判断攻击是否只在
        # 主视角可见。
        primary_pixels: torch.Tensor = primary_inputs["pixel_values"]
        wrist_pixels: torch.Tensor = wrist_inputs["pixel_values"]
        primary_features: torch.Tensor = extract_primary_siglip_features(
            model,
            primary_pixels,
        )
        wrist_features: torch.Tensor = extract_primary_siglip_features(
            model,
            wrist_pixels,
        )

        primary_inputs["pixel_values"] = torch.cat(
            [primary_pixels, wrist_pixels],
            dim=1,
        )
        assert cfg.unnorm_key is not None
        proprio_stats: dict[str, Any] = model.norm_stats[cfg.unnorm_key][
            "proprio"
        ]
        normalized_proprio: np.ndarray = normalize_proprio(
            observation.robot_state.copy(),
            proprio_stats,
        )
        predicted_actions, _ = model.predict_action(
            **primary_inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=normalized_proprio,
            proprio_projector=proprio_projector,
            noisy_action_projector=None,
            action_head=action_head,
            use_film=cfg.use_film,
        )

    actions: np.ndarray = np.asarray(predicted_actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(
            "OFT action chunk 应为 [T, 7]，实际 shape="
            f"{actions.shape}"
        )
    return ModelResponse(
        primary_siglip=primary_features.float().cpu().numpy(),
        wrist_siglip=wrist_features.float().cpu().numpy(),
        actions=actions,
    )


def _pair_metrics(
    clean_observation: CapturedObservation,
    adversarial_observation: CapturedObservation,
    clean_response: ModelResponse,
    adversarial_response: ModelResponse,
) -> dict[str, Any]:
    """形成单个 state 的分层诊断记录。"""
    clean_combined: np.ndarray = np.concatenate(
        [clean_response.primary_siglip, clean_response.wrist_siglip],
        axis=1,
    )
    adversarial_combined: np.ndarray = np.concatenate(
        [
            adversarial_response.primary_siglip,
            adversarial_response.wrist_siglip,
        ],
        axis=1,
    )
    return {
        "observation": {
            "primary_rgb": metrics_to_dict(
                compute_distance(
                    clean_observation.primary_image,
                    adversarial_observation.primary_image,
                )
            ),
            "wrist_rgb": metrics_to_dict(
                compute_distance(
                    clean_observation.wrist_image,
                    adversarial_observation.wrist_image,
                )
            ),
            "robot_state": metrics_to_dict(
                compute_distance(
                    clean_observation.robot_state,
                    adversarial_observation.robot_state,
                )
            ),
        },
        "siglip": {
            "primary": metrics_to_dict(
                compute_distance(
                    clean_response.primary_siglip,
                    adversarial_response.primary_siglip,
                )
            ),
            "wrist": metrics_to_dict(
                compute_distance(
                    clean_response.wrist_siglip,
                    adversarial_response.wrist_siglip,
                )
            ),
            "combined": metrics_to_dict(
                compute_distance(clean_combined, adversarial_combined)
            ),
        },
        "action": metrics_to_dict(
            compute_action_diagnostics(
                clean_response.actions,
                adversarial_response.actions,
            )
        ),
        "shapes": {
            "primary_rgb": list(clean_observation.primary_image.shape),
            "robot_state": list(clean_observation.robot_state.shape),
            "primary_siglip": list(clean_response.primary_siglip.shape),
            "wrist_siglip": list(clean_response.wrist_siglip.shape),
            "actions": list(clean_response.actions.shape),
        },
        # 后续跨模型梯度审计需要用相同 proprio 条件复现 OFT action head。
        # clean/adv state 已由上方距离指标验证完全相同，只保存一份即可。
        "robot_state": clean_observation.robot_state.tolist(),
        "clean_first_action": clean_response.actions[0].tolist(),
        "adversarial_first_action": adversarial_response.actions[0].tolist(),
    }


def _flatten_numeric(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """把嵌套指标展开，供跨 state 计算 mean/median/max。"""
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix: str = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, prefix=child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def _aggregate_state_metrics(
    state_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """对每个标量指标汇总 mean、median 与 max。"""
    buckets: dict[str, list[float]] = {}
    for record in state_records:
        for key, value in _flatten_numeric(record["metrics"]).items():
            buckets.setdefault(key, []).append(value)
    return {
        key: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for key, values in sorted(buckets.items())
    }


def _save_pair_images(
    output_dir: Path,
    state_id: int,
    clean: CapturedObservation,
    adversarial: CapturedObservation,
) -> None:
    """保存真正送入 policy 的成对输入，便于人工核对。"""
    image_dir: Path = output_dir / "paired_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(clean.primary_image).save(
        image_dir / f"state_{state_id:02d}_clean_primary.png"
    )
    Image.fromarray(adversarial.primary_image).save(
        image_dir / f"state_{state_id:02d}_adversarial_primary.png"
    )
    Image.fromarray(clean.wrist_image).save(
        image_dir / f"state_{state_id:02d}_clean_wrist.png"
    )
    Image.fromarray(adversarial.wrist_image).save(
        image_dir / f"state_{state_id:02d}_adversarial_wrist.png"
    )


def run_diagnostic(cfg: DiagnosticConfig) -> Path:
    """执行固定状态的 clean/adv OFT 响应诊断并返回 JSON 路径。"""
    if cfg.num_images_in_input != 2:
        raise ValueError("当前诊断固定使用 OFT 的主视角+腕部双图像输入")
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    set_seed_everywhere(cfg.seed)

    texture_info = validate_active_texture(cfg.texture_path)
    xml_path: Path = _resolve_object_xml(cfg)
    output_dir: Path = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_types: dict[str, Any] = benchmark.get_benchmark_dict()
    if cfg.task_suite_name not in benchmark_types:
        raise ValueError(f"未知 task suite: {cfg.task_suite_name}")
    task_suite: Any = benchmark_types[cfg.task_suite_name]()
    if cfg.task_id < 0 or cfg.task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id 越界: {cfg.task_id}")
    task: Any = task_suite.get_task(cfg.task_id)
    initial_states: np.ndarray = task_suite.get_task_init_states(cfg.task_id)
    state_ids: tuple[int, ...] = parse_eval_state_ids(
        cfg.state_ids,
        total_states=len(initial_states),
        maximum_count=len(initial_states),
    )

    print(f"[INFO] Loading OFT checkpoint: {cfg.pretrained_checkpoint}")
    model: Any = get_model(cfg)
    processor: Any = get_processor(cfg)
    _check_unnorm_key(cfg, model)
    action_head: torch.nn.Module = get_action_head(cfg, model.llm_dim)
    proprio_projector: torch.nn.Module = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,
    )

    clean_observations: dict[int, CapturedObservation] = {}
    adversarial_observations: dict[int, CapturedObservation] = {}
    task_description: Optional[str] = None
    print(f"[INFO] Collecting clean states: {list(state_ids)}")
    for state_id in state_ids:
        clean_observations[state_id], description = _collect_observation(
            cfg,
            task,
            initial_states[state_id],
        )
        task_description = description

    # SIGTERM 转为 Python 异常，使 XML 事务有机会恢复；SIGINT 本身会触发
    # KeyboardInterrupt 并走同一个 finally 路径。
    previous_sigterm_handler: Any = signal.getsignal(signal.SIGTERM)

    def _raise_on_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"收到信号 {signum}")

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        with TemporaryTextureActivation(xml_path) as transaction:
            transaction.activate(
                object_name=cfg.object_name,
                active_texture_path=texture_info.path,
            )
            print(
                "[INFO] Collecting adversarial states with direct MuJoCo texture: "
                f"sha256={texture_info.sha256}"
            )
            for state_id in state_ids:
                adversarial_observations[state_id], description = (
                    _collect_observation(
                        cfg,
                        task,
                        initial_states[state_id],
                    )
                )
                if task_description != description:
                    raise RuntimeError("clean/adv task description 不一致")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    if task_description is None:
        raise RuntimeError("没有采集到 task description")

    state_records: list[dict[str, Any]] = []
    for state_id in state_ids:
        print(f"[INFO] Querying OFT response for state {state_id}")
        clean_observation: CapturedObservation = clean_observations[state_id]
        adversarial_observation: CapturedObservation = (
            adversarial_observations[state_id]
        )
        clean_response: ModelResponse = _query_model_response(
            cfg,
            model,
            processor,
            action_head,
            proprio_projector,
            clean_observation,
            task_description,
        )
        adversarial_response: ModelResponse = _query_model_response(
            cfg,
            model,
            processor,
            action_head,
            proprio_projector,
            adversarial_observation,
            task_description,
        )
        _save_pair_images(
            output_dir,
            state_id,
            clean_observation,
            adversarial_observation,
        )
        state_records.append(
            {
                "state_id": state_id,
                "metrics": _pair_metrics(
                    clean_observation,
                    adversarial_observation,
                    clean_response,
                    adversarial_response,
                ),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "openvla_oft_clean_adversarial_response",
        "config": asdict(cfg),
        "task_description": task_description,
        "object_xml_path": str(xml_path),
        "texture": asdict(texture_info),
        "state_ids": list(state_ids),
        "states": state_records,
        "aggregate": _aggregate_state_metrics(state_records),
        "interpretation": {
            "small_siglip_change": "优先检查共享视觉目标和预处理对齐",
            "siglip_changes_but_action_stable": "优先调整代理损失与决策裕量",
            "action_changes_but_rollout_stable": "再检查轨迹时序和训练状态覆盖",
        },
    }
    # ActiveTextureInfo.path 是 Path，显式转换后避免依赖自定义 JSON encoder。
    report["texture"]["path"] = str(texture_info.path)
    report_path: Path = output_dir / "oft_transfer_response.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Diagnostic report: {report_path}")
    return report_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_diagnostic(_parse_args(argv))


if __name__ == "__main__":
    main()
