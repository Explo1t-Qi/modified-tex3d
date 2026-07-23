"""OpenVLA 在 LIBERO 上进行对抗纹理训练与鲁棒性评估的命令行入口。

本文件只编排 run、task 和 episode 生命周期。OpenVLA 模型专用的数据转换、
Attack Training、可微渲染、优化、评估状态机、Attack Artifact 与 Clean Asset
恢复分别位于 ``openvla_attack`` package。
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TextIO, Union

import draccus
import tqdm
import wandb

# 必须在导入 LIBERO/Robosuite 前指定 headless renderer backend。
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 直接执行该文件和通过 importlib 加载该文件时，sys.path 的初始值不同。
# 先显式加入当前目录，确保两种入口都能导入 OpenVLA 专用 module。
LIBERO_EXPERIMENT_DIR: str = str(Path(__file__).parent)
if LIBERO_EXPERIMENT_DIR not in sys.path:
    sys.path.append(LIBERO_EXPERIMENT_DIR)

from openvla_attack.assets import (
    LIBERO_ROOT,
    OBJECT_ASSETS,
    MeshScale,
    ObjectAssetSpec,
    parse_mesh_scale,
)

if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)

from libero.libero import benchmark

from libero_utils import get_libero_env, save_rollout_video
from openvla_attack.artifacts import AttackArtifactStore
from openvla_attack.evaluation import (
    LiberoEpisodeRunner,
    RolloutResult,
)
from openvla_attack.frame_collection import TrainingProcessor
from openvla_attack.renderer import (
    AdversarialTextureLoadResult,
    DifferentiableRenderer,
)
from openvla_attack.runtime_assets import RuntimeAssetTransaction
from openvla_attack.scene import SearchKeywords
from openvla_attack.training import (
    AttackTrainer,
    AttackTrainingModel,
)

sys.path.append(str(Path(__file__).parent.parent))

OPENVLA_REPO_ROOT: str = str(Path(__file__).resolve().parents[3])
if OPENVLA_REPO_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_REPO_ROOT)

from openvla_utils import get_processor
from robot_utils import (
    DATE_TIME,
    get_model,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "./openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    object_name: str = "akita_black_bowl"
    override_mesh_path: Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path: Optional[str] = None

    task_suite_name: str = "libero_spatial"
    task_id: Optional[int] = 7
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    enable_attack: bool = True
    attack_iters: int = 10
    attack_lr: float = 0.05
    num_frames_to_attack: int = 20
    num_train_init_states: int = 10
    train_frames_per_state: int = 1
    alpha_action: float = 1.0
    alpha_feature: float = 10.0
    frame_collect_with_policy: bool = False
    collect_grasp_frames: bool = False
    grasp_pre_frames: int = 40
    grasp_post_frames: int = 0
    grasp_max_steps: int = 400
    grasp_qpos_threshold: float = 0.02

    photometric_calib_frames: int = 5

    live_test_enabled: bool = True
    live_test_every_n_iters: int = 20
    live_test_resolution: int = 256
    live_test_max_steps: int = 300

    save_attack_artifacts: bool = True
    load_texture_path: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_project: str = "openvla_attack"
    wandb_entity: str = "user"

    seed: int = 7
    run_id_note: Optional[str] = None
    unnorm_key: Optional[str] = None


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)

    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(
            f"未知物体 '{cfg.object_name}'，可选: {list(OBJECT_ASSETS.keys())}"
        )
    obj_cfg: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    mesh_path: str = cfg.override_mesh_path or obj_cfg["mesh"]
    texture_path: str = (
        cfg.override_texture_path or obj_cfg["texture"]
    )
    xml_path: str = cfg.override_xml_path or obj_cfg["xml"]
    search_kw: SearchKeywords = obj_cfg["search"]
    task_suite_name: str = (
        cfg.task_suite_name or obj_cfg["task_suite"]
    )

    scale_xyz: MeshScale = parse_mesh_scale(xml_path)

    run_id: str = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id = f"{cfg.run_id_note}-{run_id}"
    artifact_store: AttackArtifactStore = AttackArtifactStore.prepare(
        local_log_dir=cfg.local_log_dir,
        run_id=run_id,
        create_attack_directory=(
            cfg.enable_attack and cfg.save_attack_artifacts
        ),
    )
    log_file: TextIO = artifact_store.open_run_log()

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_id)

    runtime_assets: RuntimeAssetTransaction = (
        RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=texture_path,
            object_name=cfg.object_name,
            backup_tag=DATE_TIME,
        )
    )

    benchmark_dict: dict[str, Any] = benchmark.get_benchmark_dict()
    task_suite_obj: Any = benchmark_dict[task_suite_name]()
    target_tasks: Iterable[int] = (
        [cfg.task_id]
        if cfg.task_id is not None
        else range(task_suite_obj.n_tasks)
    )

    model: AttackTrainingModel = get_model(cfg)
    processor: Optional[TrainingProcessor] = (
        get_processor(cfg)
        if cfg.model_family == "openvla"
        else None
    )

    renderer: Optional[DifferentiableRenderer] = None
    if cfg.enable_attack:
        print("[INFO] Initializing DifferentiableRenderer...")
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=scale_xyz,
        ).to(model.device)
    total_episodes: int = 0
    total_successes: int = 0
    video_resolution: int = 512
    episode_runner: LiberoEpisodeRunner = LiberoEpisodeRunner(
        cfg=cfg,
        model=model,
        processor=processor,
        renderer=renderer,
        search_keywords=search_kw,
        video_resolution=video_resolution,
        max_steps=300,
    )
    attack_trainer: Optional[AttackTrainer] = None
    if renderer is not None:
        if processor is None:
            raise RuntimeError(
                "OpenVLA 攻击训练需要 Hugging Face processor"
            )
        attack_trainer = AttackTrainer(
            cfg=cfg,
            model=model,
            processor=processor,
            renderer=renderer,
            artifact_store=artifact_store,
            runtime_assets=runtime_assets,
            search_keywords=search_kw,
        )

    try:
        task_id: int
        for task_id in tqdm.tqdm(target_tasks, desc="Tasks"):
            runtime_assets.restore(
                context=f"Before Task {task_id}",
                remove_backups=False,
            )

            task: Any = task_suite_obj.get_task(task_id)
            init_states: Sequence[Any] = (
                task_suite_obj.get_task_init_states(task_id)
            )

            if cfg.enable_attack and cfg.load_texture_path is not None:
                if renderer is None:
                    raise RuntimeError(
                        "攻击纹理加载需要已初始化的 renderer"
                    )
                print(
                    "[INFO] Loading pre-trained adversarial noise from "
                    f"{cfg.load_texture_path}"
                )
                load_result: AdversarialTextureLoadResult = (
                    renderer.load_adversarial_texture(
                        cfg.load_texture_path
                    )
                )
                source_label: str = (
                    ".pt"
                    if load_result.source_kind == "parameter"
                    else "PNG"
                )
                print(
                    f"[INFO] adv_noise loaded from {source_label}: "
                    f"max_delta={load_result.max_absolute_delta:.4f}, "
                    f"nonzero={load_result.nonzero_percentage:.1f}%"
                )
                adv_tex_inj_path: Path = artifact_store.save_loaded_texture(
                    task_id=task_id,
                    renderer=renderer,
                )
                mirrored_real_texture: bool = (
                    runtime_assets.activate_texture(
                        adv_tex_inj_path,
                        mirror_real_texture=True,
                    )
                )
                print(
                    "[INFO] MuJoCo XML updated with adversarial texture → "
                    f"{adv_tex_inj_path}"
                )
                if mirrored_real_texture:
                    print(
                        "[INFO] Real MuJoCo texture overwritten → "
                        f"{runtime_assets.real_texture_path}"
                    )

            if cfg.enable_attack and cfg.load_texture_path is None:
                print(f"[INFO] Attack training for Task {task_id}...")
                if attack_trainer is None or renderer is None:
                    raise RuntimeError(
                        "攻击已启用，但 renderer/trainer 尚未初始化"
                    )
                renderer.reset_texture()

                dummy_env: Any
                train_task_desc: str
                dummy_env, train_task_desc = get_libero_env(
                    task, cfg.model_family, resolution=256
                )
                dummy_env.close()

                attack_trainer.train(
                    task=task,
                    task_description=train_task_desc,
                    fallback_initial_state=init_states[0],
                    task_id=task_id,
                    num_iters=cfg.attack_iters,
                    initial_states=init_states,
                )

                trained_tex_path: Path = artifact_store.save_trained_texture(
                    task_id=task_id,
                    timestamp=DATE_TIME,
                    renderer=renderer,
                )
                print(
                    f"[INFO] Task {task_id} texture saved → "
                    f"{trained_tex_path}"
                )

                mirrored_real_texture = runtime_assets.activate_texture(
                    trained_tex_path,
                    mirror_real_texture=True,
                )
                print(f"[INFO] XML updated for Task {task_id}.")
                if mirrored_real_texture:
                    print(
                        "[INFO] Real MuJoCo texture overwritten → "
                        f"{runtime_assets.real_texture_path}"
                    )

            n_eval: int = min(
                cfg.num_trials_per_task,
                len(init_states),
            )
            episode_index: int
            for episode_index in tqdm.tqdm(
                range(n_eval),
                desc=f"Task {task_id} Episodes",
            ):
                rollout_result: RolloutResult = episode_runner.run(
                    task=task,
                    initial_state=init_states[episode_index],
                    task_id=task_id,
                    episode_index=episode_index,
                )
                if rollout_result.success:
                    total_successes += 1

                total_episodes += 1
                log_str: str = (
                    f"Task: {task_id} | Ep: {episode_index} | "
                    f"Success: {rollout_result.success}"
                )
                print(log_str)
                log_file.write(log_str + "\n")
                log_file.flush()

                save_rollout_video(
                    rollout_result.replay_images,
                    total_episodes,
                    success=rollout_result.success,
                    task_description=rollout_result.task_description,
                    log_file=log_file,
                )

        average_success_rate: float = (
            total_successes / total_episodes
            if total_episodes > 0
            else 0.0
        )
        print(
            f"\n[DONE] Episodes: {total_episodes} | "
            f"Attack success rate: {average_success_rate:.2%}"
        )
        log_file.write(
            "\nFINAL AVG SUCCESS RATE: "
            f"{average_success_rate:.2%}\n"
        )

    finally:
        runtime_assets.close(context="Final cleanup")
        log_file.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    eval_libero()
