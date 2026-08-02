"""OpenVLA 在 LIBERO 上进行对抗纹理训练与鲁棒性评估的命令行入口。

本文件只编排 run、task 和 episode 生命周期。OpenVLA 模型专用的数据转换、
Attack Training、可微渲染、优化、评估状态机、Attack Artifact 与 Clean Asset
恢复分别位于 ``openvla_attack`` package。
"""

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TextIO

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
from openvla_attack.configuration import (
    FeatureObjectiveKind,
    FeatureViewModeKind,
    GenerateConfig,
    TextureParameterizationKind,
    resolve_feature_objective,
    resolve_feature_view_mode,
    resolve_texture_parameterization,
    validate_gradient_norm_protection,
    validate_source_action_response_audit,
)
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
from openvla_attack.state_selection import (
    InitialStatePartition,
    select_initial_state_partition,
)
from openvla_attack.training import (
    AttackTrainer,
    AttackTrainingModel,
)

sys.path.append(str(Path(__file__).parent.parent))

OPENVLA_REPO_ROOT: str = str(Path(__file__).resolve().parents[3])
if OPENVLA_REPO_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_REPO_ROOT)

from openvla_utils import get_processor
from robot_utils import DATE_TIME, get_model, set_seed_everywhere


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    texture_parameterization: TextureParameterizationKind = (
        resolve_texture_parameterization(cfg.texture_parameterization)
    )
    feature_objective: FeatureObjectiveKind = resolve_feature_objective(
        cfg.feature_objective
    )
    feature_view_mode: FeatureViewModeKind = resolve_feature_view_mode(
        cfg.feature_view_mode
    )
    if (
        feature_view_mode == "primary_wrist"
        and feature_objective != "siglip_patch"
    ):
        raise ValueError(
            "feature_view_mode='primary_wrist' 要求 "
            "feature_objective='siglip_patch'"
        )
    validate_gradient_norm_protection(
        cfg,
        texture_parameterization=texture_parameterization,
        feature_objective=feature_objective,
        feature_view_mode=feature_view_mode,
    )
    validate_source_action_response_audit(
        cfg,
        texture_parameterization=texture_parameterization,
        feature_objective=feature_objective,
        feature_view_mode=feature_view_mode,
    )
    if (
        cfg.spectral_gradient_audit_only
        and not cfg.spectral_gradient_audit_enabled
    ):
        raise ValueError(
            "spectral_gradient_audit_only=True 要求同时启用 "
            "spectral_gradient_audit_enabled"
        )
    if (
        cfg.spectral_gradient_audit_reference_path is not None
        and not cfg.spectral_gradient_audit_enabled
    ):
        raise ValueError(
            "spectral_gradient_audit_reference_path 要求同时启用 "
            "spectral_gradient_audit_enabled"
        )
    if cfg.spectral_gradient_audit_enabled:
        if not cfg.enable_attack:
            raise ValueError("谱基梯度审计要求 enable_attack=True")
        if texture_parameterization != "spectral":
            raise ValueError(
                "谱基梯度审计要求 texture_parameterization='spectral'"
            )
        if feature_objective != "siglip_patch":
            raise ValueError(
                "谱基梯度审计要求 feature_objective='siglip_patch'"
            )
        if (
            cfg.spectral_gradient_audit_top_k <= 0
            or cfg.spectral_gradient_audit_top_k
            > cfg.spectral_basis_count
        ):
            raise ValueError(
                "spectral_gradient_audit_top_k 必须位于 "
                f"[1, {cfg.spectral_basis_count}]"
            )
    # 1. 解析配置
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

    # 2.1 创建共享资产事务，确保异常退出时也能恢复 XML 和真实纹理。
    runtime_assets: RuntimeAssetTransaction = (
        RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=texture_path,
            object_name=cfg.object_name,
            backup_tag=DATE_TIME,
        )
    )

    # 2.2 加载 LIBERO benchmark，并确定本次需要遍历的 task。
    benchmark_dict: dict[str, Any] = benchmark.get_benchmark_dict()
    task_suite_obj: Any = benchmark_dict[task_suite_name]()
    target_tasks: Iterable[int] = (
        [cfg.task_id]
        if cfg.task_id is not None
        else range(task_suite_obj.n_tasks)
    )

    # 2.3 加载策略模型及其输入 processor。
    model: AttackTrainingModel = get_model(cfg)
    processor: Optional[TrainingProcessor] = (
        get_processor(cfg)
        if cfg.model_family == "openvla"
        else None
    )

    # 2.4 攻击模式下创建可微 renderer；clean 评估不需要该对象。
    renderer: Optional[DifferentiableRenderer] = None
    if cfg.enable_attack:
        print("[INFO] Initializing DifferentiableRenderer...")
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=scale_xyz,
            epsilon=cfg.attack_epsilon,
            texture_parameterization=texture_parameterization,
            spectral_basis_path=cfg.spectral_basis_path,
            spectral_basis_count=cfg.spectral_basis_count,
        ).to(model.device)
        texture_parameter_count: int = int(
            renderer.get_texture_param().numel()
        )
        print(
            "[INFO] Texture parameterization: "
            f"{renderer.get_texture_parameterization_name()}, "
            f"parameters={texture_parameter_count:,}, "
            f"epsilon={cfg.attack_epsilon:.6f}"
        )
        print(
            "[INFO] Attack objective: "
            f"action_weight={cfg.alpha_action:.6f}, "
            f"feature={feature_objective}, "
            f"feature_views={feature_view_mode}, "
            f"feature_weight={cfg.alpha_feature:.6f}"
        )
        if cfg.gradient_norm_protection_enabled:
            print(
                "[INFO] Gradient norm protection: enabled, "
                "weighted_feature/action_ratio_limit="
                f"{cfg.feature_gradient_norm_ratio_limit:.6f}"
            )
    total_episodes: int = 0
    total_successes: int = 0
    video_resolution: int = 512
    episode_runner: LiberoEpisodeRunner = LiberoEpisodeRunner(
        cfg=cfg,
        model=model,
        processor=processor,
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
            feature_objective=feature_objective,
            feature_view_mode=feature_view_mode,
        )

    # 每个 task 都遵循“恢复干净资产→训练纹理→激活纹理→评估”的生命周期。
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
            )   # 获取当前 task 对应的 init_states
            state_partition: InitialStatePartition = (
                select_initial_state_partition(
                    init_states,
                    num_train_states=cfg.num_train_init_states,
                    num_eval_states=cfg.num_trials_per_task,
                    train_state_specification=cfg.train_init_state_ids,
                    eval_state_specification=cfg.eval_init_state_ids,
                    reserve_training_states=cfg.enable_attack,
                    require_disjoint=cfg.require_disjoint_init_states,
                )
            )
            print(
                "[INFO] Initial-state partition: "
                f"train={list(state_partition.train_state_ids)}, "
                f"eval={list(state_partition.eval_state_ids)}"
            )
            # 已有对抗纹理的处理逻辑
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

                # 3. 在 train 内采集训练帧；4. 使用这些帧优化对抗纹理。
                attack_trainer.train(
                    task=task,
                    task_description=train_task_desc,
                    fallback_initial_state=state_partition.train_states[0],
                    task_id=task_id,
                    num_iters=cfg.attack_iters,
                    initial_states=state_partition.train_states,
                    initial_state_ids=(
                        state_partition.train_state_ids
                    ),
                )
                if (
                    cfg.spectral_gradient_audit_only
                    or cfg.source_action_response_audit_enabled
                ):
                    diagnostic_name: str = (
                        "SPECTRAL-AUDIT"
                        if cfg.spectral_gradient_audit_only
                        else "ACTION-RESPONSE"
                    )
                    print(
                        f"[{diagnostic_name}] Task {task_id} 诊断完成；"
                        "跳过纹理训练产物激活与 held-out rollout"
                    )
                    continue

                # 5.1 保存长期保留的最终攻击纹理。
                trained_tex_path: Path = artifact_store.save_trained_texture(
                    task_id=task_id,
                    timestamp=DATE_TIME,
                    renderer=renderer,
                )
                print(
                    f"[INFO] Task {task_id} texture saved → "
                    f"{trained_tex_path}"
                )

                # 5.2 将最终纹理激活到 MuJoCo 运行资产。
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
                len(state_partition.eval_states),
            )
            episode_index: int
            for episode_index in tqdm.tqdm(
                range(n_eval),
                desc=f"Task {task_id} Episodes",
            ):
                # 6. 在 held-out initial state 上评估训练后的纹理。
                rollout_result: RolloutResult = episode_runner.run(
                    task=task,
                    initial_state=state_partition.eval_states[episode_index],
                    task_id=task_id,
                    episode_index=episode_index,
                )
                if rollout_result.success:
                    total_successes += 1

                total_episodes += 1
                log_str: str = (
                    f"Task: {task_id} | Ep: {episode_index} | "
                    f"State: {state_partition.eval_state_ids[episode_index]} | "
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

        if (
            cfg.spectral_gradient_audit_only
            or cfg.source_action_response_audit_enabled
        ):
            diagnostic_description: str = (
                "spectral gradient audit"
                if cfg.spectral_gradient_audit_only
                else "action-response audit"
            )
            print(f"\n[DONE] Source-only {diagnostic_description} completed.")
            log_file.write(
                f"\nSOURCE-ONLY {diagnostic_description.upper()} COMPLETED\n"
            )
        else:
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
