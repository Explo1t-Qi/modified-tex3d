"""运行 Gate 2E：单 state、单 update、bake、rollout 与资产恢复 smoke。

该命令固定使用 source OpenVLA Action objective 和 K=256 spectral 参数化。
它不运行 Feature loss、不选择 support、不调参，也不把单次 rollout 成败解释为
攻击效果。唯一目的，是证明正式 ``512→224→center crop→checkpoint processor``
训练路径能够反传到 Surface Delta，执行一次受预算约束的更新，并把 bake PNG
通过 Runtime Asset Transaction 交给真实 MuJoCo rollout 后完整恢复共享资产。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# 必须在导入 LIBERO/Robosuite 前固定 headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import PIL
import tensorflow as tf
import torch
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

from libero_utils import get_libero_env  # noqa: E402
from openvla_attack.artifacts import (  # noqa: E402
    AttackArtifactStore,
    LiveSnapshotPaths,
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import (  # noqa: E402
    SingleViewFrame,
    build_single_view_samples,
)
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.deployment_backward_audit import (  # noqa: E402
    SCHEMA_VERSION,
    DeploymentBackwardEvidence,
    GradientEvidence,
    evaluate_deployment_backward_evidence,
    write_deployment_backward_evidence,
)
from openvla_attack.evaluation import (  # noqa: E402
    LiberoEpisodeRunner,
    RolloutResult,
)
from openvla_attack.frame_collection import (  # noqa: E402
    TrainingFrame,
    TrainingFrameCollector,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.objective import (  # noqa: E402
    legacy_symmetric_target_cross_entropy,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiableDeploymentViewStages,
    DifferentiablePolicyViewTransform,
    build_policy_view_transform,
)
from openvla_attack.renderer import (  # noqa: E402
    DifferentiableRenderer,
    SurfaceDeltaGradientCapture,
)
from openvla_attack.runtime_assets import (  # noqa: E402
    RuntimeAssetTransaction,
)
from openvla_attack.texture_parameterization import (  # noqa: E402
    SurfaceStepStats,
)
from robot_utils import get_model, set_seed_everywhere  # noqa: E402
from openvla_utils import get_processor  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--spectral_basis_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--train_state_id", type=int, default=0)
    parser.add_argument("--rollout_state_id", type=int, default=10)
    parser.add_argument("--rollout_max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--unnorm_key",
        default="libero_spatial_no_noops",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    configuration_hashes: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
    ):
        candidate = path / name
        if candidate.is_file():
            configuration_hashes[name] = _sha256_file(candidate)
    weight_files = sorted(
        tuple(path.glob("*.safetensors")) + tuple(path.glob("*.bin"))
    )
    inventory = [
        {"name": item.name, "size": item.stat().st_size}
        for item in weight_files
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path.resolve()),
        "configuration_sha256": configuration_hashes,
        "weight_name_size_inventory": inventory,
        "weight_name_size_inventory_sha256": inventory_sha256,
    }


def _state_fingerprint(state: Any) -> str:
    """绑定 LIBERO 初始状态 dtype、shape 与逐字节内容。"""

    array = np.ascontiguousarray(np.asarray(state))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _single_view_frame(frame: TrainingFrame) -> SingleViewFrame:
    mvp = frame["mvp"]
    if mvp is None:
        raise RuntimeError("Gate 2E train state 没有找到目标物体 MVP")
    return {
        "mvp": mvp,
        "bg_tensor": frame["bg_tensor"],
        "bg_tensor_no_obj": frame["bg_tensor_no_obj"],
        "model_rot": frame["model_rot"],
    }


def run_gate_2e(args: argparse.Namespace) -> Path:
    """执行真实 Gate 2E 并返回 evidence JSON path。"""

    if args.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {args.object_name}")
    if (
        len(args.code_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.code_commit
        )
    ):
        raise ValueError("code_commit 必须是40位小写十六进制 Git SHA")
    if args.rollout_max_steps <= 0:
        raise ValueError("rollout_max_steps 必须为正数")
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark

    set_seed_everywhere(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "deployment_backward_evidence.json"
    if evidence_path.exists():
        raise FileExistsError(
            f"拒绝覆盖已有 Gate 2E evidence: {evidence_path}"
        )

    asset: ObjectAssetSpec = OBJECT_ASSETS[args.object_name]
    xml_path = Path(asset["xml"])
    real_texture_path = Path(asset["texture"])
    mesh_path = Path(asset["mesh"])
    basis_path = Path(args.spectral_basis_path)
    for required_path in (
        xml_path,
        real_texture_path,
        mesh_path,
        basis_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        model_family="openvla",
        center_crop=True,
        object_name=args.object_name,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        num_steps_wait=10,
        enable_attack=True,
        attack_iters=1,
        attack_surface_step=2.0 / 255.0,
        attack_epsilon=128.0 / 255.0,
        texture_parameterization="spectral",
        spectral_basis_path=str(basis_path),
        spectral_basis_count=256,
        num_frames_to_attack=1,
        num_train_init_states=1,
        train_frames_per_state=1,
        alpha_action=1.0,
        alpha_feature=0.0,
        feature_objective="last_hidden",
        feature_view_mode="primary",
        frame_collect_with_policy=False,
        collect_grasp_frames=False,
        photometric_calib_frames=1,
        live_test_enabled=False,
        save_attack_artifacts=True,
        local_log_dir=str(output_dir),
        use_wandb=False,
        seed=args.seed,
        unnorm_key=args.unnorm_key,
    )

    benchmark_class = benchmark.get_benchmark_dict()[args.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    for state_id in (args.train_state_id, args.rollout_state_id):
        if not 0 <= state_id < len(initial_states):
            raise ValueError(f"state_id 越界: {state_id}")

    model: Any = get_model(cfg)
    model.eval()
    # Gate 2E 只求输入/纹理梯度；冻结7B模型参数不会改变 forward 或输入 VJP，
    # 并避免为无用的权重梯度分配显存。
    model.requires_grad_(False)
    processor: Any = get_processor(cfg)
    image_preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model,
        processor=processor,
    )
    model_input_height, model_input_width = image_preprocessor.output_size
    if model_input_height != model_input_width:
        raise RuntimeError("Gate 2E 只支持正方形 checkpoint 输入")
    policy_view_transform: DifferentiablePolicyViewTransform = (
        build_policy_view_transform(
            source_resolution=POLICY_SOURCE_RESOLUTION,
            model_input_resolution=model_input_height,
        )
    )
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=real_texture_path,
        device=str(model.device),
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=cfg.attack_epsilon,
        texture_parameterization="spectral",
        spectral_basis_path=basis_path,
        spectral_basis_count=256,
    ).to(model.device)
    renderer.reset_texture()
    parameter = renderer.get_texture_param()
    if tuple(parameter.shape) != (256, 3):
        raise RuntimeError(
            f"Gate 2E 要求 spectral [256,3]，收到 {tuple(parameter.shape)}"
        )

    description_env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()
    collector = TrainingFrameCollector(
        cfg=cfg,
        model=model,
        processor=processor,
        renderer=renderer,
        image_preprocessor=image_preprocessor,
        search_keywords=asset["search"],
        feature_objective="last_hidden",
        feature_view_mode="primary",
        render_resolution=POLICY_SOURCE_RESOLUTION,
        policy_view_transform=policy_view_transform,
    )
    frames = collector.collect(
        task=task,
        task_description=task_description,
        fallback_initial_state=initial_states[args.train_state_id],
        initial_states=(initial_states[args.train_state_id],),
        initial_state_ids=(args.train_state_id,),
    )
    if len(frames) != 1:
        raise RuntimeError(f"Gate 2E 必须且只采集一帧，收到 {len(frames)}")
    frame = frames[0]

    parameter_before = parameter.detach().clone()
    parameter.grad = None
    with renderer.capture_surface_delta_gradients() as surface_capture:
        source_rgb = build_single_view_samples(
            renderer,
            _single_view_frame(frame),
            POLICY_SOURCE_RESOLUTION,
        )[0]
        source_rgb.retain_grad()
        deployment_stages: DifferentiableDeploymentViewStages = (
            policy_view_transform.build_stages(source_rgb)
        )
        deployment_stages.pre_crop_canvas.retain_grad()
        deployment_stages.effective_view.retain_grad()
        pixel_values = image_preprocessor.build_fused_pixel_values(
            deployment_stages.effective_view
        )
        with autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=frame["clean_output_ids"],
                attention_mask=torch.ones_like(frame["clean_output_ids"]),
                pixel_values=pixel_values.to(torch.bfloat16),
                output_hidden_states=True,
            )
        action_loss = legacy_symmetric_target_cross_entropy(
            outputs.logits,
            frame["clean_output_ids"],
        )
        if not bool(torch.isfinite(action_loss).item()):
            raise RuntimeError("Gate 2E Action loss 非有限")
        action_loss.backward()

    gradients = {
        "source": source_rgb.grad,
        "pre_crop": deployment_stages.pre_crop_canvas.grad,
        "effective": deployment_stages.effective_view.grad,
        "parameter": parameter.grad,
    }
    missing_gradients = [
        name for name, gradient in gradients.items() if gradient is None
    ]
    if missing_gradients:
        raise RuntimeError(f"Gate 2E 缺少梯度: {missing_gradients}")
    surface_gradient = surface_capture.summed_gradient()
    assert source_rgb.grad is not None
    assert deployment_stages.pre_crop_canvas.grad is not None
    assert deployment_stages.effective_view.grad is not None
    assert parameter.grad is not None

    with torch.no_grad():
        step_stats: SurfaceStepStats = (
            renderer.step_surface_parameterization_(
                parameter.grad,
                cfg.attack_surface_step,
            )
        )
    parameter_after = parameter.detach().clone()
    parameter_difference = parameter_after - parameter_before

    artifact_store = AttackArtifactStore.prepare(
        local_log_dir=output_dir,
        run_id="gate2e",
        create_attack_directory=True,
    )
    snapshot: LiveSnapshotPaths = artifact_store.save_live_snapshot(
        episode_index=args.task_id,
        iteration=1,
        renderer=renderer,
    )
    baked_texture_path = snapshot.texture_path.resolve()
    baked_texture_sha256 = _sha256_file(baked_texture_path)

    xml_sha256_before = _sha256_file(xml_path)
    real_texture_sha256_before = _sha256_file(real_texture_path)
    transaction: Optional[RuntimeAssetTransaction] = None
    rollout_completed = False
    rollout_success = False
    active_texture_sha256 = ""
    backup_paths: tuple[Path, ...] = ()
    try:
        transaction = RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=real_texture_path,
            object_name=args.object_name,
            backup_tag=f"gate2e_{args.code_commit[:12]}",
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
        mirrored = transaction.activate_texture(
            baked_texture_path,
            mirror_real_texture=True,
        )
        if not mirrored:
            raise RuntimeError("Gate 2E 未同步激活真实 MuJoCo texture")
        active_texture_sha256 = _sha256_file(real_texture_path)
        episode_runner = LiberoEpisodeRunner(
            cfg=cfg,
            model=model,
            processor=processor,
            video_resolution=POLICY_SOURCE_RESOLUTION,
            max_steps=args.rollout_max_steps,
        )
        rollout_result: RolloutResult = episode_runner.run(
            task=task,
            initial_state=initial_states[args.rollout_state_id],
            task_id=args.task_id,
            episode_index=args.rollout_state_id,
        )
        rollout_completed = True
        rollout_success = rollout_result.success
    finally:
        if transaction is not None:
            transaction.close(context="Gate 2E final cleanup")

    xml_sha256_after_restore = _sha256_file(xml_path)
    real_texture_sha256_after_restore = _sha256_file(real_texture_path)
    backup_paths_removed = bool(backup_paths) and not any(
        path.exists() for path in backup_paths
    )
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    evidence = DeploymentBackwardEvidence(
        schema_version=SCHEMA_VERSION,
        code_commit=args.code_commit,
        state_id=args.train_state_id,
        objective="source_openvla_action_only",
        action_loss=float(action_loss.detach().item()),
        parameter_shape=tuple(int(size) for size in parameter.shape),
        parameter_changed_count=int(
            torch.count_nonzero(parameter_difference).item()
        ),
        parameter_linf_change=float(
            parameter_difference.abs().amax().item()
        ),
        source_rgb_gradient=GradientEvidence.from_tensor(source_rgb.grad),
        pre_crop_gradient=GradientEvidence.from_tensor(
            deployment_stages.pre_crop_canvas.grad
        ),
        effective_view_gradient=GradientEvidence.from_tensor(
            deployment_stages.effective_view.grad
        ),
        surface_delta_gradient=GradientEvidence.from_tensor(
            surface_gradient
        ),
        parameter_gradient=GradientEvidence.from_tensor(parameter.grad),
        actual_surface_step=step_stats.actual_surface_step,
        max_surface_delta=step_stats.max_abs_delta,
        baked_texture_path=str(baked_texture_path),
        baked_texture_sha256=baked_texture_sha256,
        active_texture_sha256=active_texture_sha256,
        rollout_completed=rollout_completed,
        rollout_success=rollout_success,
        xml_sha256_before=xml_sha256_before,
        xml_sha256_after_restore=xml_sha256_after_restore,
        real_texture_sha256_before=real_texture_sha256_before,
        real_texture_sha256_after_restore=(
            real_texture_sha256_after_restore
        ),
        backup_paths_removed=backup_paths_removed,
        provenance={
            "command": command,
            "checkpoint": _checkpoint_provenance(
                Path(args.pretrained_checkpoint)
            ),
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "object_name": args.object_name,
            "rollout_state_id": args.rollout_state_id,
            "state_fingerprints": {
                str(args.train_state_id): _state_fingerprint(
                    initial_states[args.train_state_id]
                ),
                str(args.rollout_state_id): _state_fingerprint(
                    initial_states[args.rollout_state_id]
                ),
            },
            "spectral_basis_path": str(basis_path.resolve()),
            "spectral_basis_sha256": _sha256_file(basis_path),
            "asset_sha256_before": {
                "xml": xml_sha256_before,
                "mesh": _sha256_file(mesh_path),
                "texture": real_texture_sha256_before,
            },
            "versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "tensorflow": tf.__version__,
                "torch": torch.__version__,
            },
            "deployment": {
                "policy_source_resolution": POLICY_SOURCE_RESOLUTION,
                "policy_pre_crop_resolution": model_input_height,
                "center_crop_area": 0.9,
            },
        },
    )
    written_path = write_deployment_backward_evidence(
        evidence_path,
        evidence,
    )
    decision = evaluate_deployment_backward_evidence(evidence)
    print(
        "[GATE-2E] "
        f"gate_pass={decision.gate_pass} "
        f"failures={list(decision.failures)}"
    )
    print(f"[GATE-2E] evidence={written_path}")
    if not decision.gate_pass:
        raise RuntimeError("Gate 2E 未通过严格判定")
    return written_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_gate_2e(_parse_args(argv))


if __name__ == "__main__":
    main()
