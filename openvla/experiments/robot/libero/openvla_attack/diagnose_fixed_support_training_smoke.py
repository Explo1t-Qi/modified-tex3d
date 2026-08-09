"""运行正式Fixed-Support Action+Spectral trainer的两步GPU工程smoke。

Step 0从零Surface Delta开始，必须严格退化为Action-only；Step 1在第一次更新后
重新计算states 0--9完整目标，并验证非零谱梯度通过冻结lambda进入唯一一次
surface-normalized update。最后bake PNG、由MuJoCo加载Active Texture并恢复资产。
本命令不运行rollout，不把Action loss变化解释为方法效果。
"""

from __future__ import annotations

import argparse
import json
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

from openvla_attack.artifacts import AttackArtifactStore  # noqa: E402
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.diagnose_spectral_guard_calibration import (  # noqa: E402
    AllStateActionGradientProvider,
    capture_action_frame,
    loaded_legacy_optimizer_modules,
    sha256_array,
    write_jsonl,
)
from openvla_attack.fixed_support_training import (  # noqa: E402
    CombinedGradientUpdate,
    FixedSupportTrainerCore,
)
from openvla_attack.fixed_support_training_smoke import (  # noqa: E402
    FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION,
    evaluate_fixed_support_training_smoke_bundle,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    DifferentiablePolicyViewTransform,
    build_policy_view_transform,
)
from openvla_attack.production_support import (  # noqa: E402
    array_sha256,
    load_production_support_artifact,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.runtime_assets import RuntimeAssetTransaction  # noqa: E402
from openvla_attack.seed_score_audit import file_sha256  # noqa: E402
from openvla_attack.spectral_guard import (  # noqa: E402
    MeanActionGradient,
    SpectralGuardTerms,
    SpectralNaturalnessRegularizer,
)
from openvla_attack.spectral_guard_evidence import (  # noqa: E402
    evaluate_spectral_guard_bundle,
)
from openvla_attack.spectral_naturalness import (  # noqa: E402
    load_rho_nat_calibration_artifact,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass(frozen=True)
class FixedSupportTrainingSmokeConfig:
    pretrained_checkpoint: str
    production_support_path: str
    rho_nat_calibration_path: str
    spectral_basis_path: str
    spectral_guard_manifest_path: str
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


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> FixedSupportTrainingSmokeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--production_support_path", required=True)
    parser.add_argument("--rho_nat_calibration_path", required=True)
    parser.add_argument("--spectral_basis_path", required=True)
    parser.add_argument("--spectral_guard_manifest_path", required=True)
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
    parser.add_argument(
        "--attack_surface_step",
        type=float,
        default=2.0 / 255.0,
    )
    return FixedSupportTrainingSmokeConfig(**vars(parser.parse_args(argv)))


def _validate_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit必须是40位小写Git SHA")


def _gradient_row(
    *,
    step: int,
    action_sample: MeanActionGradient,
    terms: SpectralGuardTerms,
    update: CombinedGradientUpdate,
    configured_surface_step: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "action_loss": action_sample.loss,
        "num_action_frames": action_sample.num_frames,
        "action_state_ids": list(action_sample.state_ids),
        "action_state_fingerprints": list(action_sample.state_fingerprints),
        "total_energy": float(terms.total_energy.detach().item()),
        "low_energy": float(terms.low_energy.detach().item()),
        "high_energy": float(terms.high_energy.detach().item()),
        "high_ratio": float(terms.high_ratio.detach().item()),
        "diagnostic_high_ratio": float(
            terms.diagnostic_high_ratio.detach().item()
        ),
        "hinge": float(terms.hinge.detach().item()),
        "penalty": float(terms.penalty.detach().item()),
        "hinge_active": bool(terms.hinge.detach().item() > 0.0),
        "action_gradient_l2": update.action_gradient_l2,
        "spectral_gradient_l2": update.spectral_gradient_l2,
        "weighted_spectral_gradient_l2": (
            update.weighted_spectral_gradient_l2
        ),
        "total_gradient_l2": update.total_gradient_l2,
        "action_spectral_cosine": update.action_spectral_cosine,
        "action_total_cosine": update.action_total_cosine,
        "weighted_spectral_action_ratio": (
            update.weighted_spectral_action_ratio
        ),
        "combination_residual_linf": update.combination_residual_linf,
        "configured_surface_step": configured_surface_step,
        "surface_step_stats": asdict(update.surface_step_stats),
    }


def run_fixed_support_training_smoke(
    cfg: FixedSupportTrainingSmokeConfig,
) -> Path:
    """执行两步联合训练、bake和Active Texture资产事务。"""

    _validate_commit(cfg.code_commit)
    if tuple(parse_state_ids(cfg.state_ids, field_name="state_ids") or ()) != tuple(
        range(10)
    ):
        raise ValueError("正式training smoke必须精确使用states 0-9")
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait不得为负数")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知object_name: {cfg.object_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed-Support training smoke需要真实CUDA")
    legacy_modules = loaded_legacy_optimizer_modules()
    if legacy_modules:
        raise RuntimeError(f"training smoke进程加载了legacy optimizer: {legacy_modules}")
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
    steps_path = output_dir / "fixed_support_training_steps.jsonl"
    frames_path = output_dir / "fixed_support_training_action_frames.jsonl"
    arrays_path = output_dir / "fixed_support_training_arrays.npz"
    manifest_path = output_dir / "fixed_support_training_smoke_manifest.json"
    if any(
        path.exists()
        for path in (steps_path, frames_path, arrays_path, manifest_path)
    ):
        raise FileExistsError("拒绝覆盖已有Fixed-Support training smoke证据")

    support_path = Path(cfg.production_support_path)
    rho_path = Path(cfg.rho_nat_calibration_path)
    basis_path = Path(cfg.spectral_basis_path)
    guard_manifest_path = Path(cfg.spectral_guard_manifest_path)
    support = load_production_support_artifact(support_path)
    rho_calibration = load_rho_nat_calibration_artifact(rho_path)
    guard_decision = evaluate_spectral_guard_bundle(guard_manifest_path)
    if not guard_decision.gate_pass:
        raise RuntimeError(
            "Spectral Guard calibration未通过独立复核: "
            + "; ".join(guard_decision.failures)
        )
    guard_manifest = json.loads(guard_manifest_path.read_text(encoding="utf-8"))
    input_hashes = {
        "production_support": file_sha256(support_path),
        "rho_nat_calibration": file_sha256(rho_path),
        "spectral_basis": file_sha256(basis_path),
    }
    if guard_manifest.get("input_sha256", {}).get(
        "production_support"
    ) != input_hashes["production_support"]:
        raise RuntimeError("lambda校准未绑定当前Production Support")
    if guard_manifest.get("input_sha256", {}).get(
        "rho_nat_calibration"
    ) != input_hashes["rho_nat_calibration"]:
        raise RuntimeError("lambda校准未绑定当前rho_nat artifact")
    if guard_manifest.get("input_sha256", {}).get(
        "spectral_basis"
    ) != input_hashes["spectral_basis"]:
        raise RuntimeError("lambda校准未绑定当前谱基")
    if rho_calibration.production_support_artifact_sha256 != input_hashes[
        "production_support"
    ]:
        raise RuntimeError("rho_nat校准未绑定当前Production Support")
    lambda_spec = float(guard_manifest["lambda_spec"])
    if not 0.0 < lambda_spec <= 1.0:
        raise RuntimeError("冻结lambda_spec不在(0,1]")

    set_seed_everywhere(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"])
    mesh_path = Path(asset["mesh"])
    texture_path = Path(asset["texture"])
    checkpoint_path = Path(cfg.pretrained_checkpoint)
    for path in (
        support_path,
        rho_path,
        basis_path,
        guard_manifest_path,
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
        raise RuntimeError("training smoke只支持正方形checkpoint输入")
    policy_view_transform: DifferentiablePolicyViewTransform = (
        build_policy_view_transform(
            source_resolution=POLICY_SOURCE_RESOLUTION,
            model_input_resolution=model_height,
        )
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
    parameter = renderer.get_texture_param()
    expected_shape = (len(support.support_vertex_indices), 3)
    if tuple(parameter.shape) != expected_shape:
        raise RuntimeError("Fixed-Support紧凑参数shape与artifact不一致")

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
    trainer = FixedSupportTrainerCore(
        renderer,
        surface_step=cfg.attack_surface_step,
    )
    regularizer = SpectralNaturalnessRegularizer.from_artifacts(
        rho_path,
        basis_path,
        device=model.device,
        dtype=torch.float32,
    )

    action_gradients: list[np.ndarray] = []
    spectral_gradients: list[np.ndarray] = []
    weighted_gradients: list[np.ndarray] = []
    total_gradients: list[np.ndarray] = []
    geometry_deltas: list[np.ndarray] = [
        trainer.geometry_delta().detach().cpu().numpy().copy()
    ]
    step_rows: list[dict[str, Any]] = []
    for step in range(2):
        action_sample = provider()
        if action_sample.state_ids != tuple(range(10)):
            raise RuntimeError("正式trainer每步必须完整按序消费states 0-9")
        if action_sample.state_fingerprints != tuple(
            state_fingerprints[state_id] for state_id in range(10)
        ):
            raise RuntimeError("正式trainer state fingerprint绑定漂移")
        terms = regularizer(trainer.geometry_delta())
        spectral_gradient = torch.autograd.grad(
            terms.penalty,
            parameter,
        )[0]
        update = trainer.apply_action_spectral_gradients(
            action_sample.gradient.detach(),
            spectral_gradient.detach(),
            lambda_spec=lambda_spec,
        )
        step_rows.append(
            _gradient_row(
                step=step,
                action_sample=action_sample,
                terms=terms,
                update=update,
                configured_surface_step=cfg.attack_surface_step,
            )
        )
        action_gradients.append(
            action_sample.gradient.detach().cpu().numpy().copy()
        )
        spectral_gradients.append(
            spectral_gradient.detach().cpu().numpy().copy()
        )
        weighted_gradients.append(
            update.weighted_spectral_gradient.cpu().numpy().copy()
        )
        total_gradients.append(update.total_gradient.cpu().numpy().copy())
        geometry_deltas.append(
            trainer.geometry_delta().detach().cpu().numpy().copy()
        )

    if np.count_nonzero(spectral_gradients[0]) != 0:
        raise RuntimeError("Step 0谱梯度未严格退化为零")
    if np.count_nonzero(spectral_gradients[1]) == 0:
        raise RuntimeError("Step 1未得到非零谱梯度")
    if not step_rows[1]["hinge_active"]:
        raise RuntimeError("Step 1谱hinge未激活，无法验收联合路径")
    if trainer.update_count != 2:
        raise RuntimeError("两步smoke的Surface update次数不为2")

    write_jsonl(steps_path, step_rows)
    write_jsonl(frames_path, provider.frame_evidence_history)
    np.savez_compressed(
        arrays_path,
        action_gradients=np.stack(action_gradients).astype(np.float32),
        spectral_gradients=np.stack(spectral_gradients).astype(np.float32),
        weighted_spectral_gradients=np.stack(weighted_gradients).astype(np.float32),
        total_gradients=np.stack(total_gradients).astype(np.float32),
        geometry_deltas=np.stack(geometry_deltas).astype(np.float32),
        support_vertex_indices=support.support_vertex_indices.astype(np.int64),
    )

    artifact_store = AttackArtifactStore.prepare(
        local_log_dir=output_dir,
        run_id="fixed-support-training-smoke",
        create_attack_directory=True,
    )
    snapshot = artifact_store.save_live_snapshot(
        episode_index=cfg.task_id,
        iteration=2,
        renderer=renderer,
    )
    baked_path = snapshot.texture_path.resolve()
    xml_before = file_sha256(xml_path)
    texture_before = file_sha256(texture_path)
    transaction: Optional[RuntimeAssetTransaction] = None
    backup_paths: tuple[Path, ...] = ()
    active_texture_sha256 = ""
    active_environment_loaded = False
    active_observation_sha256 = ""
    try:
        transaction = RuntimeAssetTransaction.begin(
            xml_path=xml_path,
            real_texture_path=texture_path,
            object_name=cfg.object_name,
            backup_tag=f"fixed_support_smoke_{cfg.code_commit[:12]}",
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
        if not transaction.activate_texture(
            baked_path,
            mirror_real_texture=True,
        ):
            raise RuntimeError("training smoke未同步激活真实MuJoCo texture")
        active_texture_sha256 = file_sha256(texture_path)
        active_env, active_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=POLICY_SOURCE_RESOLUTION,
        )
        try:
            if active_description != task_description:
                raise RuntimeError("Active Texture环境task description改变")
            active_env.reset()
            active_env.set_init_state(initial_states[0])
            active_env.env.sim.forward()
            active_observation, _, _, _ = active_env.step(
                get_libero_dummy_action(cfg.model_family)
            )
            active_rgb = get_libero_image(
                active_observation,
                POLICY_SOURCE_RESOLUTION,
            )
            active_observation_sha256 = sha256_array(active_rgb)
            active_environment_loaded = True
        finally:
            active_env.close()
    finally:
        if transaction is not None:
            transaction.close(context="Fixed-Support training smoke cleanup")

    xml_after = file_sha256(xml_path)
    texture_after = file_sha256(texture_path)
    backup_paths_removed = bool(backup_paths) and not any(
        path.exists() for path in backup_paths
    )
    if loaded_legacy_optimizer_modules():
        raise RuntimeError("training smoke期间加载了legacy optimizer")

    input_paths = {
        "production_support": str(support_path),
        "rho_nat_calibration": str(rho_path),
        "spectral_basis": str(basis_path),
        "spectral_guard_manifest": str(guard_manifest_path),
        "mesh": str(mesh_path),
        "texture": str(texture_path),
    }
    all_input_hashes = {
        **input_hashes,
        "spectral_guard_manifest": file_sha256(guard_manifest_path),
        "mesh": file_sha256(mesh_path),
        "texture": texture_after,
    }
    manifest = {
        "schema_version": FIXED_SUPPORT_TRAINING_SMOKE_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "input_paths": input_paths,
        "input_sha256": all_input_hashes,
        "state_ids": list(range(10)),
        "state_fingerprints": state_fingerprints,
        "static_scene_fingerprints": {
            frame.state_id: frame.static_scene_sha256 for frame in frames
        },
        "parameter_shape": list(expected_shape),
        "num_geometry_vertices": support.num_geometry_vertices,
        "support_vertex_indices_sha256": array_sha256(
            support.support_vertex_indices
        ),
        "lambda_spec": lambda_spec,
        "rho_nat": regularizer.rho_nat,
        "num_steps": 2,
        "trainer_update_count": trainer.update_count,
        "steps_relative_path": steps_path.name,
        "steps_sha256": file_sha256(steps_path),
        "action_frames_relative_path": frames_path.name,
        "action_frames_sha256": file_sha256(frames_path),
        "arrays_relative_path": arrays_path.name,
        "arrays_sha256": file_sha256(arrays_path),
        "baked_texture_relative_path": str(
            baked_path.relative_to(output_dir.resolve())
        ),
        "baked_texture_sha256": file_sha256(baked_path),
        "active_texture_sha256": active_texture_sha256,
        "active_environment_loaded": active_environment_loaded,
        "active_observation_sha256": active_observation_sha256,
        "xml_sha256_before": xml_before,
        "xml_sha256_after_restore": xml_after,
        "texture_sha256_before": texture_before,
        "texture_sha256_after_restore": texture_after,
        "backup_paths_removed": backup_paths_removed,
        "objective_components": [
            "untargeted_clean_action_margin_hinge",
            "spectral_naturalness_hinge_squared",
        ],
        "feature_loss_computed": False,
        "wrist_used": False,
        "oft_loaded": False,
        "legacy_optimizer_loaded": False,
        "rho_nat_calibrated": True,
        "lambda_spec_calibrated": True,
        "formal_training_allowed": True,
        "next_required_gate": "fixed_support_action_spectral_source_training",
        "gate_pass": True,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    decision = evaluate_fixed_support_training_smoke_bundle(manifest_path)
    if not decision.gate_pass:
        raise RuntimeError(
            "Fixed-Support training smoke独立复核失败: "
            + "; ".join(decision.failures)
        )
    print(
        json.dumps(
            {
                "gate_pass": True,
                "lambda_spec": lambda_spec,
                "step_0_spectral_gradient_l2": step_rows[0][
                    "spectral_gradient_l2"
                ],
                "step_1_spectral_gradient_l2": step_rows[1][
                    "spectral_gradient_l2"
                ],
                "manifest_sha256": file_sha256(manifest_path),
                "formal_training_allowed": True,
            },
            sort_keys=True,
        )
    )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(
        "[FIXED-SUPPORT-TRAINING-SMOKE] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_fixed_support_training_smoke(_parse_args(argv))


if __name__ == "__main__":
    main()
