"""运行 states 0--9 Dense Seed Gradient Audit 并保存完整 raw ``G_s``。

本命令复用已通过 Objective GPU Audit 的公共 state capture：零 Surface Delta、
全部共享纹理实例、MuJoCo front-most alpha、visibility-masked compositor、精确
center-crop/checkpoint BPDA，以及固定 clean prefix 的 Untargeted Clean-Action
Margin hinge。每个 state 保存一个 float32 ``[N_v,3]`` NPZ。

该 runner 不导入 legacy ``optimization.py``，不计算 Feature/wrist/OFT gradient，
也不生成 Support Seed Score、density、coverage 或 Fixed Vertex Support。
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

from openvla_attack.action_objective_audit import (  # noqa: E402
    ACTION_OBJECTIVE_NAME,
)
from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.dense_seed_audit import (  # noqa: E402
    DENSE_SEED_AUDIT_SCHEMA_VERSION,
    DenseSeedCaptureMetadata,
    DenseSeedGradientEvidence,
    evaluate_dense_seed_gradient_evidence,
    summarize_dense_seed_gradient_evidence,
    write_dense_seed_gradient_evidence,
    write_dense_seed_state_artifact,
)
from openvla_attack.diagnose_action_objective import (  # noqa: E402
    ActionObjectiveAuditConfig,
    ActionObjectiveStateCapture,
    collect_action_objective_state_capture,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    build_policy_view_transform,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass(frozen=True)
class DenseSeedAuditConfig:
    """正式 Dense Seed Gradient Audit 的 CLI schema。"""

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


def _parse_args(argv: Optional[Sequence[str]] = None) -> DenseSeedAuditConfig:
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
    return DenseSeedAuditConfig(**vars(parser.parse_args(argv)))


def _validate_code_commit(code_commit: str) -> None:
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise ValueError("code_commit 必须是40位小写十六进制 Git SHA")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            fingerprints[name] = _file_sha256(candidate)
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


def _objective_config(cfg: DenseSeedAuditConfig) -> ActionObjectiveAuditConfig:
    """显式构造共享 runtime 配置，不接受任何 Feature/optimizer 字段。"""

    return ActionObjectiveAuditConfig(
        pretrained_checkpoint=cfg.pretrained_checkpoint,
        output_dir=cfg.output_dir,
        code_commit=cfg.code_commit,
        task_suite_name=cfg.task_suite_name,
        task_id=cfg.task_id,
        object_name=cfg.object_name,
        state_ids=cfg.state_ids,
        num_steps_wait=cfg.num_steps_wait,
        seed=cfg.seed,
        unnorm_key=cfg.unnorm_key,
        model_family=cfg.model_family,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        center_crop=cfg.center_crop,
    )


def _metadata_from_capture(
    capture: ActionObjectiveStateCapture,
) -> DenseSeedCaptureMetadata:
    return DenseSeedCaptureMetadata(
        mesh_sha256=capture.mesh_sha256,
        render_to_geometry_sha256=capture.render_to_geometry_sha256,
        policy_source_rgb_sha256=capture.policy_source_rgb_sha256,
        effective_view_rgb_sha256=capture.effective_view_rgb_sha256,
        mujoco_instance_alpha_sha256=(
            capture.mujoco_instance_alpha_sha256
        ),
        renderer_visibility_sha256=capture.renderer_visibility_sha256,
        shared_instance_body_ids=capture.shared_instance_body_ids,
        shared_instance_body_names=capture.shared_instance_body_names,
    )


def _loaded_legacy_optimizer_modules() -> tuple[str, ...]:
    """返回任意 import 路径下已加载的 OpenVLA legacy optimizer。"""

    return tuple(
        sorted(
            module_name
            for module_name in sys.modules
            if module_name == "openvla_attack.optimization"
            or module_name.endswith(".openvla_attack.optimization")
        )
    )


def run_dense_seed_gradient_audit(cfg: DenseSeedAuditConfig) -> Path:
    """执行完整 raw ``G_s`` audit，返回禁止覆盖的 manifest。"""

    _validate_code_commit(cfg.code_commit)
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("Dense Seed Audit state_ids 不得为空")
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
        raise RuntimeError("Dense Seed Audit 需要真实 CUDA nvdiffrast device")

    # 用户护栏：本进程不得加载 legacy Action+Feature optimizer。使用 sys.modules
    # 做运行时否定证据，避免未来间接 import 静默改变 seed gradient 语义。
    legacy_modules = _loaded_legacy_optimizer_modules()
    if legacy_modules:
        raise RuntimeError(
            "Dense Seed Audit 禁止加载 legacy optimization.py: "
            f"{legacy_modules}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "dense_seed_metrics.jsonl"
    manifest_path = output_dir / "dense_seed_manifest.json"
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有 Dense Seed Audit 权威文件")

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
    model_height, model_width = image_preprocessor.output_size
    if model_height != model_width:
        raise RuntimeError("Dense Seed Audit 只支持正方形 checkpoint 输入")
    policy_view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_height,
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
        raise ValueError("Dense Seed Audit state_id 超出初始状态范围")
    description_env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()
    runtime_cfg = _objective_config(cfg)

    rows: list[DenseSeedGradientEvidence] = []
    for state_id in state_ids:
        print(f"[DENSE-SEED-AUDIT] state {state_id}")
        capture = collect_action_objective_state_capture(
            cfg=runtime_cfg,
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
        row = write_dense_seed_state_artifact(
            output_root=output_dir,
            objective_evidence=capture.evidence,
            dense_geometry_gradient=capture.dense_geometry_gradient,
            metadata=_metadata_from_capture(capture),
        )
        decision = evaluate_dense_seed_gradient_evidence(
            row,
            artifact_root=output_dir,
        )
        rows.append(row)
        print(
            "[DENSE-SEED-AUDIT] "
            f"state={state_id} shape={capture.dense_geometry_gradient.shape} "
            f"grad_l2={capture.evidence.dense_geometry_gradient.l2_norm:.6e} "
            f"artifact={row.artifact_relative_path} pass={decision.gate_pass}"
        )
        if not decision.gate_pass:
            print(
                "[DENSE-SEED-AUDIT] state artifact 失败，停止后续 states: "
                + "; ".join(decision.failures)
            )
            break

    legacy_modules = _loaded_legacy_optimizer_modules()
    if legacy_modules:
        raise RuntimeError(
            "Dense Seed Audit 运行中加载了 legacy optimization.py: "
            f"{legacy_modules}"
        )
    write_dense_seed_gradient_evidence(
        metrics_path,
        rows,
        artifact_root=output_dir,
    )
    summary = summarize_dense_seed_gradient_evidence(
        rows,
        artifact_root=output_dir,
    )
    artifact_sha256 = {
        row.artifact_relative_path: row.artifact_sha256 for row in rows
    }
    manifest = {
        "schema_version": DENSE_SEED_AUDIT_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "config": asdict(cfg),
        "summary": asdict(summary),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": _file_sha256(metrics_path),
        "artifact_sha256": artifact_sha256,
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "asset_sha256": {
            "xml": _file_sha256(xml_path),
            "mesh": _file_sha256(mesh_path),
            "texture": _file_sha256(texture_path),
        },
        "objective_components": [ACTION_OBJECTIVE_NAME],
        "legacy_optimization_module_loaded": False,
        "feature_gradient_computed": False,
        "wrist_gradient_computed": False,
        "oft_loaded": False,
        "seed_score_computed": False,
        "density_computed": False,
        "coverage_computed": False,
        "production_support_constructed": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"[DENSE-SEED-AUDIT] manifest={manifest_path}")
    if not summary.gate_pass:
        raise RuntimeError(
            "Dense Seed Audit 未通过: " + "; ".join(summary.failures)
        )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    print(
        "[DENSE-SEED-AUDIT] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_dense_seed_gradient_audit(cfg)


if __name__ == "__main__":
    main()
