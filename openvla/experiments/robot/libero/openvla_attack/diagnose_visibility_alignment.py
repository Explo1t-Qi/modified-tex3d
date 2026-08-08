"""采集 states 0–9 Visibility/Alignment audit 的真实 MuJoCo/CUDA 证据。

本命令不加载 OpenVLA/OFT 权重，不读取 Action 梯度，不构造 Support，也不修改
纹理资产。每个 state 在 dummy wait 后开启一个静止 evidence transaction，在
同一 transaction 内为 Primary 与 wrist source-crop proxy 分别采集：

* 单次 front-most MuJoCo object-type/object-ID segmentation；
* 全部共享 Active Texture 实例的 pose/MVP；
* 同次 nvdiffrast adv/clean/mask/raw raster；
* 512→224→effective view 的 observation 与 soft alignment。

权威输出是 ``visibility_alignment_metrics.jsonl``。结构通过只表示行集合、静止
事务和预注册基础条件成立；候选门槛仍需人工检查 overlay 与指标分布后冻结。
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

# 必须在导入 LIBERO/Robosuite 前固定 headless backend。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import PIL
import torch
from PIL import Image


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

from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.compositing import TextureRenderInstance  # noqa: E402
from openvla_attack.instance_renderer_evidence import (  # noqa: E402
    SharedTextureRendererEvidence,
    render_shared_texture_instances,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.renderer_correspondence import (  # noqa: E402
    RasterSupportCorrespondence,
)
from openvla_attack.scene import (  # noqa: E402
    TargetBodyPose,
    compute_render_mvp,
    find_target_body_poses,
)
from openvla_attack.state_selection import parse_state_ids  # noqa: E402
from openvla_attack.visibility_alignment_audit import (  # noqa: E402
    VISIBILITY_ALIGNMENT_SCHEMA_VERSION,
    VisibilityAlignmentAuditRow,
    VisibilityAuditTransactionEvidence,
    VisibilityAuditViewName,
    summarize_visibility_alignment_rows,
    write_visibility_alignment_jsonl,
    write_visibility_alignment_manifest,
)
from openvla_attack.visibility_capture import (  # noqa: E402
    CapturedInstanceSegmentation,
    capture_instance_segmentation,
    static_scene_evidence_transaction,
)
from openvla_attack.visibility_evidence import (  # noqa: E402
    VisibilityThresholdCandidates,
)
from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
)
from openvla_attack.visibility_view_evidence import (  # noqa: E402
    ViewVisibilityEvidence,
    build_view_visibility_evidence,
)


POLICY_SOURCE_RESOLUTION = 512


@dataclass(frozen=True)
class VisibilityAlignmentConfig:
    output_dir: str
    code_commit: str
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    state_ids: str = "0-9"
    num_steps_wait: int = 10
    seed: int = 7
    model_family: str = "openvla"
    device: str = "cuda:0"


@dataclass(frozen=True)
class _CollectedView:
    view_name: VisibilityAuditViewName
    camera_name: str
    segmentation: CapturedInstanceSegmentation
    renderer: SharedTextureRendererEvidence
    correspondence: RasterSupportCorrespondence
    visibility: ViewVisibilityEvidence
    model_view_projections: torch.Tensor


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> VisibilityAlignmentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--object_name", default="akita_black_bowl")
    parser.add_argument("--state_ids", default="0-9")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_family", default="openvla")
    parser.add_argument("--device", default="cuda:0")
    return VisibilityAlignmentConfig(**vars(parser.parse_args(argv)))


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


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return _sha256_array(tensor.detach().contiguous().cpu().numpy())


def _state_fingerprint(state: Any) -> str:
    return _sha256_array(np.asarray(state))


def _save_mask(path: Path, alpha: torch.Tensor) -> str:
    array = (
        alpha.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)
    return _sha256_file(path)


def _save_overlay(
    path: Path,
    mujoco_alpha: torch.Tensor,
    renderer_alpha: torch.Tensor,
) -> str:
    """保存绿=MuJoCo only、红=renderer only、黄=soft overlap。"""

    mujoco = mujoco_alpha.detach().float().cpu()
    renderer = renderer_alpha.detach().float().cpu()
    overlap = torch.minimum(mujoco, renderer)
    red = renderer - overlap + overlap
    green = mujoco - overlap + overlap
    blue = torch.zeros_like(red)
    overlay = (
        torch.stack((red, green, blue), dim=-1)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(overlay).save(path)
    return _sha256_file(path)


def _build_instances(
    env: Any,
    target_poses: tuple[TargetBodyPose, ...],
    *,
    camera_name: str,
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
                camera_name=camera_name,
            ),
            "model_rot": pose.model_matrix[:3, :3].to(device),
        }
        for pose in target_poses
    )


def _collect_view(
    *,
    env: Any,
    renderer: DifferentiableRenderer,
    target_poses: tuple[TargetBodyPose, ...],
    instance_roots: tuple[TargetInstanceRoot, ...],
    view_name: VisibilityAuditViewName,
    camera_name: str,
    device: torch.device,
    thresholds: VisibilityThresholdCandidates,
) -> _CollectedView:
    segmentation = capture_instance_segmentation(
        env,
        instance_roots,
        camera_name=camera_name,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    instances = _build_instances(
        env,
        target_poses,
        camera_name=camera_name,
        device=device,
    )
    with torch.no_grad():
        renderer_evidence = render_shared_texture_instances(
            renderer,
            instances,
            resolution=(
                POLICY_SOURCE_RESOLUTION,
                POLICY_SOURCE_RESOLUTION,
            ),
        )
        mapping = renderer_evidence.render_to_geometry
        all_geometry_support = torch.ones(
            int(mapping.max().item()) + 1,
            dtype=torch.bool,
            device=mapping.device,
        )
        correspondence = renderer_evidence.decode_support(
            all_geometry_support
        )
        valid_control = correspondence.support_control[
            correspondence.valid_mask
        ]
        if valid_control.numel() > 0 and not torch.allclose(
            valid_control,
            torch.ones_like(valid_control),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("真实 raster 未满足全 Support w=1 不变量")
        visibility = build_view_visibility_evidence(
            segmentation.parsed.instance_alpha.to(device),
            renderer_evidence.visibility_mask,
            thresholds=thresholds,
        )
    return _CollectedView(
        view_name=view_name,
        camera_name=camera_name,
        segmentation=segmentation,
        renderer=renderer_evidence,
        correspondence=correspondence,
        visibility=visibility,
        model_view_projections=torch.stack(
            [instance["mvp"] for instance in instances],
            dim=0,
        ),
    )


def _persist_view_artifacts(
    *,
    output_dir: Path,
    state_id: int,
    collected: _CollectedView,
) -> tuple[str, dict[str, str]]:
    arrays_dir = output_dir / "arrays"
    images_dir = output_dir / "images"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"state_{state_id:02d}_{collected.view_name}"
    arrays_path = arrays_dir / f"{prefix}.npz"
    if arrays_path.exists():
        raise FileExistsError(arrays_path)

    visibility = collected.visibility
    mujoco_effective = visibility.mujoco_stages.effective.alpha
    renderer_effective = visibility.renderer_effective_alpha
    np.savez_compressed(
        arrays_path,
        schema_version=np.asarray(VISIBILITY_ALIGNMENT_SCHEMA_VERSION),
        state_id=np.asarray(state_id, dtype=np.int64),
        view_name=np.asarray(collected.view_name),
        camera_name=np.asarray(collected.camera_name),
        oriented_segmentation=collected.segmentation.oriented_segmentation,
        mujoco_alpha_source=(
            visibility.mujoco_stages.source.alpha.detach().cpu().numpy()
        ),
        mujoco_alpha_pre_crop=(
            visibility.mujoco_stages.pre_crop.alpha.detach().cpu().numpy()
        ),
        mujoco_alpha_effective=mujoco_effective.detach().cpu().numpy(),
        renderer_mask_source=(
            collected.renderer.visibility_mask.detach().cpu().numpy()
        ),
        renderer_alpha_effective=renderer_effective.detach().cpu().numpy(),
        triangle_indices=(
            collected.correspondence.triangle_indices.detach().cpu().numpy()
        ),
        barycentric=(
            collected.correspondence.barycentric.detach().cpu().numpy()
        ),
        geometry_corner_indices=(
            collected.correspondence.geometry_corner_indices
            .detach()
            .cpu()
            .numpy()
        ),
        model_view_projections=(
            collected.model_view_projections.detach().cpu().numpy()
        ),
    )
    arrays_relative_path = str(arrays_path.relative_to(output_dir))
    arrays_sha256 = _sha256_file(arrays_path)
    artifact_hashes: dict[str, str] = {
        arrays_relative_path: arrays_sha256
    }
    mujoco_raw_union = visibility.mujoco_stages.source.alpha.sum(dim=0)[0]
    renderer_raw_union = 1.0 - torch.prod(
        1.0 - collected.renderer.visibility_mask,
        dim=0,
    )[0]
    mujoco_effective_union = mujoco_effective.sum(dim=0)[0]
    renderer_effective_union = 1.0 - torch.prod(
        1.0 - renderer_effective,
        dim=0,
    )[0]
    image_evidence = {
        "mujoco_raw_mask": mujoco_raw_union,
        "renderer_raw_mask": renderer_raw_union,
        "mujoco_effective_mask": mujoco_effective_union,
        "renderer_effective_mask": renderer_effective_union,
    }
    for artifact_name, alpha in image_evidence.items():
        image_path = images_dir / f"{prefix}_{artifact_name}.png"
        artifact_hashes[str(image_path.relative_to(output_dir))] = _save_mask(
            image_path,
            alpha,
        )
    overlay_path = images_dir / f"{prefix}_effective_overlay.png"
    artifact_hashes[str(overlay_path.relative_to(output_dir))] = _save_overlay(
        overlay_path,
        mujoco_effective_union,
        renderer_effective_union,
    )
    return arrays_sha256, artifact_hashes


def run_visibility_alignment_audit(
    cfg: VisibilityAlignmentConfig,
) -> Path:
    """运行真实 audit，并返回 manifest path。"""

    _validate_code_commit(cfg.code_commit)
    if cfg.object_name not in OBJECT_ASSETS:
        raise ValueError(f"未知 object_name: {cfg.object_name}")
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait 不能为负数")
    selected_state_ids = parse_state_ids(cfg.state_ids, field_name="state_ids")
    if selected_state_ids is None or not selected_state_ids:
        raise ValueError("visibility audit state_ids 不得为空")
    state_ids = tuple(selected_state_ids)
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import get_libero_dummy_action, get_libero_env

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("visibility audit 需要真实 CUDA nvdiffrast device")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "visibility_alignment_metrics.jsonl"
    manifest_path = output_dir / "visibility_alignment_manifest.json"
    if metrics_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有 visibility audit 权威文件")

    asset: ObjectAssetSpec = OBJECT_ASSETS[cfg.object_name]
    xml_path = Path(asset["xml"])
    mesh_path = Path(asset["mesh"])
    texture_path = Path(asset["texture"])
    for required_path in (xml_path, mesh_path, texture_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    mesh_sha256 = _sha256_file(mesh_path)
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=device,
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=128.0 / 255.0,
        texture_parameterization="geometry_vertex",
    ).to(device)
    renderer.reset_texture()
    renderer_faces_sha256 = _sha256_tensor(renderer.faces)
    mapping_sha256 = _sha256_tensor(
        renderer.get_render_to_geometry_mapping()
    )

    benchmark_class = benchmark.get_benchmark_dict()[cfg.task_suite_name]
    task_suite = benchmark_class()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if max(state_ids) >= len(initial_states) or min(state_ids) < 0:
        raise ValueError("visibility audit state_id 超出初始状态范围")

    thresholds = VisibilityThresholdCandidates()
    rows: list[VisibilityAlignmentAuditRow] = []
    state_fingerprints: dict[str, str] = {}
    task_description: Optional[str] = None
    for state_id in state_ids:
        print(f"[VISIBILITY] state {state_id}")
        initial_state = initial_states[state_id]
        initial_state_sha256 = _state_fingerprint(initial_state)
        state_fingerprints[str(state_id)] = initial_state_sha256
        env, current_task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=POLICY_SOURCE_RESOLUTION,
        )
        task_description = current_task_description
        try:
            env.reset()
            env.set_init_state(initial_state)
            env.env.sim.forward()
            for _ in range(cfg.num_steps_wait):
                env.step(get_libero_dummy_action(cfg.model_family))

            target_poses = find_target_body_poses(
                env,
                asset["search"],
                device,
            )
            if not target_poses:
                raise RuntimeError(
                    f"state {state_id} 未找到共享纹理目标实例"
                )
            if any(pose.body_name is None for pose in target_poses):
                raise RuntimeError("目标实例缺少 MuJoCo body name")
            instance_roots = tuple(
                TargetInstanceRoot(
                    body_id=pose.body_id,
                    body_name=str(pose.body_name),
                )
                for pose in target_poses
            )
            body_ids = tuple(pose.body_id for pose in target_poses)
            collected_views: list[_CollectedView] = []
            with static_scene_evidence_transaction(
                env,
                body_ids,
            ) as transaction:
                # 在 transaction 内重新读取 pose，防止使用等待前缓存。
                transaction_poses = find_target_body_poses(
                    env,
                    asset["search"],
                    device,
                )
                if tuple(pose.body_id for pose in transaction_poses) != body_ids:
                    raise RuntimeError("static transaction 内目标实例集合改变")
                collected_views.extend(
                    (
                        _collect_view(
                            env=env,
                            renderer=renderer,
                            target_poses=transaction_poses,
                            instance_roots=instance_roots,
                            view_name="primary",
                            camera_name="agentview",
                            device=device,
                            thresholds=thresholds,
                        ),
                        _collect_view(
                            env=env,
                            renderer=renderer,
                            target_poses=transaction_poses,
                            instance_roots=instance_roots,
                            view_name="wrist_source_crop_proxy",
                            camera_name="robot0_eye_in_hand",
                            device=device,
                            thresholds=thresholds,
                        ),
                    )
                )
            if transaction.after is None or not transaction.verified:
                raise RuntimeError("static transaction 未生成通过证据")
            transaction_evidence = VisibilityAuditTransactionEvidence(
                verified=transaction.verified,
                before_fingerprint_sha256=(
                    transaction.before.fingerprint_sha256
                ),
                after_fingerprint_sha256=(
                    transaction.after.fingerprint_sha256
                ),
                maximum_position_delta=float(
                    transaction.maximum_position_delta
                ),
                maximum_quaternion_delta=float(
                    transaction.maximum_quaternion_delta
                ),
            )
            for collected in collected_views:
                arrays_sha256, artifact_sha256 = _persist_view_artifacts(
                    output_dir=output_dir,
                    state_id=state_id,
                    collected=collected,
                )
                evaluation = collected.visibility.evaluation
                rows.append(
                    VisibilityAlignmentAuditRow(
                        schema_version=(
                            VISIBILITY_ALIGNMENT_SCHEMA_VERSION
                        ),
                        code_commit=cfg.code_commit,
                        state_id=state_id,
                        view_name=collected.view_name,
                        status=evaluation.status,
                        thresholds=thresholds,
                        observations=collected.visibility.observations,
                        union_alignment=evaluation.union_alignment,
                        instances=evaluation.instances,
                        instance_body_ids=body_ids,
                        instance_body_names=tuple(
                            str(pose.body_name) for pose in target_poses
                        ),
                        transaction=transaction_evidence,
                        initial_state_sha256=initial_state_sha256,
                        arrays_npz_sha256=arrays_sha256,
                        oriented_segmentation_sha256=_sha256_array(
                            collected.segmentation.oriented_segmentation
                        ),
                        renderer_faces_sha256=renderer_faces_sha256,
                        render_to_geometry_sha256=mapping_sha256,
                        mesh_sha256=mesh_sha256,
                        backend=asdict(collected.segmentation.backend),
                        artifact_sha256=artifact_sha256,
                    )
                )
        finally:
            env.close()

    summary = summarize_visibility_alignment_rows(
        rows,
        expected_state_ids=state_ids,
    )
    metrics_sha256 = write_visibility_alignment_jsonl(
        rows,
        output_path=metrics_path,
    )
    metadata = {
        "code_commit": cfg.code_commit,
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "task_name": str(getattr(task, "name", task_description)),
        "task_description": task_description,
        "object_name": cfg.object_name,
        "state_ids": list(state_ids),
        "state_fingerprints": state_fingerprints,
        "seed": cfg.seed,
        "num_steps_wait": cfg.num_steps_wait,
        "device": str(device),
        "threshold_candidates": asdict(thresholds),
        "deployment": {
            "policy_source_resolution": POLICY_SOURCE_RESOLUTION,
            "policy_pre_crop_resolution": 224,
            "center_crop_area": 0.9,
            "wrist_semantics": "wrist_source_crop_proxy",
        },
        "asset_sha256": {
            "xml": _sha256_file(xml_path),
            "mesh": mesh_sha256,
            "texture": _sha256_file(texture_path),
        },
        "renderer_topology_sha256": {
            "faces": renderer_faces_sha256,
            "render_to_geometry": mapping_sha256,
        },
        "versions": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
        },
        "command": " ".join(
            shlex.quote(argument) for argument in sys.argv
        ),
        "manual_review_required": True,
    }
    manifest_sha256 = write_visibility_alignment_manifest(
        summary=summary,
        metadata=metadata,
        metrics_jsonl_sha256=metrics_sha256,
        output_path=manifest_path,
    )
    print(
        json.dumps(
            {
                "structural_pass": summary.structural_pass,
                "failures": list(summary.failures),
                "metrics_sha256": metrics_sha256,
                "manifest_sha256": manifest_sha256,
                "manual_review_required": True,
            },
            sort_keys=True,
        )
    )
    if not summary.structural_pass:
        raise RuntimeError(
            "Visibility/Alignment audit 结构验收失败；完整证据已写入"
        )
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_visibility_alignment_audit(_parse_args(argv))


if __name__ == "__main__":
    main()
