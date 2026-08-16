"""采集 Gate 6j Radial-vs-Matched-Box terminal-local 静态响应。

本 runner 把已通过 Gate 6i 的完整目录复制为自包含 parent，只读复用其中的
双终态、十 state Action gradient、Support radial step 和 clean target 绑定。
它不重新 backward、不运行训练或 rollout，只构造 coordinatewise box step，
匹配 parent radial 的实际 Surface-Linf，再在同一次 GPU 进程中重新采集
``baseline/radial_support/matched_box_support`` 共 60 条静态响应。

新 baseline/radial raw evidence 必须严格重放 parent Gate 6i；失败只令本次
audit invalid，不自动把失败原因归类为代码错误或 backend nondeterminism。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass, replace
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

from openvla_attack.assets import (  # noqa: E402
    LIBERO_ROOT,
    OBJECT_ASSETS,
    ObjectAssetSpec,
    parse_mesh_scale,
)
from openvla_attack.configuration import GenerateConfig  # noqa: E402
from openvla_attack.diagnose_terminal_endpoint_action import (  # noqa: E402
    TerminalEndpointActionConfig,
    _capture_response,
    _checkpoint_fingerprints,
    _load_clean_targets,
    _loaded_legacy_optimizer_modules,
    _validate_config,
)
from openvla_attack.fixed_support_source_training import (  # noqa: E402
    verify_executing_commit,
)
from openvla_attack.image_preprocessing import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_attack.policy_view import (  # noqa: E402
    POLICY_SOURCE_RESOLUTION,
    build_policy_view_transform,
)
from openvla_attack.renderer import DifferentiableRenderer  # noqa: E402
from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    ENDPOINT_NAMES,
    EXPECTED_STATE_IDS,
    EndpointResponseEvidence,
    load_endpoint_response_npz,
)
from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    FORMAL_SURFACE_EPSILON,
    EndpointStepEvidence,
    array_sha256,
    evaluate_terminal_endpoint_bundle,
    file_sha256,
    json_sha256,
    load_endpoint_step_npz,
    write_json_atomically,
)
from openvla_attack.terminal_projection_counterfactual import (  # noqa: E402
    PROJECTION_BUNDLE_SCHEMA_VERSION,
    PROJECTION_RESPONSE_ARMS,
    PROJECTION_SMOKE_SCHEMA_VERSION,
    MatchedBoxEndpointEvidence,
    build_matched_box_endpoint_evidence,
    evaluate_projection_response_evidence,
    frozen_projection_configuration,
    publish_terminal_projection_bundle,
    write_matched_box_endpoint_npz,
    write_projection_response_npz,
)
from openvla_utils import get_processor  # noqa: E402
from robot_utils import get_model, set_seed_everywhere  # noqa: E402


@dataclass(frozen=True)
class TerminalProjectionConfig:
    """Gate 6j CLI；科学配置从parent Gate 6i runtime严格继承。"""

    parent_bundle_path: str
    output_dir: str
    code_commit: str
    smoke_only: bool = False


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> TerminalProjectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_bundle_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument(
        "--smoke_only",
        action="store_true",
        help="只采集双endpoint的state 0三臂工程smoke，不发布正式bundle",
    )
    return TerminalProjectionConfig(**vars(parser.parse_args(argv)))


def _validate_cli(cfg: TerminalProjectionConfig) -> None:
    if len(cfg.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in cfg.code_commit
    ):
        raise ValueError("code_commit必须是40位小写Git SHA")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("manifest根必须为object")
    return value


def _parent_runtime_config(
    parent: dict[str, Any],
    *,
    cfg: TerminalProjectionConfig,
) -> TerminalEndpointActionConfig:
    provenance = parent.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("parent缺少provenance")
    raw_runtime = provenance.get("runtime_config")
    if not isinstance(raw_runtime, dict):
        raise RuntimeError("parent缺少冻结runtime_config")
    parent_cfg = TerminalEndpointActionConfig(**raw_runtime)
    runtime_cfg = replace(
        parent_cfg,
        code_commit=cfg.code_commit,
        output_dir=cfg.output_dir,
    )
    _validate_config(runtime_cfg)
    return runtime_cfg


def _copy_parent_bundle(
    parent_manifest: Path,
    *,
    output_dir: Path,
    runtime_cfg: TerminalEndpointActionConfig,
) -> tuple[Path, Path, Path]:
    """复制完整parent目录，并要求manifest bytes/SHA严格保持。"""

    source_root = parent_manifest.parent.resolve()
    copied_root = output_dir / "parent_gate6i"
    try:
        output_dir.resolve().relative_to(source_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("output_dir不能位于parent bundle目录内部")
    shutil.copytree(source_root, copied_root)
    copied_manifest = copied_root / parent_manifest.name
    original_parent_sha = file_sha256(parent_manifest)
    copied_parent_sha = file_sha256(copied_manifest)
    if copied_parent_sha != original_parent_sha:
        raise RuntimeError("strict parent copy SHA漂移")

    dense_source = Path(runtime_cfg.dense_seed_metrics_path).resolve()
    support_source = Path(runtime_cfg.production_support_path).resolve()
    inputs_root = output_dir / "parent_inputs"
    copied_dense_root = inputs_root / "dense_seed"
    shutil.copytree(dense_source.parent, copied_dense_root)
    copied_dense_metrics = copied_dense_root / dense_source.name
    inputs_root.mkdir(parents=True, exist_ok=True)
    copied_support = inputs_root / support_source.name
    shutil.copy2(support_source, copied_support)
    if file_sha256(copied_dense_metrics) != file_sha256(dense_source):
        raise RuntimeError("strict Dense Seed metrics copy SHA漂移")
    if file_sha256(copied_support) != file_sha256(support_source):
        raise RuntimeError("strict Production Support copy SHA漂移")
    copied_decision = evaluate_terminal_endpoint_bundle(
        copied_manifest,
        source_dense_seed_metrics_path=copied_dense_metrics,
        production_support_path=copied_support,
    )
    if not copied_decision.audit_valid:
        raise RuntimeError(
            "复制后的parent Gate 6i复核失败: "
            + "; ".join(copied_decision.failures)
        )
    return copied_manifest, copied_dense_metrics, copied_support


def run_terminal_projection_counterfactual(
    cfg: TerminalProjectionConfig,
) -> Path:
    """执行response-only Gate 6j，并返回原子发布的成功manifest。"""

    _validate_cli(cfg)
    verify_executing_commit(cfg.code_commit)
    if LIBERO_ROOT not in sys.path:
        sys.path.insert(0, LIBERO_ROOT)
    from libero.libero import benchmark
    from libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Gate 6j需要真实CUDA nvdiffrast device")
    if _loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6j禁止加载legacy optimization.py")
    parent_manifest = Path(cfg.parent_bundle_path).resolve()
    if not parent_manifest.is_file():
        raise FileNotFoundError(parent_manifest)
    parent_decision = evaluate_terminal_endpoint_bundle(parent_manifest)
    if not parent_decision.audit_valid:
        raise RuntimeError(
            "parent Gate 6i复核失败: " + "; ".join(parent_decision.failures)
        )
    parent_json = _load_json(parent_manifest)
    runtime_cfg = _parent_runtime_config(parent_json, cfg=cfg)

    set_seed_everywhere(runtime_cfg.seed)
    random.seed(runtime_cfg.seed)
    np.random.seed(runtime_cfg.seed)
    torch.manual_seed(runtime_cfg.seed)

    output_dir = Path(cfg.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    (
        copied_parent_manifest,
        copied_dense_metrics,
        copied_support,
    ) = _copy_parent_bundle(
        parent_manifest,
        output_dir=output_dir,
        runtime_cfg=runtime_cfg,
    )
    copied_parent_sha = file_sha256(copied_parent_manifest)
    copied_parent_json = _load_json(copied_parent_manifest)

    dense_metrics_path = Path(runtime_cfg.dense_seed_metrics_path).resolve()
    clean_targets = _load_clean_targets(dense_metrics_path)
    checkpoint_path = Path(runtime_cfg.pretrained_checkpoint).resolve()
    asset: ObjectAssetSpec = OBJECT_ASSETS[runtime_cfg.object_name]
    xml_path = Path(asset["xml"]).resolve()
    mesh_path = Path(asset["mesh"]).resolve()
    texture_path = Path(asset["texture"]).resolve()
    for required in (
        dense_metrics_path,
        checkpoint_path,
        xml_path,
        mesh_path,
        texture_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    model_cfg = GenerateConfig(
        pretrained_checkpoint=runtime_cfg.pretrained_checkpoint,
        model_family=runtime_cfg.model_family,
        center_crop=True,
        object_name=runtime_cfg.object_name,
        task_suite_name=runtime_cfg.task_suite_name,
        task_id=runtime_cfg.task_id,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=runtime_cfg.unnorm_key,
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
    if (model_height, model_width) != (224, 224):
        raise RuntimeError("Gate 6j schema固定224x224 checkpoint输入")
    view_transform = build_policy_view_transform(
        source_resolution=POLICY_SOURCE_RESOLUTION,
        model_input_resolution=model_height,
    )
    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(model.device),
        scale_xyz=parse_mesh_scale(xml_path),
        epsilon=FORMAL_SURFACE_EPSILON,
        texture_parameterization="geometry_vertex",
    ).to(model.device)

    parent_steps: dict[str, EndpointStepEvidence] = {}
    matched_steps: dict[str, MatchedBoxEndpointEvidence] = {}
    matched_records: list[dict[str, Any]] = []
    for record in copied_parent_json["endpoint_steps"]:
        endpoint = str(record["endpoint"])
        parent_step_path = (
            copied_parent_manifest.parent / record["npz_relative_path"]
        )
        parent_step = load_endpoint_step_npz(parent_step_path)
        if file_sha256(parent_step_path) != record["npz_sha256"]:
            raise RuntimeError(f"{endpoint} parent step SHA漂移")
        matched = build_matched_box_endpoint_evidence(
            parent_bundle_sha256=copied_parent_sha,
            parent_step_npz_sha256=record["npz_sha256"],
            parent_step=parent_step,
        )
        parent_steps[endpoint] = parent_step
        matched_steps[endpoint] = matched
        relative = Path("steps") / f"{endpoint}_matched_box.npz"
        digest = write_matched_box_endpoint_npz(
            matched,
            output_path=output_dir / relative,
        )
        matched_records.append(
            {
                "endpoint": endpoint,
                "npz_relative_path": str(relative),
                "npz_sha256": digest,
                "parent_step_npz_sha256": record["npz_sha256"],
            }
        )
    if set(parent_steps) != set(ENDPOINT_NAMES):
        raise RuntimeError("parent endpoint step inventory不完整")
    geometry_shape = parent_steps[ENDPOINT_NAMES[0]].support_step.realized_endpoint.shape
    if renderer.get_texture_param().shape != geometry_shape:
        raise RuntimeError("parent Surface与renderer geometry shape不一致")

    benchmark_class = benchmark.get_benchmark_dict()[runtime_cfg.task_suite_name]
    suite = benchmark_class()
    task = suite.get_task(runtime_cfg.task_id)
    initial_states = suite.get_task_init_states(runtime_cfg.task_id)
    description_env, task_description = get_libero_env(
        task,
        runtime_cfg.model_family,
        resolution=POLICY_SOURCE_RESOLUTION,
    )
    description_env.close()

    response_records: list[dict[str, Any]] = []
    captured_responses: list[EndpointResponseEvidence] = []
    response_state_ids = (0,) if cfg.smoke_only else EXPECTED_STATE_IDS
    for endpoint in ENDPOINT_NAMES:
        parent_step = parent_steps[endpoint]
        matched = matched_steps[endpoint]
        baseline_surface = parent_step.support_step.realized_endpoint
        radial_surface = np.ascontiguousarray(
            baseline_surface + parent_step.support_step.executed_step,
            dtype=np.float32,
        )
        surfaces = {
            "baseline": baseline_surface,
            "radial_support": radial_surface,
            "matched_box_support": matched.matched_surface_delta,
        }
        expected_hashes = {
            "baseline": matched.realized_endpoint_sha256,
            "radial_support": matched.radial_surface_delta_sha256,
            "matched_box_support": matched.matched_surface_delta_sha256,
        }
        for arm, surface in surfaces.items():
            if array_sha256(surface) != expected_hashes[arm]:
                raise RuntimeError(f"{endpoint}/{arm} Surface SHA漂移")
        for state_id in response_state_ids:
            for arm in PROJECTION_RESPONSE_ARMS:
                print(
                    f"[GATE-6J] response endpoint={endpoint} "
                    f"state={state_id} arm={arm}"
                )
                evidence = _capture_response(
                    cfg=runtime_cfg,
                    endpoint=endpoint,
                    state_id=state_id,
                    arm=arm,
                    surface=surfaces[arm],
                    initial_state=initial_states[state_id],
                    clean_target=clean_targets[state_id],
                    task=task,
                    task_description=task_description,
                    asset=asset,
                    model=model,
                    processor=processor,
                    image_preprocessor=image_preprocessor,
                    view_transform=view_transform,
                    renderer=renderer,
                    get_dummy_action=get_libero_dummy_action,
                    get_libero_env=get_libero_env,
                    get_libero_image=get_libero_image,
                )
                relative = Path("responses") / (
                    f"{endpoint}_state_{state_id:02d}_{arm}.npz"
                )
                digest = write_projection_response_npz(
                    evidence,
                    output_path=output_dir / relative,
                )
                captured_responses.append(evidence)
                response_records.append(
                    {
                        "endpoint": endpoint,
                        "state_id": state_id,
                        "arm": arm,
                        "npz_relative_path": str(relative),
                        "npz_sha256": digest,
                    }
                )

    if _loaded_legacy_optimizer_modules():
        raise RuntimeError("Gate 6j运行中加载了legacy optimization.py")
    parent_record = {
        "manifest_relative_path": str(
            copied_parent_manifest.relative_to(output_dir)
        ),
        "manifest_sha256": copied_parent_sha,
        "source_dense_seed_metrics_relative_path": str(
            copied_dense_metrics.relative_to(output_dir)
        ),
        "source_dense_seed_metrics_sha256": file_sha256(
            copied_dense_metrics
        ),
        "production_support_relative_path": str(
            copied_support.relative_to(output_dir)
        ),
        "production_support_sha256": file_sha256(copied_support),
    }
    provenance_record = {
        "gradient_recomputed": False,
        "training_or_rollout_run": False,
        "feature_gradient": False,
        "wrist_gradient": False,
        "oft_gradient": False,
    }
    runtime_record = {
        "cli": asdict(cfg),
        "inherited_parent_runtime": asdict(runtime_cfg),
        "checkpoint_fingerprints": _checkpoint_fingerprints(checkpoint_path),
        "processor_specification": asdict(image_preprocessor),
        "processor_specification_sha256": json_sha256(
            asdict(image_preprocessor)
        ),
    }
    if cfg.smoke_only:
        parent_responses: list[EndpointResponseEvidence] = []
        for record in copied_parent_json["response_records"]:
            if record["state_id"] != 0 or record["arm"] not in (
                "baseline",
                "support",
            ):
                continue
            artifact = (
                copied_parent_manifest.parent / record["npz_relative_path"]
            )
            if file_sha256(artifact) != record["npz_sha256"]:
                raise RuntimeError("smoke parent response SHA漂移")
            parent_responses.append(load_endpoint_response_npz(artifact))
        smoke_decision = evaluate_projection_response_evidence(
            captured_responses,
            parent_responses=parent_responses,
            matched_steps_by_endpoint=matched_steps,
            expected_state_ids=response_state_ids,
        )
        if not smoke_decision.audit_valid:
            raise RuntimeError(
                "Gate 6j smoke复核失败: "
                + "; ".join(smoke_decision.failures)
            )
        smoke_path = output_dir / "terminal_projection_smoke.json"
        write_json_atomically(
            {
                "schema_version": PROJECTION_SMOKE_SCHEMA_VERSION,
                "status": "complete",
                "code_commit": cfg.code_commit,
                "formal_bundle": False,
                "state_ids": list(response_state_ids),
                "parent": parent_record,
                "endpoint_steps": matched_records,
                "response_records": response_records,
                "provenance": provenance_record,
                "runtime": runtime_record,
                "derived": asdict(smoke_decision),
            },
            output_path=smoke_path,
        )
        print(f"[GATE-6J-SMOKE] manifest={smoke_path}")
        return smoke_path

    configuration = frozen_projection_configuration()
    payload = {
        "schema_version": PROJECTION_BUNDLE_SCHEMA_VERSION,
        "code_commit": cfg.code_commit,
        "configuration": configuration,
        "config_sha256": json_sha256(configuration),
        "parent": parent_record,
        "endpoint_steps": matched_records,
        "response_records": response_records,
        "provenance": provenance_record,
        "runtime": runtime_record,
    }
    manifest_path = output_dir / "terminal_projection_manifest.json"
    publish_terminal_projection_bundle(payload, output_path=manifest_path)
    print(f"[GATE-6J] manifest={manifest_path}")
    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    print(
        "[GATE-6J] command="
        + " ".join(shlex.quote(argument) for argument in sys.argv)
    )
    run_terminal_projection_counterfactual(cfg)


if __name__ == "__main__":
    main()
