"""Gate 6i endpoint step artifact 与 bundle 的纯 CPU 合同测试。"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_endpoint_action_evidence import (  # noqa: E402
    ENDPOINT_BUNDLE_SCHEMA_VERSION,
    FORMAL_SURFACE_EPSILON,
    FORMAL_SURFACE_STEP,
    TerminalEndpointActionEvidenceError,
    build_endpoint_step_evidence,
    evaluate_endpoint_step_evidence,
    evaluate_terminal_endpoint_bundle,
    file_sha256,
    frozen_bundle_configuration,
    json_sha256,
    load_endpoint_step_npz,
    publish_terminal_endpoint_bundle,
    write_endpoint_step_npz,
)
from openvla_attack.action_objective_audit import (  # noqa: E402
    ACTION_OBJECTIVE_NAME,
    ACTION_OBJECTIVE_SCHEMA_VERSION,
    ActionObjectiveAuditEvidence,
)
from openvla_attack.dense_seed_audit import (  # noqa: E402
    DenseSeedCaptureMetadata,
    array_sha256 as dense_array_sha256,
    write_dense_seed_gradient_evidence,
    write_dense_seed_state_artifact,
)
from openvla_attack.deployment_backward_audit import (  # noqa: E402
    GradientEvidence,
)
from openvla_attack.production_support import (  # noqa: E402
    PRODUCTION_SUPPORT_SCHEMA_VERSION,
    FrozenProductionSupport,
    ProductionSupportProvenance,
    array_sha256 as support_array_sha256,
    write_production_support_artifact,
)
from openvla_attack.terminal_endpoint_action_audit import (  # noqa: E402
    EndpointGradientEvidence,
    EndpointResponseEvidence,
    compute_action_hinge,
    load_endpoint_response_npz,
    write_endpoint_gradient_npz,
    write_endpoint_response_npz,
)
from openvla_attack.terminal_projection_counterfactual import (  # noqa: E402
    PROJECTION_BUNDLE_SCHEMA_VERSION,
    build_matched_box_endpoint_evidence,
    evaluate_terminal_projection_bundle,
    frozen_projection_configuration,
    publish_terminal_projection_bundle,
    write_matched_box_endpoint_npz,
    write_projection_response_npz,
)


def _step_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gradient = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    support_mask = np.asarray([True, False, True, False])
    return endpoint, gradient, support_mask


def test_endpoint_step_npz_round_trip_and_independent_recompute(
    tmp_path: Path,
) -> None:
    endpoint, gradient, support_mask = _step_inputs()
    evidence = build_endpoint_step_evidence(
        endpoint="action_spectral",
        endpoint_surface_delta=endpoint,
        aggregate_gradient=gradient,
        support_mask=support_mask,
        epsilon=1.0,
        surface_step=0.2,
    )
    path = tmp_path / "endpoint-step.npz"

    digest = write_endpoint_step_npz(evidence, output_path=path)
    loaded = load_endpoint_step_npz(path)
    decision = evaluate_endpoint_step_evidence(
        loaded,
        aggregate_gradient=gradient,
        support_mask=support_mask,
    )

    assert len(digest) == 64
    assert decision.audit_valid, decision.failures
    assert decision.endpoint == "action_spectral"
    assert len(decision.arm_metrics) == 2
    dense, support = decision.arm_metrics
    assert dense.arm == "dense"
    assert support.arm == "support"
    assert np.isclose(dense.executed_linf, 0.2)
    # endpoint边界上的全局投影可以让实际步小于归一化目标。
    assert 0.0 < support.executed_linf <= 0.2
    assert dense.gradient_step_inner_product < 0.0
    assert support.gradient_step_inner_product < 0.0
    assert dense.descent_alignment_cosine > 0.0
    assert support.descent_alignment_cosine > 0.0
    assert loaded.dense_surface_delta_sha256 != (
        loaded.support_surface_delta_sha256
    )


def test_endpoint_step_rejects_projection_residual_or_support_hash_drift() -> None:
    endpoint, gradient, support_mask = _step_inputs()
    evidence = build_endpoint_step_evidence(
        endpoint="action_only_control",
        endpoint_surface_delta=endpoint,
        aggregate_gradient=gradient,
        support_mask=support_mask,
        epsilon=1.0,
        surface_step=0.2,
    )
    residual = evidence.support_step.projection_residual.copy()
    residual[0, 0] += np.float32(0.1)
    drifted_step = replace(
        evidence.support_step,
        projection_residual=residual,
    )

    residual_decision = evaluate_endpoint_step_evidence(
        replace(evidence, support_step=drifted_step),
        aggregate_gradient=gradient,
        support_mask=support_mask,
    )
    hash_decision = evaluate_endpoint_step_evidence(
        replace(evidence, support_mask_sha256="f" * 64),
        aggregate_gradient=gradient,
        support_mask=support_mask,
    )

    assert not residual_decision.audit_valid
    assert any(
        "projection_residual" in failure
        for failure in residual_decision.failures
    )
    assert not hash_decision.audit_valid
    assert any(
        "support mask SHA" in failure for failure in hash_decision.failures
    )


def _gradient_stats(*shape: int) -> GradientEvidence:
    return GradientEvidence.from_tensor(torch.ones(shape))


def _zero_gradient(state_id: int) -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(4, 3) + state_id + 1.0


def _objective(
    state_id: int,
    gradient: np.ndarray,
) -> ActionObjectiveAuditEvidence:
    return ActionObjectiveAuditEvidence(
        schema_version=ACTION_OBJECTIVE_SCHEMA_VERSION,
        code_commit="a" * 40,
        state_id=state_id,
        initial_state_sha256=f"{state_id + 1:064x}",
        objective=ACTION_OBJECTIVE_NAME,
        parameterization="geometry_vertex_dense",
        zero_surface_delta_linf=0.0,
        teacher_forced_clean_prefix=True,
        clean_action_token_ids=[31_744, 31_745, 31_999],
        clean_classes=[0, 1, 255],
        margins=[2.0, 0.0, 1.0],
        hinge_values=[2.0, 0.0, 1.0],
        action_loss=1.0,
        parameter_shape=(4, 3),
        source_rgb_gradient=_gradient_stats(1, 3, 8, 8),
        pre_crop_gradient=_gradient_stats(1, 3, 4, 4),
        effective_view_gradient=_gradient_stats(1, 3, 4, 4),
        render_surface_delta_gradient=_gradient_stats(6, 3),
        dense_geometry_gradient=GradientEvidence.from_tensor(
            torch.from_numpy(gradient)
        ),
        dense_geometry_gradient_sha256=dense_array_sha256(gradient),
    )


def _write_source_inputs(root: Path) -> tuple[Path, Path, np.ndarray]:
    source_root = root / "source-dense"
    metadata = DenseSeedCaptureMetadata(
        mesh_sha256="b" * 64,
        render_to_geometry_sha256="c" * 64,
        policy_source_rgb_sha256="d" * 64,
        effective_view_rgb_sha256="e" * 64,
        mujoco_instance_alpha_sha256="f" * 64,
        renderer_visibility_sha256="1" * 64,
        shared_instance_body_ids=(17, 23),
        shared_instance_body_names=("bowl_main", "bowl_secondary"),
    )
    rows = []
    for state_id in range(10):
        gradient = _zero_gradient(state_id)
        rows.append(
            write_dense_seed_state_artifact(
                output_root=source_root,
                objective_evidence=_objective(state_id, gradient),
                dense_geometry_gradient=gradient,
                metadata=metadata,
            )
        )
    metrics_path = source_root / "dense_seed_metrics.jsonl"
    write_dense_seed_gradient_evidence(
        metrics_path,
        rows,
        artifact_root=source_root,
    )

    support_mask = np.asarray([False, True, False, True], dtype=np.bool_)
    provenance = ProductionSupportProvenance(
        code_commit="1" * 40,
        object_name="akita_black_bowl",
        source_support_manifest_sha256="2" * 64,
        source_candidate_artifact_sha256="3" * 64,
        source_coverage_artifact_sha256="4" * 64,
        seed_score_manifest_sha256="5" * 64,
        seed_score_artifact_sha256="6" * 64,
        visibility_manifest_sha256="7" * 64,
        visibility_metrics_sha256="8" * 64,
        mesh_file_sha256="b" * 64,
        mesh_array_sha256="9" * 64,
        renderer_faces_sha256="a" * 64,
        render_to_geometry_sha256="c" * 64,
    )
    support = FrozenProductionSupport(
        schema_version=PRODUCTION_SUPPORT_SCHEMA_VERSION,
        provenance=provenance,
        selected_candidate_index=0,
        num_geometry_vertices=4,
        support_mask=support_mask,
        support_vertex_indices=np.asarray([1, 3], dtype=np.int64),
        support_mask_sha256=support_array_sha256(support_mask),
        compact_coordinate_order="ascending_geometry_vertex_id",
        num_regions=1,
        seed_vertex_ids=(1,),
        total_surface_area=10.0,
        target_area_fraction=0.1,
        target_mass=1.0,
        actual_mass=1.0,
        actual_area_fraction=0.1,
        primary_coverage_min_threshold=0.2,
        primary_state_ids=(0,),
        primary_statuses=("valid",),
        primary_source_coverage=(0.3,),
        primary_effective_coverage=(0.3,),
        wrist_state_ids=(0,),
        wrist_statuses=("valid",),
        wrist_source_coverage=(0.1,),
        wrist_effective_coverage=(0.1,),
        naturalness_k_nonconstant=128,
        production_support_constructed=True,
        fixed_support_frozen=True,
        rho_nat_calibrated=False,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )
    support_path = root / "production_fixed_support.npz"
    write_production_support_artifact(support_path, support)
    return metrics_path, support_path, support_mask


def _response(
    *,
    endpoint: str,
    state_id: int,
    arm: str,
    loss: float,
    surface_sha256: str,
) -> EndpointResponseEvidence:
    clean_classes = np.asarray([0, 1, 255], dtype=np.int64)
    teacher = np.zeros((3, 256), dtype=np.float32)
    teacher[np.arange(3), clean_classes] = np.float32(loss)
    objective = compute_action_hinge(teacher, clean_classes)
    generation = np.zeros((3, 256), dtype=np.float32)
    generation[np.arange(3), clean_classes] = np.float32(1.0)
    return EndpointResponseEvidence(
        endpoint=endpoint,
        state_id=state_id,
        arm=arm,
        state_fingerprint=f"{state_id + 1:064x}",
        surface_delta_sha256=surface_sha256,
        clean_action_token_ids=clean_classes + 31_744,
        clean_classes=clean_classes,
        action_loss=objective.action_loss,
        margins=objective.margins,
        hinge_values=objective.hinge_values,
        generated_token_ids=clean_classes + 31_744,
        generated_classes=clean_classes.copy(),
        generation_logits=generation,
        teacher_logits=teacher,
        decoded_action=np.zeros(3, dtype=np.float32),
        effective_view_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        processor_bf16_bits=np.zeros((1, 6, 224, 224), dtype=np.uint16),
    )


def _bundle_payload(root: Path) -> dict[str, object]:
    metrics_path, support_path, support_mask = _write_source_inputs(root)
    artifact_root = root / "artifacts"
    gradient_records = []
    response_records = []
    step_records = []
    endpoint_sources = []
    aggregate_gradient = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
    endpoint_surface = np.zeros((4, 3), dtype=np.float32)
    for endpoint_index, endpoint in enumerate(
        ("action_spectral", "action_only_control")
    ):
        step = build_endpoint_step_evidence(
            endpoint=endpoint,
            endpoint_surface_delta=endpoint_surface,
            aggregate_gradient=aggregate_gradient,
            support_mask=support_mask,
            epsilon=FORMAL_SURFACE_EPSILON,
            surface_step=FORMAL_SURFACE_STEP,
        )
        step_path = artifact_root / f"{endpoint}_step.npz"
        step_sha = write_endpoint_step_npz(step, output_path=step_path)
        step_records.append(
            {
                "endpoint": endpoint,
                "npz_relative_path": str(step_path.relative_to(root)),
                "npz_sha256": step_sha,
            }
        )
        arm_hashes = {
            "baseline": step.realized_endpoint_sha256,
            "dense": step.dense_surface_delta_sha256,
            "support": step.support_surface_delta_sha256,
        }
        for state_id in range(10):
            gradient_evidence = EndpointGradientEvidence(
                endpoint=endpoint,
                state_id=state_id,
                state_fingerprint=f"{state_id + 1:064x}",
                realized_endpoint_sha256=step.realized_endpoint_sha256,
                dense_surface_gradient=aggregate_gradient.copy(),
            )
            gradient_path = artifact_root / (
                f"{endpoint}_state_{state_id:02d}_gradient.npz"
            )
            gradient_sha = write_endpoint_gradient_npz(
                gradient_evidence,
                output_path=gradient_path,
            )
            gradient_records.append(
                {
                    "endpoint": endpoint,
                    "state_id": state_id,
                    "state_fingerprint": gradient_evidence.state_fingerprint,
                    "npz_relative_path": str(gradient_path.relative_to(root)),
                    "npz_sha256": gradient_sha,
                }
            )
            for arm, loss in (
                ("baseline", 5.0),
                ("dense", 3.0),
                ("support", 4.0),
            ):
                response = _response(
                    endpoint=endpoint,
                    state_id=state_id,
                    arm=arm,
                    loss=loss,
                    surface_sha256=arm_hashes[arm],
                )
                response_path = artifact_root / (
                    f"{endpoint}_state_{state_id:02d}_{arm}.npz"
                )
                response_sha = write_endpoint_response_npz(
                    response,
                    output_path=response_path,
                )
                response_records.append(
                    {
                        "endpoint": endpoint,
                        "state_id": state_id,
                        "arm": arm,
                        "npz_relative_path": str(
                            response_path.relative_to(root)
                        ),
                        "npz_sha256": response_sha,
                    }
                )
        formal_path = root / f"{endpoint_index}_formal.json"
        formal_path.write_text("{}\n", encoding="utf-8")
        compact_path = root / f"{endpoint_index}_compact.bin"
        compact_path.write_bytes(endpoint.encode("ascii"))
        endpoint_sources.append(
            {
                "endpoint": endpoint,
                "formal_manifest_relative_path": str(
                    formal_path.relative_to(root)
                ),
                "formal_manifest_sha256": file_sha256(formal_path),
                "compact_parameter_relative_path": str(
                    compact_path.relative_to(root)
                ),
                "compact_parameter_sha256": file_sha256(compact_path),
            }
        )
    configuration = frozen_bundle_configuration()
    processor_specification = {
        "output_size": [224, 224],
        "branches": [
            {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        ],
    }
    return {
        "schema_version": ENDPOINT_BUNDLE_SCHEMA_VERSION,
        "code_commit": "f" * 40,
        "configuration": configuration,
        "config_sha256": json_sha256(configuration),
        "provenance": {
            "source_dense_seed_metrics_path": str(metrics_path),
            "source_dense_seed_metrics_sha256": file_sha256(metrics_path),
            "production_support_path": str(support_path),
            "production_support_sha256": file_sha256(support_path),
            "endpoint_sources": endpoint_sources,
            "processor_specification": processor_specification,
            "processor_specification_sha256": json_sha256(
                processor_specification
            ),
            "gradient_modalities": ["source_openvla_primary_action"],
            "teacher_forced_clean_prefix": True,
            "shared_texture_instances_aggregated": True,
            "render_to_geometry_mapping": "strict_face_corner_scatter_add",
            "feature_gradient": False,
            "wrist_gradient": False,
            "oft_gradient": False,
            "training_or_rollout_run": False,
            "legacy_optimizer_modules": [],
        },
        "gradient_cases": gradient_records,
        "response_records": response_records,
        "endpoint_steps": step_records,
    }


def test_bundle_publish_recomputes_sources_inventory_steps_and_derived(
    tmp_path: Path,
) -> None:
    payload = _bundle_payload(tmp_path)
    manifest_path = tmp_path / "terminal_endpoint_manifest.json"

    digest = publish_terminal_endpoint_bundle(
        payload,
        output_path=manifest_path,
    )
    decision = evaluate_terminal_endpoint_bundle(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert decision.audit_valid, decision.failures
    assert decision.gradient_case_count == 20
    assert decision.response_record_count == 60
    assert decision.endpoint_step_count == 2
    assert manifest["status"] == "complete"
    assert "support_bottleneck" not in json.dumps(manifest["derived"])
    assert not (tmp_path / "terminal_endpoint_manifest_candidate.json").exists()
    assert not (
        tmp_path / "terminal_endpoint_manifest_raw_candidate.json"
    ).exists()

    cross_platform = json.loads(json.dumps(manifest))
    original_cosine = cross_platform["derived"]["evidence"][
        "aggregate_endpoint_gradient_cosine"
    ]
    cross_platform["derived"]["evidence"][
        "aggregate_endpoint_gradient_cosine"
    ] = float(np.nextafter(original_cosine, np.inf))
    cross_platform_path = tmp_path / "cross_platform_derived.json"
    cross_platform_path.write_text(
        json.dumps(cross_platform, sort_keys=True),
        encoding="utf-8",
    )
    cross_platform_decision = evaluate_terminal_endpoint_bundle(
        cross_platform_path
    )
    assert cross_platform_decision.audit_valid, (
        cross_platform_decision.failures
    )

    manifest["derived"]["evidence"][
        "aggregate_endpoint_gradient_cosine"
    ] = original_cosine + 1e-9
    tampered_path = tmp_path / "tampered_derived.json"
    tampered_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    tampered = evaluate_terminal_endpoint_bundle(tampered_path)
    assert not tampered.audit_valid
    assert any("derived" in failure for failure in tampered.failures)


def test_invalid_bundle_is_not_published_as_success(tmp_path: Path) -> None:
    payload = _bundle_payload(tmp_path)
    responses = list(payload["response_records"])
    responses[-1] = responses[0]
    payload["response_records"] = responses
    manifest_path = tmp_path / "invalid_manifest.json"

    with pytest.raises(
        TerminalEndpointActionEvidenceError,
        match="raw candidate复核失败",
    ):
        publish_terminal_endpoint_bundle(payload, output_path=manifest_path)

    assert not manifest_path.exists()
    assert not (tmp_path / "invalid_manifest_raw_candidate.json").exists()
    assert not (tmp_path / "invalid_manifest_candidate.json").exists()


def test_projection_child_bundle_replays_parent_and_reports_each_endpoint(
    tmp_path: Path,
) -> None:
    """Gate 6j child必须自包含parent，并原子发布60条严格三臂证据。"""

    child_root = tmp_path / "projection-child"
    parent_root = child_root / "parent_gate6i"
    parent_payload = _bundle_payload(parent_root)
    parent_manifest = parent_root / "terminal_endpoint_manifest.json"
    publish_terminal_endpoint_bundle(
        parent_payload,
        output_path=parent_manifest,
    )
    parent_json = json.loads(parent_manifest.read_text(encoding="utf-8"))

    matched_records = []
    matched_by_endpoint = {}
    for record in parent_json["endpoint_steps"]:
        parent_step_path = parent_root / record["npz_relative_path"]
        parent_step = load_endpoint_step_npz(parent_step_path)
        matched = build_matched_box_endpoint_evidence(
            parent_bundle_sha256=file_sha256(parent_manifest),
            parent_step_npz_sha256=record["npz_sha256"],
            parent_step=parent_step,
        )
        matched_by_endpoint[matched.endpoint] = matched
        relative = Path("steps") / f"{matched.endpoint}_matched_box.npz"
        digest = write_matched_box_endpoint_npz(
            matched,
            output_path=child_root / relative,
        )
        matched_records.append(
            {
                "endpoint": matched.endpoint,
                "npz_relative_path": str(relative),
                "npz_sha256": digest,
                "parent_step_npz_sha256": record["npz_sha256"],
            }
        )

    response_records = []
    for record in parent_json["response_records"]:
        if record["arm"] not in ("baseline", "support"):
            continue
        parent_response = load_endpoint_response_npz(
            parent_root / record["npz_relative_path"]
        )
        arm = (
            "baseline"
            if parent_response.arm == "baseline"
            else "radial_support"
        )
        response = replace(parent_response, arm=arm)
        relative = Path("responses") / (
            f"{response.endpoint}_state_{response.state_id:02d}_{arm}.npz"
        )
        digest = write_projection_response_npz(
            response,
            output_path=child_root / relative,
        )
        response_records.append(
            {
                "endpoint": response.endpoint,
                "state_id": response.state_id,
                "arm": arm,
                "npz_relative_path": str(relative),
                "npz_sha256": digest,
            }
        )
        if arm == "radial_support":
            matched = matched_by_endpoint[response.endpoint]
            matched_response = _response(
                endpoint=response.endpoint,
                state_id=response.state_id,
                arm="matched_box_support",
                loss=2.0,
                surface_sha256=matched.matched_surface_delta_sha256,
            )
            matched_relative = Path("responses") / (
                f"{response.endpoint}_state_{response.state_id:02d}_"
                "matched_box_support.npz"
            )
            matched_digest = write_projection_response_npz(
                matched_response,
                output_path=child_root / matched_relative,
            )
            response_records.append(
                {
                    "endpoint": matched_response.endpoint,
                    "state_id": matched_response.state_id,
                    "arm": matched_response.arm,
                    "npz_relative_path": str(matched_relative),
                    "npz_sha256": matched_digest,
                }
            )

    configuration = frozen_projection_configuration()
    payload = {
        "schema_version": PROJECTION_BUNDLE_SCHEMA_VERSION,
        "code_commit": "e" * 40,
        "configuration": configuration,
        "config_sha256": json_sha256(configuration),
        "parent": {
            "manifest_relative_path": str(parent_manifest.relative_to(child_root)),
            "manifest_sha256": file_sha256(parent_manifest),
            "source_dense_seed_metrics_relative_path": str(
                Path(parent_json["provenance"]["source_dense_seed_metrics_path"])
                .resolve()
                .relative_to(child_root.resolve())
            ),
            "source_dense_seed_metrics_sha256": parent_json["provenance"][
                "source_dense_seed_metrics_sha256"
            ],
            "production_support_relative_path": str(
                Path(parent_json["provenance"]["production_support_path"])
                .resolve()
                .relative_to(child_root.resolve())
            ),
            "production_support_sha256": parent_json["provenance"][
                "production_support_sha256"
            ],
        },
        "endpoint_steps": matched_records,
        "response_records": response_records,
        "provenance": {
            "gradient_recomputed": False,
            "training_or_rollout_run": False,
            "feature_gradient": False,
            "wrist_gradient": False,
            "oft_gradient": False,
        },
    }
    output = child_root / "terminal_projection_manifest.json"

    digest = publish_terminal_projection_bundle(payload, output_path=output)
    decision = evaluate_terminal_projection_bundle(output)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert decision.audit_valid, decision.failures
    assert decision.response_record_count == 60
    assert decision.endpoint_step_count == 2
    assert tuple(
        metric.endpoint for metric in decision.response_decision.endpoint_metrics
    ) == ("action_spectral", "action_only_control")
    assert "projection_harm" not in json.dumps(manifest["derived"])
    assert manifest["status"] == "complete"

    incomplete = json.loads(json.dumps(manifest))
    del incomplete["derived"]
    incomplete_path = child_root / "incomplete_projection.json"
    incomplete_path.write_text(
        json.dumps(incomplete, sort_keys=True),
        encoding="utf-8",
    )
    incomplete_decision = evaluate_terminal_projection_bundle(incomplete_path)
    assert not incomplete_decision.audit_valid
    assert any("derived" in failure for failure in incomplete_decision.failures)

    invalid_commit = json.loads(json.dumps(manifest))
    invalid_commit["code_commit"] = "not-a-commit"
    invalid_commit_path = child_root / "invalid_commit_projection.json"
    invalid_commit_path.write_text(
        json.dumps(invalid_commit, sort_keys=True),
        encoding="utf-8",
    )
    invalid_commit_decision = evaluate_terminal_projection_bundle(
        invalid_commit_path
    )
    assert not invalid_commit_decision.audit_valid
    assert any(
        "code_commit" in failure
        for failure in invalid_commit_decision.failures
    )

    invalid_status = json.loads(json.dumps(manifest))
    invalid_status["status"] = "running"
    invalid_status_path = child_root / "invalid_status_projection.json"
    invalid_status_path.write_text(
        json.dumps(invalid_status, sort_keys=True),
        encoding="utf-8",
    )
    invalid_status_decision = evaluate_terminal_projection_bundle(
        invalid_status_path
    )
    assert not invalid_status_decision.audit_valid
    assert any("status" in failure for failure in invalid_status_decision.failures)

    cross_platform = json.loads(json.dumps(manifest))
    original_l2 = cross_platform["derived"]["step_metrics"][0]["matched_l2"]
    cross_platform["derived"]["step_metrics"][0]["matched_l2"] = float(
        np.nextafter(original_l2, np.inf)
    )
    cross_platform_path = child_root / "cross_platform_projection.json"
    cross_platform_path.write_text(
        json.dumps(cross_platform, sort_keys=True),
        encoding="utf-8",
    )
    cross_platform_decision = evaluate_terminal_projection_bundle(
        cross_platform_path
    )
    assert cross_platform_decision.audit_valid, (
        cross_platform_decision.failures
    )

    cross_platform["derived"]["step_metrics"][0]["matched_l2"] = (
        original_l2 + 1e-9
    )
    tampered_path = child_root / "tampered_projection.json"
    tampered_path.write_text(
        json.dumps(cross_platform, sort_keys=True),
        encoding="utf-8",
    )
    tampered = evaluate_terminal_projection_bundle(tampered_path)
    assert not tampered.audit_valid
    assert any("derived" in failure for failure in tampered.failures)
