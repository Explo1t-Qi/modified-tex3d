"""Visibility/Alignment audit 权威 schema 的结构判定测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
OPENVLA_ROOT = Path(__file__).resolve().parents[3] / "openvla"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
sys.path.insert(0, str(OPENVLA_ROOT))

from openvla_attack.visibility_alignment_audit import (  # noqa: E402
    VISIBILITY_ALIGNMENT_SCHEMA_VERSION,
    VisibilityAlignmentAuditRow,
    VisibilityAuditTransactionEvidence,
    summarize_visibility_alignment_rows,
    write_visibility_alignment_jsonl,
)
from openvla_attack.visibility_evidence import (  # noqa: E402
    InstanceVisibilityEvidence,
    ObservationEvidence,
    SoftAlignmentMetrics,
    VisibilityThresholdCandidates,
)
from openvla_attack.visibility_view_evidence import (  # noqa: E402
    VisibilityObservationStages,
)


GIT_COMMIT = "b" * 40
SHA256 = "a" * 64


def _row(
    state_id: int,
    view_name: str,
    *,
    status: str = "valid",
    observation_area: float = 0.1,
) -> VisibilityAlignmentAuditRow:
    observation = ObservationEvidence(
        equivalent_visible_pixels=observation_area * 100.0,
        observation_area=observation_area,
    )
    alignment = SoftAlignmentMetrics(
        overlap=10.0,
        mujoco_visible_pixels=10.0,
        renderer_visible_pixels=10.0,
        union_visible_pixels=10.0,
        recall_visible=1.0,
        precision_visible=1.0,
        iou_visible=1.0,
    )
    return VisibilityAlignmentAuditRow(
        schema_version=VISIBILITY_ALIGNMENT_SCHEMA_VERSION,
        code_commit=GIT_COMMIT,
        state_id=state_id,
        view_name=view_name,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        thresholds=VisibilityThresholdCandidates(),
        observations=VisibilityObservationStages(
            source=observation,
            pre_crop=observation,
            effective=observation,
            per_instance_source=(observation,),
            per_instance_pre_crop=(observation,),
            per_instance_effective=(observation,),
        ),
        union_alignment=alignment,
        instances=(
            InstanceVisibilityEvidence(0, observation, alignment),
        ),
        instance_body_ids=(2,),
        instance_body_names=("target",),
        transaction=VisibilityAuditTransactionEvidence(
            verified=True,
            before_fingerprint_sha256=SHA256,
            after_fingerprint_sha256=SHA256,
            maximum_position_delta=0.0,
            maximum_quaternion_delta=0.0,
        ),
        initial_state_sha256=SHA256,
        arrays_npz_sha256=SHA256,
        oriented_segmentation_sha256=SHA256,
        renderer_faces_sha256=SHA256,
        render_to_geometry_sha256=SHA256,
        mesh_sha256=SHA256,
        backend={"simulation_class": "fake"},
        artifact_sha256={"overlay": SHA256},
    )


def test_complete_rows_pass_structure_without_freezing_thresholds() -> None:
    rows = [_row(0, "primary"), _row(0, "wrist_source_crop_proxy")]

    summary = summarize_visibility_alignment_rows(
        rows,
        expected_state_ids=(0,),
    )

    assert summary.structural_pass
    assert summary.row_count == 2
    assert summary.primary_valid_state_ids == (0,)
    assert summary.threshold_freeze_requires_manual_review


def test_primary_invalid_and_boundary_observation_fail_structure() -> None:
    rows = [
        _row(0, "primary", status="invalid_alignment"),
        _row(
            0,
            "wrist_source_crop_proxy",
            observation_area=1e-3,
        ),
    ]

    summary = summarize_visibility_alignment_rows(
        rows,
        expected_state_ids=(0,),
    )

    assert not summary.structural_pass
    assert summary.primary_invalid_state_ids == (0,)
    assert summary.a_obs_boundary_cases == (
        (0, "wrist_source_crop_proxy"),
    )


def test_missing_or_duplicate_rows_fail_structure() -> None:
    primary = _row(0, "primary")

    missing = summarize_visibility_alignment_rows(
        [primary],
        expected_state_ids=(0,),
    )
    duplicate = summarize_visibility_alignment_rows(
        [primary, primary, _row(0, "wrist_source_crop_proxy")],
        expected_state_ids=(0,),
    )

    assert not missing.structural_pass
    assert any("missing_rows" in failure for failure in missing.failures)
    assert not duplicate.structural_pass
    assert "duplicate_state_view_rows" in duplicate.failures


def test_sha256_length_is_not_accepted_as_git_commit() -> None:
    primary = replace(_row(0, "primary"), code_commit=SHA256)
    wrist = replace(
        _row(0, "wrist_source_crop_proxy"),
        code_commit=SHA256,
    )

    summary = summarize_visibility_alignment_rows(
        [primary, wrist],
        expected_state_ids=(0,),
    )

    assert not summary.structural_pass
    assert "invalid_or_mixed_code_commit" in summary.failures


def test_jsonl_writer_uses_deterministic_state_view_order(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.jsonl"

    digest = write_visibility_alignment_jsonl(
        [_row(0, "wrist_source_crop_proxy"), _row(0, "primary")],
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert '"view_name":"primary"' in lines[0]
    assert '"view_name":"wrist_source_crop_proxy"' in lines[1]
    assert len(digest) == 64
