"""Gate 2R renderer-to-bake response 的纯 CPU 单元测试。"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.renderer_bake_response_audit import (  # noqa: E402
    PROBE_CHANNELS,
    RendererBakeResponseEvidence,
    RendererBakeResponseProvenance,
    compute_untargeted_clean_action_margins,
    compute_weighted_response_metrics,
    evaluate_renderer_bake_response_evidence,
    summarize_renderer_bake_response_rows,
    write_renderer_bake_response_csv,
    write_renderer_bake_response_jsonl,
    write_renderer_bake_response_manifest,
    write_renderer_bake_response_npz,
)


def _evidence(
    state_id: int = 0,
    probe_channel: str = "r",
) -> RendererBakeResponseEvidence:
    alpha = np.asarray([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
    response = np.zeros((2, 2, 3), dtype=np.float32)
    response[..., PROBE_CHANNELS.index(probe_channel)] = 0.01
    margins = np.asarray([2.0, 1.0], dtype=np.float64)
    return RendererBakeResponseEvidence(
        state_id=state_id,
        probe_channel=probe_channel,  # type: ignore[arg-type]
        alpha=alpha,
        d_sur=response,
        d_bake=response.copy(),
        clean_action_margins=margins,
        surrogate_action_margins=margins - 0.1,
        bake_action_margins=margins - 0.2,
    )


def _provenance() -> RendererBakeResponseProvenance:
    artifact_sha256 = {"arrays/state_00_probe_r.npz": "2" * 64}
    for role in (
        "alpha",
        "d_sur",
        "d_bake",
        "difference",
        "weighted_scatter",
    ):
        artifact_sha256[f"visualizations/state_00_probe_r_{role}.png"] = (
            "3" * 64
        )
    return RendererBakeResponseProvenance(
        code_commit="a" * 40,
        config_sha256="b" * 64,
        evidence_sha256="c" * 64,
        initial_state_sha256="d" * 64,
        clean_sim_state_sha256="4" * 64,
        bake_sim_state_sha256="4" * 64,
        asset_xml_path="assets/task.xml",
        real_texture_path="assets/texture.png",
        transaction_verified=True,
        xml_sha256_before="e" * 64,
        xml_sha256_after_restore="e" * 64,
        clean_texture_sha256="f" * 64,
        baked_texture_sha256="1" * 64,
        active_texture_sha256="1" * 64,
        restored_texture_sha256="f" * 64,
        arrays_npz_path="arrays/state_00_probe_r.npz",
        arrays_npz_sha256="2" * 64,
        artifact_sha256=artifact_sha256,
    )


def _provenance_with_npz_path(path: str) -> RendererBakeResponseProvenance:
    provenance = _provenance()
    artifacts = {
        artifact_path: fingerprint
        for artifact_path, fingerprint in provenance.artifact_sha256.items()
        if not artifact_path.endswith(".npz")
    }
    artifacts[path] = provenance.arrays_npz_sha256
    return replace(
        provenance,
        arrays_npz_path=path,
        artifact_sha256=artifacts,
    )


def test_identical_response_has_unit_cosine_and_passes_row_gate() -> None:
    evidence = _evidence()
    metrics = compute_weighted_response_metrics(
        evidence.alpha,
        evidence.d_sur,
        evidence.d_bake,
    )
    row = evaluate_renderer_bake_response_evidence(evidence, _provenance())

    assert np.isclose(metrics.cosine, 1.0)
    assert metrics.relative_l2 == 0.0
    assert metrics.sign_consistency == 1.0
    assert row["status"] == "valid"
    assert row["gate_pass"]


def test_opposite_response_is_sufficient_but_fails_direction_gate() -> None:
    evidence = _evidence()
    opposite = replace(evidence, d_bake=-evidence.d_bake)
    row = evaluate_renderer_bake_response_evidence(opposite, _provenance())

    assert np.isclose(row["cos_alpha"], -1.0)
    assert row["status"] == "valid"
    assert not row["gate_pass"]
    assert "cos_alpha_not_positive" in row["failures"]


def test_sim_state_mismatch_invalidates_otherwise_valid_evidence() -> None:
    provenance = replace(_provenance(), bake_sim_state_sha256="5" * 64)

    row = evaluate_renderer_bake_response_evidence(_evidence(), provenance)

    assert row["status"] == "invalid_evidence"
    assert not row["gate_pass"]
    assert "clean_bake_sim_state_mismatch" in row["failures"]


def test_insufficient_response_does_not_report_unstable_cosine() -> None:
    evidence = _evidence()
    tiny = replace(evidence, d_sur=evidence.d_sur * 1e-8)
    row = evaluate_renderer_bake_response_evidence(tiny, _provenance())

    assert row["status"] == "insufficient_probe_response"
    assert row["cos_alpha"] is None
    assert row["rel_l2_alpha"] is None
    assert not row["gate_pass"]


def test_non_valid_visibility_keeps_explicit_row_without_unstable_metrics() -> None:
    evidence = replace(
        _evidence(),
        alpha=np.zeros((2, 2), dtype=np.float32),
        visibility_status="not_observable",
    )

    row = evaluate_renderer_bake_response_evidence(evidence, _provenance())

    assert row["status"] == "not_observable"
    assert row["surrogate_rms"] is None
    assert row["bake_rms"] is None
    assert row["cos_alpha"] is None
    assert row["failures"] == ["visibility_status=not_observable"]
    assert not row["gate_pass"]


def test_alpha_weighting_ignores_background_and_sign_double_zeros() -> None:
    alpha = np.asarray([[1.0, 0.0]], dtype=np.float32)
    d_sur = np.asarray([[[1.0, 0.0, 0.0], [1e6, 1e6, 1e6]]])
    d_bake = np.asarray([[[1.0, 2.0, 0.0], [-1e6, -1e6, -1e6]]])
    metrics = compute_weighted_response_metrics(alpha, d_sur, d_bake)

    # 可见像素 R 同号，G 只有 bake 非零，B 双零被排除：分母恰为两个分量。
    assert metrics.sign_denominator_component_count == 2
    assert metrics.sign_denominator_weight == 2.0
    assert metrics.sign_consistent_weight == 1.0
    assert metrics.sign_consistency == 0.5
    assert np.isclose(metrics.surrogate_rms, np.sqrt(1.0 / 3.0))


def test_clean_action_margin_uses_clean_class_against_best_other() -> None:
    logits = np.asarray([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]])
    clean_classes = np.asarray([1, 2], dtype=np.int64)

    margins = compute_untargeted_clean_action_margins(logits, clean_classes)

    assert margins.tolist() == [2.0, -1.0]


def test_summary_requires_complete_unique_state_probe_grid() -> None:
    rows = []
    for state_id in (0, 1):
        for channel in PROBE_CHANNELS:
            provenance = _provenance_with_npz_path(
                f"arrays/state_{state_id:02d}_{channel}.npz"
            )
            rows.append(
                evaluate_renderer_bake_response_evidence(
                    _evidence(state_id, channel),
                    provenance,
                )
            )
    summary = summarize_renderer_bake_response_rows(
        rows,
        expected_state_ids=(0, 1),
    )
    assert summary["structural_pass"]
    assert summary["gate_pass"]
    assert summary["per_probe"]["r"]["minimum_cosine_state_ids"] == [0, 1]

    incomplete = summarize_renderer_bake_response_rows(
        [*rows[:-1], rows[0]],
        expected_state_ids=(0, 1),
    )
    assert not incomplete["structural_pass"]
    assert incomplete["missing_keys"] == [[1, "b"]]
    assert incomplete["duplicate_keys"] == [[0, "r"]]

    altered_row = dict(rows[0])
    altered_row["probe_surface_delta"] = 1.0 / 255.0
    altered = summarize_renderer_bake_response_rows(
        [altered_row, *rows[1:]],  # type: ignore[list-item]
        expected_state_ids=(0, 1),
    )
    assert not altered["structural_pass"]
    assert "mixed_or_invalid_numeric_contract" in altered[
        "structural_failures"
    ]


def test_npz_and_table_writers_are_no_pickle_and_deterministic(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    npz_path = tmp_path / "arrays" / "state_00_r.npz"
    npz_sha = write_renderer_bake_response_npz(evidence, output_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive["alpha"].dtype == np.float32
        assert archive["d_sur"].shape == (2, 2, 3)
        assert archive["alpha_dtype"].item() == "float32"

    provenance = replace(
        _provenance(),
        arrays_npz_path="arrays/state_00_r.npz",
        arrays_npz_sha256=npz_sha,
        artifact_sha256={
            **{
                path: fingerprint
                for path, fingerprint in _provenance().artifact_sha256.items()
                if path.endswith(".png")
            },
            "arrays/state_00_r.npz": npz_sha,
        },
    )
    row = evaluate_renderer_bake_response_evidence(evidence, provenance)
    summary = summarize_renderer_bake_response_rows(
        [
            evaluate_renderer_bake_response_evidence(
                _evidence(0, channel),
                replace(
                    _provenance_with_npz_path(
                        f"arrays/state_00_{channel}.npz"
                    ),
                    arrays_npz_sha256=npz_sha,
                    artifact_sha256={
                        **{
                            path: fingerprint
                            for path, fingerprint in provenance.artifact_sha256.items()
                            if path.endswith(".png")
                        },
                        f"arrays/state_00_{channel}.npz": npz_sha,
                    },
                ),
            )
            for channel in PROBE_CHANNELS
        ],
        expected_state_ids=(0,),
    )
    jsonl_path = tmp_path / "response_metrics.jsonl"
    csv_path = tmp_path / "response_metrics.csv"
    jsonl_sha = write_renderer_bake_response_jsonl(
        [row], output_path=jsonl_path
    )
    csv_sha = write_renderer_bake_response_csv([row], output_path=csv_path)
    manifest_path = tmp_path / "response_manifest.json"
    write_renderer_bake_response_manifest(
        summary=summary,
        metadata={"code_commit": "a" * 40},
        metrics_jsonl_sha256=jsonl_sha,
        derived_csv_sha256=csv_sha,
        output_path=manifest_path,
    )

    parsed = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert parsed["state_id"] == 0
    assert csv_path.read_text(encoding="utf-8").splitlines()[1].startswith(
        "0,r,valid,"
    )
    assert jsonl_sha in manifest_path.read_text(encoding="utf-8")
