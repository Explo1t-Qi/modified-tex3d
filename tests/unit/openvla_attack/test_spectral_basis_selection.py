"""source-only 梯度选基产物的 CPU 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openvla.experiments.robot.libero.openvla_attack.spectral_basis_selection import (
    SpectralBasisSelectionError,
    build_gradient_selected_spectral_basis,
)
from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    load_spectral_basis,
    save_spectral_basis,
)


def _write_candidate_basis(path: Path) -> Path:
    """写入含一个常数列和两个非恒定列的正交 toy basis。"""
    return save_spectral_basis(
        path,
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        mass=np.ones(3, dtype=np.float64),
        eigenvalues_with_constant=np.asarray(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
        basis_with_constant=np.eye(3, dtype=np.float64),
        metadata={"source": "selection-unit-test"},
    )


def _write_audit(
    path: Path,
    *,
    state_ids: np.ndarray,
    eigenvalues: np.ndarray | None = None,
    ranked_indices: np.ndarray | None = None,
) -> Path:
    """写入选基所需的最小完整审计数组。"""
    number_of_samples: int = int(state_ids.size)
    np.savez_compressed(
        path,
        state_ids=state_ids,
        eigenvalues=(
            np.asarray([1.0, 2.0], dtype=np.float64)
            if eigenvalues is None
            else eigenvalues
        ),
        basis_linf=np.ones(2, dtype=np.float64),
        feature_gradients=np.ones(
            (number_of_samples, 2, 3),
            dtype=np.float64,
        ),
        feature_ranked_indices=(
            np.asarray([1, 0], dtype=np.int64)
            if ranked_indices is None
            else ranked_indices
        ),
        feature_stable_score=np.asarray(
            [0.25, 0.75],
            dtype=np.float64,
        ),
    )
    return path


def test_build_selected_basis_preserves_rank_order_and_provenance(
    tmp_path: Path,
) -> None:
    candidate_path: Path = _write_candidate_basis(
        tmp_path / "candidate.npz"
    )
    audit_path: Path = _write_audit(
        tmp_path / "audit.npz",
        state_ids=np.asarray([0, 1], dtype=np.int64),
    )

    result = build_gradient_selected_spectral_basis(
        candidate_basis_path=candidate_path,
        audit_path=audit_path,
        output_path=tmp_path / "selected.npz",
        objective="feature",
        top_k=1,
        expected_state_ids=(0, 1),
    )
    loaded = load_spectral_basis(result.output_path, max_basis=1)

    np.testing.assert_array_equal(
        result.selected_source_mode_indices,
        [1],
    )
    np.testing.assert_array_equal(result.selected_eigenvalues, [2.0])
    np.testing.assert_array_equal(
        loaded.basis,
        np.asarray([[0.0], [0.0], [1.0]]),
    )
    np.testing.assert_array_equal(loaded.eigenvalues, [2.0])
    assert loaded.metadata["selection_scope"] == "source_openvla_only"
    assert loaded.metadata["selection_method"] == (
        "feature_surface_stable_score_descending"
    )
    assert loaded.metadata["selected_source_mode_indices"] == [1]
    assert loaded.metadata["source_state_ids"] == [0, 1]
    assert result.mass_orthogonality_error == pytest.approx(0.0)


def test_selection_rejects_unexpected_source_states(
    tmp_path: Path,
) -> None:
    candidate_path: Path = _write_candidate_basis(
        tmp_path / "candidate.npz"
    )
    audit_path: Path = _write_audit(
        tmp_path / "audit.npz",
        state_ids=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(
        SpectralBasisSelectionError,
        match="expected_state_ids",
    ):
        build_gradient_selected_spectral_basis(
            candidate_basis_path=candidate_path,
            audit_path=audit_path,
            output_path=tmp_path / "selected.npz",
            objective="feature",
            top_k=1,
            expected_state_ids=(0, 1),
        )


@pytest.mark.parametrize(
    ("eigenvalues", "ranked_indices", "error_match"),
    [
        (
            np.asarray([1.0, 3.0], dtype=np.float64),
            None,
            "eigenvalues",
        ),
        (
            None,
            np.asarray([1, 1], dtype=np.int64),
            "permutation",
        ),
    ],
)
def test_selection_rejects_audit_from_different_candidate(
    tmp_path: Path,
    eigenvalues: np.ndarray | None,
    ranked_indices: np.ndarray | None,
    error_match: str,
) -> None:
    candidate_path: Path = _write_candidate_basis(
        tmp_path / "candidate.npz"
    )
    audit_path: Path = _write_audit(
        tmp_path / "audit.npz",
        state_ids=np.asarray([0, 1], dtype=np.int64),
        eigenvalues=eigenvalues,
        ranked_indices=ranked_indices,
    )

    with pytest.raises(
        SpectralBasisSelectionError,
        match=error_match,
    ):
        build_gradient_selected_spectral_basis(
            candidate_basis_path=candidate_path,
            audit_path=audit_path,
            output_path=tmp_path / "selected.npz",
            objective="feature",
            top_k=1,
            expected_state_ids=(0, 1),
        )
