"""曲面谱基生成与拓扑映射的 CPU 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    SpectralGeometryError,
    build_cotangent_laplacian_and_mass,
    build_render_to_geometry_map,
    load_obj_geometry,
    load_spectral_basis,
    save_spectral_basis,
    solve_manifold_harmonics,
    spectral_numerical_errors,
    validate_basis_geometry,
)


def test_render_mapping_shares_uv_seam_vertices() -> None:
    geometry_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    geometry_faces = np.asarray(
        [[0, 1, 2], [0, 2, 3]],
        dtype=np.int64,
    )
    render_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    render_faces = np.asarray(
        [[0, 1, 2], [3, 4, 5]],
        dtype=np.int64,
    )

    mapping = build_render_to_geometry_map(
        geometry_vertices,
        geometry_faces,
        render_vertices,
        render_faces,
    )

    np.testing.assert_array_equal(mapping, [0, 1, 2, 0, 2, 3])


def test_render_mapping_rejects_reordered_faces() -> None:
    geometry_vertices = np.eye(3, dtype=np.float64)
    geometry_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(SpectralGeometryError, match="无法建立严格映射"):
        build_render_to_geometry_map(
            geometry_vertices,
            geometry_faces,
            geometry_vertices,
            geometry_faces[:, ::-1],
        )


def test_obj_loader_uses_geometry_indices_across_uv_seams(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "seam.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "vt 0.5 0.5",
                "f 1/1 2/2 3/3",
                "f 1/4 3/3 2/2",
            ]
        ),
        encoding="utf-8",
    )

    vertices, faces = load_obj_geometry(obj_path)

    assert vertices.shape == (3, 3)
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 1]])


def test_generated_modes_are_mass_orthonormal_eigenfunctions() -> None:
    vertices = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    laplacian, mass, valid_faces = (
        build_cotangent_laplacian_and_mass(vertices, faces)
    )
    eigenvalues, basis = solve_manifold_harmonics(
        laplacian,
        mass,
        number_of_nonconstant_basis=2,
    )
    orthogonality_error, residuals = spectral_numerical_errors(
        laplacian,
        mass,
        eigenvalues,
        basis,
    )

    np.testing.assert_array_equal(valid_faces, faces)
    assert abs(eigenvalues[0]) < 1e-12
    assert orthogonality_error < 1e-12
    assert float(residuals.max()) < 1e-10


def test_basis_round_trip_excludes_constant_and_validates_geometry(
    tmp_path: Path,
) -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mass = np.ones(3, dtype=np.float64)
    basis_with_constant = np.asarray(
        [[1.0, 1.0], [1.0, 0.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    artifact_path = save_spectral_basis(
        tmp_path / "spectral_basis.npz",
        vertices=vertices,
        faces=faces,
        mass=mass,
        eigenvalues_with_constant=np.asarray([0.0, 1.0]),
        basis_with_constant=basis_with_constant,
        metadata={"source": "unit-test"},
    )

    loaded = load_spectral_basis(artifact_path, max_basis=1)

    assert loaded.basis.shape == (3, 1)
    assert loaded.metadata["source"] == "unit-test"
    np.testing.assert_array_equal(loaded.eigenvalues, [1.0])
    validate_basis_geometry(loaded, vertices, faces)

    changed_vertices = vertices.copy()
    changed_vertices[0, 0] = 1e-12
    with pytest.raises(SpectralGeometryError, match="不完全一致"):
        validate_basis_geometry(loaded, changed_vertices, faces)
