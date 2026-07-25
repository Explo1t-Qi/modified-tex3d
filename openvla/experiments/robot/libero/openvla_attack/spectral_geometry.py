"""曲面谱基的生成、持久化与几何一致性校验。

本模块只处理 CPU 上的 mesh 数值计算，不依赖 OpenVLA、LIBERO 或
nvdiffrast。谱基定义在 OBJ 的“几何顶点”上，而渲染器可能因为 UV seam
复制顶点，因此这里同时提供严格的拓扑映射：

``geometry_delta [N, 3] -> render_delta [V, 3]``

其中 ``N`` 是 OBJ 几何顶点数，``V`` 是渲染 mesh 顶点数。映射完全由对齐的
三角形角点建立；不会使用最近邻近似，以免把相邻但不相同的曲面位置混淆。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh


PathLike: TypeAlias = str | Path
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class SpectralGeometryError(ValueError):
    """谱基文件、mesh 拓扑或数值结果不满足约束。"""


def load_obj_geometry(obj_path: PathLike) -> tuple[FloatArray, IntArray]:
    """读取 OBJ 的几何顶点与三角形几何索引。

    UV/法线索引在此处有意忽略。函数支持正、负 OBJ 索引，并按文件顺序对
    多边形做扇形三角化。

    Returns:
        ``vertices`` 为 float64 ``[num_geometry_vertices, 3]``；
        ``faces`` 为 int64 ``[num_faces, 3]``。
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    resolved_path: Path = Path(obj_path)

    with resolved_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line_number, raw_line in enumerate(obj_file, start=1):
            line: str = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields: list[str] = line.split()
            if fields[0] == "v":
                if len(fields) < 4:
                    raise SpectralGeometryError(
                        f"OBJ 顶点不完整: {resolved_path}:{line_number}: {line}"
                    )
                vertices.append(
                    [float(fields[1]), float(fields[2]), float(fields[3])]
                )
            elif fields[0] == "f":
                polygon: list[int] = []
                for token in fields[1:]:
                    vertex_token: str = token.split("/", maxsplit=1)[0]
                    if not vertex_token:
                        raise SpectralGeometryError(
                            f"OBJ face 索引无效: "
                            f"{resolved_path}:{line_number}: {line}"
                        )
                    obj_index: int = int(vertex_token)
                    polygon.append(
                        obj_index - 1
                        if obj_index > 0
                        else len(vertices) + obj_index
                    )
                if len(polygon) < 3:
                    raise SpectralGeometryError(
                        f"OBJ face 少于三个顶点: "
                        f"{resolved_path}:{line_number}: {line}"
                    )
                for corner_index in range(1, len(polygon) - 1):
                    faces.append(
                        [
                            polygon[0],
                            polygon[corner_index],
                            polygon[corner_index + 1],
                        ]
                    )

    vertex_array: FloatArray = np.asarray(vertices, dtype=np.float64)
    face_array: IntArray = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise SpectralGeometryError(f"{resolved_path} 中没有有效三维顶点")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise SpectralGeometryError(f"{resolved_path} 中没有有效三角形")
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise SpectralGeometryError(f"{resolved_path} 中存在越界 face 索引")
    return vertex_array, face_array


def mesh_array_sha256(vertices: FloatArray, faces: IntArray) -> str:
    """计算与平台字节序无关的 mesh 数组哈希。"""
    digest = hashlib.sha256()
    canonical_vertices: FloatArray = np.ascontiguousarray(vertices, dtype="<f8")
    canonical_faces: IntArray = np.ascontiguousarray(faces, dtype="<i8")
    digest.update(np.asarray(canonical_vertices.shape, dtype="<i8").tobytes())
    digest.update(canonical_vertices.tobytes())
    digest.update(np.asarray(canonical_faces.shape, dtype="<i8").tobytes())
    digest.update(canonical_faces.tobytes())
    return digest.hexdigest()


def build_cotangent_laplacian_and_mass(
    vertices: FloatArray,
    faces: IntArray,
    *,
    area_epsilon: float = 1e-14,
) -> tuple[sparse.csr_matrix, FloatArray, IntArray]:
    """构造 cotangent stiffness matrix 与 barycentric lumped mass。

    Returns:
        ``laplacian`` 为稀疏 float64 ``[N, N]``；
        ``mass`` 为 float64 ``[N]``；
        ``valid_faces`` 为过滤退化三角形后的 int64 ``[F_valid, 3]``。
    """
    vertex_array: FloatArray = np.asarray(vertices, dtype=np.float64)
    face_array: IntArray = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise SpectralGeometryError("vertices 必须为 [N, 3]")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise SpectralGeometryError("faces 必须为 [F, 3]")

    vertex_i: IntArray
    vertex_j: IntArray
    vertex_k: IntArray
    vertex_i, vertex_j, vertex_k = face_array.T
    position_i: FloatArray = vertex_array[vertex_i]
    position_j: FloatArray = vertex_array[vertex_j]
    position_k: FloatArray = vertex_array[vertex_k]
    twice_area: FloatArray = np.linalg.norm(
        np.cross(position_j - position_i, position_k - position_i),
        axis=1,
    )
    valid_mask: NDArray[np.bool_] = twice_area > area_epsilon
    valid_faces: IntArray = face_array[valid_mask]
    if len(valid_faces) == 0:
        raise SpectralGeometryError("mesh 不包含非退化三角形")

    vertex_i, vertex_j, vertex_k = valid_faces.T
    position_i = vertex_array[vertex_i]
    position_j = vertex_array[vertex_j]
    position_k = vertex_array[vertex_k]
    twice_area = twice_area[valid_mask]

    def rowwise_cotangent(
        first: FloatArray,
        second: FloatArray,
    ) -> FloatArray:
        numerator: FloatArray = np.einsum("ij,ij->i", first, second)
        denominator: FloatArray = np.linalg.norm(
            np.cross(first, second),
            axis=1,
        )
        return numerator / np.maximum(denominator, area_epsilon)

    cotangent_i: FloatArray = rowwise_cotangent(
        position_j - position_i,
        position_k - position_i,
    )
    cotangent_j: FloatArray = rowwise_cotangent(
        position_k - position_j,
        position_i - position_j,
    )
    cotangent_k: FloatArray = rowwise_cotangent(
        position_i - position_k,
        position_j - position_k,
    )

    rows: IntArray = np.concatenate(
        [vertex_j, vertex_k, vertex_k, vertex_i, vertex_i, vertex_j]
    )
    columns: IntArray = np.concatenate(
        [vertex_k, vertex_j, vertex_i, vertex_k, vertex_j, vertex_i]
    )
    weights: FloatArray = np.concatenate(
        [
            0.5 * cotangent_i,
            0.5 * cotangent_i,
            0.5 * cotangent_j,
            0.5 * cotangent_j,
            0.5 * cotangent_k,
            0.5 * cotangent_k,
        ]
    )
    weight_matrix: sparse.csr_matrix = sparse.coo_matrix(
        (weights, (rows, columns)),
        shape=(len(vertex_array), len(vertex_array)),
    ).tocsr()
    weight_matrix = 0.5 * (weight_matrix + weight_matrix.T)
    degree: FloatArray = np.asarray(weight_matrix.sum(axis=1)).ravel()
    laplacian: sparse.csr_matrix = (
        sparse.diags(degree, format="csr") - weight_matrix
    )

    mass: FloatArray = np.zeros(len(vertex_array), dtype=np.float64)
    triangle_vertex_mass: FloatArray = (0.5 * twice_area) / 3.0
    np.add.at(mass, vertex_i, triangle_vertex_mass)
    np.add.at(mass, vertex_j, triangle_vertex_mass)
    np.add.at(mass, vertex_k, triangle_vertex_mass)
    if np.any(mass <= 0.0):
        bad_vertex_ids: IntArray = np.flatnonzero(mass <= 0.0)
        raise SpectralGeometryError(
            "mesh 包含孤立或零质量顶点，前十个 ID: "
            f"{bad_vertex_ids[:10].tolist()}"
        )
    return laplacian.tocsr(), mass, valid_faces


def solve_manifold_harmonics(
    laplacian: sparse.csr_matrix,
    mass: FloatArray,
    number_of_nonconstant_basis: int,
    *,
    tolerance: float = 1e-9,
) -> tuple[FloatArray, FloatArray]:
    """求解 ``L φ = λ M φ``，返回常数模态加 K 个非恒定模态。

    返回的 ``basis`` 为 float64 ``[N, K + 1]``，第 0 列是常数模态。每列按
    lumped mass 做 M-归一化，并固定符号以保证重复生成时更稳定。
    """
    requested_count: int = number_of_nonconstant_basis + 1
    if number_of_nonconstant_basis <= 0:
        raise SpectralGeometryError("非恒定谱基数量必须为正数")
    if requested_count >= laplacian.shape[0]:
        raise SpectralGeometryError(
            f"{laplacian.shape[0]} 个顶点无法求解 {requested_count} 个模态"
        )
    if mass.shape != (laplacian.shape[0],) or np.any(mass <= 0):
        raise SpectralGeometryError("mass 必须为与 Laplacian 匹配的正向量")

    mass_matrix: sparse.csr_matrix = sparse.diags(mass, format="csr")
    eigenvalues: FloatArray
    basis: FloatArray
    eigenvalues, basis = eigsh(
        laplacian,
        k=requested_count,
        M=mass_matrix,
        sigma=1e-8,
        which="LM",
        tol=tolerance,
        maxiter=max(10000, 20 * laplacian.shape[0]),
        v0=np.random.default_rng(0).standard_normal(laplacian.shape[0]),
    )
    order: IntArray = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]

    for column_index in range(basis.shape[1]):
        mass_norm: float = float(
            np.sqrt(np.sum(mass * basis[:, column_index] ** 2))
        )
        basis[:, column_index] /= mass_norm
        maximum_index: int = int(
            np.argmax(np.abs(basis[:, column_index]))
        )
        if basis[maximum_index, column_index] < 0:
            basis[:, column_index] *= -1.0

    # shift-invert 返回的向量通常比 λ 更精确，因此用广义 Rayleigh quotient
    # 重新计算特征值，随后数值残差才具有可解释性。
    eigenvalues = np.asarray(
        [
            float(
                phi @ (laplacian @ phi)
                / np.sum(mass * phi**2)
            )
            for phi in basis.T
        ],
        dtype=np.float64,
    )
    return eigenvalues, basis


def spectral_numerical_errors(
    laplacian: sparse.csr_matrix,
    mass: FloatArray,
    eigenvalues: FloatArray,
    basis: FloatArray,
) -> tuple[float, FloatArray]:
    """返回最大 M-正交误差及逐模态特征方程残差。"""
    gram: FloatArray = basis.T @ (mass[:, None] * basis)
    orthogonality_error: float = float(
        np.max(np.abs(gram - np.eye(gram.shape[0])))
    )
    residuals: list[float] = []
    for eigenvalue, phi in zip(eigenvalues, basis.T):
        left: FloatArray = laplacian @ phi
        right: FloatArray = eigenvalue * mass * phi
        absolute_error: float = float(np.linalg.norm(left - right))
        if abs(eigenvalue) < 1e-10:
            residuals.append(absolute_error)
        else:
            denominator: float = max(
                float(np.linalg.norm(left)),
                float(np.linalg.norm(right)),
                1e-15,
            )
            residuals.append(absolute_error / denominator)
    return orthogonality_error, np.asarray(residuals, dtype=np.float64)


@dataclass(frozen=True)
class SpectralBasisData:
    """通过校验的非恒定曲面谱基及其源几何。"""

    vertices: FloatArray  # [N, 3]
    faces: IntArray  # [F, 3]
    mass: FloatArray  # [N]
    eigenvalues: FloatArray  # [K]
    basis: FloatArray  # [N, K]
    metadata: dict[str, Any]

    @property
    def num_geometry_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.basis.shape[1])

    @property
    def mesh_sha256(self) -> str:
        return mesh_array_sha256(self.vertices, self.faces)


def save_spectral_basis(
    output_path: PathLike,
    *,
    vertices: FloatArray,
    faces: IntArray,
    mass: FloatArray,
    eigenvalues_with_constant: FloatArray,
    basis_with_constant: FloatArray,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """保存稳定的 NPZ 谱基产物。

    文件同时保留含常数模态的完整数组和不含常数模态的攻击数组。MVP 默认只
    加载 ``nonconstant_basis``，从参数空间中排除全局 RGB 偏移。
    """
    resolved_path: Path = Path(output_path)
    if resolved_path.suffix.lower() != ".npz":
        raise SpectralGeometryError("谱基产物必须使用 .npz 后缀")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_metadata: dict[str, Any] = dict(metadata or {})
    resolved_metadata.update(
        {
            "format_version": 1,
            "mesh_array_sha256": mesh_array_sha256(vertices, faces),
            "num_geometry_vertices": int(vertices.shape[0]),
            "num_faces": int(faces.shape[0]),
            "num_nonconstant_basis": int(basis_with_constant.shape[1] - 1),
        }
    )
    np.savez_compressed(
        resolved_path,
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        mass=np.asarray(mass, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues_with_constant, dtype=np.float64),
        basis_with_constant=np.asarray(
            basis_with_constant,
            dtype=np.float64,
        ),
        nonconstant_basis=np.asarray(
            basis_with_constant[:, 1:],
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(resolved_metadata, ensure_ascii=False, sort_keys=True)
        ),
    )
    return resolved_path


def load_spectral_basis(
    basis_path: PathLike,
    *,
    max_basis: Optional[int] = None,
    include_constant: bool = False,
) -> SpectralBasisData:
    """加载 NPZ 谱基，并进行 shape、有限值和元数据校验。"""
    resolved_path: Path = Path(basis_path)
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)

    with np.load(resolved_path, allow_pickle=False) as archive:
        payload: dict[str, NDArray[Any]] = {
            key: archive[key] for key in archive.files
        }

    required_keys: set[str] = {"vertices", "faces", "mass", "eigenvalues"}
    missing_keys: list[str] = sorted(required_keys.difference(payload))
    if missing_keys:
        raise SpectralGeometryError(
            f"{resolved_path} 缺少数组: {missing_keys}"
        )

    basis_key: str = (
        "basis_with_constant" if include_constant else "nonconstant_basis"
    )
    eigenvalue_offset: int = 0 if include_constant else 1
    if basis_key not in payload:
        raise SpectralGeometryError(
            f"{resolved_path} 缺少 {basis_key!r}"
        )

    vertices: FloatArray = np.asarray(payload["vertices"], dtype=np.float64)
    faces: IntArray = np.asarray(payload["faces"], dtype=np.int64)
    mass: FloatArray = np.asarray(payload["mass"], dtype=np.float64)
    all_eigenvalues: FloatArray = np.asarray(
        payload["eigenvalues"],
        dtype=np.float64,
    )
    basis: FloatArray = np.asarray(payload[basis_key], dtype=np.float64)

    if max_basis is not None:
        if max_basis <= 0:
            raise SpectralGeometryError("max_basis 必须为正数")
        if max_basis > basis.shape[1]:
            raise SpectralGeometryError(
                f"请求 {max_basis} 个模态，但产物只有 {basis.shape[1]} 个"
            )
        basis = basis[:, :max_basis]
    eigenvalues: FloatArray = all_eigenvalues[
        eigenvalue_offset : eigenvalue_offset + basis.shape[1]
    ]

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise SpectralGeometryError(
            f"vertices 应为 [N, 3]，实际为 {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise SpectralGeometryError(
            f"faces 应为 [F, 3]，实际为 {faces.shape}"
        )
    if basis.ndim != 2 or basis.shape[0] != vertices.shape[0]:
        raise SpectralGeometryError(
            f"basis 应为 [{len(vertices)}, K]，实际为 {basis.shape}"
        )
    if mass.shape != (len(vertices),) or np.any(mass <= 0):
        raise SpectralGeometryError(
            f"mass 应为正向量 [{len(vertices)}]"
        )
    if eigenvalues.shape != (basis.shape[1],):
        raise SpectralGeometryError("特征值数量与谱基列数不一致")
    finite_arrays: tuple[FloatArray, ...] = (
        vertices,
        mass,
        eigenvalues,
        basis,
    )
    if not all(np.isfinite(array).all() for array in finite_arrays):
        raise SpectralGeometryError(f"{resolved_path} 包含非有限值")

    loaded_metadata: dict[str, Any] = {
        "artifact_path": str(resolved_path.resolve()),
        "include_constant": include_constant,
        "mesh_array_sha256": mesh_array_sha256(vertices, faces),
    }
    if "metadata_json" in payload:
        try:
            parsed_metadata: Any = json.loads(
                str(payload["metadata_json"].item())
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise SpectralGeometryError(
                f"{resolved_path} 的 metadata_json 无效: {error}"
            ) from error
        if not isinstance(parsed_metadata, dict):
            raise SpectralGeometryError("metadata_json 必须编码 JSON object")
        loaded_metadata.update(parsed_metadata)

    expected_hash: Any = loaded_metadata.get("mesh_array_sha256")
    actual_hash: str = mesh_array_sha256(vertices, faces)
    if expected_hash != actual_hash:
        raise SpectralGeometryError(
            "谱基元数据中的 mesh hash 与数组内容不一致"
        )

    return SpectralBasisData(
        vertices=np.ascontiguousarray(vertices),
        faces=np.ascontiguousarray(faces),
        mass=np.ascontiguousarray(mass),
        eigenvalues=np.ascontiguousarray(eigenvalues),
        basis=np.ascontiguousarray(basis),
        metadata=loaded_metadata,
    )


def validate_basis_geometry(
    basis_data: SpectralBasisData,
    geometry_vertices: FloatArray,
    geometry_faces: IntArray,
) -> None:
    """要求谱基 mesh 与当前原始 OBJ 几何逐元素相同。"""
    vertices: FloatArray = np.asarray(geometry_vertices, dtype=np.float64)
    faces: IntArray = np.asarray(geometry_faces, dtype=np.int64)
    if basis_data.vertices.shape != vertices.shape:
        raise SpectralGeometryError(
            "谱基/OBJ 顶点数量不一致: "
            f"{basis_data.vertices.shape} vs {vertices.shape}"
        )
    if basis_data.faces.shape != faces.shape:
        raise SpectralGeometryError(
            "谱基/OBJ face 数量不一致: "
            f"{basis_data.faces.shape} vs {faces.shape}"
        )
    if not np.array_equal(basis_data.vertices, vertices):
        maximum_error: float = float(
            np.max(np.abs(basis_data.vertices - vertices))
        )
        raise SpectralGeometryError(
            "谱基顶点与当前 OBJ 不完全一致，"
            f"最大绝对误差为 {maximum_error:.3e}"
        )
    if not np.array_equal(basis_data.faces, faces):
        mismatch_count: int = int(
            np.count_nonzero(basis_data.faces != faces)
        )
        raise SpectralGeometryError(
            f"谱基 face 与当前 OBJ 不一致，共 {mismatch_count} 个索引不同"
        )


def build_render_to_geometry_map(
    geometry_vertices: FloatArray,
    geometry_faces: IntArray,
    render_vertices: FloatArray,
    render_faces: IntArray,
) -> IntArray:
    """将 UV seam 复制的渲染顶点严格映射回 OBJ 几何顶点。

    当前 trimesh 路径保持三角形及角点顺序，因此每个对齐角点都给出
    ``render_vertex -> geometry_vertex`` 的确定映射。坐标变化、face 重排、
    冲突映射和未使用的渲染顶点均直接拒绝。
    """
    geometry_vertex_array: FloatArray = np.asarray(
        geometry_vertices,
        dtype=np.float64,
    )
    geometry_face_array: IntArray = np.asarray(
        geometry_faces,
        dtype=np.int64,
    )
    render_vertex_array: FloatArray = np.asarray(
        render_vertices,
        dtype=np.float64,
    )
    render_face_array: IntArray = np.asarray(render_faces, dtype=np.int64)

    if geometry_face_array.shape != render_face_array.shape:
        raise SpectralGeometryError(
            "OBJ 与 renderer 的三角形数组不对齐: "
            f"{geometry_face_array.shape} vs {render_face_array.shape}"
        )
    if geometry_face_array.ndim != 2 or geometry_face_array.shape[1] != 3:
        raise SpectralGeometryError("仅支持对齐的三角 mesh")
    if (
        geometry_face_array.min() < 0
        or geometry_face_array.max() >= len(geometry_vertex_array)
    ):
        raise SpectralGeometryError("OBJ face 索引越界")
    if (
        render_face_array.min() < 0
        or render_face_array.max() >= len(render_vertex_array)
    ):
        raise SpectralGeometryError("renderer face 索引越界")

    geometry_corner_positions: FloatArray = geometry_vertex_array[
        geometry_face_array
    ]
    render_corner_positions: FloatArray = render_vertex_array[
        render_face_array
    ]
    if not np.array_equal(
        geometry_corner_positions,
        render_corner_positions,
    ):
        first_mismatch: IntArray = np.argwhere(
            geometry_corner_positions != render_corner_positions
        )[0]
        face_index, corner_index, axis = (
            int(value) for value in first_mismatch
        )
        raise SpectralGeometryError(
            "renderer 改变了 face/角点顺序或顶点坐标，无法建立严格映射；"
            f"首个差异: face={face_index}, corner={corner_index}, axis={axis}"
        )

    mapping: IntArray = np.full(
        len(render_vertex_array),
        -1,
        dtype=np.int64,
    )
    render_ids: IntArray = render_face_array.reshape(-1)
    geometry_ids: IntArray = geometry_face_array.reshape(-1)
    for render_id, geometry_id in zip(render_ids, geometry_ids):
        previous_geometry_id: int = int(mapping[render_id])
        if previous_geometry_id >= 0 and previous_geometry_id != geometry_id:
            raise SpectralGeometryError(
                f"渲染顶点 {render_id} 同时映射到几何顶点 "
                f"{previous_geometry_id} 与 {geometry_id}"
            )
        mapping[render_id] = geometry_id

    unmapped_ids: IntArray = np.flatnonzero(mapping < 0)
    if len(unmapped_ids):
        raise SpectralGeometryError(
            f"{len(unmapped_ids)} 个渲染顶点未被 face 使用，前十个 ID: "
            f"{unmapped_ids[:10].tolist()}"
        )
    if not np.array_equal(
        render_vertex_array,
        geometry_vertex_array[mapping],
    ):
        raise SpectralGeometryError("渲染顶点到几何顶点的最终坐标校验失败")
    return mapping
