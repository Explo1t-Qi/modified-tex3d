"""从 Visibility Alignment NPZ 重放逐顶点投影 coverage contribution。

Coverage 对 support membership 是线性的，但 source→effective 变换必须作用在
``alpha * w_S`` 上。为了无需为每个几何顶点单独变换一张 512 图，本模块对已
冻结的 area-downsample + center-crop 线性算子计算一次 source-space 求和伴随
权重，再把 ``alpha * beta`` 严格 scatter 到原始 OBJ 顶点。零/全 support、逐
实例求和和保存的 effective alpha 都会独立重放校验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Mapping, TypeAlias

import numpy as np
import torch
import torch.nn.functional as torch_functional
from numpy.typing import NDArray

from experiments.robot.openvla_image_transform import torch_center_crop_float

from .coverage_evidence import (
    COVERAGE_INVARIANT_ATOL,
    CoverageEvidenceSpecification,
    transform_premultiplied_evidence,
)
from .support_construction import ProjectionCoverageEvidence
from .visibility_evidence import VisibilityEvidenceStatus


FloatArray: TypeAlias = NDArray[np.float64]
SUPPORT_COVERAGE_CONTRIBUTION_SCHEMA_VERSION: Final[str] = (
    "openvla-support-coverage-contribution-v1"
)
VISIBILITY_ALIGNMENT_SCHEMA_VERSION: Final[str] = (
    "openvla-visibility-alignment-v1"
)
ADJOINT_SUM_ATOL: Final[float] = 2e-3


class SupportCoverageContributionError(ValueError):
    """Visibility NPZ 无法满足 coverage contribution 重放契约。"""


@dataclass(frozen=True)
class CoverageContributionDiagnostics:
    """一个 ``(state, view)`` 的线性重放数值不变量。"""

    state_id: int
    view_name: str
    num_instances: int
    num_vertices: int
    source_denominator: float
    effective_denominator: float
    effective_adjoint_denominator_absolute_error: float
    source_full_support_numerator: float
    effective_full_support_numerator: float
    source_full_support_coverage: float | None
    effective_full_support_coverage: float | None
    source_union_instance_max_error: float
    effective_union_instance_max_error: float


def _scalar_string(array: NDArray[np.generic], *, name: str) -> str:
    if array.ndim != 0:
        raise SupportCoverageContributionError(f"{name} 必须是 scalar")
    return str(array.item())


@lru_cache(maxsize=8)
def _effective_sum_source_weights_cached(
    source_resolution: int,
    pre_crop_resolution: int,
    crop_input_resolution: int,
    crop_output_resolution: int,
    crop_area: float,
) -> FloatArray:
    """返回 ``sum(T(x))`` 对每个 source pixel 的 float64 伴随权重。"""

    from experiments.robot.openvla_image_transform import (
        CenterCropSpecification,
    )

    source = torch.ones(
        (1, 1, source_resolution, source_resolution),
        dtype=torch.float32,
        requires_grad=True,
    )
    pre_crop = torch_functional.interpolate(
        source,
        size=(pre_crop_resolution, pre_crop_resolution),
        mode="area",
    )
    effective = torch_center_crop_float(
        pre_crop,
        specification=CenterCropSpecification(
            input_resolution=crop_input_resolution,
            output_resolution=crop_output_resolution,
            crop_area=crop_area,
        ),
    )
    gradient = torch.autograd.grad(effective.sum(), source)[0]
    weights = gradient[0, 0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("coverage transform 伴随权重不是有限非负数")
    expected_sum = float(crop_output_resolution * crop_output_resolution)
    if not math.isclose(
        float(weights.sum(dtype=np.float64)),
        expected_sum,
        rel_tol=0.0,
        abs_tol=ADJOINT_SUM_ATOL,
    ):
        raise RuntimeError("coverage transform 伴随权重不保持常量图求和")
    weights.setflags(write=False)
    return weights


def effective_sum_source_weights(
    specification: CoverageEvidenceSpecification,
) -> FloatArray:
    """返回只读 float64 ``[Hsource,Wsource]`` effective-sum 伴随权重。"""

    return _effective_sum_source_weights_cached(
        specification.source_resolution,
        specification.pre_crop_resolution,
        specification.center_crop.input_resolution,
        specification.center_crop.output_resolution,
        specification.center_crop.crop_area,
    )


def _validate_correspondence(
    *,
    triangle_indices: NDArray[np.int64],
    barycentric: NDArray[np.float32],
    geometry_corner_indices: NDArray[np.int64],
    num_instances: int,
    source_resolution: int,
    num_vertices: int,
) -> NDArray[np.bool_]:
    expected_pixel_shape = (
        num_instances,
        source_resolution,
        source_resolution,
    )
    if triangle_indices.dtype != np.int64 or triangle_indices.shape != expected_pixel_shape:
        raise SupportCoverageContributionError(
            "triangle_indices 必须为 int64 [K,Hsource,Wsource]"
        )
    if barycentric.dtype != np.float32 or barycentric.shape != expected_pixel_shape + (3,):
        raise SupportCoverageContributionError(
            "barycentric 必须为 float32 [K,Hsource,Wsource,3]"
        )
    if (
        geometry_corner_indices.dtype != np.int64
        or geometry_corner_indices.shape != expected_pixel_shape + (3,)
    ):
        raise SupportCoverageContributionError(
            "geometry_corner_indices 必须为 int64 [K,Hsource,Wsource,3]"
        )
    valid = triangle_indices >= 0
    corner_valid = np.all(geometry_corner_indices >= 0, axis=-1)
    if not np.array_equal(valid, corner_valid):
        raise SupportCoverageContributionError("triangle 与 geometry corner valid mask 不一致")
    if np.any(geometry_corner_indices[valid] >= num_vertices):
        raise SupportCoverageContributionError("geometry corner 顶点索引越界")
    if np.any(geometry_corner_indices[~valid] != -1):
        raise SupportCoverageContributionError("背景 geometry corner 必须为 -1")
    if np.any(barycentric[~valid] != 0.0):
        raise SupportCoverageContributionError("背景 barycentric 必须为零")
    valid_barycentric = barycentric[valid].astype(np.float64)
    if valid_barycentric.size > 0:
        if np.any(valid_barycentric < -COVERAGE_INVARIANT_ATOL) or np.any(
            valid_barycentric > 1.0 + COVERAGE_INVARIANT_ATOL
        ):
            raise SupportCoverageContributionError("有效 barycentric 超出 [0,1]")
        if not np.allclose(
            valid_barycentric.sum(axis=-1),
            1.0,
            rtol=0.0,
            atol=COVERAGE_INVARIANT_ATOL,
        ):
            raise SupportCoverageContributionError("有效 barycentric 和不为 1")
    return valid


def _scatter_instance_vertex_numerators(
    *,
    alpha_source: NDArray[np.float32],
    barycentric: NDArray[np.float32],
    geometry_corner_indices: NDArray[np.int64],
    valid: NDArray[np.bool_],
    spatial_sum_weights: FloatArray,
    num_vertices: int,
) -> FloatArray:
    """把逐像素 ``weight * alpha * beta`` scatter 为 float64 ``[K,N_v]``。"""

    num_instances = alpha_source.shape[0]
    result = np.zeros((num_instances, num_vertices), dtype=np.float64)
    for instance_index in range(num_instances):
        pixel_weight = (
            alpha_source[instance_index, 0].astype(np.float64)
            * spatial_sum_weights
        )
        instance_valid = valid[instance_index]
        for corner_index in range(3):
            vertex_ids = geometry_corner_indices[
                instance_index, :, :, corner_index
            ][instance_valid]
            values = (
                pixel_weight[instance_valid]
                * barycentric[instance_index, :, :, corner_index][
                    instance_valid
                ].astype(np.float64)
            )
            np.add.at(result[instance_index], vertex_ids, values)
    return result


def build_projection_coverage_evidence(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    status: VisibilityEvidenceStatus,
    num_vertices: int,
    specification: CoverageEvidenceSpecification = CoverageEvidenceSpecification(),
) -> tuple[ProjectionCoverageEvidence, CoverageContributionDiagnostics]:
    """从一个 Visibility NPZ mapping 构造可加的逐顶点 coverage evidence。"""

    required = {
        "schema_version",
        "state_id",
        "view_name",
        "mujoco_alpha_source",
        "mujoco_alpha_pre_crop",
        "mujoco_alpha_effective",
        "triangle_indices",
        "barycentric",
        "geometry_corner_indices",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise SupportCoverageContributionError(f"Visibility NPZ 缺少 keys: {missing}")
    if _scalar_string(arrays["schema_version"], name="schema_version") != VISIBILITY_ALIGNMENT_SCHEMA_VERSION:
        raise SupportCoverageContributionError("Visibility NPZ schema 不匹配")
    state_id_array = arrays["state_id"]
    if state_id_array.ndim != 0:
        raise SupportCoverageContributionError("state_id 必须是 scalar")
    state_id = int(state_id_array.item())
    view_name = _scalar_string(arrays["view_name"], name="view_name")

    alpha_source = np.asarray(arrays["mujoco_alpha_source"])
    if (
        alpha_source.dtype != np.float32
        or alpha_source.ndim != 4
        or alpha_source.shape[0] <= 0
        or alpha_source.shape[1:] != (
            1,
            specification.source_resolution,
            specification.source_resolution,
        )
    ):
        raise SupportCoverageContributionError(
            "mujoco_alpha_source 必须为 float32 [K,1,Hsource,Wsource]"
        )
    if not np.isfinite(alpha_source).all() or np.any(
        alpha_source < -COVERAGE_INVARIANT_ATOL
    ) or np.any(alpha_source > 1.0 + COVERAGE_INVARIANT_ATOL):
        raise SupportCoverageContributionError("mujoco alpha 非法")
    num_instances = int(alpha_source.shape[0])
    triangle_indices = np.asarray(arrays["triangle_indices"])
    barycentric = np.asarray(arrays["barycentric"])
    geometry_corner_indices = np.asarray(arrays["geometry_corner_indices"])
    valid = _validate_correspondence(
        triangle_indices=triangle_indices,
        barycentric=barycentric,
        geometry_corner_indices=geometry_corner_indices,
        num_instances=num_instances,
        source_resolution=specification.source_resolution,
        num_vertices=num_vertices,
    )

    alpha_tensor = torch.from_numpy(np.ascontiguousarray(alpha_source))
    full_control_source = np.where(
        valid,
        barycentric.astype(np.float64).sum(axis=-1),
        0.0,
    ).astype(np.float32)[:, None]
    full_stages = transform_premultiplied_evidence(
        alpha_tensor,
        alpha_tensor * torch.from_numpy(full_control_source),
        specification=specification,
    )
    stored_pre_crop = np.asarray(arrays["mujoco_alpha_pre_crop"])
    stored_effective = np.asarray(arrays["mujoco_alpha_effective"])
    expected_pre_crop = full_stages.pre_crop.alpha.detach().cpu().numpy()
    expected_effective = full_stages.effective.alpha.detach().cpu().numpy()
    if not np.allclose(
        stored_pre_crop,
        expected_pre_crop,
        rtol=0.0,
        atol=COVERAGE_INVARIANT_ATOL,
    ) or not np.allclose(
        stored_effective,
        expected_effective,
        rtol=0.0,
        atol=COVERAGE_INVARIANT_ATOL,
    ):
        raise SupportCoverageContributionError(
            "保存的 pre-crop/effective alpha 不能由 source evidence 重放"
        )

    source_weights = np.ones(
        (specification.source_resolution, specification.source_resolution),
        dtype=np.float64,
    )
    effective_weights = effective_sum_source_weights(specification)
    source_instances = _scatter_instance_vertex_numerators(
        alpha_source=alpha_source,
        barycentric=barycentric,
        geometry_corner_indices=geometry_corner_indices,
        valid=valid,
        spatial_sum_weights=source_weights,
        num_vertices=num_vertices,
    )
    effective_instances = _scatter_instance_vertex_numerators(
        alpha_source=alpha_source,
        barycentric=barycentric,
        geometry_corner_indices=geometry_corner_indices,
        valid=valid,
        spatial_sum_weights=effective_weights,
        num_vertices=num_vertices,
    )
    source_instance_denominators = alpha_source.astype(np.float64).sum(
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    effective_instance_denominators = stored_effective.astype(np.float64).sum(
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    adjoint_effective_denominators = (
        alpha_source[:, 0].astype(np.float64) * effective_weights[None]
    ).sum(axis=(1, 2), dtype=np.float64)
    denominator_error = float(
        np.max(
            np.abs(
                adjoint_effective_denominators
                - effective_instance_denominators
            ),
            initial=0.0,
        )
    )
    if denominator_error > ADJOINT_SUM_ATOL:
        raise SupportCoverageContributionError(
            "effective denominator 的线性伴随重放误差超限: "
            f"{denominator_error} > {ADJOINT_SUM_ATOL}"
        )

    source_union = source_instances.sum(axis=0, dtype=np.float64)
    effective_union = effective_instances.sum(axis=0, dtype=np.float64)
    source_denominator = float(source_instance_denominators.sum())
    effective_denominator = float(effective_instance_denominators.sum())
    source_full = float(source_union.sum(dtype=np.float64))
    effective_full = float(effective_union.sum(dtype=np.float64))
    direct_source_full = float(
        full_stages.source.alpha_times_control.sum().item()
    )
    direct_effective_full = float(
        full_stages.effective.alpha_times_control.sum().item()
    )
    if abs(source_full - direct_source_full) > ADJOINT_SUM_ATOL or abs(
        effective_full - direct_effective_full
    ) > ADJOINT_SUM_ATOL:
        raise SupportCoverageContributionError(
            "全 support contribution 与直接 premultiplied transform 不一致"
        )

    evidence = ProjectionCoverageEvidence(
        state_id=state_id,
        view_name=view_name,
        status=status,
        source_vertex_numerator=source_union,
        effective_vertex_numerator=effective_union,
        source_denominator=source_denominator,
        effective_denominator=effective_denominator,
        source_instance_vertex_numerator=source_instances,
        effective_instance_vertex_numerator=effective_instances,
        source_instance_denominators=source_instance_denominators,
        effective_instance_denominators=effective_instance_denominators,
    )
    diagnostics = CoverageContributionDiagnostics(
        state_id=state_id,
        view_name=view_name,
        num_instances=num_instances,
        num_vertices=num_vertices,
        source_denominator=source_denominator,
        effective_denominator=effective_denominator,
        effective_adjoint_denominator_absolute_error=denominator_error,
        source_full_support_numerator=source_full,
        effective_full_support_numerator=effective_full,
        source_full_support_coverage=(
            source_full / source_denominator if source_denominator > 0.0 else None
        ),
        effective_full_support_coverage=(
            effective_full / effective_denominator
            if effective_denominator > 0.0
            else None
        ),
        source_union_instance_max_error=float(
            np.max(
                np.abs(source_union - source_instances.sum(axis=0)),
                initial=0.0,
            )
        ),
        effective_union_instance_max_error=float(
            np.max(
                np.abs(effective_union - effective_instances.sum(axis=0)),
                initial=0.0,
            )
        ),
    )
    return evidence, diagnostics


def load_projection_coverage_evidence_npz(
    path: str | Path,
    *,
    status: VisibilityEvidenceStatus,
    num_vertices: int,
    specification: CoverageEvidenceSpecification = CoverageEvidenceSpecification(),
) -> tuple[ProjectionCoverageEvidence, CoverageContributionDiagnostics]:
    """安全读取单个 NPZ；禁用 pickle 后调用完整重放。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return build_projection_coverage_evidence(
        arrays,
        status=status,
        num_vertices=num_vertices,
        specification=specification,
    )
