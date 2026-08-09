"""Fixed Support 谱自然性频带与 ``rho_nat`` 校准契约。

本模块只执行 CPU 数值计算，不加载 VLA、LIBERO 或 renderer。它把不可变
Production Fixed Support 与连续几何谱基绑定，并使用 uniform-support probe
``delta_probe = 1_S`` 计算 ``rho_nat = r_high(delta_probe)``。校准 artifact
不包含可学习参数，也不放行 ``lambda_spec`` 校准或正式训练。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from .production_support import (
    FrozenProductionSupport,
    array_sha256,
    load_production_support_artifact,
)
from .seed_score_audit import file_sha256
from .spectral_geometry import (
    SpectralBasisData,
    load_spectral_basis,
)


NATURALNESS_CALIBRATION_SCHEMA_VERSION: Final[str] = (
    "openvla-spectral-naturalness-calibration-v1"
)
NATURALNESS_K_NONCONSTANT: Final[int] = 128
NATURALNESS_NUM_LOW_MODES: Final[int] = 129
NATURALNESS_ENERGY_EPSILON: Final[float] = 1e-12
MASS_ORTHONORMALITY_TOLERANCE: Final[float] = 1e-8
CONSTANT_MODE_TOLERANCE: Final[float] = 1e-8


class SpectralNaturalnessError(ValueError):
    """自然性频带、uniform probe 或校准 artifact 不满足冻结契约。"""


@dataclass(frozen=True)
class NaturalnessEnergy:
    """一个几何顶点 RGB Surface Delta 的谱能量分解。"""

    total: float
    low: float
    high: float
    high_ratio: float


@dataclass(frozen=True)
class NaturalnessBandAudit:
    """连续低频带的数值与尺度审计。"""

    mass_orthonormality_linf: float
    constant_mode_linf: float
    total_surface_area: float
    cutoff_eigenvalue: float
    dimensionless_cutoff: float


@dataclass(frozen=True)
class SpectralNaturalnessCalibration:
    """训练前只读的 ``rho_nat`` 校准结果。"""

    schema_version: str
    code_commit: str
    object_name: str
    production_support_artifact_sha256: str
    support_mask_sha256: str
    spectral_basis_artifact_sha256: str
    mesh_array_sha256: str
    mass_sha256: str
    low_basis_sha256: str
    low_band_eigenvalues_sha256: str
    num_geometry_vertices: int
    num_support_vertices: int
    naturalness_k_nonconstant: int
    num_low_modes_including_constant: int
    low_band_eigenvalues: NDArray[np.float64]
    total_surface_area: float
    cutoff_eigenvalue: float
    dimensionless_cutoff: float
    mass_orthonormality_linf: float
    constant_mode_linf: float
    support_probe_sha256: str
    energy_total: float
    energy_low: float
    energy_high: float
    energy_epsilon: float
    rho_nat: float
    rho_nat_calibrated: bool
    lambda_spec_calibrated: bool
    formal_training_allowed: bool


@dataclass(frozen=True)
class SpectralNaturalnessDecision:
    """校准 artifact 对输入 Support 与谱基的独立复核结果。"""

    gate_pass: bool
    failures: tuple[str, ...]


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def audit_naturalness_band(
    basis_data: SpectralBasisData,
) -> NaturalnessBandAudit:
    """验收常数模态加前128个连续非恒定模态。

    ``basis_data.basis`` 必须为 float64 ``[num_vertices, 129]``，列顺序与
    ``eigenvalues`` 严格一致。常数模态允许整体符号相反，但不得混入空间变化。
    """

    basis = np.asarray(basis_data.basis, dtype=np.float64)
    mass = np.asarray(basis_data.mass, dtype=np.float64)
    eigenvalues = np.asarray(basis_data.eigenvalues, dtype=np.float64)
    if basis.shape != (basis_data.num_geometry_vertices, NATURALNESS_NUM_LOW_MODES):
        raise SpectralNaturalnessError(
            "自然性频带必须包含常数模态和128个非恒定模态"
        )
    if eigenvalues.shape != (NATURALNESS_NUM_LOW_MODES,):
        raise SpectralNaturalnessError("自然性频带特征值数量不匹配")
    if not bool(basis_data.metadata.get("include_constant", False)):
        raise SpectralNaturalnessError("自然性频带必须显式加载常数模态")
    if np.any(np.diff(eigenvalues) < -1e-8):
        raise SpectralNaturalnessError("自然性频带特征值未按升序连续排列")
    if np.any(eigenvalues[1:] <= 0.0):
        raise SpectralNaturalnessError("非恒定自然性模态特征值必须为正")

    total_surface_area = float(mass.sum())
    expected_constant = np.full(
        basis_data.num_geometry_vertices,
        1.0 / math.sqrt(total_surface_area),
        dtype=np.float64,
    )
    constant_column = basis[:, 0]
    if float(np.dot(mass * constant_column, expected_constant)) < 0.0:
        constant_column = -constant_column
    constant_error = float(
        np.max(np.abs(constant_column - expected_constant))
    )
    gram = basis.T @ (mass[:, None] * basis)
    orthonormality_error = float(
        np.max(np.abs(gram - np.eye(NATURALNESS_NUM_LOW_MODES)))
    )
    if constant_error > CONSTANT_MODE_TOLERANCE:
        raise SpectralNaturalnessError(
            "自然性频带首列不是mass-normalized常数模态"
        )
    if orthonormality_error > MASS_ORTHONORMALITY_TOLERANCE:
        raise SpectralNaturalnessError("自然性频带未通过M-正交归一门槛")

    cutoff = float(eigenvalues[-1])
    return NaturalnessBandAudit(
        mass_orthonormality_linf=orthonormality_error,
        constant_mode_linf=constant_error,
        total_surface_area=total_surface_area,
        cutoff_eigenvalue=cutoff,
        dimensionless_cutoff=cutoff * total_surface_area,
    )


def compute_naturalness_energy(
    geometry_delta: NDArray[np.floating[Any]],
    *,
    mass: NDArray[np.floating[Any]],
    low_basis: NDArray[np.floating[Any]],
    epsilon: float = NATURALNESS_ENERGY_EPSILON,
) -> NaturalnessEnergy:
    """计算RGB整体谱能量和无 ``stopgrad`` 的同值高频比例。

    Args:
        geometry_delta: float ``[num_geometry_vertices, 3]``。
        mass: 正的lumped mass，float ``[num_geometry_vertices]``。
        low_basis: M-正交低频带，float ``[num_geometry_vertices, 129]``。
        epsilon: 只用于零能量附近数值稳定的冻结常数。
    """

    delta = np.asarray(geometry_delta, dtype=np.float64)
    mass_array = np.asarray(mass, dtype=np.float64)
    basis = np.asarray(low_basis, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 3:
        raise SpectralNaturalnessError("geometry_delta 必须为 [N, 3]")
    if mass_array.shape != (delta.shape[0],) or np.any(mass_array <= 0.0):
        raise SpectralNaturalnessError("mass 必须为与delta匹配的正向量")
    if basis.ndim != 2 or basis.shape[0] != delta.shape[0]:
        raise SpectralNaturalnessError("low_basis 顶点维度与delta不匹配")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise SpectralNaturalnessError("energy epsilon 必须为有限正数")
    if not all(
        np.isfinite(array).all() for array in (delta, mass_array, basis)
    ):
        raise SpectralNaturalnessError("谱能量输入包含非有限值")

    weighted_delta = mass_array[:, None] * delta
    total = float(np.sum(weighted_delta * delta))
    coefficients = basis.T @ weighted_delta
    low = float(np.sum(coefficients * coefficients))
    numerical_slack = max(1e-12, 1e-10 * total)
    if low > total + numerical_slack:
        raise SpectralNaturalnessError("低频投影能量显著超过总能量")
    high = max(total - low, 0.0)
    return NaturalnessEnergy(
        total=total,
        low=low,
        high=high,
        high_ratio=high / (total + epsilon),
    )


def build_rho_nat_calibration(
    *,
    production_support_path: str | Path,
    spectral_basis_path: str | Path,
    code_commit: str,
    object_name: str,
) -> SpectralNaturalnessCalibration:
    """从冻结 Support 与连续K512产物构造K_nat=128校准。"""

    if not _is_hex(code_commit, 40):
        raise SpectralNaturalnessError("code_commit 必须是40位小写Git SHA")
    support_path = Path(production_support_path)
    basis_path = Path(spectral_basis_path)
    support: FrozenProductionSupport = load_production_support_artifact(
        support_path
    )
    if support.provenance.object_name != object_name:
        raise SpectralNaturalnessError("Production Support object_name不匹配")
    if support.naturalness_k_nonconstant != NATURALNESS_K_NONCONSTANT:
        raise SpectralNaturalnessError("Production Support K_nat不是128")
    basis_data = load_spectral_basis(
        basis_path,
        max_basis=NATURALNESS_NUM_LOW_MODES,
        include_constant=True,
    )
    if basis_data.mesh_sha256 != support.provenance.mesh_array_sha256:
        raise SpectralNaturalnessError("谱基mesh与Production Support不匹配")
    band_audit = audit_naturalness_band(basis_data)
    if not math.isclose(
        band_audit.total_surface_area,
        support.total_surface_area,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise SpectralNaturalnessError("谱基mass总面积与Support不匹配")

    # delta_probe: float64 [num_geometry_vertices, 3]。三个通道在Support内均为1，
    # Support外严格为0；RGB整体能量使比例不依赖重复通道数。
    support_probe = np.repeat(
        support.support_mask.astype(np.float64)[:, None],
        3,
        axis=1,
    )
    energy = compute_naturalness_energy(
        support_probe,
        mass=basis_data.mass,
        low_basis=basis_data.basis,
    )
    if energy.total <= 0.0:
        raise SpectralNaturalnessError("uniform Support probe总能量必须为正")
    eigenvalues = np.ascontiguousarray(
        basis_data.eigenvalues,
        dtype=np.float64,
    )
    return SpectralNaturalnessCalibration(
        schema_version=NATURALNESS_CALIBRATION_SCHEMA_VERSION,
        code_commit=code_commit,
        object_name=object_name,
        production_support_artifact_sha256=file_sha256(support_path),
        support_mask_sha256=support.support_mask_sha256,
        spectral_basis_artifact_sha256=file_sha256(basis_path),
        mesh_array_sha256=basis_data.mesh_sha256,
        mass_sha256=array_sha256(basis_data.mass),
        low_basis_sha256=array_sha256(basis_data.basis),
        low_band_eigenvalues_sha256=array_sha256(eigenvalues),
        num_geometry_vertices=support.num_geometry_vertices,
        num_support_vertices=len(support.support_vertex_indices),
        naturalness_k_nonconstant=NATURALNESS_K_NONCONSTANT,
        num_low_modes_including_constant=NATURALNESS_NUM_LOW_MODES,
        low_band_eigenvalues=eigenvalues,
        total_surface_area=band_audit.total_surface_area,
        cutoff_eigenvalue=band_audit.cutoff_eigenvalue,
        dimensionless_cutoff=band_audit.dimensionless_cutoff,
        mass_orthonormality_linf=band_audit.mass_orthonormality_linf,
        constant_mode_linf=band_audit.constant_mode_linf,
        support_probe_sha256=array_sha256(support_probe),
        energy_total=energy.total,
        energy_low=energy.low,
        energy_high=energy.high,
        energy_epsilon=NATURALNESS_ENERGY_EPSILON,
        rho_nat=energy.high_ratio,
        rho_nat_calibrated=True,
        lambda_spec_calibrated=False,
        formal_training_allowed=False,
    )


def _validate_calibration(value: SpectralNaturalnessCalibration) -> None:
    if value.schema_version != NATURALNESS_CALIBRATION_SCHEMA_VERSION:
        raise SpectralNaturalnessError("自然性校准schema不匹配")
    if not _is_hex(value.code_commit, 40):
        raise SpectralNaturalnessError("校准code_commit无效")
    for name in (
        "production_support_artifact_sha256",
        "support_mask_sha256",
        "spectral_basis_artifact_sha256",
        "mesh_array_sha256",
        "mass_sha256",
        "low_basis_sha256",
        "low_band_eigenvalues_sha256",
        "support_probe_sha256",
    ):
        if not _is_hex(str(getattr(value, name)), 64):
            raise SpectralNaturalnessError(f"{name}必须是64位SHA-256")
    if value.naturalness_k_nonconstant != NATURALNESS_K_NONCONSTANT:
        raise SpectralNaturalnessError("自然性校准K_nat必须为128")
    if value.num_low_modes_including_constant != NATURALNESS_NUM_LOW_MODES:
        raise SpectralNaturalnessError("自然性校准必须记录129个低频模态")
    if value.low_band_eigenvalues.shape != (NATURALNESS_NUM_LOW_MODES,):
        raise SpectralNaturalnessError("自然性校准特征值shape无效")
    if array_sha256(value.low_band_eigenvalues) != (
        value.low_band_eigenvalues_sha256
    ):
        raise SpectralNaturalnessError("自然性校准特征值semantic hash不匹配")
    for name in (
        "total_surface_area",
        "cutoff_eigenvalue",
        "dimensionless_cutoff",
        "energy_total",
        "energy_low",
        "energy_high",
        "energy_epsilon",
        "rho_nat",
    ):
        number = float(getattr(value, name))
        if not math.isfinite(number) or number < 0.0:
            raise SpectralNaturalnessError(f"{name}必须为有限非负数")
    if value.energy_total <= 0.0 or value.energy_epsilon <= 0.0:
        raise SpectralNaturalnessError("校准总能量与epsilon必须为正")
    if not math.isclose(
        value.energy_high,
        max(value.energy_total - value.energy_low, 0.0),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise SpectralNaturalnessError("校准高频能量分解不一致")
    if not math.isclose(
        value.rho_nat,
        value.energy_high / (value.energy_total + value.energy_epsilon),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise SpectralNaturalnessError("rho_nat与能量不一致")
    if (
        not value.rho_nat_calibrated
        or value.lambda_spec_calibrated
        or value.formal_training_allowed
    ):
        raise SpectralNaturalnessError("rho_nat阶段flags非法")


def write_rho_nat_calibration_artifact(
    path: str | Path,
    calibration: SpectralNaturalnessCalibration,
) -> str:
    """拒绝覆盖地写入校准NPZ，并返回文件SHA-256。"""

    _validate_calibration(calibration)
    output_path = Path(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        name: getattr(calibration, name)
        for name in calibration.__dataclass_fields__
    }
    np.savez_compressed(output_path, **payload)
    return file_sha256(output_path)


def _scalar(archive: Mapping[str, NDArray[Any]], name: str) -> Any:
    value = archive[name]
    if value.ndim != 0:
        raise SpectralNaturalnessError(f"{name}必须是scalar")
    return value.item()


def load_rho_nat_calibration_artifact(
    path: str | Path,
) -> SpectralNaturalnessCalibration:
    """使用 ``allow_pickle=False`` 严格加载校准artifact。"""

    resolved_path = Path(path)
    with np.load(resolved_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    expected_names = set(SpectralNaturalnessCalibration.__dataclass_fields__)
    if set(payload) != expected_names:
        raise SpectralNaturalnessError("自然性校准artifact字段集合不匹配")
    value = SpectralNaturalnessCalibration(
        schema_version=str(_scalar(payload, "schema_version")),
        code_commit=str(_scalar(payload, "code_commit")),
        object_name=str(_scalar(payload, "object_name")),
        production_support_artifact_sha256=str(
            _scalar(payload, "production_support_artifact_sha256")
        ),
        support_mask_sha256=str(_scalar(payload, "support_mask_sha256")),
        spectral_basis_artifact_sha256=str(
            _scalar(payload, "spectral_basis_artifact_sha256")
        ),
        mesh_array_sha256=str(_scalar(payload, "mesh_array_sha256")),
        mass_sha256=str(_scalar(payload, "mass_sha256")),
        low_basis_sha256=str(_scalar(payload, "low_basis_sha256")),
        low_band_eigenvalues_sha256=str(
            _scalar(payload, "low_band_eigenvalues_sha256")
        ),
        num_geometry_vertices=int(_scalar(payload, "num_geometry_vertices")),
        num_support_vertices=int(_scalar(payload, "num_support_vertices")),
        naturalness_k_nonconstant=int(
            _scalar(payload, "naturalness_k_nonconstant")
        ),
        num_low_modes_including_constant=int(
            _scalar(payload, "num_low_modes_including_constant")
        ),
        low_band_eigenvalues=np.asarray(
            payload["low_band_eigenvalues"], dtype=np.float64
        ),
        total_surface_area=float(_scalar(payload, "total_surface_area")),
        cutoff_eigenvalue=float(_scalar(payload, "cutoff_eigenvalue")),
        dimensionless_cutoff=float(_scalar(payload, "dimensionless_cutoff")),
        mass_orthonormality_linf=float(
            _scalar(payload, "mass_orthonormality_linf")
        ),
        constant_mode_linf=float(_scalar(payload, "constant_mode_linf")),
        support_probe_sha256=str(_scalar(payload, "support_probe_sha256")),
        energy_total=float(_scalar(payload, "energy_total")),
        energy_low=float(_scalar(payload, "energy_low")),
        energy_high=float(_scalar(payload, "energy_high")),
        energy_epsilon=float(_scalar(payload, "energy_epsilon")),
        rho_nat=float(_scalar(payload, "rho_nat")),
        rho_nat_calibrated=bool(_scalar(payload, "rho_nat_calibrated")),
        lambda_spec_calibrated=bool(
            _scalar(payload, "lambda_spec_calibrated")
        ),
        formal_training_allowed=bool(
            _scalar(payload, "formal_training_allowed")
        ),
    )
    _validate_calibration(value)
    return value


def evaluate_rho_nat_calibration(
    calibration_path: str | Path,
    *,
    production_support_path: str | Path,
    spectral_basis_path: str | Path,
) -> SpectralNaturalnessDecision:
    """从两个输入artifact重建预期值并逐字段验收校准结果。"""

    failures: list[str] = []
    try:
        actual = load_rho_nat_calibration_artifact(calibration_path)
        expected = build_rho_nat_calibration(
            production_support_path=production_support_path,
            spectral_basis_path=spectral_basis_path,
            code_commit=actual.code_commit,
            object_name=actual.object_name,
        )
        for name in actual.__dataclass_fields__:
            actual_value = getattr(actual, name)
            expected_value = getattr(expected, name)
            if isinstance(actual_value, np.ndarray):
                if not np.array_equal(actual_value, expected_value):
                    failures.append(f"{name}与输入artifact重算不一致")
            elif actual_value != expected_value:
                failures.append(f"{name}与输入artifact重算不一致")
    except (OSError, KeyError, TypeError, ValueError) as error:
        failures.append(f"rho_nat校准artifact无法严格复核: {error}")
    return SpectralNaturalnessDecision(not failures, tuple(failures))
