"""Gate 2C：OpenVLA center-crop 跨框架等价性审计。

本模块把第 44 项冻结决策中的确定性 case 固化为可直接运行的 CPU 审计。它
比较 TensorFlow deployment oracle 与 PyTorch differentiable surrogate 的：

* forward 输出；
* 固定 crop box 下的输入 RGB VJP ``dL/dI``。

权威输出为逐 case JSONL。每行都记录输入/输出 shape、dtype、seed、framework
版本、crop specification 以及 relative L2、cosine、max-absolute error。这里的
候选门槛只用于判断实现是否明显偏离预注册值；正式冻结仍需把完整分布写入状态
文档，不能根据 support 或攻击结果反向调整。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Final, Literal, Sequence, TypedDict

import numpy as np
import tensorflow as tf
import torch
from numpy.typing import NDArray

from experiments.robot.openvla_image_transform import (
    CenterCropSpecification,
    tensorflow_center_crop_float,
    torch_center_crop_float,
)


CENTER_CROP_AUDIT_SCHEMA_VERSION: Final[str] = "openvla-gate-2c-v1"
VJP_RELATIVE_L2_CANDIDATE_MAX: Final[float] = 1e-5
VJP_COSINE_CANDIDATE_MIN: Final[float] = 0.99999
FORWARD_RELATIVE_L2_ENGINEERING_MAX: Final[float] = 1e-5
FORWARD_COSINE_ENGINEERING_MIN: Final[float] = 0.99999
FORWARD_RANDOM_SEED: Final[int] = 17
VJP_RANDOM_SEED: Final[int] = 23


class CenterCropAuditResult(TypedDict):
    """单个 forward 或 VJP case 的权威 JSONL schema。"""

    schema_version: str
    code_commit: str
    case_name: str
    audit_kind: Literal["forward", "vjp"]
    input_shape: list[int]
    output_shape: list[int]
    dtype: str
    seed: int | None
    input_resolution: int
    output_resolution: int
    crop_area: float
    tensorflow_version: str
    torch_version: str
    numpy_version: str
    relative_l2: float
    cosine: float
    max_abs: float
    relative_l2_limit: float
    cosine_limit: float
    candidate_pass: bool


def _spatial_ramp(size: int) -> NDArray[np.float32]:
    """返回 float32 HWC RGB 横、纵和对角 ramp。"""

    y_values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x_values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    return np.stack(
        (x_grid, y_grid, (x_grid + y_grid) / 2.0),
        axis=-1,
    ).astype(np.float32, copy=False)


def _checkerboard(size: int) -> NDArray[np.float32]:
    """返回高频 float32 HWC RGB checkerboard。"""

    y_grid, x_grid = np.indices((size, size))
    checker = ((x_grid + y_grid) % 2).astype(np.float32)
    return np.stack(
        (checker, 1.0 - checker, checker * 0.5),
        axis=-1,
    )


def _impulse(
    size: int,
    *,
    row: int,
    column: int,
) -> NDArray[np.float32]:
    """返回指定像素具有非零三通道值的 float32 HWC impulse。"""

    image = np.zeros((size, size, 3), dtype=np.float32)
    image[row, column] = np.asarray((1.0, 0.5, 0.25), dtype=np.float32)
    return image


def _compute_metrics(
    candidate: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> tuple[float, float, float]:
    """计算 candidate 相对 TensorFlow reference 的三项冻结指标。"""

    candidate_float64 = candidate.astype(np.float64, copy=False)
    reference_float64 = reference.astype(np.float64, copy=False)
    difference = candidate_float64 - reference_float64
    reference_norm = float(np.linalg.norm(reference_float64.reshape(-1)))
    candidate_norm = float(np.linalg.norm(candidate_float64.reshape(-1)))
    relative_l2 = float(
        np.linalg.norm(difference.reshape(-1)) / (reference_norm + 1e-12)
    )
    cosine = float(
        np.vdot(candidate_float64.reshape(-1), reference_float64.reshape(-1))
        / (candidate_norm * reference_norm + 1e-12)
    )
    max_abs = float(np.max(np.abs(difference)))
    return relative_l2, cosine, max_abs


def _build_result(
    *,
    case_name: str,
    audit_kind: Literal["forward", "vjp"],
    seed: int | None,
    specification: CenterCropSpecification,
    candidate: NDArray[np.float32],
    reference: NDArray[np.float32],
    code_commit: str,
) -> CenterCropAuditResult:
    """把一个跨框架比较转换为稳定 JSONL 行。"""

    relative_l2: float
    cosine: float
    max_abs: float
    relative_l2, cosine, max_abs = _compute_metrics(candidate, reference)
    if audit_kind == "vjp":
        relative_l2_limit = VJP_RELATIVE_L2_CANDIDATE_MAX
        cosine_limit = VJP_COSINE_CANDIDATE_MIN
    else:
        # forward 尚无单独科学门槛；这里使用同量级的严格工程 guard，只用于
        # 暴露坐标交换、半像素偏移或 align_corners 错误。
        relative_l2_limit = FORWARD_RELATIVE_L2_ENGINEERING_MAX
        cosine_limit = FORWARD_COSINE_ENGINEERING_MIN
    candidate_pass = bool(
        math.isfinite(relative_l2)
        and math.isfinite(cosine)
        and math.isfinite(max_abs)
        and relative_l2 <= relative_l2_limit
        and cosine >= cosine_limit
    )
    return {
        "schema_version": CENTER_CROP_AUDIT_SCHEMA_VERSION,
        "code_commit": code_commit,
        "case_name": case_name,
        "audit_kind": audit_kind,
        "input_shape": [
            specification.input_resolution,
            specification.input_resolution,
            3,
        ],
        "output_shape": [
            specification.output_resolution,
            specification.output_resolution,
            3,
        ],
        "dtype": "float32",
        "seed": seed,
        "input_resolution": specification.input_resolution,
        "output_resolution": specification.output_resolution,
        "crop_area": specification.crop_area,
        "tensorflow_version": tf.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs": max_abs,
        "relative_l2_limit": relative_l2_limit,
        "cosine_limit": cosine_limit,
        "candidate_pass": candidate_pass,
    }


def _forward_cases(
    specification: CenterCropSpecification,
) -> tuple[tuple[str, NDArray[np.float32], int | None], ...]:
    """构造冻结的五类 forward 空间图案。"""

    size = specification.input_resolution
    side_offset = (1.0 - math.sqrt(specification.crop_area)) / 2.0
    crop_boundary = int(round(side_offset * (size - 1)))
    random_rgb = np.random.default_rng(FORWARD_RANDOM_SEED).random(
        (size, size, 3),
        dtype=np.float32,
    )
    return (
        ("forward_spatial_ramp", _spatial_ramp(size), None),
        ("forward_checkerboard", _checkerboard(size), None),
        (
            "forward_center_impulse",
            _impulse(size, row=size // 2, column=size // 2),
            None,
        ),
        (
            "forward_crop_boundary_impulse",
            _impulse(
                size,
                row=crop_boundary,
                column=size // 2,
            ),
            None,
        ),
        ("forward_random_rgb", random_rgb, FORWARD_RANDOM_SEED),
    )


def _vjp_cases(
    specification: CenterCropSpecification,
) -> tuple[tuple[str, NDArray[np.float32], int | None], ...]:
    """构造冻结的五类输出上游梯度。"""

    size = specification.output_resolution
    random_upstream = np.random.default_rng(VJP_RANDOM_SEED).standard_normal(
        (size, size, 3),
        dtype=np.float32,
    )
    return (
        ("vjp_spatial_ramp", _spatial_ramp(size), None),
        (
            "vjp_center_impulse",
            _impulse(size, row=size // 2, column=size // 2),
            None,
        ),
        (
            "vjp_boundary_impulse",
            _impulse(size, row=0, column=size // 2),
            None,
        ),
        ("vjp_corner_impulse", _impulse(size, row=0, column=0), None),
        ("vjp_random", random_upstream, VJP_RANDOM_SEED),
    )


def run_center_crop_equivalence_audit(
    *,
    specification: CenterCropSpecification,
    code_commit: str = "uncommitted",
) -> list[CenterCropAuditResult]:
    """运行第 44 项规定的全部 forward/VJP 确定性 case。

    返回顺序固定为五个 forward case 后接五个 VJP case；写入 JSONL 时不得重新
    排序或静默省略失败 case。
    """

    results: list[CenterCropAuditResult] = []
    for case_name, source_hwc, seed in _forward_cases(specification):
        tensorflow_output = tensorflow_center_crop_float(
            tf.convert_to_tensor(source_hwc),
            specification=specification,
        ).numpy()
        source_nchw = torch.from_numpy(source_hwc).permute(2, 0, 1)[None, ...]
        torch_output = torch_center_crop_float(
            source_nchw,
            specification=specification,
        )[0].permute(1, 2, 0).detach().numpy()
        results.append(
            _build_result(
                case_name=case_name,
                audit_kind="forward",
                seed=seed,
                specification=specification,
                candidate=torch_output,
                reference=tensorflow_output,
                code_commit=code_commit,
            )
        )

    # 固定 bilinear crop 对输入是线性算子，VJP 与 source 数值无关。仍使用非零
    # ramp 作为统一 source，便于出现异常时同时查看 forward。
    source_hwc = _spatial_ramp(specification.input_resolution)
    for case_name, upstream_hwc, seed in _vjp_cases(specification):
        source_tensorflow = tf.Variable(source_hwc)
        with tf.GradientTape() as tape:
            output_tensorflow = tensorflow_center_crop_float(
                source_tensorflow,
                specification=specification,
            )
            loss_tensorflow = tf.reduce_sum(
                output_tensorflow * tf.convert_to_tensor(upstream_hwc)
            )
        gradient_tensorflow_tensor = tape.gradient(
            loss_tensorflow,
            source_tensorflow,
        )
        if gradient_tensorflow_tensor is None:
            raise RuntimeError(f"TensorFlow VJP 缺失：{case_name}")
        gradient_tensorflow = gradient_tensorflow_tensor.numpy()

        source_torch = (
            torch.from_numpy(source_hwc)
            .permute(2, 0, 1)[None, ...]
            .requires_grad_(True)
        )
        upstream_torch = torch.from_numpy(upstream_hwc).permute(2, 0, 1)[
            None,
            ...,
        ]
        output_torch = torch_center_crop_float(
            source_torch,
            specification=specification,
        )
        torch.sum(output_torch * upstream_torch).backward()
        if source_torch.grad is None:
            raise RuntimeError(f"PyTorch VJP 缺失：{case_name}")
        gradient_torch = (
            source_torch.grad[0].permute(1, 2, 0).detach().numpy()
        )
        results.append(
            _build_result(
                case_name=case_name,
                audit_kind="vjp",
                seed=seed,
                specification=specification,
                candidate=gradient_torch,
                reference=gradient_tensorflow,
                code_commit=code_commit,
            )
        )
    return results


def write_center_crop_audit_jsonl(
    results: Sequence[CenterCropAuditResult],
    *,
    output_path: Path,
) -> str:
    """按传入顺序写入无省略 JSONL，并返回文件 SHA-256。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_lines = [
        json.dumps(result, ensure_ascii=False, sort_keys=True)
        for result in results
    ]
    payload = ("\n".join(serialized_lines) + "\n").encode("utf-8")
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 OpenVLA Gate 2C center-crop 跨框架 CPU 审计",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--code-commit", type=str, required=True)
    parser.add_argument("--input-resolution", type=int, default=224)
    parser.add_argument("--output-resolution", type=int, default=224)
    parser.add_argument("--crop-area", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    """CLI 入口；任一 case 未通过候选工程 guard 时返回非零。"""

    args = _parse_args()
    specification = CenterCropSpecification(
        input_resolution=args.input_resolution,
        output_resolution=args.output_resolution,
        crop_area=args.crop_area,
    )
    results = run_center_crop_equivalence_audit(
        specification=specification,
        code_commit=args.code_commit,
    )
    output_sha256 = write_center_crop_audit_jsonl(
        results,
        output_path=args.output_jsonl,
    )
    worst_relative_l2 = max(result["relative_l2"] for result in results)
    worst_cosine = min(result["cosine"] for result in results)
    worst_max_abs = max(result["max_abs"] for result in results)
    all_pass = all(result["candidate_pass"] for result in results)
    print(
        json.dumps(
            {
                "schema_version": CENTER_CROP_AUDIT_SCHEMA_VERSION,
                "case_count": len(results),
                "all_candidate_pass": all_pass,
                "worst_relative_l2": worst_relative_l2,
                "worst_cosine": worst_cosine,
                "worst_max_abs": worst_max_abs,
                "output_jsonl": str(args.output_jsonl),
                "output_sha256": output_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
