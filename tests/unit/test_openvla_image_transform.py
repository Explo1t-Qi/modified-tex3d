"""OpenVLA 部署 center-crop 的跨框架纯 CPU 契约测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch


OPENVLA_ROOT = Path(__file__).resolve().parents[2] / "openvla"
sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.openvla_image_transform import (  # noqa: E402
    CenterCropSpecification,
    deployment_center_crop_uint8,
    tensorflow_center_crop_float,
    torch_center_crop_float,
)
from experiments.robot.openvla_image_transform_audit import (  # noqa: E402
    CENTER_CROP_AUDIT_SCHEMA_VERSION,
    run_center_crop_equivalence_audit,
    write_center_crop_audit_jsonl,
)


def _spatial_ramp(size: int) -> np.ndarray:
    """返回 float32 HWC RGB 横纵 ramp。"""
    y_values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x_values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    return np.stack(
        [x_grid, y_grid, (x_grid + y_grid) / 2.0],
        axis=-1,
    )


def test_deployment_center_crop_uint8_matches_legacy_tensorflow_path() -> None:
    """共享 exact helper 必须逐值复现现有 TF uint8 部署代码。"""
    source_uint8 = np.rint(_spatial_ramp(8) * 255.0).astype(np.uint8)
    specification = CenterCropSpecification(
        input_resolution=8,
        output_resolution=8,
        crop_area=0.9,
    )

    actual = deployment_center_crop_uint8(
        source_uint8,
        specification=specification,
    )

    source_tf = tf.convert_to_tensor(source_uint8)
    source_float = tf.image.convert_image_dtype(source_tf, tf.float32)
    side_scale = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
    offset = (1.0 - side_scale) / 2.0
    expected_float = tf.image.crop_and_resize(
        source_float[None, ...],
        boxes=tf.reshape(
            tf.stack([offset, offset, offset + side_scale, offset + side_scale]),
            (1, 4),
        ),
        box_indices=tf.constant([0], dtype=tf.int32),
        crop_size=(8, 8),
    )[0]
    expected = tf.image.convert_image_dtype(
        tf.clip_by_value(expected_float, 0.0, 1.0),
        tf.uint8,
        saturate=True,
    ).numpy()

    np.testing.assert_array_equal(actual, expected)


def test_torch_center_crop_forward_matches_tensorflow_float_geometry() -> None:
    source_hwc = _spatial_ramp(9)
    specification = CenterCropSpecification(
        input_resolution=9,
        output_resolution=7,
        crop_area=0.9,
    )

    tensorflow_output = tensorflow_center_crop_float(
        tf.convert_to_tensor(source_hwc),
        specification=specification,
    ).numpy()
    source_nchw = torch.from_numpy(source_hwc).permute(2, 0, 1).unsqueeze(0)
    torch_output = torch_center_crop_float(
        source_nchw,
        specification=specification,
    )[0].permute(1, 2, 0).detach().numpy()

    np.testing.assert_allclose(
        torch_output,
        tensorflow_output,
        rtol=1e-5,
        atol=1e-6,
    )


def test_torch_center_crop_preserves_batch_order() -> None:
    """多 state/view batch 不得在显式 gather 中发生维度或顺序混淆。"""
    first = _spatial_ramp(9)
    second = np.ascontiguousarray(first[::-1, ::-1])
    source_nhwc = np.stack((first, second), axis=0)
    specification = CenterCropSpecification(
        input_resolution=9,
        output_resolution=7,
        crop_area=0.9,
    )

    tensorflow_output = tensorflow_center_crop_float(
        tf.convert_to_tensor(source_nhwc),
        specification=specification,
    ).numpy()
    source_nchw = torch.from_numpy(source_nhwc).permute(0, 3, 1, 2)
    torch_output = torch_center_crop_float(
        source_nchw,
        specification=specification,
    ).permute(0, 2, 3, 1).detach().numpy()

    np.testing.assert_array_equal(torch_output, tensorflow_output)


def test_torch_center_crop_input_vjp_matches_tensorflow() -> None:
    source_hwc = _spatial_ramp(9)
    upstream_hwc = np.random.default_rng(17).standard_normal(
        (7, 7, 3),
        dtype=np.float32,
    )
    specification = CenterCropSpecification(
        input_resolution=9,
        output_resolution=7,
        crop_area=0.9,
    )

    source_tf = tf.Variable(source_hwc)
    with tf.GradientTape() as tape:
        output_tf = tensorflow_center_crop_float(
            source_tf,
            specification=specification,
        )
        loss_tf = tf.reduce_sum(output_tf * upstream_hwc)
    gradient_tf = tape.gradient(loss_tf, source_tf).numpy()

    source_torch = (
        torch.from_numpy(source_hwc)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .requires_grad_(True)
    )
    upstream_torch = (
        torch.from_numpy(upstream_hwc).permute(2, 0, 1).unsqueeze(0)
    )
    output_torch = torch_center_crop_float(
        source_torch,
        specification=specification,
    )
    torch.sum(output_torch * upstream_torch).backward()
    gradient_torch = (
        source_torch.grad[0].permute(1, 2, 0).detach().numpy()
    )

    difference = gradient_torch - gradient_tf
    relative_l2 = float(
        np.linalg.norm(difference)
        / (np.linalg.norm(gradient_tf) + 1e-12)
    )
    cosine = float(
        np.vdot(gradient_torch, gradient_tf)
        / (
            np.linalg.norm(gradient_torch)
            * np.linalg.norm(gradient_tf)
            + 1e-12
        )
    )

    assert relative_l2 <= 1e-5
    assert cosine >= 0.99999


def test_center_crop_audit_covers_frozen_cases_and_candidate_thresholds(
    tmp_path: Path,
) -> None:
    """Gate 2C 必须覆盖冻结图案并留下逐 case JSONL 证据。"""
    specification = CenterCropSpecification(
        input_resolution=224,
        output_resolution=224,
        crop_area=0.9,
    )

    results = run_center_crop_equivalence_audit(
        specification=specification,
    )

    assert {result["case_name"] for result in results} == {
        "forward_spatial_ramp",
        "forward_checkerboard",
        "forward_center_impulse",
        "forward_crop_boundary_impulse",
        "forward_random_rgb",
        "vjp_spatial_ramp",
        "vjp_center_impulse",
        "vjp_boundary_impulse",
        "vjp_corner_impulse",
        "vjp_random",
    }
    assert all(
        result["schema_version"] == CENTER_CROP_AUDIT_SCHEMA_VERSION
        for result in results
    )
    assert all(result["candidate_pass"] for result in results)
    assert all(result["relative_l2"] <= 1e-5 for result in results)
    assert all(result["cosine"] >= 0.99999 for result in results)

    output_path = tmp_path / "center_crop_metrics.jsonl"
    output_sha256 = write_center_crop_audit_jsonl(
        results,
        output_path=output_path,
    )
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results)
    assert len(output_sha256) == 64
