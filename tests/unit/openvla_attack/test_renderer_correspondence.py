"""nvdiffrast raster 到几何 Support correspondence 的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.renderer_correspondence import (  # noqa: E402
    decode_raster_support_correspondence,
)


def _topology() -> tuple[torch.Tensor, torch.Tensor]:
    # Renderer face 采用非平凡顺序；首个 face 的三个 corner 映射到不同几何
    # 顶点，以侦测任意 corner 互换。renderer vertex 1/4 另为同一 geometry
    # vertex 的 UV seam 副本，用第二个 face 锁定 seam 映射语义。
    faces = torch.tensor([[3, 1, 2], [0, 2, 4]], dtype=torch.int32)
    render_to_geometry = torch.tensor([0, 1, 3, 2, 1], dtype=torch.int64)
    return faces, render_to_geometry


@pytest.mark.parametrize(
    ("uv", "support_vertex", "expected_geometry_corners"),
    [
        ((1.0, 0.0), 2, [2, 1, 3]),
        ((0.0, 1.0), 1, [2, 1, 3]),
        ((0.0, 0.0), 3, [2, 1, 3]),
    ],
)
def test_corner_one_hot_matches_face_triplet_order(
    uv: tuple[float, float],
    support_vertex: int,
    expected_geometry_corners: list[int],
) -> None:
    faces, mapping = _topology()
    raster = torch.tensor([[[[uv[0], uv[1], 0.0, 1.0]]]])
    support = torch.zeros(4, dtype=torch.bool)
    support[support_vertex] = True

    result = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        support,
    )

    assert result.triangle_indices.tolist() == [[[0]]]
    assert result.geometry_corner_indices.tolist() == [
        [[expected_geometry_corners]]
    ]
    assert result.valid_mask.tolist() == [[[[True]]]]
    assert result.support_control.item() == 1.0


def test_triangle_id_is_one_based_and_selects_second_face() -> None:
    faces, mapping = _topology()
    raster = torch.tensor([[[[0.25, 0.50, 0.0, 2.0]]]])
    support = torch.tensor([True, False, False, False])

    result = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        support,
    )

    assert result.triangle_indices.item() == 1
    assert result.barycentric.tolist() == [[[[0.25, 0.5, 0.25]]]]
    assert result.geometry_corner_indices.tolist() == [[[[0, 3, 1]]]]
    assert result.support_control.item() == 0.25


def test_background_is_explicitly_invalid_with_zero_evidence() -> None:
    faces, mapping = _topology()
    raster = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    support = torch.ones(4, dtype=torch.bool)

    result = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        support,
    )

    assert result.triangle_indices.tolist() == [[[-1, -1]]]
    assert not bool(result.valid_mask.any())
    assert not bool(result.barycentric.any())
    assert (result.geometry_corner_indices == -1).all()
    assert not bool(result.support_control.any())


def test_empty_and_full_support_are_zero_and_one_on_valid_pixels() -> None:
    faces, mapping = _topology()
    raster = torch.tensor(
        [[[[0.2, 0.3, 0.0, 1.0], [0.3, 0.4, 0.0, 2.0]]]]
    )

    empty = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        torch.zeros(4, dtype=torch.bool),
    )
    full = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        torch.ones(4, dtype=torch.bool),
    )

    assert torch.equal(empty.support_control, torch.zeros_like(empty.support_control))
    assert torch.allclose(full.support_control, torch.ones_like(full.support_control))


def test_support_inclusion_is_pixelwise_monotone() -> None:
    faces, mapping = _topology()
    raster = torch.tensor(
        [[[[0.2, 0.3, 0.0, 1.0], [0.3, 0.4, 0.0, 2.0]]]]
    )
    small_support = torch.tensor([False, False, True, False])
    large_support = torch.tensor([True, True, True, False])

    small = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        small_support,
    )
    large = decode_raster_support_correspondence(
        raster,
        faces,
        mapping,
        large_support,
    )

    assert bool((large.support_control >= small.support_control).all())
    assert bool((large.support_control > small.support_control).any())


@pytest.mark.parametrize(
    ("raster", "error"),
    [
        (
            torch.tensor([[[[0.2, 0.2, 0.0, 0.5]]]]),
            "triangle ID 不是容差内整数",
        ),
        (
            torch.tensor([[[[0.2, 0.2, 0.0, 3.0]]]]),
            "triangle ID 超出",
        ),
        (
            torch.tensor([[[[0.8, 0.8, 0.0, 1.0]]]]),
            "重心坐标超出",
        ),
    ],
)
def test_invalid_raster_fails_fast(
    raster: torch.Tensor,
    error: str,
) -> None:
    faces, mapping = _topology()

    with pytest.raises(ValueError, match=error):
        decode_raster_support_correspondence(
            raster,
            faces,
            mapping,
            torch.ones(4, dtype=torch.bool),
        )


def test_invalid_topology_or_support_fails_fast() -> None:
    faces, mapping = _topology()
    raster = torch.tensor([[[[0.2, 0.2, 0.0, 1.0]]]])

    with pytest.raises(ValueError, match="renderer 顶点索引"):
        decode_raster_support_correspondence(
            raster,
            torch.tensor([[3, 1, 9]], dtype=torch.int32),
            mapping,
            torch.ones(4, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="几何顶点索引"):
        decode_raster_support_correspondence(
            raster,
            faces,
            torch.tensor([0, 1, 1, 2, 8]),
            torch.ones(4, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="bool dtype"):
        decode_raster_support_correspondence(
            raster,
            faces,
            mapping,
            torch.ones(4, dtype=torch.float32),
        )
