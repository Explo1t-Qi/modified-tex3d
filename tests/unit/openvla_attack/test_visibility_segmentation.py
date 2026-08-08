"""MuJoCo segmentation schema 与 instance mapping 的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.visibility_segmentation import (  # noqa: E402
    TargetInstanceRoot,
    build_instance_geometry_mappings,
    parse_instance_segmentation,
)


GEOM_TYPE = 5
BACKGROUND_TYPE = -1


def _model() -> SimpleNamespace:
    # world=0；实例A root=1, child=2；实例B root=3, child=4。
    return SimpleNamespace(
        nbody=5,
        ngeom=6,
        body_parentid=np.array([0, 0, 1, 0, 3], dtype=np.int32),
        geom_bodyid=np.array([0, 1, 2, 3, 4, 4], dtype=np.int32),
    )


def _roots() -> tuple[TargetInstanceRoot, ...]:
    return (
        TargetInstanceRoot(body_id=1, body_name="bowl_1"),
        TargetInstanceRoot(body_id=3, body_name="bowl_2"),
    )


def test_instance_mapping_includes_recursive_bodies_and_geometries() -> None:
    mappings = build_instance_geometry_mappings(_model(), _roots())

    assert mappings[0].subtree_body_ids == (1, 2)
    assert mappings[0].geometry_ids == (1, 2)
    assert mappings[0].geometry_body_pairs == ((1, 1), (2, 2))
    assert mappings[1].subtree_body_ids == (3, 4)
    assert mappings[1].geometry_ids == (3, 4, 5)
    assert mappings[1].geometry_body_pairs == (
        (3, 3),
        (4, 4),
        (5, 4),
    )


def test_parse_segmentation_splits_one_frontmost_id_map_by_instance() -> None:
    # [object_type, object_id]；geom0是非目标，-1是背景。
    segmentation = np.array(
        [
            [[GEOM_TYPE, 1], [GEOM_TYPE, 2], [GEOM_TYPE, 0]],
            [[GEOM_TYPE, 3], [GEOM_TYPE, 5], [BACKGROUND_TYPE, -1]],
        ],
        dtype=np.int32,
    )

    parsed = parse_instance_segmentation(
        segmentation,
        model=_model(),
        instance_roots=_roots(),
        geom_object_type=GEOM_TYPE,
    )

    expected_alpha = torch.tensor(
        [
            [[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        parsed.instance_alpha,
        expected_alpha,
        rtol=0,
        atol=0,
    )
    assert parsed.object_type_histogram == {-1: 1, GEOM_TYPE: 5}
    assert parsed.object_id_histograms_by_type == {
        -1: {-1: 1},
        GEOM_TYPE: {0: 1, 1: 1, 2: 1, 3: 1, 5: 1},
    }
    assert parsed.geometry_id_histogram == {0: 1, 1: 1, 2: 1, 3: 1, 5: 1}
    assert parsed.non_geometry_pixel_count == 1
    assert parsed.non_target_geometry_pixel_count == 1
    assert bool((parsed.instance_alpha.sum(dim=0) <= 1).all())


@pytest.mark.parametrize(
    ("segmentation", "message"),
    [
        (np.zeros((2, 2), dtype=np.int32), "shape"),
        (np.zeros((2, 2, 3), dtype=np.int32), "shape"),
        (np.zeros((2, 2, 2), dtype=np.float32), "必须为整数"),
        (
            np.array([[[GEOM_TYPE, 99]]], dtype=np.int32),
            "越界 geom ID",
        ),
    ],
)
def test_parse_segmentation_rejects_invalid_backend_schema(
    segmentation: np.ndarray,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        parse_instance_segmentation(
            segmentation,
            model=_model(),
            instance_roots=_roots(),
            geom_object_type=GEOM_TYPE,
        )


def test_instance_mapping_rejects_nested_instance_roots() -> None:
    with pytest.raises(ValueError, match="subtrees 重叠"):
        build_instance_geometry_mappings(
            _model(),
            (
                TargetInstanceRoot(body_id=1, body_name="parent"),
                TargetInstanceRoot(body_id=2, body_name="child"),
            ),
        )


def test_instance_mapping_rejects_instance_without_geometry() -> None:
    model = SimpleNamespace(
        nbody=2,
        ngeom=1,
        body_parentid=np.array([0, 0], dtype=np.int32),
        geom_bodyid=np.array([0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="没有 geom"):
        build_instance_geometry_mappings(
            model,
            (TargetInstanceRoot(body_id=1, body_name="empty"),),
        )
