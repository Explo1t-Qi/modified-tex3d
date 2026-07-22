"""OpenVLA LIBERO 资产注册表与 mesh 元数据解析测试。"""

from __future__ import annotations

import sys
from pathlib import Path


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.assets import OBJECT_ASSETS, parse_mesh_scale


def test_mesh_scale_parsing_preserves_scalar_vector_and_default_rules(
    tmp_path: Path,
) -> None:
    cases: dict[str, tuple[str, list[float]]] = {
        "vector.xml": (
            '<mujoco><asset><mesh scale="1 2 3"/></asset></mujoco>',
            [1.0, 2.0, 3.0],
        ),
        "scalar.xml": (
            '<mujoco><asset><mesh scale="0.5"/></asset></mujoco>',
            [0.5, 0.5, 0.5],
        ),
        "default.xml": (
            "<mujoco><asset><mesh/></asset></mujoco>",
            [1.0, 1.0, 1.0],
        ),
    }

    for filename, (xml_text, expected_scale) in cases.items():
        xml_path = tmp_path / filename
        xml_path.write_text(xml_text, encoding="utf-8")
        assert parse_mesh_scale(xml_path) == expected_scale


def test_asset_registry_preserves_spatial_and_object_task_mapping() -> None:
    assert len(OBJECT_ASSETS) == 11

    spatial_bowl = OBJECT_ASSETS["akita_black_bowl"]
    assert spatial_bowl["task_suite"] == "libero_spatial"
    assert spatial_bowl["task_id"] == 0
    assert spatial_bowl["search"] == [["akita_black_bowl"], ["bowl"]]
    assert spatial_bowl["mesh"].endswith("/akita_black_bowl/akita_black_bowl.obj")

    object_specs = {
        name: spec
        for name, spec in OBJECT_ASSETS.items()
        if spec["task_suite"] == "libero_object"
    }
    assert len(object_specs) == 10
    assert {spec["task_id"] for spec in object_specs.values()} == set(range(10))
    assert object_specs["cream_cheese"]["mesh"].endswith(
        "/cream_cheese/cream_cheese.obj"
    )
    assert object_specs["butter"]["mesh"].endswith("/butter/butter.obj")
