"""OpenVLA-OFT 直接 Active Texture 迁移辅助测试。"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


OFT_LIBERO_DIR = (
    Path(__file__).resolve().parents[3]
    / "openvla-oft/experiments/robot/libero"
)
sys.path.insert(0, str(OFT_LIBERO_DIR))

from transfer_evaluation import (  # noqa: E402
    activate_texture_in_xml,
    parse_eval_state_ids,
    validate_active_texture,
)


def test_validates_png_without_resampling_pixels(tmp_path: Path) -> None:
    texture_path = tmp_path / "attack.png"
    pixels = np.asarray(
        [[[0, 127, 255], [64, 128, 192]]],
        dtype=np.uint8,
    )
    Image.fromarray(pixels).save(texture_path)
    original_bytes = texture_path.read_bytes()

    info = validate_active_texture(texture_path)

    assert info.path == texture_path.resolve()
    assert (info.width, info.height) == (2, 1)
    assert len(info.sha256) == 64
    assert texture_path.read_bytes() == original_bytes


def test_activates_exact_external_path_in_xml(tmp_path: Path) -> None:
    xml_path = tmp_path / "object.xml"
    xml_path.write_text(
        """<mujoco><asset>
        <texture name="tex-akita_black_bowl" file="clean.png"/>
        <material name="mat-akita_black_bowl" texuniform="true"/>
        </asset></mujoco>""",
        encoding="utf-8",
    )
    texture_path = tmp_path / "attack.png"
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(texture_path)

    activate_texture_in_xml(
        xml_path=xml_path,
        object_name="akita_black_bowl",
        active_texture_path=texture_path,
    )

    root = ET.parse(xml_path).getroot()
    texture_element = root.find(".//texture")
    material_element = root.find(".//material")
    assert texture_element is not None
    assert material_element is not None
    assert texture_element.get("file") == str(texture_path.resolve())
    assert texture_element.get("type") == "2d"
    assert material_element.get("texuniform") == "false"


def test_parses_held_out_eval_states() -> None:
    assert parse_eval_state_ids(
        "10-49",
        total_states=50,
        maximum_count=10,
    ) == tuple(range(10, 20))


def test_rejects_out_of_range_eval_state() -> None:
    with pytest.raises(ValueError, match="越界"):
        parse_eval_state_ids(
            "49-50",
            total_states=50,
            maximum_count=10,
        )
