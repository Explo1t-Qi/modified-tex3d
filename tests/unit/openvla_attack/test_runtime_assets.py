"""LIBERO 运行时 XML 与真实纹理事务的单元测试。"""

from __future__ import annotations

import signal
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import pytest


LIBERO_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3] / "openvla/experiments/robot/libero"
)
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

import openvla_attack.runtime_assets as runtime_assets
from openvla_attack.runtime_assets import RuntimeAssetTransaction


def _write_clean_xml(xml_path: Path, texture_path: Path) -> str:
    """写入包含目标 texture/material 的最小 MuJoCo XML。"""
    xml_text = (
        "<mujoco><asset>"
        '<texture name="tex-akita_black_bowl" '
        f'file="{texture_path}" type="2d" />'
        '<material name="mat-akita_black_bowl" texuniform="true" />'
        "</asset></mujoco>"
    )
    xml_path.write_text(xml_text)
    return xml_text


def _read_target_xml(xml_path: Path) -> tuple[ET.Element, ET.Element]:
    """返回测试 XML 中唯一的 texture 和 material 元素。"""
    root: ET.Element = ET.parse(xml_path).getroot()
    texture_element: ET.Element | None = root.find("asset/texture")
    material_element: ET.Element | None = root.find("asset/material")
    assert texture_element is not None
    assert material_element is not None
    return texture_element, material_element


def test_transaction_activates_and_restores_xml_and_real_texture(
    tmp_path: Path,
) -> None:
    xml_path: Path = tmp_path / "akita.xml"
    real_texture_path: Path = tmp_path / "clean.png"
    clean_xml: str = _write_clean_xml(xml_path, real_texture_path)
    real_texture_path.write_bytes(b"clean-texture")

    transaction: RuntimeAssetTransaction = RuntimeAssetTransaction.begin(
        xml_path=xml_path,
        real_texture_path=real_texture_path,
        object_name="akita_black_bowl",
        backup_tag="2026_07_23-12_00_00",
        install_process_handlers=False,
    )

    assert transaction.xml_backup_path.name == (
        "akita_clean_backup_2026_07_23-12_00_00.xml"
    )
    assert transaction.real_texture_backup_path is not None
    assert transaction.real_texture_backup_path.name == (
        "texture_clean_backup_2026_07_23-12_00_00.png"
    )
    assert transaction.xml_backup_path.read_text() == clean_xml
    assert (
        transaction.real_texture_backup_path.read_bytes()
        == b"clean-texture"
    )

    live_texture_path: Path = tmp_path / "live.png"
    live_texture_path.write_bytes(b"live-texture")
    mirrored: bool = transaction.activate_texture(
        live_texture_path,
        mirror_real_texture=False,
    )
    texture_element, material_element = _read_target_xml(xml_path)
    assert texture_element.get("file") == str(live_texture_path.resolve())
    assert texture_element.get("type") == "2d"
    assert material_element.get("texuniform") == "false"
    assert mirrored is False
    assert real_texture_path.read_bytes() == b"clean-texture"

    trained_texture_path: Path = tmp_path / "trained.png"
    trained_texture_path.write_bytes(b"trained-texture")
    mirrored = transaction.activate_texture(
        trained_texture_path,
        mirror_real_texture=True,
    )
    assert mirrored is True
    assert real_texture_path.read_bytes() == b"trained-texture"

    transaction.restore(context="Before Task 1", remove_backups=False)
    assert xml_path.read_text() == clean_xml
    assert real_texture_path.read_bytes() == b"clean-texture"
    assert transaction.xml_backup_path.exists()
    assert transaction.real_texture_backup_path.exists()

    transaction.close(context="Final cleanup")
    assert xml_path.read_text() == clean_xml
    assert real_texture_path.read_bytes() == b"clean-texture"
    assert not transaction.xml_backup_path.exists()
    assert not transaction.real_texture_backup_path.exists()

    # close 必须幂等，finally 与 atexit 同时触发时不应再次操作已删除的备份。
    transaction.close(context="Repeated cleanup")


def test_transaction_activates_hope_texture_by_xml_reference(
    tmp_path: Path,
) -> None:
    """HOPE object 的 XML 节点名不一定包含 object_name。"""
    xml_path: Path = tmp_path / "alphabet_soup.xml"
    real_texture_path: Path = tmp_path / "texture_map.png"
    clean_xml: str = (
        "<mujoco><asset>"
        '<texture name="tex-textured" file="texture_map.png" type="2d" />'
        '<material name="textured" texture="tex-textured" '
        'texuniform="true" />'
        "</asset></mujoco>"
    )
    xml_path.write_text(clean_xml)
    real_texture_path.write_bytes(b"clean-texture")

    transaction: RuntimeAssetTransaction = RuntimeAssetTransaction.begin(
        xml_path=xml_path,
        real_texture_path=real_texture_path,
        object_name="alphabet_soup",
        backup_tag="test",
        install_process_handlers=False,
    )
    adversarial_texture_path: Path = tmp_path / "adversarial.png"
    adversarial_texture_path.write_bytes(b"adversarial")

    transaction.activate_texture(
        adversarial_texture_path,
        mirror_real_texture=False,
    )

    texture_element, material_element = _read_target_xml(xml_path)
    assert texture_element.get("file") == str(
        adversarial_texture_path.resolve()
    )
    assert material_element.get("texuniform") == "false"

    transaction.close(context="Final cleanup")
    assert xml_path.read_text() == clean_xml


def test_transaction_keeps_missing_real_texture_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xml_path: Path = tmp_path / "akita.xml"
    missing_texture_path: Path = tmp_path / "missing.png"
    clean_xml: str = _write_clean_xml(xml_path, missing_texture_path)
    adversarial_texture_path: Path = tmp_path / "adversarial.png"
    adversarial_texture_path.write_bytes(b"adversarial")

    transaction: RuntimeAssetTransaction = RuntimeAssetTransaction.begin(
        xml_path=xml_path,
        real_texture_path=missing_texture_path,
        object_name="akita_black_bowl",
        backup_tag="test",
        install_process_handlers=False,
    )
    mirrored: bool = transaction.activate_texture(
        adversarial_texture_path,
        mirror_real_texture=True,
    )

    assert transaction.real_texture_backup_path is None
    assert mirrored is False
    assert not missing_texture_path.exists()
    assert "Real MuJoCo texture not found" in capsys.readouterr().out

    transaction.close(context="Final cleanup")
    assert xml_path.read_text() == clean_xml


def test_signal_handler_restores_assets_and_previous_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xml_path: Path = tmp_path / "akita.xml"
    texture_path: Path = tmp_path / "texture.png"
    clean_xml: str = _write_clean_xml(xml_path, texture_path)
    texture_path.write_bytes(b"clean")

    previous_handlers: dict[int, object] = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
    }
    current_handlers: dict[int, object] = dict(previous_handlers)
    registered_exit_handlers: list[Callable[[], None]] = []
    unregistered_exit_handlers: list[Callable[[], None]] = []

    def fake_getsignal(signal_number: int) -> object:
        return current_handlers[signal_number]

    def fake_signal(signal_number: int, handler: object) -> object:
        previous: object = current_handlers[signal_number]
        current_handlers[signal_number] = handler
        return previous

    def fake_register(handler: Callable[[], None]) -> Callable[[], None]:
        registered_exit_handlers.append(handler)
        return handler

    def fake_unregister(handler: Callable[[], None]) -> None:
        unregistered_exit_handlers.append(handler)

    monkeypatch.setattr(runtime_assets.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(runtime_assets.signal, "signal", fake_signal)
    monkeypatch.setattr(runtime_assets.atexit, "register", fake_register)
    monkeypatch.setattr(runtime_assets.atexit, "unregister", fake_unregister)

    transaction: RuntimeAssetTransaction = RuntimeAssetTransaction.begin(
        xml_path=xml_path,
        real_texture_path=texture_path,
        object_name="akita_black_bowl",
        backup_tag="test",
    )
    adversarial_texture_path: Path = tmp_path / "adversarial.png"
    adversarial_texture_path.write_bytes(b"adversarial")
    transaction.activate_texture(
        adversarial_texture_path,
        mirror_real_texture=True,
    )

    assert len(registered_exit_handlers) == 1
    installed_handler: Any = current_handlers[signal.SIGTERM]
    with pytest.raises(SystemExit, match="Caught signal"):
        installed_handler(signal.SIGTERM, None)

    assert xml_path.read_text() == clean_xml
    assert texture_path.read_bytes() == b"clean"
    assert not transaction.xml_backup_path.exists()
    assert transaction.real_texture_backup_path is not None
    assert not transaction.real_texture_backup_path.exists()
    assert current_handlers == previous_handlers
    assert unregistered_exit_handlers == registered_exit_handlers
