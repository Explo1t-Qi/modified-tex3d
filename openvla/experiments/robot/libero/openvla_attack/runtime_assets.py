"""LIBERO 评估期间 XML 与真实纹理的运行时事务。

OpenVLA 攻击会把某张对抗纹理写入目标物体的 MuJoCo XML；为了让录制视频与
XML 引用保持一致，部分路径还会覆盖 LIBERO asset 目录中的真实纹理文件。
这些操作直接修改共享资源，必须在下一个 task、正常退出、异常退出或收到
SIGINT/SIGTERM 时恢复。

:class:`RuntimeAssetTransaction` 把以下实现细节隐藏在一个生命周期 interface
之后：

1. 建立 XML 和真实纹理的 clean backup；
2. 将对抗纹理注入指定 object 的 texture/material；
3. 按调用场景决定是否同步覆盖真实纹理；
4. 恢复资源、删除 backup，并复原进程原有的 signal handler。

实验入口只表达何时 ``activate``、``restore`` 和 ``close``。本 module 不负责
生成对抗纹理，也不负责保存实验产物。
"""

from __future__ import annotations

import atexit
import shutil
import signal
import xml.etree.ElementTree as ET
from pathlib import Path
from types import FrameType
from typing import Any, Callable, NoReturn, Optional, TypeAlias


PathLike: TypeAlias = str | Path
ExitHandler: TypeAlias = Callable[[], None]


class RuntimeAssetTransaction:
    """管理一次评估运行对 LIBERO 共享资源造成的临时修改。

    正确调用顺序为 ``begin -> activate/restore -> close``。``close`` 是幂等的，
    会恢复干净资源、删除 backup，并撤销本事务安装的进程 handler。
    """

    def __init__(
        self,
        *,
        xml_path: Path,
        real_texture_path: Path,
        object_name: str,
        backup_tag: str,
    ) -> None:
        self.xml_path: Path = xml_path
        self.real_texture_path: Path = real_texture_path
        self.xml_backup_path: Path = xml_path.with_name(
            f"{xml_path.stem}_clean_backup_{backup_tag}{xml_path.suffix}"
        )
        self.real_texture_backup_path: Optional[Path] = None

        self._texture_name: str = f"tex-{object_name}"
        self._material_name: str = f"mat-{object_name}"
        self._backup_tag: str = backup_tag
        self._closed: bool = False
        self._exit_handler: Optional[ExitHandler] = None
        self._previous_signal_handlers: dict[int, Any] = {}

    @classmethod
    def begin(
        cls,
        *,
        xml_path: PathLike,
        real_texture_path: PathLike,
        object_name: str,
        backup_tag: str,
        install_process_handlers: bool = True,
    ) -> "RuntimeAssetTransaction":
        """建立 clean backup，并可选安装 atexit/signal 清理。

        相对真实纹理路径继续按重构前行为，以当前工作目录为基准解析。XML
        不存在时在创建任何 backup 前抛出 :class:`FileNotFoundError`。
        """
        resolved_xml_path: Path = Path(xml_path)
        if not resolved_xml_path.exists():
            raise FileNotFoundError(
                f"XML asset not found at {resolved_xml_path}"
            )

        resolved_texture_path: Path = Path(real_texture_path)
        if not resolved_texture_path.is_absolute():
            resolved_texture_path = (
                Path.cwd() / resolved_texture_path
            ).resolve()

        transaction: RuntimeAssetTransaction = cls(
            xml_path=resolved_xml_path,
            real_texture_path=resolved_texture_path,
            object_name=object_name,
            backup_tag=backup_tag,
        )
        transaction._create_clean_backups()
        if install_process_handlers:
            transaction._install_process_handlers()
        return transaction

    def _create_clean_backups(self) -> None:
        """复制事务开始时的 XML 和真实纹理。"""
        shutil.copy(self.xml_path, self.xml_backup_path)

        if self.real_texture_path.exists():
            self.real_texture_backup_path = (
                self.real_texture_path.with_name(
                    f"texture_clean_backup_{self._backup_tag}.png"
                )
            )
            shutil.copy(
                self.real_texture_path,
                self.real_texture_backup_path,
            )
            return

        print(
            "[WARNING] Real MuJoCo texture not found at "
            f"{self.real_texture_path}, MuJoCo video may look clean"
        )

    def _install_process_handlers(self) -> None:
        """注册退出清理，并保存当前 SIGTERM/SIGINT handler。"""
        if self._exit_handler is not None:
            return

        exit_handler: ExitHandler = self._handle_process_exit
        self._exit_handler = exit_handler
        atexit.register(exit_handler)

        signal_number: int
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            self._previous_signal_handlers[signal_number] = (
                signal.getsignal(signal_number)
            )
            signal.signal(signal_number, self._handle_signal)

    def _uninstall_process_handlers(self) -> None:
        """撤销本事务注册的清理逻辑，恢复调用方原有 signal handler。"""
        if self._exit_handler is not None:
            atexit.unregister(self._exit_handler)
            self._exit_handler = None

        signal_number: int
        previous_handler: Any
        for signal_number, previous_handler in (
            self._previous_signal_handlers.items()
        ):
            signal.signal(signal_number, previous_handler)
        self._previous_signal_handlers.clear()

    def _handle_process_exit(self) -> None:
        """在 Python 正常退出阶段恢复并删除 backup。"""
        self.close(context="(exit handler)")

    def _handle_signal(
        self,
        signal_number: int,
        frame: Optional[FrameType],
    ) -> NoReturn:
        """处理中断信号；frame 只用于匹配 ``signal.signal`` interface。"""
        del frame
        self.close(context="(exit handler)")
        raise SystemExit(
            f"Caught signal {signal_number}, exiting cleanly."
        )

    def activate_texture(
        self,
        texture_path: PathLike,
        *,
        mirror_real_texture: bool,
    ) -> bool:
        """激活一张对抗纹理，并返回是否同步覆盖了真实纹理。

        XML 中目标 ``texture.file`` 被替换为对抗纹理的绝对路径，类型强制为
        ``2d``；引用该 texture 的 material 之 ``texuniform`` 被设为
        ``false``。节点优先通过 XML 中的真实文件引用和 texture/material
        关联定位，而不是假定节点名称一定包含 ``object_name``；HOPE object
        常使用 ``tex-textured`` / ``textured``。

        仅当 ``mirror_real_texture`` 为真，且事务开始时存在真实纹理（或运行中
        该路径后来出现）时才执行文件覆盖。这保留了缺失真实纹理时只修改 XML
        的现有行为。
        """
        resolved_texture_path: Path = Path(texture_path).resolve()
        tree: ET.ElementTree = ET.parse(self.xml_path)
        root: ET.Element = tree.getroot()

        # texture_elements: XML 中候选纹理节点，长度通常为 1。
        texture_elements: list[ET.Element] = root.findall("asset/texture")
        target_texture: Optional[ET.Element] = None
        real_texture_path: Path = self.real_texture_path.resolve()
        texture_element: ET.Element
        for texture_element in texture_elements:
            texture_file: Optional[str] = texture_element.get("file")
            if texture_file is None:
                continue
            referenced_path: Path = Path(texture_file)
            if not referenced_path.is_absolute():
                referenced_path = self.xml_path.parent / referenced_path
            if referenced_path.resolve() == real_texture_path:
                target_texture = texture_element
                break

        # override_texture_path 可能与 XML 原始 file 不同；保留旧命名作为 fallback。
        if target_texture is None:
            target_texture = next(
                (
                    element
                    for element in texture_elements
                    if element.get("name") == self._texture_name
                ),
                None,
            )
        if target_texture is None and len(texture_elements) == 1:
            target_texture = texture_elements[0]
        if target_texture is None:
            raise ValueError(
                "Unable to identify target texture in XML "
                f"{self.xml_path} for {self.real_texture_path}"
            )

        target_texture_name: Optional[str] = target_texture.get("name")
        target_texture.set("file", str(resolved_texture_path))
        target_texture.set("type", "2d")

        material_elements: list[ET.Element] = root.findall(".//material")
        target_material: Optional[ET.Element] = next(
            (
                element
                for element in material_elements
                if target_texture_name is not None
                and element.get("texture") == target_texture_name
            ),
            None,
        )
        if target_material is None:
            target_material = next(
                (
                    element
                    for element in material_elements
                    if element.get("name") == self._material_name
                ),
                None,
            )
        if target_material is None and len(material_elements) == 1:
            target_material = material_elements[0]
        if target_material is None:
            raise ValueError(
                "Unable to identify material referencing texture "
                f"{target_texture_name!r} in XML {self.xml_path}"
            )
        target_material.set("texuniform", "false")
        tree.write(self.xml_path)

        should_mirror: bool = (
            mirror_real_texture
            and (
                self.real_texture_path.exists()
                or self.real_texture_backup_path is not None
            )
        )
        if should_mirror:
            shutil.copy(
                resolved_texture_path,
                self.real_texture_path,
            )
            return True
        return False

    def restore(
        self,
        *,
        context: str,
        remove_backups: bool,
    ) -> None:
        """从 clean backup 恢复共享资源。

        ``remove_backups=False`` 用于 task 之间的恢复，backup 保留给后续 task；
        ``True`` 只应在最终清理时使用。
        """
        if self.xml_backup_path.exists():
            shutil.copy(self.xml_backup_path, self.xml_path)
            if remove_backups:
                self.xml_backup_path.unlink(missing_ok=True)
            print(f"[INFO] {context} Original XML restored.")

        real_texture_backup_path: Optional[Path] = (
            self.real_texture_backup_path
        )
        if (
            real_texture_backup_path is not None
            and real_texture_backup_path.exists()
        ):
            shutil.copy(
                real_texture_backup_path,
                self.real_texture_path,
            )
            if remove_backups:
                real_texture_backup_path.unlink(missing_ok=True)
            print(f"[INFO] {context} Real MuJoCo texture restored.")

    def close(self, *, context: str) -> None:
        """最终恢复资源、删除 backup 并复原进程 handler；可重复调用。"""
        if self._closed:
            return

        # 如果 restore 抛错，保留 handler 和未关闭状态，让 atexit 仍有机会重试。
        self.restore(context=context, remove_backups=True)
        self._uninstall_process_handlers()
        self._closed = True
