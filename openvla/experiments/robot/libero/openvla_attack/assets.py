"""OpenVLA LIBERO 实验使用的目标物体资产注册表。

本模块只描述“一个可攻击物体在哪里、如何在 MuJoCo 中找到它、属于哪个任务”。
它不加载 mesh 或 texture，也不创建 LIBERO 环境。把资产元数据集中后，实验入口
不再同时承担路径拼接、对象到任务映射和 XML 元数据解析三种职责。

各字段的数据流为：

- ``xml`` → ``parse_mesh_scale`` / ``RuntimeAssetTransaction``；
- ``mesh`` → ``DifferentiableRenderer``；
- ``texture`` → ``DifferentiableRenderer`` / ``RuntimeAssetTransaction``；
- ``search`` → frame collector / episode runner，用于定位 MuJoCo body；
- ``task_suite`` / ``task_id`` → LIBERO benchmark。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final, Literal, Optional, TypeAlias, TypedDict


TaskSuiteName: TypeAlias = Literal["libero_spatial", "libero_object"]
MeshScale: TypeAlias = list[float]

# 该服务器上的 LIBERO 源码与资产根目录。路径仍作为 str 暴露，保持现有
# sys.path、ElementTree 和 renderer 调用处的运行时行为。
LIBERO_ROOT: Final[str] = "/home/xiaomengqi/src/github/paper_code/LIBERO"
SCANNED_ASSET_ROOT: Final[str] = (
    f"{LIBERO_ROOT}/libero/libero/assets/stable_scanned_objects"
)
HOPE_ASSET_ROOT: Final[str] = (
    f"{LIBERO_ROOT}/libero/libero/assets/stable_hope_objects"
)


class AssetFiles(TypedDict):
    """renderer 和 MuJoCo 使用的一组物体文件。"""

    xml: str
    mesh: str
    texture: str


class ObjectAssetSpec(AssetFiles):
    """一个目标物体的完整实验元数据。

    ``search`` 的形状可理解为 ``[num_fallbacks, num_keywords]``。外层列表按
    优先级依次尝试，内层字符串必须同时出现在 MuJoCo body name 中。
    """

    search: list[list[str]]
    task_suite: TaskSuiteName
    task_id: int


def _scanned_asset_files(
    name: str,
    mesh_file: Optional[str] = None,
    texture_file: str = "texture.png",
) -> AssetFiles:
    """构造 stable scanned object 的三个资产路径。"""
    asset_directory: str = f"{SCANNED_ASSET_ROOT}/{name}"
    resolved_mesh_file: str = mesh_file if mesh_file is not None else f"{name}.obj"
    return {
        "xml": f"{asset_directory}/{name}.xml",
        "mesh": f"{asset_directory}/{resolved_mesh_file}",
        "texture": f"{asset_directory}/{texture_file}",
    }


def _hope_asset_files(
    name: str,
    mesh_file: Optional[str] = None,
    texture_file: str = "texture_map.png",
) -> AssetFiles:
    """构造 stable HOPE object 的三个资产路径。"""
    asset_directory: str = f"{HOPE_ASSET_ROOT}/{name}"
    resolved_mesh_file: str = mesh_file if mesh_file is not None else "textured.obj"
    return {
        "xml": f"{asset_directory}/{name}.xml",
        "mesh": f"{asset_directory}/{resolved_mesh_file}",
        "texture": f"{asset_directory}/{texture_file}",
    }


def _object_asset_spec(
    files: AssetFiles,
    search: list[list[str]],
    task_suite: TaskSuiteName,
    task_id: int,
) -> ObjectAssetSpec:
    """把文件路径与 LIBERO 任务定位信息合并为一个注册表条目。"""
    return {
        "xml": files["xml"],
        "mesh": files["mesh"],
        "texture": files["texture"],
        "search": search,
        "task_suite": task_suite,
        "task_id": task_id,
    }


# 键是命令行 ``--object_name`` 接受的稳定名称。Spatial task 0 使用 scanned
# bowl；Object suite 的 task 0-9 分别使用对应 HOPE 物体。
OBJECT_ASSETS: Final[dict[str, ObjectAssetSpec]] = {
    "akita_black_bowl": _object_asset_spec(
        _scanned_asset_files("akita_black_bowl"),
        search=[["akita_black_bowl"], ["bowl"]],
        task_suite="libero_spatial",
        task_id=0,
    ),
    "alphabet_soup": _object_asset_spec(
        _hope_asset_files("alphabet_soup", mesh_file="textured.obj"),
        search=[["alphabet_soup"], ["soup"]],
        task_suite="libero_object",
        task_id=0,
    ),
    "cream_cheese": _object_asset_spec(
        _hope_asset_files("cream_cheese", mesh_file="cream_cheese.obj"),
        search=[["cream_cheese"], ["cheese"]],
        task_suite="libero_object",
        task_id=1,
    ),
    "salad_dressing": _object_asset_spec(
        _hope_asset_files("salad_dressing", mesh_file="textured.obj"),
        search=[["salad_dressing"], ["dressing"]],
        task_suite="libero_object",
        task_id=2,
    ),
    "bbq_sauce": _object_asset_spec(
        _hope_asset_files("bbq_sauce", mesh_file="bbq_sauce.obj"),
        search=[["bbq_sauce"], ["bbq"], ["sauce"]],
        task_suite="libero_object",
        task_id=3,
    ),
    "ketchup": _object_asset_spec(
        _hope_asset_files("ketchup", mesh_file="textured.obj"),
        search=[["ketchup"]],
        task_suite="libero_object",
        task_id=4,
    ),
    "tomato_sauce": _object_asset_spec(
        _hope_asset_files("tomato_sauce"),
        search=[["tomato_sauce"], ["tomato"]],
        task_suite="libero_object",
        task_id=5,
    ),
    "butter": _object_asset_spec(
        _hope_asset_files("butter", mesh_file="butter.obj"),
        search=[["butter"]],
        task_suite="libero_object",
        task_id=6,
    ),
    "milk": _object_asset_spec(
        _hope_asset_files("milk", mesh_file="textured.obj"),
        search=[["milk"]],
        task_suite="libero_object",
        task_id=7,
    ),
    "chocolate_pudding": _object_asset_spec(
        _hope_asset_files("chocolate_pudding", mesh_file="textured.obj"),
        search=[["chocolate_pudding"], ["chocolate"], ["pudding"]],
        task_suite="libero_object",
        task_id=8,
    ),
    "orange_juice": _object_asset_spec(
        _hope_asset_files("orange_juice", mesh_file="textured.obj"),
        search=[["orange_juice"], ["orange"], ["juice"]],
        task_suite="libero_object",
        task_id=9,
    ),
}


def parse_mesh_scale(xml_path: str | Path) -> MeshScale:
    """读取 MuJoCo XML 中第一个带 ``scale`` 的 mesh 缩放值。

    Args:
        xml_path: 目标物体的 MuJoCo XML 文件路径。

    Returns:
        长度始终为 3 的 ``list[float]``，依次对应 mesh 的 ``[x, y, z]``
        缩放。XML 提供三个值时原样返回；提供其他数量时沿用原实现，将第一个
        值复制到三轴；没有任何 mesh scale 时返回 ``[1.0, 1.0, 1.0]``。

    Raises:
        xml.etree.ElementTree.ParseError: XML 内容无法解析。
        ValueError: scale 中存在无法转换为浮点数的文本。
        IndexError: scale 属性只有空白字符，无法取得第一个缩放值。以上错误
            保持底层解析异常，不在资产注册层静默修复错误文件。
    """
    xml_tree: ET.ElementTree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    for mesh_element in xml_root.findall(".//mesh"):
        scale_text: Optional[str] = mesh_element.get("scale")
        if scale_text:
            scale_values: list[float] = [
                float(value) for value in scale_text.strip().split()
            ]
            if len(scale_values) == 3:
                return scale_values

            uniform_scale: float = scale_values[0]
            return [uniform_scale, uniform_scale, uniform_scale]

    return [1.0, 1.0, 1.0]
