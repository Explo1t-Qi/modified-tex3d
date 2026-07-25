"""OpenVLA-OFT 对外部 Attack Artifact 的直接迁移评估辅助。

迁移实验的目标是检验同一张 bake PNG 在目标模型上的效果。PNG 已经是完整的
UV texture，不应再经过 PNG→顶点颜色→PNG，也不应在 rollout 时用
nvdiffrast 合成图替换 MuJoCo policy observation。本模块维护这两个约束，并
提供可独立测试的状态索引与 XML 激活逻辑。
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias

from PIL import Image


PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class ActiveTextureInfo:
    """经过验证、可直接激活的外部 PNG。"""

    path: Path
    width: int
    height: int
    sha256: str


def validate_active_texture(texture_path: PathLike) -> ActiveTextureInfo:
    """验证外部 Attack Artifact 是可读取的 RGB PNG，并计算内容哈希。"""
    resolved_path: Path = Path(texture_path).resolve()
    if resolved_path.suffix.lower() != ".png":
        raise ValueError(
            "直接 MuJoCo Active Texture 迁移评估只接受 .png 产物"
        )
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    with Image.open(resolved_path) as image:
        image.verify()
    with Image.open(resolved_path) as image:
        rgb_image: Image.Image = image.convert("RGB")
        width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ValueError("Active Texture 的图像尺寸必须为正数")

    digest = hashlib.sha256()
    with resolved_path.open("rb") as texture_file:
        for chunk in iter(lambda: texture_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return ActiveTextureInfo(
        path=resolved_path,
        width=width,
        height=height,
        sha256=digest.hexdigest(),
    )


def activate_texture_in_xml(
    *,
    xml_path: PathLike,
    object_name: str,
    active_texture_path: PathLike,
) -> None:
    """让目标物体 material 引用给定 Active Texture。

    该函数只更新调用方已经纳入备份事务的 XML，不复制或修改纹理文件。
    """
    resolved_xml_path: Path = Path(xml_path)
    resolved_texture_path: Path = Path(active_texture_path).resolve()
    tree: ET.ElementTree = ET.parse(resolved_xml_path)
    root: ET.Element = tree.getroot()
    texture_name: str = f"tex-{object_name}"
    material_name: str = f"mat-{object_name}"

    texture_found: bool = False
    for texture_element in root.findall(".//texture"):
        if texture_element.get("name") == texture_name:
            texture_element.set("file", str(resolved_texture_path))
            texture_element.set("type", "2d")
            texture_found = True
    if not texture_found:
        raise ValueError(
            f"{resolved_xml_path} 中未找到 texture {texture_name!r}"
        )

    for material_element in root.findall(".//material"):
        if material_element.get("name") == material_name:
            material_element.set("texuniform", "false")
    tree.write(resolved_xml_path)


def parse_eval_state_ids(
    specification: Optional[str],
    *,
    total_states: int,
    maximum_count: int,
) -> tuple[int, ...]:
    """解析 held-out eval 状态，支持 ``"10-49"`` 和逗号组合。

    未显式提供时保持 OFT 旧行为，从 state 0 开始。谱迁移正式实验应显式使用
    ``10-49``。
    """
    if total_states <= 0:
        raise ValueError("当前 task 没有初始状态")
    if maximum_count <= 0:
        raise ValueError("maximum_count 必须为正数")
    if specification is None:
        return tuple(range(min(total_states, maximum_count)))

    state_ids: list[int] = []
    for raw_token in specification.split(","):
        token: str = raw_token.strip()
        if not token:
            raise ValueError("eval_init_state_ids 包含空 token")
        if "-" in token:
            fields: list[str] = token.split("-")
            if len(fields) != 2:
                raise ValueError(f"状态区间 {token!r} 无效")
            start_id: int = int(fields[0])
            end_id: int = int(fields[1])
            if start_id < 0 or end_id < start_id:
                raise ValueError(f"状态区间 {token!r} 必须非负且递增")
            state_ids.extend(range(start_id, end_id + 1))
        else:
            state_id: int = int(token)
            if state_id < 0:
                raise ValueError("状态 ID 不能为负数")
            state_ids.append(state_id)

    if len(state_ids) != len(set(state_ids)):
        raise ValueError("eval_init_state_ids 包含重复 ID")
    selected_ids: tuple[int, ...] = tuple(state_ids[:maximum_count])
    out_of_range_ids: list[int] = [
        state_id
        for state_id in selected_ids
        if state_id >= total_states
    ]
    if out_of_range_ids:
        raise ValueError(
            f"评估状态越界（总数 {total_states}）: {out_of_range_ids}"
        )
    if not selected_ids:
        raise ValueError("没有可用评估状态")
    return selected_ids
