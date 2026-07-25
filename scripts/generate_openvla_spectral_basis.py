"""为 OpenVLA Tex3D 目标物体生成曲面谱基 NPZ 产物。

示例：

    python scripts/generate_openvla_spectral_basis.py \
      --mesh /path/to/akita_black_bowl.obj \
      --num-basis 128 \
      --output /path/to/akita_black_bowl_k128.npz

脚本只进行 CPU 稀疏特征分解，不加载 VLA 模型或占用 GPU。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

# 直接以 ``python scripts/...`` 调用时，Python 只把 scripts/ 放入 sys.path。
# 显式加入仓库根目录，使脚本不依赖调用方额外配置 PYTHONPATH。
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openvla.experiments.robot.libero.openvla_attack.spectral_geometry import (
    build_cotangent_laplacian_and_mass,
    load_obj_geometry,
    save_spectral_basis,
    solve_manifold_harmonics,
    spectral_numerical_errors,
)


DEFAULT_NUM_BASIS: Final[int] = 128


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成 Tex3D 使用的非恒定曲面谱基",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="用于谱分解的原始 OBJ 路径",
    )
    parser.add_argument(
        "--num-basis",
        type=int,
        default=DEFAULT_NUM_BASIS,
        help="非恒定谱基数量 K，默认 128",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 .npz 路径",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="ARPACK eigsh 收敛容差",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """生成谱基、报告数值误差并保存产物。"""
    args: argparse.Namespace = parse_args(argv)
    if args.num_basis <= 0:
        raise ValueError("--num-basis 必须为正数")

    # vertices: float64 [N, 3]；faces: int64 [F, 3]。
    vertices, faces = load_obj_geometry(args.mesh)
    laplacian, mass, valid_faces = build_cotangent_laplacian_and_mass(
        vertices,
        faces,
    )
    eigenvalues, basis_with_constant = solve_manifold_harmonics(
        laplacian,
        mass,
        args.num_basis,
        tolerance=args.tolerance,
    )
    orthogonality_error, residuals = spectral_numerical_errors(
        laplacian,
        mass,
        eigenvalues,
        basis_with_constant,
    )
    output_path: Path = save_spectral_basis(
        args.output,
        vertices=vertices,
        faces=faces,
        mass=mass,
        eigenvalues_with_constant=eigenvalues,
        basis_with_constant=basis_with_constant,
        metadata={
            "source_mesh": str(args.mesh.resolve()),
            "solver": "scipy.sparse.linalg.eigsh",
            "tolerance": float(args.tolerance),
            "num_degenerate_faces_removed": int(
                faces.shape[0] - valid_faces.shape[0]
            ),
            "max_m_orthogonality_error": orthogonality_error,
            "max_eigen_residual": float(residuals.max()),
        },
    )

    summary: dict[str, int | float | str] = {
        "output": str(output_path.resolve()),
        "num_geometry_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_nonconstant_basis": int(args.num_basis),
        "max_m_orthogonality_error": orthogonality_error,
        "max_eigen_residual": float(residuals.max()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
