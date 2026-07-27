"""从 source-only 梯度审计中生成 OpenVLA 非连续谱基产物。

示例：

    python scripts/select_openvla_spectral_basis.py \
      --candidate-basis experiments/spectral_basis/akita_black_bowl_k512.npz \
      --audit /path/to/Ep0_Spectral_Gradient_Audit.npz \
      --objective feature \
      --top-k 128 \
      --expected-state-ids 0-9 \
      --output experiments/spectral_basis/akita_black_bowl_feature_top128.npz

脚本只读取 NumPy 产物并在 CPU 上复制谱基列，不加载 VLA 或 LIBERO。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openvla.experiments.robot.libero.openvla_attack.spectral_basis_selection import (
    GradientSelectedBasisResult,
    SelectionObjective,
    build_gradient_selected_spectral_basis,
)
from openvla.experiments.robot.libero.openvla_attack.state_selection import (
    parse_state_ids,
)


DEFAULT_TOP_K: Final[int] = 128


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析非连续谱基选择参数。"""
    parser = argparse.ArgumentParser(
        description="按 source-only stable score 选择曲面谱基",
    )
    parser.add_argument(
        "--candidate-basis",
        type=Path,
        required=True,
        help="包含完整候选模态的 K=M 谱基 NPZ",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
        help="同一候选谱基产生的 Spectral Gradient Audit NPZ",
    )
    parser.add_argument(
        "--objective",
        choices=("action", "feature"),
        default="feature",
        help="使用哪一项独立 stable-score 排名，默认 feature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="选择的非恒定谱基数量，默认 128",
    )
    parser.add_argument(
        "--expected-state-ids",
        type=str,
        required=True,
        help="审计必须精确匹配的 source states，例如 0-9",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出的梯度选择谱基 .npz",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """生成选基产物并打印可复查摘要。"""
    args: argparse.Namespace = parse_args(argv)
    expected_state_ids = parse_state_ids(
        args.expected_state_ids,
        field_name="expected_state_ids",
    )
    if expected_state_ids is None:
        raise ValueError("--expected-state-ids 不能为空")
    objective: SelectionObjective = args.objective
    result: GradientSelectedBasisResult = (
        build_gradient_selected_spectral_basis(
            candidate_basis_path=args.candidate_basis,
            audit_path=args.audit,
            output_path=args.output,
            objective=objective,
            top_k=args.top_k,
            expected_state_ids=expected_state_ids,
        )
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "selection_scope": "source_openvla_only",
                "objective": objective,
                "num_selected_basis": result.num_selected_basis,
                "source_state_ids": result.source_state_ids.tolist(),
                "selected_source_mode_indices": (
                    result.selected_source_mode_indices.tolist()
                ),
                "max_m_orthogonality_error": (
                    result.mass_orthogonality_error
                ),
                "candidate_sha256": result.candidate_sha256,
                "audit_sha256": result.audit_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
