"""分析主视角 Action、主视角 Feature 与腕部 Feature 的谱系数梯度冲突。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _row_norms(vectors: FloatArray) -> FloatArray:
    """返回二维 ``[num_samples, flattened_dim]`` 向量的逐行 L2 norm。"""
    return np.linalg.norm(vectors, axis=1)


def _row_cosines(left: FloatArray, right: FloatArray) -> FloatArray:
    """安全计算同状态向量 cosine；零向量对应值记为0。"""
    denominator: FloatArray = _row_norms(left) * _row_norms(right)
    numerator: FloatArray = np.sum(left * right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).tiny,
    )


def _cross_state_consistency(vectors: FloatArray) -> float:
    """计算 ``||mean(g)|| / mean(||g||)``，范围为 ``[0,1]``。"""
    mean_norm: float = float(_row_norms(vectors).mean())
    if mean_norm <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.linalg.norm(vectors.mean(axis=0)) / mean_norm)


def _mean_pairwise_cosine(vectors: FloatArray) -> float:
    """返回不同 states 两两 cosine 的均值；单样本时返回1。"""
    if vectors.shape[0] <= 1:
        return 1.0
    norms: FloatArray = _row_norms(vectors)
    normalized: FloatArray = np.divide(
        vectors,
        norms[:, None],
        out=np.zeros_like(vectors),
        where=norms[:, None] > np.finfo(np.float64).tiny,
    )
    cosine_matrix: FloatArray = normalized @ normalized.T
    upper_triangle: tuple[NDArray[np.int64], NDArray[np.int64]] = (
        np.triu_indices(vectors.shape[0], k=1)
    )
    return float(cosine_matrix[upper_triangle].mean())


def summarize_dual_view_gradients(
    *,
    state_ids: IntArray,
    action_gradients: FloatArray,
    combined_feature_gradients: FloatArray,
    primary_feature_gradients: FloatArray,
    wrist_feature_gradients: FloatArray,
    action_weight: float,
    feature_weight: float,
) -> dict[str, object]:
    """汇总三条 source-only 梯度的方向、范数和跨状态稳定性。

    四组输入 shape 均为 ``[num_samples, num_basis, 3]``。优化使用梯度下降，
    三条 loss gradient 最终都会同时取负，因此这里直接比较 ``dL/dC`` 的
    cosine，不会改变方向关系。
    """
    gradient_arrays: tuple[FloatArray, ...] = (
        action_gradients,
        combined_feature_gradients,
        primary_feature_gradients,
        wrist_feature_gradients,
    )
    expected_shape: tuple[int, ...] = action_gradients.shape
    if (
        len(expected_shape) != 3
        or expected_shape[2] != 3
        or any(array.shape != expected_shape for array in gradient_arrays)
    ):
        raise ValueError("四组梯度必须具有相同的 [S,K,3] shape")
    if state_ids.shape != (expected_shape[0],):
        raise ValueError("state_ids 数量必须与梯度样本数一致")
    if not np.isfinite(action_weight) or not np.isfinite(feature_weight):
        raise ValueError("loss weights 必须为有限值")
    if not all(np.isfinite(array).all() for array in gradient_arrays):
        raise ValueError("梯度数组包含 NaN/Inf")

    # flattened_*: float64 [num_samples, num_basis * rgb_channels]。
    flattened: dict[str, FloatArray] = {
        "action": action_gradients.reshape(expected_shape[0], -1),
        "feature": combined_feature_gradients.reshape(
            expected_shape[0], -1
        ),
        "primary_feature": primary_feature_gradients.reshape(
            expected_shape[0], -1
        ),
        "wrist_feature": wrist_feature_gradients.reshape(
            expected_shape[0], -1
        ),
    }
    expected_combined: FloatArray = (
        flattened["primary_feature"] + flattened["wrist_feature"]
    ) / 2.0
    combined_identity_error: float = float(
        np.max(np.abs(flattened["feature"] - expected_combined))
    )

    weighted_action: FloatArray = action_weight * flattened["action"]
    weighted_feature: FloatArray = feature_weight * flattened["feature"]
    weighted_total: FloatArray = weighted_action + weighted_feature
    action_norms: FloatArray = _row_norms(flattened["action"])
    primary_norms: FloatArray = _row_norms(flattened["primary_feature"])
    wrist_norms: FloatArray = _row_norms(flattened["wrist_feature"])
    weighted_action_norms: FloatArray = _row_norms(weighted_action)
    weighted_feature_norms: FloatArray = _row_norms(weighted_feature)

    per_state: list[dict[str, float | int]] = []
    state_index: int
    for state_index, state_id in enumerate(state_ids):
        action_denominator: float = max(
            float(weighted_action_norms[state_index]),
            np.finfo(np.float64).tiny,
        )
        per_state.append(
            {
                "state_id": int(state_id),
                "action_norm": float(action_norms[state_index]),
                "primary_feature_norm": float(primary_norms[state_index]),
                "wrist_feature_norm": float(wrist_norms[state_index]),
                "weighted_feature_to_action_norm_ratio": float(
                    weighted_feature_norms[state_index]
                    / action_denominator
                ),
                "action_primary_feature_cosine": float(
                    _row_cosines(
                        flattened["action"],
                        flattened["primary_feature"],
                    )[state_index]
                ),
                "action_wrist_feature_cosine": float(
                    _row_cosines(
                        flattened["action"],
                        flattened["wrist_feature"],
                    )[state_index]
                ),
                "primary_wrist_feature_cosine": float(
                    _row_cosines(
                        flattened["primary_feature"],
                        flattened["wrist_feature"],
                    )[state_index]
                ),
                "action_combined_feature_cosine": float(
                    _row_cosines(
                        flattened["action"],
                        flattened["feature"],
                    )[state_index]
                ),
                "weighted_total_action_cosine": float(
                    _row_cosines(
                        weighted_total,
                        flattened["action"],
                    )[state_index]
                ),
            }
        )

    pair_names: tuple[tuple[str, str], ...] = (
        ("action", "primary_feature"),
        ("action", "wrist_feature"),
        ("primary_feature", "wrist_feature"),
        ("action", "feature"),
    )
    mean_cosines: dict[str, float] = {
        f"{left}_to_{right}": float(
            _row_cosines(flattened[left], flattened[right]).mean()
        )
        for left, right in pair_names
    }
    mean_cosines["weighted_total_to_action"] = float(
        _row_cosines(weighted_total, flattened["action"]).mean()
    )

    consistency_inputs: dict[str, FloatArray] = {
        **flattened,
        "weighted_total": weighted_total,
    }
    return {
        "scope": "source_openvla_only_zero_surface_delta",
        "num_samples": int(expected_shape[0]),
        "num_basis": int(expected_shape[1]),
        "state_ids": state_ids.tolist(),
        "action_weight": float(action_weight),
        "feature_weight": float(feature_weight),
        "combined_feature_identity_max_abs_error": (
            combined_identity_error
        ),
        "mean_raw_norms": {
            name: float(_row_norms(vectors).mean())
            for name, vectors in flattened.items()
        },
        "mean_weighted_feature_to_action_norm_ratio": float(
            np.mean(
                weighted_feature_norms
                / np.maximum(
                    weighted_action_norms,
                    np.finfo(np.float64).tiny,
                )
            )
        ),
        "mean_cosines": mean_cosines,
        "cross_state_consistency": {
            name: _cross_state_consistency(vectors)
            for name, vectors in consistency_inputs.items()
        },
        "mean_cross_state_pairwise_cosine": {
            name: _mean_pairwise_cosine(vectors)
            for name, vectors in consistency_inputs.items()
        },
        "per_state": per_state,
    }


def analyze_archive(
    audit_path: str | Path,
    *,
    action_weight: float,
    feature_weight: float,
) -> dict[str, object]:
    """读取扩展 Spectral Gradient Audit NPZ 并返回诊断摘要。"""
    required_keys: tuple[str, ...] = (
        "state_ids",
        "action_gradients",
        "feature_gradients",
        "primary_feature_gradients",
        "wrist_feature_gradients",
    )
    with np.load(Path(audit_path), allow_pickle=False) as archive:
        missing_keys: list[str] = [
            key for key in required_keys if key not in archive.files
        ]
        if missing_keys:
            raise ValueError(
                f"审计产物缺少双视角字段: {missing_keys}"
            )
        return summarize_dual_view_gradients(
            state_ids=np.asarray(archive["state_ids"], dtype=np.int64),
            action_gradients=np.asarray(
                archive["action_gradients"], dtype=np.float64
            ),
            combined_feature_gradients=np.asarray(
                archive["feature_gradients"], dtype=np.float64
            ),
            primary_feature_gradients=np.asarray(
                archive["primary_feature_gradients"], dtype=np.float64
            ),
            wrist_feature_gradients=np.asarray(
                archive["wrist_feature_gradients"], dtype=np.float64
            ),
            action_weight=action_weight,
            feature_weight=feature_weight,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action_weight", type=float, default=0.1)
    parser.add_argument("--feature_weight", type=float, default=4.0)
    arguments = parser.parse_args(argv)

    summary: dict[str, object] = analyze_archive(
        arguments.audit,
        action_weight=arguments.action_weight,
        feature_weight=arguments.feature_weight,
    )
    output_path: Path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Dual-view spectral gradient diagnosis: {output_path}")


if __name__ == "__main__":
    main()
