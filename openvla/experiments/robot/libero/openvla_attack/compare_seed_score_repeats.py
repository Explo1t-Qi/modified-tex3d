"""比较两次已通过的 Seed Score artifact，不注册稳定性阈值。

本命令只报告连续score、全顶点rank高分集合和局部峰候选的repeat稳定性。它不
选择production seed、不构造区域、不计算coverage，也不据结果自动调整算法。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openvla.experiments.robot.libero.openvla_attack.seed_score_audit import (
    SEED_SCORE_SCHEMA_VERSION,
    SeedScoreAuditError,
    compare_seed_score_stability,
    file_sha256,
    load_seed_score_artifact_arrays,
    seed_score_evidence_from_mapping,
)


@dataclass(frozen=True)
class RepeatComparisonConfig:
    """两个score run目录和唯一输出JSON。"""

    first_score_dir: str
    second_score_dir: str
    output_path: str


def _parse_args(argv: Optional[Sequence[str]] = None) -> RepeatComparisonConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first_score_dir", required=True)
    parser.add_argument("--second_score_dir", required=True)
    parser.add_argument("--output_path", required=True)
    return RepeatComparisonConfig(**vars(parser.parse_args(argv)))


def _load_run(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "seed_score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SEED_SCORE_SCHEMA_VERSION:
        raise SeedScoreAuditError(f"{root} schema不匹配")
    if manifest.get("decision", {}).get("gate_pass") is not True:
        raise SeedScoreAuditError(f"{root} score Gate未通过")
    if manifest.get("production_support_constructed") is not False:
        raise SeedScoreAuditError(f"{root} 非纯score artifact")
    evidence = seed_score_evidence_from_mapping(manifest["evidence"])
    artifact_path = root / evidence.artifact_relative_path
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if file_sha256(artifact_path) != evidence.artifact_sha256:
        raise SeedScoreAuditError(f"{root} artifact SHA-256不匹配")
    return manifest, artifact_path


def run_repeat_comparison(cfg: RepeatComparisonConfig) -> Path:
    """输出只读稳定性报告；兼容性失败时拒绝产生可接受结论。"""

    first_root = Path(cfg.first_score_dir)
    second_root = Path(cfg.second_score_dir)
    output_path = Path(cfg.output_path)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖repeat报告: {output_path}")
    first_manifest, first_artifact = _load_run(first_root)
    second_manifest, second_artifact = _load_run(second_root)
    first_arrays = load_seed_score_artifact_arrays(first_artifact)
    second_arrays = load_seed_score_artifact_arrays(second_artifact)
    report = compare_seed_score_stability(first_arrays, second_arrays)
    payload = {
        "schema_version": "openvla-seed-score-repeat-v1",
        "first": {
            "directory": str(first_root.resolve()),
            "manifest_sha256": file_sha256(
                first_root / "seed_score_manifest.json"
            ),
            "artifact_sha256": file_sha256(first_artifact),
            "dense_code_commit": first_manifest["evidence"]["provenance"][
                "dense_code_commit"
            ],
        },
        "second": {
            "directory": str(second_root.resolve()),
            "manifest_sha256": file_sha256(
                second_root / "seed_score_manifest.json"
            ),
            "artifact_sha256": file_sha256(second_artifact),
            "dense_code_commit": second_manifest["evidence"]["provenance"][
                "dense_code_commit"
            ],
        },
        "report": asdict(report),
        "production_support_constructed": False,
        "stability_threshold_registered": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload["report"], indent=2, sort_keys=True))
    if not report.compatible:
        raise RuntimeError(
            "Seed Score repeat不兼容: " + "; ".join(report.failures)
        )
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_repeat_comparison(_parse_args(argv))


if __name__ == "__main__":
    main()
