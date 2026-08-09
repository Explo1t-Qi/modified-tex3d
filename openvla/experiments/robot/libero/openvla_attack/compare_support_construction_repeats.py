"""只读比较 canonical/repeat Seed Score 导出的最终 Support candidates。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[5]
OPENVLA_ROOT = THIS_FILE.parents[4]
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
for import_path in (REPOSITORY_ROOT, OPENVLA_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from openvla_attack.support_construction_audit import (  # noqa: E402
    compare_support_repeat_artifacts,
    file_sha256,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first_dir", required=True)
    parser.add_argument("--second_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    first_root = Path(args.first_dir)
    second_root = Path(args.second_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "support_repeat_comparison.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    first_manifest_path = first_root / "support_construction_manifest.json"
    second_manifest_path = second_root / "support_construction_manifest.json"
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    if not first_manifest["decision"]["gate_pass"] or not second_manifest[
        "decision"
    ]["gate_pass"]:
        raise ValueError("两个 Support audit 必须先通过 artifact gate")
    if first_manifest["fixed_support_frozen"] or second_manifest[
        "fixed_support_frozen"
    ]:
        raise ValueError("repeat comparison 只接受未冻结的 audit candidates")
    first_candidate = first_root / first_manifest["evidence"][
        "candidate_artifact_relative_path"
    ]
    second_candidate = second_root / second_manifest["evidence"][
        "candidate_artifact_relative_path"
    ]
    if file_sha256(first_candidate) != first_manifest["evidence"][
        "candidate_artifact_sha256"
    ] or file_sha256(second_candidate) != second_manifest["evidence"][
        "candidate_artifact_sha256"
    ]:
        raise ValueError("candidate artifact SHA-256 与 manifest 不一致")
    report = compare_support_repeat_artifacts(first_candidate, second_candidate)
    output = {
        "schema_version": "openvla-support-repeat-v1",
        "first": {
            "directory": str(first_root.resolve()),
            "manifest_sha256": file_sha256(first_manifest_path),
            "candidate_artifact_sha256": file_sha256(first_candidate),
        },
        "second": {
            "directory": str(second_root.resolve()),
            "manifest_sha256": file_sha256(second_manifest_path),
            "candidate_artifact_sha256": file_sha256(second_candidate),
        },
        "report": asdict(report),
        "threshold_registered": False,
        "production_support_constructed": False,
        "fixed_support_frozen": False,
    }
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["report"], sort_keys=True))
    if not report.compatible:
        raise RuntimeError("Support repeat inputs 不兼容；报告已写入")


if __name__ == "__main__":
    main()
