"""纯 CPU 生成Action-only κ feasibility/margin-drift报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


LIBERO_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
if str(LIBERO_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.terminal_margin_calibration import (  # noqa: E402
    analyze_terminal_margin_calibration_bundle,
    margin_calibration_report_dict,
    write_margin_calibration_report,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    report = analyze_terminal_margin_calibration_bundle(args.manifest_path)
    payload = margin_calibration_report_dict(report)
    if args.output_path is not None:
        digest = write_margin_calibration_report(
            report,
            output_path=args.output_path,
        )
        payload["written_report_sha256"] = digest
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.audit_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
