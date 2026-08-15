"""纯 CPU 复核 Gate 6i Terminal Endpoint Action evidence bundle。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .terminal_endpoint_action_evidence import (
    evaluate_terminal_endpoint_bundle,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument(
        "--source_dense_seed_metrics_path",
        help="rsync 后覆盖 manifest 中服务器 Dense Seed metrics 绝对路径",
    )
    parser.add_argument(
        "--production_support_path",
        help="rsync 后覆盖 manifest 中服务器 Production Support 绝对路径",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    decision = evaluate_terminal_endpoint_bundle(
        Path(args.manifest_path),
        source_dense_seed_metrics_path=args.source_dense_seed_metrics_path,
        production_support_path=args.production_support_path,
    )
    print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    if not decision.audit_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
