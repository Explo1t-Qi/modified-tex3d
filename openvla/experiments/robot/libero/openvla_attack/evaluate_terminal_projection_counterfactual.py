"""纯 CPU 复核 Gate 6j Radial-vs-Matched-Box evidence bundle。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .terminal_projection_counterfactual import (
    evaluate_terminal_projection_bundle,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    decision = evaluate_terminal_projection_bundle(Path(args.manifest_path))
    print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    if not decision.audit_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
