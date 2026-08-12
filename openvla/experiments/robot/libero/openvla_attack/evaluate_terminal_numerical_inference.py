"""在CPU上独立复核Gate 6g numerical inference bundle。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .terminal_numerical_inference_audit import (
    evaluate_numerical_inference_bundle,
    numerical_decision_payload,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest_path)
    decision = evaluate_numerical_inference_bundle(manifest_path)
    print(json.dumps(numerical_decision_payload(decision), indent=2, sort_keys=True))
    if not decision.bundle_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
