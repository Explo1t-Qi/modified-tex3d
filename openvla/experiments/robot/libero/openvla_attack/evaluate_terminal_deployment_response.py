"""在CPU上独立复核Gate 6g smoke或formal deployment response bundle。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .terminal_deployment_response_audit import (
    evaluate_terminal_response_bundle,
    evaluate_terminal_response_smoke_bundle,
    resolve_terminal_response_run_protocol,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--audit_scope", default="formal")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    protocol = resolve_terminal_response_run_protocol(args.audit_scope)
    manifest_path = Path(args.manifest_path)
    evaluator = (
        evaluate_terminal_response_bundle
        if protocol.scope == "formal"
        else evaluate_terminal_response_smoke_bundle
    )
    decision = evaluator(manifest_path)
    print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    if not decision.audit_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
