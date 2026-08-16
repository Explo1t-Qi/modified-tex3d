"""在CPU上独立复核已同步的Action-only+κ两步工程smoke。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .action_margin_kappa_smoke import (
    evaluate_action_margin_kappa_smoke_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    args = parser.parse_args(argv)
    decision = evaluate_action_margin_kappa_smoke_bundle(args.manifest_path)
    print(
        json.dumps(
            {
                "engineering_valid": decision.gate_pass,
                "scientific_gate": False,
                "failures": list(decision.failures),
                "shallow_crossed_active_tokens": (
                    decision.shallow_crossed_active_tokens
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not decision.gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
