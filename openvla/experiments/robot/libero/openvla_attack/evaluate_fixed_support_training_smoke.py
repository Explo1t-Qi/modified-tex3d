"""独立复核已同步的Fixed-Support Action+Spectral两步smoke bundle。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


THIS_FILE = Path(__file__).resolve()
LIBERO_EXPERIMENT_DIR = THIS_FILE.parents[1]
if str(LIBERO_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from openvla_attack.fixed_support_training_smoke import (  # noqa: E402
    evaluate_fixed_support_training_smoke_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path")
    args = parser.parse_args(argv)
    decision = evaluate_fixed_support_training_smoke_bundle(args.manifest_path)
    print(
        json.dumps(
            {
                "gate_pass": decision.gate_pass,
                "failures": list(decision.failures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not decision.gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
