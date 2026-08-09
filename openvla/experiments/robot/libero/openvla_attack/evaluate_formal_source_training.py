"""在CPU环境独立复核正式Fixed-Support source training bundle。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .formal_source_training_evidence import (
    evaluate_formal_source_training_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    args = parser.parse_args(argv)
    decision = evaluate_formal_source_training_bundle(args.manifest_path)
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
