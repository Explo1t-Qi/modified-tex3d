"""独立复核已同步的Spectral Guard GPU Calibration bundle。"""

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

from openvla_attack.spectral_guard_evidence import (  # noqa: E402
    evaluate_spectral_guard_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path")
    args = parser.parse_args(argv)
    decision = evaluate_spectral_guard_bundle(args.manifest_path)
    print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    if not decision.gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
