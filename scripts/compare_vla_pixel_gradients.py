"""比较 OpenVLA 与 OpenVLA-OFT 的同状态像素攻击梯度。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.vla_pixel_gradient_audit import (
    PixelGradientArtifact,
    compare_pixel_gradient_artifacts,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_action_weight", type=float, default=0.1)
    parser.add_argument("--source_feature_weight", type=float, default=4.0)
    arguments = parser.parse_args(argv)

    source = PixelGradientArtifact.load(arguments.source)
    target = PixelGradientArtifact.load(arguments.target)
    summary = compare_pixel_gradient_artifacts(
        source,
        target,
        source_action_weight=arguments.source_action_weight,
        source_feature_weight=arguments.source_feature_weight,
    )
    output_path: Path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Cross-model pixel gradient summary: {output_path}")


if __name__ == "__main__":
    main()
