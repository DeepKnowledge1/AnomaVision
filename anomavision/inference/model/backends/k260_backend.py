"""AMD Kria K26/KV260 board backend for complete Hailo-8 anomaly HEFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hailo_backend import HailoAnomalyRuntime, HailoBackend


class K260Backend(HailoBackend):
    """Run a complete Hailo-8 AnomaVision HEF on a K26/KV260 Linux target."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "k260",
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__(model_path, device=device, input_size=input_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("hailo_result.json"))
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1:
        parser.error("--height and --width must be positive")
    with HailoAnomalyRuntime(args.hef, input_size=(args.height, args.width)) as runtime:
        result = runtime.predict(args.image)
    serializable = {
        "hef": str(args.hef.resolve()),
        "image": str(args.image.resolve()),
        "image_score": float(np.asarray(result["image_scores"]).reshape(-1)[0]),
        "score_map_shape": list(np.asarray(result["score_map"]).shape),
        "score_map_min": float(np.min(result["score_map"])),
        "score_map_max": float(np.max(result["score_map"])),
    }
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
