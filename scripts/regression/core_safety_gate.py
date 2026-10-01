"""Run a real-model regression gate against a known-good image.

This is intentionally separate from the Studio/API layer. It validates the
actual inference contract: model loading, score, classification, and map shape.

Example:
    uv run python scripts/regression/core_safety_gate.py \
      --model model.pt \
      --image D:/01-DATA/mvtec/cable/test/bent_wire/000.png \
      --threshold 13.0 \
      --expected anomaly \
      --min-score 13.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import classification


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (array - mean) / std
    return np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="AnomaVision core regression gate")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--expected", choices=("normal", "anomaly"), required=True)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--max-score", type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"FAIL: model does not exist: {args.model}")
    if not args.image.is_file():
        raise SystemExit(f"FAIL: image does not exist: {args.image}")

    batch = _load_image(args.image)
    model = ModelWrapper(str(args.model), args.device)

    try:
        scores, maps = model.predict(batch)
    finally:
        model.close()

    scores = np.asarray(scores).reshape(-1)
    if scores.size != 1:
        raise SystemExit(f"FAIL: expected one image score, got shape {scores.shape}")

    score = float(scores[0])
    prediction = "anomaly" if int(classification(scores, args.threshold)[0]) else "normal"

    if prediction != args.expected:
        raise SystemExit(
            f"FAIL: prediction changed: expected={args.expected}, "
            f"actual={prediction}, score={score:.6f}, threshold={args.threshold:.6f}"
        )

    if args.min_score is not None and score < args.min_score:
        raise SystemExit(
            f"FAIL: score {score:.6f} is below minimum {args.min_score:.6f}"
        )

    if args.max_score is not None and score > args.max_score:
        raise SystemExit(
            f"FAIL: score {score:.6f} is above maximum {args.max_score:.6f}"
        )

    if maps is None:
        raise SystemExit("FAIL: inference returned no anomaly map")

    maps = np.asarray(maps)
    if maps.size == 0 or maps.ndim < 3:
        raise SystemExit(f"FAIL: invalid anomaly-map shape: {maps.shape}")

    print("AnomaVision Core Safety Gate: PASS")
    print(f"  model:      {args.model}")
    print(f"  image:      {args.image}")
    print(f"  prediction: {prediction}")
    print(f"  score:      {score:.6f}")
    print(f"  threshold:  {args.threshold:.6f}")
    print(f"  map shape:  {maps.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
