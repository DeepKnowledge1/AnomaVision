"""Run the real AnomaVision detect CLI against immutable golden cases."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests" / "regression" / "baseline.json"
MODEL = ROOT / "tests" / "regression" / "model" / "model.pt"
CONFIG = ROOT / "tests" / "regression" / "config.yml"


def fail(message: str) -> int:
    print(f"CORE REGRESSION FAIL: {message}")
    return 1


def main() -> int:
    if not MODEL.is_file():
        return fail(f"missing regression model: {MODEL}")
    if not BASELINE.is_file():
        return fail(f"missing baseline: {BASELINE}")
    if not CONFIG.is_file():
        return fail(f"missing config: {CONFIG}")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    cases = baseline.get("cases", [])
    if not cases:
        return fail("baseline contains no cases")

    threshold = float(baseline["threshold"])
    tolerance = float(baseline.get("score_tolerance", 0.05))

    with tempfile.TemporaryDirectory(prefix="anomavision-regression-") as tmp:
        tmp_path = Path(tmp)
        for case in cases:
            image = ROOT / case["image"]
            if not image.is_file():
                return fail(f"missing golden image: {image}")

            # detect.py consumes a directory through AnodetDataset. Use a
            # one-image temporary directory so each expected case maps to one
            # deterministic CLI invocation.
            image_dir = tmp_path / case["name"]
            image_dir.mkdir()
            copied = image_dir / image.name
            shutil.copy2(image, copied)

            output = tmp_path / f"{case['name']}.json"

            cmd = [
                sys.executable,
                "-m",
                "anomavision.cli",
                "detect",
                "--config",
                str(CONFIG),
                "--model",
                str(MODEL),
                "--img_path",
                str(image_dir),
                "--algorithm",
                str(baseline["algorithm"]),
                "--device",
                "cpu",
                "--batch_size",
                "1",
                "--thresh",
                str(threshold),
                "--regression-output",
                str(output),
            ]

            print("$", " ".join(cmd))
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr)
                return fail(f"detect command failed for {case['name']}")

            if not output.is_file():
                print(completed.stdout)
                print(completed.stderr)
                return fail(
                    f"detect completed but did not produce regression output for {case['name']}"
                )

            result = json.loads(output.read_text(encoding="utf-8"))
            scores = result.get("scores", [])
            classifications = result.get("classifications", [])
            maps = result.get("map_shapes", [])

            if len(scores) != 1 or len(classifications) != 1:
                return fail(
                    f"{case['name']}: expected exactly one score/classification, "
                    f"got scores={len(scores)} classifications={len(classifications)}"
                )

            score = float(scores[0])
            actual = "anomaly" if int(classifications[0]) else "normal"
            expected = case["expected"]

            if actual != expected:
                return fail(
                    f"{case['name']}: expected {expected}, got {actual} "
                    f"(score={score:.8f}, threshold={threshold:.8f})"
                )

            if not maps or not maps[0] or any(int(x) <= 0 for x in maps[0]):
                return fail(f"{case['name']}: invalid anomaly-map shape: {maps}")

            expected_score = case.get("expected_score")
            if expected_score is not None:
                if not math.isclose(
                    score,
                    float(expected_score),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    return fail(
                        f"{case['name']}: score changed: expected "
                        f"{float(expected_score):.8f} ± {tolerance:.8f}, got {score:.8f}"
                    )

            print(
                f"PASS {case['name']}: {actual}, score={score:.8f}, "
                f"map_shape={maps[0]}"
            )

    print("AnomaVision real detect regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
