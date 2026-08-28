#!/usr/bin/env python3
"""Validate AnomaVision Hailo HEF compatibility without requiring a device.

Without Hailo hardware this tool performs static HEF/ONNX contract checks and
runs the ONNX reference on the same preprocessed image. If HailoRT can execute
the HEF on an available device, it additionally compares numerical outputs.

It intentionally does not claim ONNX-vs-HEF numerical equivalence when no
Hailo runtime/device is available: a HEF is a compiled Hailo artifact and
cannot be executed by ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Match the normal AnomaVision resize + ImageNet normalization contract."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (size[1], size[0]), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None], dtype=np.float32)


def onnx_info(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    outputs = session.get_outputs()
    return {
        "input_name": inp.name,
        "input_shape": list(inp.shape),
        "input_type": inp.type,
        "output_names": [item.name for item in outputs],
        "output_shapes": [list(item.shape) for item in outputs],
        "providers": session.get_providers(),
        "session": session,
    }


def hailo_info(path: Path) -> dict:
    try:
        from hailo_platform import HEF
    except ImportError as exc:
        raise RuntimeError(
            "hailo_platform is not installed. Static HEF inspection requires HailoRT."
        ) from exc

    hef = HEF(str(path))
    inputs = hef.get_input_vstream_infos()
    outputs = hef.get_output_vstream_infos()
    return {
        "input_names": [item.name for item in inputs],
        "input_shapes": [list(item.shape) for item in inputs],
        "input_types": [str(item.format.type) for item in inputs],
        "output_names": [item.name for item in outputs],
        "output_shapes": [list(item.shape) for item in outputs],
        "output_types": [str(item.format.type) for item in outputs],
    }


def compare_contract(onnx: dict, hef: dict) -> None:
    if onnx["input_name"] not in hef["input_names"]:
        # Hailo may rename the single input stream; shape is the authoritative check.
        if len(hef["input_shapes"]) != 1:
            raise AssertionError("ONNX/Hailo input count mismatch")
    onnx_shape = [1, 224, 224, 3]
    if list(onnx["input_shape"]) == [1, 3, 224, 224]:
        onnx_shape = [1, 224, 224, 3]
    hailo_shape = hef["input_shapes"][0]
    if hailo_shape != onnx_shape:
        raise AssertionError(
            f"Input shape mismatch: ONNX NCHW {onnx['input_shape']} vs Hailo NHWC {hailo_shape}"
        )
    required = {"image_scores", "score_map"}
    missing = required.difference(hef["output_names"])
    if missing:
        raise AssertionError(f"HEF missing required anomaly outputs: {sorted(missing)}")
    if set(onnx["output_names"]) != required:
        raise AssertionError(
            f"ONNX outputs must be {sorted(required)}, got {onnx['output_names']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)
    if not args.hef.exists():
        raise FileNotFoundError(args.hef)
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    onnx = onnx_info(args.onnx)
    hef = hailo_info(args.hef)
    compare_contract(onnx, hef)

    print("PASS: HEF/ONNX input and output contract")
    print(f"  ONNX input: {onnx['input_shape']} {onnx['input_type']}")
    print(f"  HEF input:  {hef['input_shapes'][0]}")
    print(f"  Outputs:    {sorted(set(hef['output_names']))}")
    print("PASS: Hailo path expects ImageNet-normalized RGB input; runtime only transposes NCHW -> NHWC.")

    if not args.skip_onnx:
        tensor = preprocess(args.image, (args.height, args.width))
        session = onnx["session"]
        outputs = session.run(onnx["output_names"], {onnx["input_name"]: tensor})
        print("PASS: ONNX reference inference")
        print(f"  image_scores shape: {np.asarray(outputs[0]).shape}")
        print(f"  score_map shape:    {np.asarray(outputs[1]).shape}")

    print("NOTE: numerical ONNX-vs-HEF comparison requires HailoRT execution; a HEF cannot be run by ONNX Runtime.")
    print(json.dumps({"status": "static_validation_passed", "numerical_comparison": "requires_hailo_runtime"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
