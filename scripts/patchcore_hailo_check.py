"""Check PatchCore ONNX node names required by the Hailo parser."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


REQUIRED_NODES = {"/MaxPool", "/Squeeze"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    names = {node.name for node in model.graph.node}
    missing = REQUIRED_NODES - names

    print(f"ONNX: {args.onnx}")
    print(f"Nodes: {len(names)}")
    if missing:
        print("FAIL: missing Hailo endpoint nodes:", ", ".join(sorted(missing)))
        return 1

    print("PASS: /MaxPool and /Squeeze are available for Hailo parsing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
