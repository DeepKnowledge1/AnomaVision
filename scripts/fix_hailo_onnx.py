#!/usr/bin/env python3
"""Make an exported ONNX graph compatible with Hailo DFC Conv parsing."""

import argparse
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    output = args.output or args.input
    model = onnx.load(str(args.input))
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue

        weight = initializers.get(node.input[1])
        if weight is None or len(weight.dims) < 2:
            continue

        if any(attribute.name == "kernel_shape" for attribute in node.attribute):
            continue

        node.attribute.append(
            onnx.helper.make_attribute("kernel_shape", list(weight.dims[2:]))
        )
        changed += 1

    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    print(f"Added kernel_shape to {changed} Conv nodes: {output}")


if __name__ == "__main__":
    main()
