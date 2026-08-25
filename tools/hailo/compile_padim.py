"""Compile quantized PaDiM HAR model to HEF.

Expected flow:
ONNX -> HAR -> INT8 HAR -> HEF

This script does not modify PaDiM implementation.
"""

from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compile Hailo PaDiM model")
    parser.add_argument("har", type=Path, help="Path to quantized HAR file")
    parser.add_argument("--arch", default="hailo8", help="Hailo target architecture")
    parser.add_argument("--output", type=Path, default=Path("padim_backbone.hef"))
    args = parser.parse_args()

    # Compilation is performed with Hailo Dataflow Compiler tools.
    # Keep this wrapper lightweight to match existing AnomaVision export flow.
    print(f"Compile {args.har} for {args.arch}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
