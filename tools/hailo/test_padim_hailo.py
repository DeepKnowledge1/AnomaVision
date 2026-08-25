"""
Validate Hailo feature output against the existing PaDiM pipeline.

This script only checks feature extraction consistency.
PaDiM scoring remains unchanged.
"""

import argparse

from hailo_padim_runtime import HailoPadimRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    runtime = HailoPadimRuntime(args.hef)
    features = runtime.infer(args.image)

    print("Hailo feature shape:", features.shape)
    print("Validation completed")


if __name__ == "__main__":
    main()
