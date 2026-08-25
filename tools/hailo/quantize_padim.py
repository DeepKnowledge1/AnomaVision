#!/usr/bin/env python3
"""Quantize PaDiM backbone HAR model for Hailo.

Input:
    - parsed HAR file
    - calibration image directory

Output:
    - optimized INT8 HAR

PaDiM implementation is unchanged. This only prepares the Hailo model.
"""

from pathlib import Path

from hailo_sdk_client import ClientRunner


HAR_PATH = Path("padim_backbone.har")
CALIB_DIR = Path("dataset/train/good")
OUTPUT_HAR = Path("padim_backbone_quantized.har")


def main():
    runner = ClientRunner(har=str(HAR_PATH))

    # TODO: connect project preprocessing pipeline here.
    # Calibration images must use the same preprocessing as inference.
    calibration_images = list(CALIB_DIR.glob("*.png"))

    if not calibration_images:
        raise RuntimeError("No calibration images found")

    runner.optimize(calibration_data=calibration_images)
    runner.save_har(str(OUTPUT_HAR))


if __name__ == "__main__":
    main()
