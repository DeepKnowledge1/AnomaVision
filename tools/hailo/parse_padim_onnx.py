"""Parse PaDiM backbone ONNX model for Hailo.

The ONNX model is exported externally.
This script only handles Hailo conversion.
"""

from pathlib import Path

from hailo_sdk_client import ClientRunner


ONNX_MODEL = Path("padim_backbone.onnx")
HAR_OUTPUT = "padim_backbone.har"


# Adjust paths according to local Hailo SDK installation.
def main():
    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(
        str(ONNX_MODEL),
        model_name="padim_backbone",
    )
    runner.save_har(HAR_OUTPUT)
    print(f"Saved {HAR_OUTPUT}")


if __name__ == "__main__":
    main()
