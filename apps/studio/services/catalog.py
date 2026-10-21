"""Small static catalogs exposed by the Studio UI.

Keep these values descriptive only. Training/deployment support must still be
validated by the underlying AnomaVision engine before execution.
"""

ALGORITHMS = {
    "patchcore": {
        "name": "PatchCore",
        "description": "Memory-bank based visual anomaly detection.",
    },
    "padim": {
        "name": "PaDiM",
        "description": "Patch distribution modeling for industrial inspection.",
    },
}

# Keep this list aligned with deployment_service.TARGET_FORMATS plus
# its CPU validation target. Hardware targets without export adapters must
# not be offered as deployable options in Studio.
DEPLOYMENT_TARGETS = {
    "cpu": "Validate existing model on CPU",
    "onnx": "Portable ONNX Runtime artifact",
    "openvino": "OpenVINO IR",
    "tensorrt": "TensorRT engine",
    "torchscript": "TorchScript artifact",
}
