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

DEPLOYMENT_TARGETS = {
    "cpu": "CPU / ONNX Runtime",
    "openvino": "OpenVINO",
    "tensorrt": "TensorRT",
    "hailo": "Hailo",
    "kv260": "KV260 / Vitis AI",
}
