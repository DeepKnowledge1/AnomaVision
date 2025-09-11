import os
from enum import Enum


class ModelType(Enum):
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"

    @classmethod
    def from_extension(cls, model_path):
        """Determine model type from file extension or directory structure"""

        # Check if it's a directory (for OpenVINO)
        if os.path.isdir(model_path):
            # Look for .xml files in the directory
            for file in os.listdir(model_path):
                if file.endswith(".xml"):
                    return cls.OPENVINO
            raise ValueError(
                f"Directory {model_path} doesn't contain OpenVINO .xml files"
            )

        # Handle file extensions
        extension_map = {
            ".pt": cls.PYTORCH,
            ".pth": cls.PYTORCH,
            ".torchscript": cls.TORCHSCRIPT,
            ".onnx": cls.ONNX,
            ".engine": cls.TENSORRT,
            ".trt": cls.TENSORRT,
            ".xml": cls.OPENVINO,
            ".bin": cls.OPENVINO,
        }

        ext = os.path.splitext(model_path)[1].lower()
        model_type = extension_map.get(ext)

        if model_type is None:
            raise ValueError(
                f"Unsupported model format: {ext}. Supported: {list(extension_map.keys())} or OpenVINO directories"
            )

        return model_type
