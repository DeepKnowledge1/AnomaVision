
from enum import Enum
import os
class ModelType(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    
    @classmethod
    def from_extension(cls, model_path):
        """Determine model type from file extension"""
        extension_map = {
            '.pt': cls.PYTORCH,
            '.pth': cls.PYTORCH,
            '.onnx': cls.ONNX,
            '.engine': cls.TENSORRT,
            '.trt': cls.TENSORRT,
            '.xml': cls.OPENVINO,
            '.bin': cls.OPENVINO,
        }
    
        ext = os.path.splitext(model_path)[1].lower()
        model_type = extension_map.get(ext)
        
        if model_type is None:
            raise ValueError(f"Unsupported model format: {ext}. Supported: {list(extension_map.keys())}")
        
        return model_type
