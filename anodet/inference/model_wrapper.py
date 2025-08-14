
import onnxruntime as ort
import torch
from .modelType import ModelType

class ModelWrapper:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_type = ModelType.from_extension(model_path)
        
        if self.model_type == ModelType.ONNX:
            self._init_onnx(model_path)
        elif self.model_type == ModelType.PYTORCH:
            self._init_pytorch(model_path)
        elif self.model_type == ModelType.TENSORRT:
            self._init_tensorrt(model_path)
        elif self.model_type == ModelType.OPENVINO:
            self._init_openvino(model_path)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
    
    def _init_onnx(self, model_path):
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.model = None
    
    def _init_pytorch(self, model_path):
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.session = None
    
    def _init_tensorrt(self, model_path):
        # Placeholder for TensorRT implementation
        raise NotImplementedError("TensorRT support not implemented yet")
    
    def _init_openvino(self, model_path):
        # Placeholder for OpenVINO implementation  
        raise NotImplementedError("OpenVINO support not implemented yet")
    
    def predict(self, batch):
        if self.model_type == ModelType.ONNX:
            return self._predict_onnx(batch)
        elif self.model_type == ModelType.PYTORCH:
            return self._predict_pytorch(batch)
        elif self.model_type == ModelType.TENSORRT:
            return self._predict_tensorrt(batch)
        elif self.model_type == ModelType.OPENVINO:
            return self._predict_openvino(batch)
    
    def _predict_onnx(self, batch):
        # print(f"ONNX input shape: {batch.shape}, dtype: {batch.dtype}")
        # print(f"ONNX input stats: min={batch.min()}, max={batch.max()}, mean={batch.mean()}")
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: batch})
        # print(f"ONNX outputs[0] (image_scores): {outputs[0]}")
        # print(f"ONNX outputs[1] (score_map) stats: min={outputs[1].min()}, max={outputs[1].max()}, mean={outputs[1].mean()}")
        
        return outputs[0], outputs[1]
    
    def _predict_pytorch(self, batch):
        # print(f"PyTorch input shape: {batch.shape}, dtype: {batch.dtype}")
        # print(f"PyTorch input stats: min={batch.min()}, max={batch.max()}, mean={batch.mean()}")
        
        batch = batch.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            scores, maps = self.model.predict(batch)
            # print(f"PyTorch scores: {scores.cpu().numpy()}")
            # print(f"PyTorch maps stats: min={maps.cpu().numpy().min()}, max={maps.cpu().numpy().max()}, mean={maps.cpu().numpy().mean()}")
            
            return scores.cpu().numpy(), maps.cpu().numpy()
    
    def _predict_tensorrt(self, batch):
        raise NotImplementedError("TensorRT prediction not implemented yet")
    
    def _predict_openvino(self, batch):
        raise NotImplementedError("OpenVINO prediction not implemented yet")
