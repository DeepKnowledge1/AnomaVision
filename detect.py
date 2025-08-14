
import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from export import export_onnx
import argparse
from anodet.test import *
import time
import onnxruntime as ort
from enum import Enum

THRESH = 13

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")
    parser.add_argument('--dataset_path', default=r"D:\01-DATA\dum\c2", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')
    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')
    parser.add_argument('--model', type=str, default='padim_model.pt',
                        help='Model file (.pt for PyTorch, .onnx for ONNX)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory for faster GPU transfers.')
    return parser.parse_args()

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
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
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
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: batch})
        return outputs[0], outputs[1]
    
    def _predict_pytorch(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        with torch.no_grad():
            scores, maps = self.model.predict(batch)
            return scores.cpu().numpy(), maps.cpu().numpy()
    
    def _predict_tensorrt(self, batch):
        raise NotImplementedError("TensorRT prediction not implemented yet")
    
    def _predict_openvino(self, batch):
        raise NotImplementedError("OpenVINO prediction not implemented yet")

def main(args):
    # Setup
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load model
    model_path = os.path.join(MODEL_DATA_PATH, args.model)
    model = ModelWrapper(model_path, device)
    
    # DataLoader
    test_dataset = anodet.AnodetDataset(DATASET_PATH)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available() and model.model_type == ModelType.PYTORCH,
        persistent_workers=args.num_workers > 0
    )
    
    print(f"Processing {len(test_dataset)} images using {model.model_type.value.upper()}")
    
    # Process
    for batch_idx, (batch, images, _, _) in enumerate(test_dataloader):
        image_scores, score_maps = model.predict(batch)
        
        score_map_classifications = anodet.classification(score_maps, THRESH)
        image_classifications = anodet.classification(image_scores, THRESH)
        
        print(f"Batch {batch_idx}: Scores: {image_scores}, Classifications: {image_classifications}")
        
        # Visualizations
        test_images = np.array(images)
        boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
        heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
        highlighted_images = anodet.visualization.highlighted_images([images[i] for i in range(len(images))], score_map_classifications, color=(128, 0, 128))

        # Show first batch only
        if batch_idx == 0:
            fig, axs = plt.subplots(1, 4, figsize=(12, 6))
            fig.suptitle('Sample Result', fontsize=14)
            axs[0].imshow(images[0])
            axs[1].imshow(boundary_images[0])
            axs[2].imshow(heatmap_images[0])
            axs[3].imshow(highlighted_images[0])
            plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)