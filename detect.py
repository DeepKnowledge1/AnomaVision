
import os
import anodet
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from anodet.inference.model_wrapper import ModelWrapper
from anodet.inference.modelType import ModelType

THRESH = 13

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")
    parser.add_argument('--dataset_path', default=r"D:\01-DATA\dum\c2", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')
    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')
    parser.add_argument('--model', type=str, default='padim_model.onnx',
                        help='Model file (.pt for PyTorch, .onnx for ONNX)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory for faster GPU transfers.')
    return parser.parse_args()

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