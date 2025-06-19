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


THRESH = 13 # Anomaly threshold

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")

    parser.add_argument('--dataset_path',default=r"D:\01-DATA\bottle", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')
    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')

    parser.add_argument('--model_data', type=str, default='padim_model.pt',
                        help='model PT model.')


    return parser.parse_args()

def get_images(DATASET_PATH):
    
    paths = [
        os.path.join(DATASET_PATH, "test/broken_large/000.png"),
        os.path.join(DATASET_PATH, "test/broken_small/000.png"),
        os.path.join(DATASET_PATH, "test/contamination/000.png"),
        os.path.join(DATASET_PATH, "test/good/000.png"),
        os.path.join(DATASET_PATH, "test/good/001.png"),
    ]

    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    return images
    

def main(args):
    # Set up paths
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padim = torch.load(os.path.join(MODEL_DATA_PATH, args.model_data))    

    images = get_images(DATASET_PATH)
    batch = anodet.to_batch(images, anodet.standard_image_transform, torch.device('cpu'))
    
    image_scores, score_maps = padim.predict(batch)
    
    
    score_map_classifications = anodet.classification(score_maps, THRESH)
    image_classifications = anodet.classification(image_scores, THRESH)
    print("Image scores:", image_scores)
    print("Image classifications:", image_classifications)    
    
    test_images = np.array(images).copy()
        
    boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
    heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

    for idx in range(1): #range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle('Image: ' + str(idx), y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        plt.show()


if __name__ == "__main__":
        args = parse_args()
        main(args)
