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



def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")

    parser.add_argument('--dataset_path',default=r"D:\01-DATA\bottle", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')

    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')

    parser.add_argument('--backbone', type=str, choices=['resnet18', 'wide_resnet50'], default='resnet18',
                        help='Backbone network to use for feature extraction.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size used during training and inference.')

    parser.add_argument('--output_model', type=str, default='padim_model.pt',
                        help='Filename to save the PT model.')

    parser.add_argument('--layer_indices', nargs='+', type=int, default=[0],
                        help='List of layer indices to extract features from. Default: [0].')

    parser.add_argument('--feat_dim', type=int, default=50,
                        help='Number of random feature dimensions to keep.')

    return parser.parse_args()



def main(args):
    # Set up paths
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padim = torch.load(os.path.join(MODEL_DATA_PATH, args.output_model))    
    
    class_name = 'bottle'
    DATASET_PATH = os.path.realpath("D:/01-DATA/")
    test_dataset = anodet.MVTecDataset(DATASET_PATH, class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    print("Number of images in dataset:", len(test_dataloader.dataset))

    # results = padim.evaluate(
    #     dataloader=test_dataloader,
    #     # threshold=13,              # Adjust as needed
    #     show_progress=True,
    #     return_details=True,       # True to get all predictions and images
    # )
    
    res = padim.evaluate(test_dataloader)
    images, image_classifications_target, masks_target, image_scores, score_maps = res
    anodet.visualize_eval_data(image_classifications_target, masks_target, image_scores, score_maps)


if __name__ == "__main__":
        args = parse_args()
        main(args)
