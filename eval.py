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

## This is to make a complete evaluation of your dataset
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")

    parser.add_argument('--dataset_path',default=r"D:\01-DATA", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')

    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size used during training and inference.')

    parser.add_argument('--model_name', type=str, default='padim_model.pt',
                        help='Filename to save the PT model.')

    return parser.parse_args()



def main(args):
    # Set up paths
    DATASET_PATH = os.path.realpath(args.dataset_path)
    class_name = 'bottle'
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    padim = torch.load(os.path.join(MODEL_DATA_PATH, args.model_name))    
    
    test_dataset = anodet.MVTecDataset(DATASET_PATH, class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    print("Number of images in dataset:", len(test_dataloader.dataset))

   
    st = time.time()
    # res = padim.evaluate(test_dataloader)
    res = padim.evaluate_memory_efficient(test_dataloader)
        
    print(f"Time about {time.time()- st:.4f} s")
    images, image_classifications_target, masks_target, image_scores, score_maps = res
    anodet.visualize_eval_data(image_classifications_target, masks_target, image_scores, score_maps)


if __name__ == "__main__":
        args = parse_args()
        main(args)
