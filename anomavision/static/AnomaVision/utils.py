"""
Provides utility functions for anomaly detection.
"""

import numpy as np
from typing import List
import cv2

def to_batch(images: List[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(img, (224, 224)) for img in images]
    normalized = [(img / 255. - np.array([0.485, 0.456, 0.406])) / 
                  np.array([0.229, 0.224, 0.225]) for img in resized]
    batch = np.stack([img.transpose(2, 0, 1) for img in normalized])  # CHW
    return batch.astype(np.float32)



def classification(image_scores: np.ndarray, thresh: float) -> np.ndarray:
    """
    Classify images as anomalous (0) or normal (1) based on threshold.
    
    Args:
        image_scores: A 1D array of image anomaly scores.
        thresh: Threshold value to determine anomaly.
    
    Returns:
        An array of classifications: 0 = anomaly, 1 = normal.
    """
    return np.where(image_scores < thresh, 1, 0)
