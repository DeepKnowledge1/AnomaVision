import logging
import os

# Add NullHandler to prevent logs when used as library
logging.getLogger(__name__).addHandler(logging.NullHandler())

"""
Provides functions for performing anomaly detection in images.
"""

from .datasets.dataset import AnodetDataset
from .datasets.mvtec_dataset import MVTecDataset
from .feature_extraction import ResnetEmbeddingsExtractor
from .padim import Padim

from .sampling_methods.kcenter_greedy import kCenterGreedy
from .test import optimal_threshold, visualize_eval_data, visualize_eval_pair
from .utils import get_logger  # Export for users
from .utils import setup_logging  # Export for users who want to enable logging
from .utils import (
    classification,
    image_score,
    mahalanobis,
    pytorch_cov,
    split_tensor_and_run_function,
    standard_image_transform,
    standard_mask_transform,
    to_batch,
)
from .visualization import *
