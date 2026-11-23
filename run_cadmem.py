
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from anomavision.algorithm.cadmem import CADMemConfig, CADMemPatchCore
from anomavision.datasets.mvtec_dataset import MVTecDataset

DATASET_ROOT = "D:/01-DATA/mvtec"
CLASS_NAME = "cable"
MODEL_PATH = Path(
    "distributions/patchcore/cable/anomav_exp/model.pt"
)

device = torch.device("cpu")

# Load the already-trained PatchCore model.
# Only load checkpoints that you trust.
base = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)
base.to_device(device)
base.eval()

# Calibration: use normal training images.
dataset = MVTecDataset(
    dataset_path=DATASET_ROOT,
    class_name=CLASS_NAME,
    is_train=True,
    resize=(224, 224),
    crop_size=None,
    normalize=True,
)

# Deterministic subset for this initial demonstration.
# For a publishable experiment, keep calibration data
# separate from the images used to fit PatchCore.
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator)
calibration_indices = indices[:max(1, len(dataset) // 5)]

calibration_loader = DataLoader(
    Subset(dataset, calibration_indices.tolist()),
    batch_size=2,
    shuffle=False,
)

cadmem = CADMemPatchCore(
    base,
    CADMemConfig(
        max_adaptive=256,
        gate_cap=0.5,
        min_support=3,
    ),
)

cadmem.calibrate(calibration_loader)

# Evaluate on the held-out MVTec test split.
test_dataset = MVTecDataset(
    dataset_path=DATASET_ROOT,
    class_name=CLASS_NAME,
    is_train=False,
    resize=(224, 224),
    crop_size=None,
    normalize=True,
)

test_loader = DataLoader(test_dataset, batch_size=2)

print("CAD-Mem initialized successfully")

for images, _, labels, _ in test_loader:
    images = images.to(device)

    with torch.no_grad():
        scores, maps = cadmem.predict(images)

    print("Labels:", labels.tolist())
    print("Scores:", scores.tolist())
    print("Maps:", tuple(maps.shape))

    break

print("CAD-Mem inference completed successfully")
