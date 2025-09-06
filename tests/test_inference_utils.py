import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Import from your detect script
# If your file is named differently, adjust this import.
import detect


def test_determine_device_basic_roundtrip():
    assert detect.determine_device("cpu") == "cpu"
    assert detect.determine_device("cuda") == "cuda"
    auto = detect.determine_device("auto")
    # Auto should resolve to a valid device string given current machine
    if torch.cuda.is_available():
        assert auto == "cuda"
    else:
        assert auto == "cpu"


def test_save_visualization_single_and_batch(tmp_path):
    from anodet.general import save_visualization
    # Single image (H,W,3)
    single = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    save_visualization(single, "single.png", str(tmp_path))
    assert (tmp_path / "single.png").exists()

    # Batch of images (N,H,W,3)
    batch = (np.random.rand(3, 8, 8, 3) * 255).astype(np.uint8)
    save_visualization(batch, "batch.png", str(tmp_path))

    # Expect 3 files like batch_batch_0.png, ...
    files = list(tmp_path.glob("batch_batch_*.png"))
    assert len(files) == 3


def test_parse_args_defaults(monkeypatch):
    # Run parse_args with no CLI args to get the defaults
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    old = sys.argv[:]
    sys.argv = [old[0]]
    try:
        args = detect.parse_args()
    finally:
        sys.argv = old

    # Check that args object exists and has expected attributes
    assert hasattr(args, 'batch_size')
    assert hasattr(args, 'thresh')
    assert hasattr(args, 'device')
    assert hasattr(args, 'enable_visualization')


def test_main_with_missing_model_file_raises(tmp_path, monkeypatch):
    """
    Ensures the 'model file not found' path raises FileNotFoundError.
    Skips if detect imports rely on external packages not available.
    """
    # Mock sys.argv to simulate command line arguments with the correct argument names
    old = sys.argv[:]
    sys.argv = [
        old[0],
        "--model", "does_not_exist.pt",
        "--device", "cpu"
    ]

    try:
        # If detect's global imports (e.g., anodet) aren't available,
        # importing detect would already have failed. But if we're here,
        # guard runtime errors unrelated to "file not found" by catching and skipping.
        with pytest.raises(FileNotFoundError):
            detect.main()
    finally:
        sys.argv = old
