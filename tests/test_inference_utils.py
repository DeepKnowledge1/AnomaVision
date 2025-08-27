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
    # Single image (H,W,3)
    single = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    detect.save_visualization(single, "single.png", str(tmp_path))
    assert (tmp_path / "single.png").exists()

    # Batch of images (N,H,W,3)
    batch = (np.random.rand(3, 8, 8, 3) * 255).astype(np.uint8)
    detect.save_visualization(batch, "batch.png", str(tmp_path))

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

    # A few sanity checks of defaults
    assert isinstance(args.batch_size, int)
    assert isinstance(args.thresh, float)
    assert args.device in {"auto", "cpu", "cuda"}
    assert isinstance(args.enable_visualization, bool)


def test_main_with_missing_model_file_raises(tmp_path):
    """
    Ensures the 'model file not found' path raises FileNotFoundError.
    Skips if detect imports rely on external packages not available.
    """
    # Build args equivalent to CLI with a guaranteed-missing model path
    class A:
        dataset_path = str(tmp_path)                  # exists, but unused before model check
        model_data_path = str(tmp_path)               # exists
        model = "does_not_exist.pt"                   # missing
        device = "cpu"
        batch_size = 1
        thresh = 13.0
        num_workers = 0
        pin_memory = False
        enable_visualization = False
        save_visualizations = False
        viz_output_dir = str(tmp_path / "viz")
        show_first_batch_only = True
        viz_alpha = 0.5
        viz_padding = 40
        viz_color = "128,0,128"
        log_level = "INFO"
        detailed_timing = False

    # If detect’s global imports (e.g., anodet) aren’t available,
    # importing detect would already have failed. But if we’re here,
    # guard runtime errors unrelated to "file not found" by catching and skipping.
    with pytest.raises(FileNotFoundError):
        detect.main(A)
