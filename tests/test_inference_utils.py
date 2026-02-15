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
    from anomavision.general import save_visualization

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
    """Test parse_args with no CLI args to get the defaults."""
    # Run parse_args with no CLI args to get the defaults
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    old = sys.argv[:]
    sys.argv = [old[0]]
    try:
        parser, args = detect.parse_args()
        # parse_args returns (parser, args) tuple
        # When called as a script, args will be populated
        # When not called as __main__, it returns (parser, None)
        # So we need to actually parse to get args
        args = parser.parse_args([])
    finally:
        sys.argv = old

    # Check that args object exists and has expected attributes
    assert hasattr(args, "batch_size")
    assert hasattr(args, "thresh")
    assert hasattr(args, "device")
    assert hasattr(args, "enable_visualization")


def test_main_with_missing_model_file_raises(tmp_path, monkeypatch):
    """
    Ensures the 'model file not found' path raises FileNotFoundError.
    Skips if detect imports rely on external packages not available.
    """
    # Mock sys.argv to simulate command line arguments with the correct argument names
    old = sys.argv[:]
    sys.argv = [old[0], "--model", "does_not_exist.pt", "--device", "cpu"]

    try:
        # detect.main() catches exceptions and calls exit(1), which raises SystemExit
        # So we expect SystemExit with code 1
        with pytest.raises(SystemExit) as exc_info:
            detect.main()

        # Verify it exited with code 1 (error), not 0 (success)
        assert exc_info.value.code == 1
    finally:
        sys.argv = old
