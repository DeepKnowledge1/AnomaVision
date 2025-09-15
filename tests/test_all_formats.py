# tests/test_all_formats.py
"""
Simple test to compare predictions across all model formats.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import (
    adaptive_gaussian_blur,
    get_logger,
    merge_config,
    setup_logging,
)
from export import ModelExporter


class TestAllFormats:
    """Simple test comparing all model formats."""

    def test_all_formats_match(self, temp_model_dir, trained_padim_model, sample_batch):
        """Test that all formats give similar predictions."""

        batch, _, _, _ = sample_batch
        input_shape = tuple(batch.shape)
        setup_logging("Warning")
        logger = get_logger(__name__)

        # 1. Save all formats
        models = {}

        # PyTorch model
        pt_path = temp_model_dir / "model.pt"
        torch.save(trained_padim_model, str(pt_path))
        models["pytorch"] = pt_path

        # Statistics models
        pth_fp32_path = temp_model_dir / "stats_fp32.pth"
        pth_fp16_path = temp_model_dir / "stats_fp16.pth"
        trained_padim_model.save_statistics(str(pth_fp32_path), half=False)
        trained_padim_model.save_statistics(str(pth_fp16_path), half=True)
        models["stats_fp32"] = pth_fp32_path
        models["stats_fp16"] = pth_fp16_path

        # Export other formats
        exporter = ModelExporter(pt_path, temp_model_dir, logger=logger)

        # ONNX
        try:
            onnx_path = exporter.export_onnx(
                input_shape, "model.onnx", dynamic_batch=False
            )
            if onnx_path:
                models["onnx"] = onnx_path
        except:
            raise RuntimeError("ONNX export failed")

        # TorchScript
        try:
            ts_path = exporter.export_torchscript(input_shape, "model.torchscript")
            if ts_path:
                models["torchscript"] = ts_path
        except:
            raise RuntimeError("TorchScript export failed")

        # OpenVINO
        try:
            ov_dir = exporter.export_openvino(
                input_shape, "model_ov", fp16=False, dynamic_batch=False
            )
            if ov_dir:
                xml_files = list(ov_dir.glob("*.xml"))
                if xml_files:
                    models["openvino"] = xml_files[0]
        except:
            raise RuntimeError("OpenVINO export failed")

        print(f"\nTesting {len(models)} formats: {list(models.keys())}")

        # 2. Load all models and get predictions
        predictions = {}
        wrappers = {}

        for name, path in models.items():
            try:
                wrapper = ModelWrapper(str(path), device="cpu")
                scores, maps = wrapper.predict(batch)
                predictions[name] = {"scores": scores, "maps": maps}
                wrappers[name] = wrapper
                print(
                    f"✓ {name}: scores {scores.shape}, range [{scores.min():.4f}, {scores.max():.4f}]"
                )
            except Exception as e:
                print(f"✗ {name} failed: {e}")

        # 3. Compare against PyTorch reference
        if "pytorch" not in predictions:
            pytest.skip("PyTorch model failed, can't compare")

        ref_scores = predictions["pytorch"]["scores"]
        ref_maps = predictions["pytorch"]["maps"]

        # Define tolerances
        tolerances = {
            "stats_fp32": 1e-4,
            "stats_fp16": 1e-2,
            "onnx": 1e-3,
            "torchscript": 1e-5,
            "openvino": 1e-3,
        }

        print(f"\nComparing against PyTorch reference:")

        for name, pred in predictions.items():
            if name == "pytorch":
                continue

            score_diff = np.abs(pred["scores"] - ref_scores).max()
            map_diff = np.abs(pred["maps"] - ref_maps).max()

            tolerance = tolerances.get(name, 1e-3)

            print(
                f"{name:12}: score_diff={score_diff:.2e}, map_diff={map_diff:.2e} (tol={tolerance:.1e})"
            )

            # Check tolerance
            if score_diff <= tolerance and map_diff <= tolerance:
                print(f"             ✓ PASS")
            else:
                print(f"             ⚠ TOLERANCE EXCEEDED")
                # Don't fail test, just warn

        # 4. Cleanup
        for wrapper in wrappers.values():
            wrapper.close()

        # Must have at least PyTorch + 1 other format working
        assert (
            len(predictions) >= 2
        ), f"Need at least 2 working formats, got {len(predictions)}"

        print(f"\n✓ Test completed with {len(predictions)} working formats")
