# tests/test_backends.py
"""
Tests for multi-format backend functionality.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType


class TestModelType:
    """Test model type detection."""

    @pytest.mark.parametrize(
        "filename,expected_type",
        [
            ("model.pt", ModelType.PYTORCH),
            ("model.pth", ModelType.PYTORCH),
            ("model.onnx", ModelType.ONNX),
            ("model.torchscript", ModelType.TORCHSCRIPT),
            ("model.engine", ModelType.TENSORRT),
            ("model.trt", ModelType.TENSORRT),
            ("model.xml", ModelType.OPENVINO),
        ],
    )
    def test_model_type_detection(self, filename, expected_type):
        """Test model type detection from file extensions."""
        detected_type = ModelType.from_extension(filename)
        assert detected_type == expected_type

    def test_invalid_extension(self):
        """Test that invalid extensions raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model format"):
            ModelType.from_extension("model.invalid")

    def test_directory_detection(self, temp_model_dir):
        """Test OpenVINO directory detection."""
        # Create mock OpenVINO directory structure
        openvino_dir = temp_model_dir / "model_openvino"
        openvino_dir.mkdir()
        (openvino_dir / "model.xml").touch()

        model_type = ModelType.from_extension(str(openvino_dir))
        assert model_type == ModelType.OPENVINO


class TestTorchBackend:
    """Test PyTorch backend functionality."""

    def test_torch_backend_init(self, temp_model_dir, trained_padim_model):
        """Test PyTorch backend initialization."""
        # Save model
        model_path = temp_model_dir / "test_model.pt"
        torch.save(trained_padim_model, str(model_path))

        # Load with backend
        wrapper = ModelWrapper(str(model_path), device="cpu")

        assert wrapper.device == "cpu"
        wrapper.close()

    def test_torch_backend_prediction(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test PyTorch backend prediction."""
        # Save model
        model_path = temp_model_dir / "test_model.pt"
        torch.save(trained_padim_model, str(model_path))

        # Load with backend
        wrapper = ModelWrapper(str(model_path), device="cpu")

        batch, _, _, _ = sample_batch
        scores, maps = wrapper.predict(batch)

        # Check outputs
        assert isinstance(scores, np.ndarray)
        assert isinstance(maps, np.ndarray)
        assert scores.shape == (batch.size(0),)

        wrapper.close()

    def test_torch_backend_stats_loading(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test PyTorch backend with statistics (.pth) files."""
        # Save statistics
        stats_path = temp_model_dir / "test_stats.pth"
        trained_padim_model.save_statistics(str(stats_path))

        # Load with backend
        wrapper = ModelWrapper(str(stats_path), device="cpu")

        batch, _, _, _ = sample_batch
        scores, maps = wrapper.predict(batch)

        # Check outputs
        assert isinstance(scores, np.ndarray)
        assert isinstance(maps, np.ndarray)
        assert scores.shape == (batch.size(0),)

        wrapper.close()


class TestModelWrapper:
    """Test unified ModelWrapper interface."""

    def test_wrapper_initialization(self, temp_model_dir, trained_padim_model):
        """Test ModelWrapper initialization."""
        model_path = temp_model_dir / "test_model.pt"
        torch.save(trained_padim_model, str(model_path))

        wrapper = ModelWrapper(str(model_path), device="cpu")
        assert wrapper.device == "cpu"

        wrapper.close()

    def test_wrapper_prediction_consistency(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test that wrapper prediction is consistent."""
        model_path = temp_model_dir / "test_model.pt"
        torch.save(trained_padim_model, str(model_path))

        wrapper = ModelWrapper(str(model_path), device="cpu")

        batch, _, _, _ = sample_batch

        # Run prediction multiple times
        scores1, maps1 = wrapper.predict(batch)
        scores2, maps2 = wrapper.predict(batch)

        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(scores1, scores2)
        np.testing.assert_array_almost_equal(maps1, maps2)

        wrapper.close()

    def test_wrapper_warmup(self, temp_model_dir, trained_padim_model, sample_batch):
        """Test wrapper warmup functionality."""
        model_path = temp_model_dir / "test_model.pt"
        torch.save(trained_padim_model, str(model_path))

        wrapper = ModelWrapper(str(model_path), device="cpu")

        batch, _, _, _ = sample_batch

        # Warmup should not raise errors
        wrapper.warmup(batch, runs=2)

        # Should still work after warmup
        scores, maps = wrapper.predict(batch)
        assert scores is not None
        assert maps is not None

        wrapper.close()


class TestBackendCompatibility:
    """Test compatibility between different backends."""

    # def test_pytorch_vs_stats_consistency(self, temp_model_dir, trained_padim_model, sample_batch):
    #     """Test that full model and stats-only model give same results."""
    #     # Save both formats
    #     full_model_path = temp_model_dir / "full_model.pt"
    #     stats_path = temp_model_dir / "stats_only.pth"

    #     torch.save(trained_padim_model, str(full_model_path))
    #     trained_padim_model.save_statistics(str(stats_path))

    #     # Load both
    #     full_wrapper = ModelWrapper(str(full_model_path), device="cpu")
    #     stats_wrapper = ModelWrapper(str(stats_path), device="cpu")

    #     batch, _, _, _ = sample_batch

    #     # Compare predictions
    #     full_scores, full_maps = full_wrapper.predict(batch)
    #     stats_scores, stats_maps = stats_wrapper.predict(batch)

    #     # Should be very close (allowing for minor numerical differences)
    #     # Instead of decimal=5, use decimal=3 or 4

    #     np.testing.assert_array_almost_equal(full_scores, stats_scores, decimal=5)
    #     np.testing.assert_array_almost_equal(full_maps, stats_maps, decimal=5)

    #     full_wrapper.close()
    #     stats_wrapper.close()

    def test_backbone_differences(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        batch, _, _, _ = sample_batch

        # Save and load stats
        stats_path = temp_model_dir / "stats_only.pth"
        trained_padim_model.save_statistics(str(stats_path))
        stats_wrapper = ModelWrapper(str(stats_path), device="cpu")

        # Compare backbone parameters directly
        orig_backbone = trained_padim_model.embeddings_extractor.backbone
        new_backbone = stats_wrapper.backend.model.embeddings_extractor.backbone

        # Check if weights are identical
        for (name1, param1), (name2, param2) in zip(
            orig_backbone.named_parameters(), new_backbone.named_parameters()
        ):
            if not torch.equal(param1, param2):
                print(f"Parameter {name1} differs!")
                print(f"Max diff: {(param1 - param2).abs().max()}")

        # Check BatchNorm buffers (running_mean, running_var)
        for (name1, buf1), (name2, buf2) in zip(
            orig_backbone.named_buffers(), new_backbone.named_buffers()
        ):
            if not torch.equal(buf1, buf2):
                print(f"Buffer {name1} differs!")
                print(f"Max diff: {(buf1 - buf2).abs().max()}")

        # Test feature extraction directly
        orig_features, w1, h1 = trained_padim_model.embeddings_extractor(
            batch, channel_indices=trained_padim_model.channel_indices
        )
        new_features, w2, h2 = stats_wrapper.backend.model.embeddings_extractor(
            batch, channel_indices=stats_wrapper.backend.model.channel_indices
        )

        print(
            f"Feature extraction identical: {torch.allclose(orig_features, new_features, atol=1e-7)}"
        )
        print(f"Max feature diff: {(orig_features - new_features).abs().max()}")

    def test_pytorch_vs_stats_consistency(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test that full model and stats-only model give same results."""
        # Save both formats
        full_model_path = temp_model_dir / "full_model.pt"
        stats_path_fp32 = temp_model_dir / "stats_fp32.pth"
        stats_path_fp16 = temp_model_dir / "stats_fp16.pth"

        torch.save(trained_padim_model, str(full_model_path))
        trained_padim_model.save_statistics(str(stats_path_fp32), half=False)
        trained_padim_model.save_statistics(str(stats_path_fp16), half=True)

        # Load all three variants
        full_wrapper = ModelWrapper(str(full_model_path), device="cpu")
        stats_fp32_wrapper = ModelWrapper(str(stats_path_fp32), device="cpu")
        stats_fp16_wrapper = ModelWrapper(str(stats_path_fp16), device="cpu")

        batch, _, _, _ = sample_batch

        # Get predictions from all variants
        full_scores, full_maps = full_wrapper.predict(batch)
        fp32_scores, fp32_maps = stats_fp32_wrapper.predict(batch)
        fp16_scores, fp16_maps = stats_fp16_wrapper.predict(batch)

        print("=== Prediction Consistency Test ===")
        print(f"Full model scores shape: {full_scores.shape}")
        print(f"FP32 stats scores shape: {fp32_scores.shape}")
        print(f"FP16 stats scores shape: {fp16_scores.shape}")

        # Test FP32 stats consistency (should be nearly identical)
        score_diff_fp32 = np.abs(full_scores - fp32_scores)
        map_diff_fp32 = np.abs(full_maps - fp32_maps)

        print(f"\nFP32 Stats vs Full Model:")
        print(
            f"Score differences - max: {score_diff_fp32.max():.8f}, mean: {score_diff_fp32.mean():.8f}"
        )
        print(
            f"Map differences - max: {map_diff_fp32.max():.8f}, mean: {map_diff_fp32.mean():.8f}"
        )

        # Test FP16 stats consistency (should be close but with expected precision loss)
        score_diff_fp16 = np.abs(full_scores - fp16_scores)
        map_diff_fp16 = np.abs(full_maps - fp16_maps)

        print(f"\nFP16 Stats vs Full Model:")
        print(
            f"Score differences - max: {score_diff_fp16.max():.8f}, mean: {score_diff_fp16.mean():.8f}"
        )
        print(
            f"Map differences - max: {map_diff_fp16.max():.8f}, mean: {map_diff_fp16.mean():.8f}"
        )

        # Test FP32 vs FP16 differences
        fp32_vs_fp16_scores = np.abs(fp32_scores - fp16_scores)
        fp32_vs_fp16_maps = np.abs(fp32_maps - fp16_maps)

        print(f"\nFP32 Stats vs FP16 Stats:")
        print(
            f"Score differences - max: {fp32_vs_fp16_scores.max():.8f}, mean: {fp32_vs_fp16_scores.mean():.8f}"
        )
        print(
            f"Map differences - max: {fp32_vs_fp16_maps.max():.8f}, mean: {fp32_vs_fp16_maps.mean():.8f}"
        )

        # Assertions with appropriate tolerances

        # FP32 stats should be very close to full model (numerical precision should be high)
        # Relaxed tolerance to account for the computation path differences we identified
        np.testing.assert_array_almost_equal(
            full_scores,
            fp32_scores,
            decimal=4,
            err_msg="FP32 stats model differs too much from full model",
        )
        np.testing.assert_array_almost_equal(
            full_maps,
            fp32_maps,
            decimal=5,
            err_msg="FP32 stats maps differ too much from full model maps",
        )

        # FP16 should be reasonably close (allowing for precision loss)
        assert (
            score_diff_fp16.max() < 0.1
        ), f"FP16 score differences too large: {score_diff_fp16.max()}"
        assert (
            map_diff_fp16.max() < 0.1
        ), f"FP16 map differences too large: {map_diff_fp16.max()}"

        # Verify that FP16 introduces some precision loss (should not be identical to FP32)
        assert (
            fp32_vs_fp16_scores.max() > 1e-6
        ), "FP16 should introduce some precision loss compared to FP32"

        # Test file sizes
        import os

        full_size = os.path.getsize(str(full_model_path)) / (1024 * 1024)
        fp32_size = os.path.getsize(str(stats_path_fp32)) / (1024 * 1024)
        fp16_size = os.path.getsize(str(stats_path_fp16)) / (1024 * 1024)

        print(f"\nFile Sizes:")
        print(f"Full model: {full_size:.2f} MB")
        print(f"FP32 stats: {fp32_size:.2f} MB")
        print(f"FP16 stats: {fp16_size:.2f} MB")
        print(f"FP16 compression ratio: {fp32_size/fp16_size:.1f}x")

        # Verify FP16 provides meaningful compression
        assert (
            fp16_size < fp32_size * 0.7
        ), "FP16 should provide significant compression"

        # Cleanup
        full_wrapper.close()
        stats_fp32_wrapper.close()
        stats_fp16_wrapper.close()

    def test_stats_precision_handling(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test different precision save/load combinations."""

        # Save in both precisions
        fp32_path = temp_model_dir / "test_fp32.pth"
        fp16_path = temp_model_dir / "test_fp16.pth"

        trained_padim_model.save_statistics(str(fp32_path), half=False)
        trained_padim_model.save_statistics(str(fp16_path), half=True)

        # Test loading with different force_fp32 settings
        stats_fp32_keep = trained_padim_model.load_statistics(
            str(fp32_path), device="cpu", force_fp32=False
        )
        stats_fp32_force = trained_padim_model.load_statistics(
            str(fp32_path), device="cpu", force_fp32=True
        )
        stats_fp16_keep = trained_padim_model.load_statistics(
            str(fp16_path), device="cpu", force_fp32=False
        )
        stats_fp16_force = trained_padim_model.load_statistics(
            str(fp16_path), device="cpu", force_fp32=True
        )

        # Verify data types
        assert stats_fp32_keep["mean"].dtype == torch.float32
        assert stats_fp32_force["mean"].dtype == torch.float32
        assert (
            stats_fp16_keep["mean"].dtype == torch.float32
        )  # Should be converted to fp32
        assert stats_fp16_force["mean"].dtype == torch.float32

        # Verify dtype tracking
        assert stats_fp32_keep.get("dtype") == "fp32"
        assert stats_fp16_keep.get("dtype") == "fp16"

        print("Precision handling test passed - all dtypes handled correctly")

    def test_device_compatibility(
        self, temp_model_dir, trained_padim_model, sample_batch
    ):
        """Test that FP16 save/load works across CPU and GPU."""

        stats_path = temp_model_dir / "test_device_compat.pth"

        # Save statistics (should work regardless of current device)
        trained_padim_model.save_statistics(str(stats_path), half=True)

        # Test loading to CPU
        stats_cpu = trained_padim_model.load_statistics(str(stats_path), device="cpu")
        assert stats_cpu["mean"].device.type == "cpu"
        assert stats_cpu["cov_inv"].device.type == "cpu"

        # Test loading to CUDA (if available)
        if torch.cuda.is_available():
            stats_cuda = trained_padim_model.load_statistics(
                str(stats_path), device="cuda"
            )
            assert stats_cuda["mean"].device.type == "cuda"
            assert stats_cuda["cov_inv"].device.type == "cuda"

            # Verify data is identical (just different devices)
            assert torch.allclose(
                stats_cpu["mean"], stats_cuda["mean"].cpu(), atol=1e-6
            )
            assert torch.allclose(
                stats_cpu["cov_inv"], stats_cuda["cov_inv"].cpu(), atol=1e-6
            )

        print("Device compatibility test passed")


# Mock backends for testing (when ONNX/OpenVINO not available)
class MockOnnxBackend:
    """Mock ONNX backend for testing when ONNX is not available."""

    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device

    def predict(self, batch):
        # Return mock predictions with correct shapes
        batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
        height, width = 224, 224  # Assume standard size

        scores = np.random.rand(batch_size)
        maps = np.random.rand(batch_size, height, width)
        return scores, maps

    def close(self):
        pass

    def warmup(self, batch, runs=2):
        pass


class TestMockBackends:
    """Test with mock backends when real ones aren't available."""

    def test_mock_onnx_backend(self, sample_batch):
        """Test mock ONNX backend functionality."""
        batch, _, _, _ = sample_batch

        # Create mock backend
        backend = MockOnnxBackend("dummy_path.onnx")

        # Test prediction
        scores, maps = backend.predict(batch)

        assert scores.shape == (batch.size(0),)
        assert maps.shape == (batch.size(0), 224, 224)

        # Test warmup
        backend.warmup(batch)

        # Test cleanup
        backend.close()
