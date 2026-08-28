from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomavision.inference.model.backends.hailo_backend import HailoAnomalyRuntime
from anomavision.quantize.model.backends.hef import graphs as hailo_graphs
from anomavision.quantize.model.backends.hef.exporter import (
    _write_calibration_manifest,
    export_onnx,
)


class _FakeExtractor(torch.nn.Module):
    def __init__(self, backbone, device):
        super().__init__()

    def forward(self, image, layer_indices=None):
        batch = image.shape[0]
        features = torch.nn.functional.adaptive_avg_pool2d(image, (4, 4))
        features = features.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return features.permute(0, 2, 3, 1).reshape(batch, 16, 4), 4, 4


def _patch_fake_extractor(monkeypatch):
    monkeypatch.setattr(hailo_graphs, "ResnetEmbeddingsExtractor", _FakeExtractor)


def test_padim_graph_contains_distance_and_reduction(monkeypatch):
    _patch_fake_extractor(monkeypatch)
    graph = hailo_graphs.PadimEndToEndGraph(
        backbone="resnet18", layer_indices=[0, 1], channel_indices=torch.arange(4),
        mean=torch.zeros(16, 4), cov_inv=torch.eye(4).repeat(16, 1, 1), input_size=(32, 32)
    ).eval()
    image_scores, score_map = graph(torch.ones(1, 3, 32, 32))
    assert image_scores.shape == (1,)
    assert score_map.shape == (1, 32, 32)
    assert torch.isfinite(image_scores).all()
    assert torch.isfinite(score_map).all()


def test_patchcore_graph_contains_memory_distance_and_reduction(monkeypatch):
    _patch_fake_extractor(monkeypatch)
    graph = hailo_graphs.PatchCoreEndToEndGraph(
        backbone="resnet18", layer_indices=[0, 1], memory_bank=torch.zeros(8, 4),
        patch_grid=4, input_size=(32, 32)
    ).eval()
    image_scores, score_map = graph(torch.ones(1, 3, 32, 32))
    assert image_scores.shape == (1,)
    assert score_map.shape == (1, 32, 32)
    assert torch.isfinite(image_scores).all()
    assert torch.isfinite(score_map).all()


def test_export_writes_end_to_end_metadata_and_calibration_manifest(tmp_path, monkeypatch):
    _patch_fake_extractor(monkeypatch)
    artifact = tmp_path / "patchcore.pt"
    torch.save({"backbone": "resnet18", "layer_indices": [0, 1], "memory_bank": torch.zeros(8, 4), "patch_grid": 4}, artifact)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(calibration / "one.png")
    output = tmp_path / "export"
    onnx_path = export_onnx("patchcore", artifact, output, (32, 32))
    assert onnx_path.exists()
    manifest = _write_calibration_manifest(calibration, output, (32, 32))
    assert manifest.exists()
    calibration_array = np.load(output / "calibration_npy" / "sample_0000.npy")
    expected = -(np.asarray([0.485, 0.456, 0.406]) / np.asarray([0.229, 0.224, 0.225]))
    np.testing.assert_allclose(calibration_array[0, 0], expected, atol=1e-6)
    assert onnx_path.name.endswith("_end_to_end.onnx")


def test_hailo_preprocessed_tensor_only_transposes():
    runtime = HailoAnomalyRuntime.__new__(HailoAnomalyRuntime)
    runtime.input_size = (32, 32)
    runtime.input_dtype = np.float32
    runtime.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    runtime.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    nchw = np.random.default_rng(42).normal(size=(1, 3, 32, 32)).astype(np.float32)
    prepared = runtime._prepare_input(nchw)
    np.testing.assert_allclose(prepared, np.transpose(nchw[0], (1, 2, 0)))


def test_hailo_raw_image_is_normalized_once():
    runtime = HailoAnomalyRuntime.__new__(HailoAnomalyRuntime)
    runtime.input_size = (32, 32)
    runtime.input_dtype = np.float32
    runtime.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    runtime.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    raw = np.full((32, 32, 3), 255, dtype=np.uint8)
    prepared = runtime._prepare_input(raw)
    expected = (1.0 - runtime.mean) / runtime.std
    np.testing.assert_allclose(prepared, expected, atol=1e-6)


def test_export_rejects_partial_artifact(tmp_path):
    artifact = tmp_path / "bad.pt"
    torch.save({"backbone": "resnet18"}, artifact)
    with pytest.raises(ValueError, match="missing keys"):
        export_onnx("padim", artifact, tmp_path / "out", (32, 32))
