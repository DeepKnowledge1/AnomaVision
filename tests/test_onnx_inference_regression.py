"""Strict end-to-end regression tests for PyTorch and ONNX inference.

These tests intentionally fail on export, ONNX graph validation, backend loading,
output-contract, non-finite-output, or numerical-equivalence errors. They do not
swallow inference exceptions or downgrade mismatches to warnings.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, TensorDataset

import anomavision.algorithm.patchcore as patchcore_module
from anomavision.export import ModelExporter
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import get_logger


LOGGER = get_logger(__name__)


class _DeterministicPatchCoreExtractor(torch.nn.Module):
    """Small export-friendly extractor for a fast, deterministic PatchCore test."""

    backbone_name = "resnet18"

    def __init__(self, backbone_name, device):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, batch, layer_indices=None):
        batch_size = batch.shape[0]
        pooled = batch.mean(dim=(1, 2, 3)).reshape(batch_size, 1, 1)
        embeddings = pooled.expand(batch_size, 16, 3)
        return embeddings, 4, 4

    def to_device(self, device):
        self.device = torch.device(device)


def _assert_pt_and_onnx_match(model, batch, tmp_path: Path, filename: str):
    """Export a model and strictly compare production wrapper outputs on CPU."""
    assert batch.ndim == 4
    assert batch.shape[0] >= 1
    assert torch.isfinite(batch).all()

    model.eval()
    pt_path = tmp_path / f"{filename}.pt"
    torch.save(model, pt_path)
    assert pt_path.is_file() and pt_path.stat().st_size > 0

    exporter = ModelExporter(pt_path, tmp_path, LOGGER, device="cpu")
    onnx_path = exporter.export_onnx(
        input_shape=tuple(batch.shape),
        output_name=f"{filename}.onnx",
        opset_version=18,
        dynamic_batch=False,
        force_precision="fp32",
    )
    assert onnx_path is not None, "ONNX export returned None"
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    # This fails on malformed graphs, including missing initializer references.
    onnx.checker.check_model(str(onnx_path))

    # Enforce the default inference contract: scores and maps only.
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_providers()[0] == "CPUExecutionProvider"
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 2

    pt_wrapper = ModelWrapper(str(pt_path), device="cpu")
    onnx_wrapper = ModelWrapper(str(onnx_path), device="cpu")
    try:
        pt_scores, pt_maps = pt_wrapper.predict(batch)
        onnx_scores, onnx_maps = onnx_wrapper.predict(batch)

        assert isinstance(pt_scores, np.ndarray)
        assert isinstance(pt_maps, np.ndarray)
        assert isinstance(onnx_scores, np.ndarray)
        assert isinstance(onnx_maps, np.ndarray)

        assert pt_scores.shape == onnx_scores.shape == (batch.shape[0],)
        assert pt_maps.shape == onnx_maps.shape == (
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
        )
        assert np.isfinite(pt_scores).all()
        assert np.isfinite(pt_maps).all()
        assert np.isfinite(onnx_scores).all()
        assert np.isfinite(onnx_maps).all()

        # Fail the test on any score or pixel-map mismatch outside tolerance.
        np.testing.assert_allclose(
            onnx_scores, pt_scores, rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            onnx_maps, pt_maps, rtol=1e-3, atol=1e-4
        )
    finally:
        pt_wrapper.close()
        onnx_wrapper.close()


def test_padim_pt_and_onnx_inference_are_strictly_equivalent(
    tmp_path, trained_padim_model, sample_batch
):
    """Exercise PaDiM training fixture, export, graph validation, and both backends."""

    batch, _, _, _ = sample_batch
    _assert_pt_and_onnx_match(
        trained_padim_model, batch.cpu(), tmp_path, "padim_regression"
    )


def test_patchcore_pt_and_onnx_inference_are_strictly_equivalent(
    tmp_path, monkeypatch
):
    """Exercise PatchCore export and inference without external datasets or GPUs."""

    monkeypatch.setattr(
        patchcore_module,
        "ResnetEmbeddingsExtractor",
        _DeterministicPatchCoreExtractor,
    )

    torch.manual_seed(42)
    images = torch.arange(6 * 3 * 8 * 8, dtype=torch.float32).reshape(6, 3, 8, 8)
    images = images / images.max()
    loader = DataLoader(TensorDataset(images), batch_size=2, shuffle=False)

    model = patchcore_module.PatchCore(
        device="cpu",
        layer_indices=[0],
        coreset_ratio=0.5,
        max_memory_patches=5,
        patch_grid=2,
        search_chunk_size=2,
        coreset_method="kcenter",
        coreset_seed=42,
    )
    model.fit(loader)
    assert model.memory_bank is not None
    assert model.memory_bank.shape[0] > 0

    _assert_pt_and_onnx_match(
        model, images[:2].contiguous(), tmp_path, "patchcore_regression"
    )
