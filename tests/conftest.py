import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# tests/conftest.py
import math

import pytest
import torch


class DummyExtractor:
    """
    A tiny stand-in for ResnetEmbeddingsExtractor used only in tests.
    It returns a fixed (B, N, D) embedding consistent with the provided stats.
    """

    def __init__(self, backbone_name: str, device: torch.device):
        self.backbone_name = backbone_name
        self.device = device

    def __call__(self, batch, *, channel_indices, layer_indices, layer_hook=None):
        # batch: (B, C, H, W) -> we don't use C; we only need B
        B = batch.shape[0]
        # We’ll infer N and D from the saved stats exposed via monkeypatch (set on the class)
        N = getattr(self, "_N", None)
        D = getattr(self, "_D", None)
        if N is None or D is None:
            raise RuntimeError("DummyExtractor._N/_D not set by the test.")
        # Create deterministic embeddings so tests are stable
        torch.manual_seed(0)
        emb = torch.randn(B, N, D, device=self.device, dtype=torch.float32)
        # width, height chosen so that N = width * height
        width = getattr(self, "_W", None)
        height = getattr(self, "_H", None)
        if width is None or height is None or width * height != N:
            # choose a sensible (W, H)
            width = int(math.sqrt(N))
            height = N // width
            if width * height != N:
                width, height = N, 1
        return emb, width, height


@pytest.fixture
def make_stats():
    """
    Create a tiny, valid PaDiM stats dict on CPU with fp16 storage (as your saver does).
    Returns: (stats_fp16, shapes) where shapes=(N, D, W, H)
    """

    def _factory(N=6, D=4, W=3, H=2):
        assert W * H == N, "W*H must equal N for a valid map."
        torch.manual_seed(1)
        mean = torch.randn(N, D, dtype=torch.float32)
        # build positive-definite covariance, then invert
        L = torch.randn(N, D, D)
        cov = L @ L.transpose(-1, -2) + 1e-3 * torch.eye(D).expand(N, D, D)
        cov_inv = torch.inverse(cov)

        stats = {
            "mean": mean.half().cpu(),
            "cov_inv": cov_inv.half().cpu(),
            "channel_indices": torch.arange(D, dtype=torch.int32),  # dummy indices
            "layer_indices": [2, 3],  # arbitrary
            "backbone": "resnet18",
            "model_version": "1.0",
        }
        return stats, (N, D, W, H)

    return _factory


@pytest.fixture
def patch_extractor(monkeypatch):
    """
    Monkeypatch the test DummyExtractor into anodet.padim_lite so PadimLite uses it.
    Also expose N/D/W/H on the extractor instance so it can return consistent embeddings.
    """
    from importlib import import_module

    padim_lite = import_module("anodet.padim_lite")

    def _apply(N, D, W, H):
        # bind a subclass that sets N/D/W/H on the instance
        class _Extractor(DummyExtractor):
            def __init__(self, backbone_name, device):
                super().__init__(backbone_name, device)
                # set the per-instance dimensions for this test
                self._N, self._D, self._W, self._H = N, D, W, H

        monkeypatch.setattr(
            padim_lite, "ResnetEmbeddingsExtractor", _Extractor, raising=True
        )
        return _Extractor

    return _apply
