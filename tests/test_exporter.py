import os
from pathlib import Path

import pytest
import torch

from anomavision.utils import get_logger, setup_logging

# Adjust this import to your actual exporter module filename if needed
# e.g., from export import ModelExporter, _ExportWrapper
from export import ModelExporter, _ExportWrapper  # noqa: F401

setup_logging("INFO")
logger = get_logger(__name__)


class TinyModel(torch.nn.Module):
    """
    Minimal model with a .predict(...) that returns (scores, maps)
    to match your _ExportWrapper's expectation.
    """

    def __init__(self, in_ch=3, h=16, w=16):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        self.h, self.w = h, w

    def forward(self, x):
        # Not used directly by exporter; _ExportWrapper calls predict(..)
        return self.predict(x)

    @torch.no_grad()
    def predict(self, x, **kwargs):
        feat = self.conv(x)  # (B,1,H,W)
        scores = feat.mean(dim=(2, 3))  # (B,1)
        maps = feat
        return scores, maps


def _save_tiny_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "tiny_model.pt"
    m = TinyModel()
    torch.save(m, model_path)

    # stats_path = model_path.with_suffix(".pth")
    # try:
    #     m.save_statistics(str(stats_path))
    #     logger.info("saved: slim statistics=%s", stats_path)
    # except Exception as e:
    #     logger.warning("saving slim statistics failed: %s", e)

    assert model_path.exists()
    return model_path


def test_export_onnx_creates_file(tmp_path):
    model_path = _save_tiny_model(tmp_path)
    exporter = ModelExporter(str(model_path), str(tmp_path), logger)

    out = exporter.export_onnx(
        input_shape=(1, 3, 16, 16),
        output_name="tiny.onnx",
        opset_version=17,
        dynamic_batch=True,
    )
    assert out is not None
    assert out.exists()
    assert out.suffix == ".onnx"
    assert out.stat().st_size > 0


def test_export_torchscript_creates_file_and_loads(tmp_path):
    model_path = _save_tiny_model(tmp_path)
    exporter = ModelExporter(str(model_path), str(tmp_path), logger)

    out = exporter.export_torchscript(
        input_shape=(1, 3, 16, 16),
        output_name="tiny.torchscript",
        optimize=False,
    )
    assert out is not None
    assert out.exists()
    assert out.suffix == ".torchscript"
    # Ensure it’s loadable
    loaded = torch.jit.load(str(out))
    assert isinstance(loaded, torch.jit.RecursiveScriptModule)


def test_export_openvino_returns_none_if_not_installed(tmp_path, monkeypatch):
    """
    If OpenVINO isn't installed (common in CI), exporter should fail gracefully
    and return None. If it IS installed, we still accept a valid export.
    """
    model_path = _save_tiny_model(tmp_path)
    exporter = ModelExporter(str(model_path), str(tmp_path), logger)

    try:
        import openvino  # noqa: F401

        # OpenVINO available: expect a real directory result
        out = exporter.export_openvino(
            input_shape=(1, 3, 16, 16),
            output_name="tiny_openvino",
            fp16=True,
            dynamic_batch=False,
        )
        assert out is not None
        assert out.exists() and out.is_dir()
        # The XML is typically saved under <dir>/<name>.xml
        xml = out / "tiny_openvino.xml"
        assert xml.exists()
    except Exception:
        # Not installed: exporter should return None (as implemented)
        out = exporter.export_openvino(
            input_shape=(1, 3, 16, 16),
            output_name="tiny_openvino",
            fp16=True,
            dynamic_batch=False,
        )
        assert out is None
