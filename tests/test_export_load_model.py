# tests/test_export_load_model.py
import torch

from anomavision.utils import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger(__name__)


def test_exporter_loads_stats_pth_and_wraps(tmp_path, make_stats, patch_extractor):
    """
    Validates the patched _load_model in export.py:
      - If torch.load returns a stats dict (.pth), it builds PadimLite
      - Returns an _ExportWrapper that calls .predict under the hood
    """
    stats, (N, D, W, H) = make_stats()
    patch_extractor(N, D, W, H)

    pth_path = tmp_path / "model_stats.pth"
    torch.save(stats, pth_path)

    from export import ModelExporter, _ExportWrapper

    exp = ModelExporter(model_path=pth_path, output_dir=tmp_path, logger=logger,device="cpu")
    m = exp._load_model()  # should be _ExportWrapper(PadimLite)
    assert isinstance(m, _ExportWrapper)
    assert hasattr(m, "forward")

    # Forward should call .predict and return (scores, maps)
    x = torch.zeros(1, 3, 96, 96)
    scores, maps = m(x)
    assert isinstance(scores, torch.Tensor) and isinstance(maps, torch.Tensor)
    assert scores.shape == (1,)
    assert maps.shape == (1, x.shape[2], x.shape[3])
