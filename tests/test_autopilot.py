import json
from pathlib import Path

from anomavision.autopilot import _format_metric, _select, _write_report, create_parser


def _candidate(image_auroc, p95, threshold=0.3):
    return {
        "threshold": threshold,
        "model_format": ".pt",
        "metrics": {"image_auroc": image_auroc, "pixel_auroc": image_auroc - 0.1},
        "latency_ms": {"median": p95 / 2, "p95": p95},
        "localization": {
            "available": True,
            "non_empty_fraction": 0.5,
            "mean_mask_area_fraction": 0.08,
            "anomaly_non_empty_fraction": 0.75,
            "normal_false_positive_fraction": 0.05,
            "anomaly_mean_mask_area_fraction": 0.12,
            "verdict": "healthy",
        },
    }


def test_autopilot_formats_unavailable_metrics_honestly():
    assert _format_metric(None) == "N/A"
    assert _format_metric(0.91234) == "0.9123"


def test_autopilot_selects_best_eligible_model():
    candidates = {"padim": _candidate(0.95, 80), "patchcore": _candidate(0.90, 20)}
    assert _select(candidates, target_latency_ms=30) == "patchcore"
    assert _select(candidates, target_latency_ms=None) == "padim"


def test_autopilot_report_contains_manifest_summary(tmp_path):
    manifest = {
        "schema_version": 1,
        "selected_model": "patchcore",
        "selected_artifact": "model.pt",
        "dataset": {"class_name": "bottle", "samples": 4},
        "preprocessing": {"resize": 224},
        "target_latency_ms": 50,
        "candidates": {"patchcore": _candidate(0.9, 20)},
        "environment": {"device": "cpu"},
    }
    _write_report(manifest, tmp_path)
    report = (tmp_path / "localization_report.md").read_text(encoding="utf-8")
    html = (tmp_path / "production_autopilot_report.html").read_text(encoding="utf-8")
    assert "Production Autopilot Report" in report
    assert "patchcore" in report
    assert "0.9000" in html
    assert "Deployment confidence, before production." in html
    assert "<html" in html
    assert "Anomaly coverage" in html
    assert "Normal images with false-positive maps" in html
    assert "healthy" in html


def test_autopilot_parser_exposes_production_controls():
    args = create_parser().parse_args(
        ["--config", "config.yml", "--target_latency_ms", "50"]
    )
    assert args.target_latency_ms == 50
    assert args.output_dir == "./production_package"
