import json
import sys
from pathlib import Path

# apps/api.py is also run as a script, so make its sibling inference module importable.
_APPS_DIR = str(Path(__file__).resolve().parents[1] / "apps")
if _APPS_DIR not in sys.path:
    sys.path.insert(0, _APPS_DIR)

import pytest

from apps import api as studio_inference_api
from apps.inference_engine import _algorithm_for_model
from apps.studio.services import catalog, deployment_service


def _write_project_config(root: Path, project_id: str, config: dict) -> Path:
    project_dir = root / project_id
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True)
    config_path = project_dir / "training.yml"
    config_path.write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in config.items())
        + "\n",
        encoding="utf-8",
    )
    (models_dir / "latest_training.json").write_text(
        json.dumps({"config": str(config_path)}), encoding="utf-8"
    )
    return project_dir


def test_disabled_drift_monitor_clears_stale_project_state(tmp_path, monkeypatch):
    project_id = "disabled-monitor"
    _write_project_config(
        tmp_path,
        project_id,
        {"algorithm": "padim", "enable_drift_monitoring": False},
    )
    monkeypatch.setenv("ANOMAVISION_STUDIO_ROOT", str(tmp_path))
    monkeypatch.setattr(studio_inference_api, "_drift_monitor", object())
    monkeypatch.setattr(studio_inference_api, "_drift_project_id", "previous-project")
    monkeypatch.setattr(studio_inference_api, "_drift_config", {"stale": True})

    studio_inference_api._apply_project_config(project_id)

    assert studio_inference_api._drift_monitor is None
    assert studio_inference_api._drift_project_id is None
    assert studio_inference_api._drift_config == {}


def test_invalid_drift_monitor_initialization_does_not_block_config(tmp_path, monkeypatch):
    project_id = "invalid-monitor"
    _write_project_config(
        tmp_path,
        project_id,
        {
            "algorithm": "patchcore",
            "enable_drift_monitoring": True,
            "thresh_patchcore": 0.42,
        },
    )
    monkeypatch.setenv("ANOMAVISION_STUDIO_ROOT", str(tmp_path))
    monkeypatch.setattr(studio_inference_api, "_drift_monitor", None)
    monkeypatch.setattr(studio_inference_api, "_drift_project_id", None)
    monkeypatch.setattr(studio_inference_api, "_drift_config", {})
    monkeypatch.setattr(
        studio_inference_api.engine,
        "ANOMALY_THRESHOLD",
        studio_inference_api.engine.ANOMALY_THRESHOLD,
    )
    monkeypatch.setattr(
        studio_inference_api,
        "_load_project_drift_monitor",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid reference")),
    )

    studio_inference_api._apply_project_config(project_id)

    assert studio_inference_api._drift_monitor is None
    assert studio_inference_api._drift_project_id is None
    assert studio_inference_api._drift_config == {}
    assert studio_inference_api.engine.ANOMALY_THRESHOLD == pytest.approx(0.42)


def test_algorithm_resolution_uses_metadata_for_model_project(tmp_path, monkeypatch):
    project_dir = _write_project_config(
        tmp_path, "patchcore-project", {"algorithm": "patchcore"}
    )
    model_path = project_dir / "deployments" / "run-1" / "onnx" / "model.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"test artifact")
    monkeypatch.setattr("apps.inference_engine.STUDIO_ROOT", str(tmp_path))

    assert _algorithm_for_model(None, str(model_path)) == "patchcore"


def test_algorithm_resolution_does_not_use_another_projects_metadata(tmp_path, monkeypatch):
    _write_project_config(tmp_path, "other-project", {"algorithm": "patchcore"})
    model_path = tmp_path / "model-padim.onnx"
    model_path.write_bytes(b"test artifact")
    monkeypatch.setattr("apps.inference_engine.STUDIO_ROOT", str(tmp_path))

    assert _algorithm_for_model(None, str(model_path)) == "padim"


def test_studio_deployment_catalog_matches_implemented_targets():
    expected = {"cpu", *deployment_service.TARGET_FORMATS.keys()}

    assert set(catalog.DEPLOYMENT_TARGETS) == expected
    assert "hailo" not in catalog.DEPLOYMENT_TARGETS
    assert "kv260" not in catalog.DEPLOYMENT_TARGETS


@pytest.mark.parametrize("target", ["hailo", "kv260"])
def test_unimplemented_hardware_targets_are_rejected(tmp_path, target):
    model = {
        "path": str(tmp_path / "model.pt"),
        "config": str(tmp_path / "config.yml"),
        "run_name": "run",
    }

    with pytest.raises(ValueError, match="not yet supported"):
        deployment_service.deploy_model(tmp_path, model, target)
