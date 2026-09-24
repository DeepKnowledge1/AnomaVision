from fastapi.testclient import TestClient

from apps.studio.api import app as studio_api


def test_health():
    client = TestClient(studio_api.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_catalog():
    client = TestClient(studio_api.app)
    response = client.get("/api/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert "patchcore" in payload["algorithms"]
    assert "cpu" in payload["deployment_targets"]


def test_projects_uses_isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_api, "store", studio_api.ProjectStore(tmp_path))
    monkeypatch.setattr(studio_api, "STORE_ROOT", tmp_path)

    client = TestClient(studio_api.app)
    response = client.post(
        "/api/projects",
        json={
            "name": "Bottle Inspection",
            "algorithm": "patchcore",
            "description": "test project",
        },
    )

    assert response.status_code == 201
    project = response.json()
    assert project["id"] == "bottle-inspection"
    assert client.get("/api/projects/bottle-inspection").status_code == 200


def test_config_reads_canonical_values(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text("""
algorithm: patchcore
resize: [224, 224]
normalize: true
thresh_padim: 13.0
thresh_patchcore: 0.25
thresh_efficientad: 1.0
stream_display_fps: true
stream_max_frames: null
drift_window: 50
drift_min_samples: 5
drift_threshold: 0.20
drift_evaluation_interval: 25
enable_drift_monitoring: false
""", encoding="utf-8")
    monkeypatch.setenv("ANOMAVISION_CONFIG", str(config))

    client = TestClient(studio_api.app)
    response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["algorithm"] == "patchcore"
    assert payload["resize"] == [224, 224]
    assert payload["thresholds"]["patchcore"] == 0.25
    assert payload["drift"]["window"] == 50
