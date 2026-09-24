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
