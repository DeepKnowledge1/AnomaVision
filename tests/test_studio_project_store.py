from pathlib import Path

from apps.studio.services.project_store import ProjectStore


def test_project_store_creates_portable_project(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Bottle Cap Inspection", "patchcore")

    assert project["id"] == "bottle-cap-inspection"
    assert project["algorithm"] == "patchcore"
    assert (tmp_path / "projects" / "bottle-cap-inspection" / "project.json").is_file()
    assert store.get("bottle-cap-inspection")["name"] == "Bottle Cap Inspection"


def test_project_store_rejects_duplicate(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    store.create("Bottle Cap Inspection")
    try:
        store.create("Bottle Cap Inspection")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate project creation to fail")
