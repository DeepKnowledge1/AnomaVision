"""Filesystem-backed project storage for AnomaVision Studio.

The first Studio release deliberately avoids a database. Project metadata is
small, portable, and easy to inspect or back up on an edge machine.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".anomavision" / "projects"


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "project"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    """Create, list and read Studio projects using JSON metadata."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or DEFAULT_ROOT).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for metadata_path in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get(self, project_id: str) -> dict[str, Any] | None:
        path = self.root / project_id / "project.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def create(self, name: str, algorithm: str = "patchcore") -> dict[str, Any]:
        project_id = _slugify(name)
        project_dir = self.root / project_id
        if project_dir.exists():
            raise ValueError(f"Project already exists: {project_id}")

        now = _now()
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "id": project_id,
            "name": name.strip(),
            "description": "",
            "algorithm": algorithm.lower(),
            "status": "created",
            "created_at": now,
            "updated_at": now,
        }
        for directory in ("datasets", "experiments", "models", "deployments", "monitoring"):
            (project_dir / directory).mkdir(parents=True, exist_ok=True)
        (project_dir / "project.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return metadata
