"""Model registry and evaluation adapters for AnomaVision Studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_models(project_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(project_dir) / "models"
    models: list[dict[str, Any]] = []
    for model_path in root.glob("*/*/*/model.pt"):
        run_dir = model_path.parent
        config_path = run_dir / "config.yml"
        metadata = {}
        if config_path.is_file():
            import yaml
            try:
                metadata = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                metadata = {}
        models.append({
            "id": str(model_path.relative_to(root)),
            "path": str(model_path),
            "algorithm": metadata.get("algorithm", model_path.parts[-4]),
            "class_name": metadata.get("class_name", model_path.parts[-3]),
            "run_name": metadata.get("run_name", model_path.parts[-2]),
            "config": str(config_path),
            "status": "trained",
        })

    metadata_path = root / "latest_training.json"
    if metadata_path.is_file():
        try:
            latest = json.loads(metadata_path.read_text(encoding="utf-8"))
            for model in models:
                if model["path"] == latest.get("model"):
                    model["status"] = latest.get("status", "trained")
        except (OSError, json.JSONDecodeError):
            pass

    return sorted(models, key=lambda item: item["run_name"], reverse=True)


def get_model(project_dir: Path | str, model_id: str) -> dict[str, Any]:
    for model in list_models(project_dir):
        if model["id"] == model_id:
            return model
    raise ValueError(f"Model not found: {model_id}")
