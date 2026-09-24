"""Thin Studio adapter around AnomaVision's existing training entry point."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from anomavision.train import run_training


def normalize_dataset_source(source: str, class_name: str = "default") -> tuple[Path, str]:
    """Map common Studio image-folder layouts to the existing train.py contract.

    Accepted layouts:
      dataset/class/train/good
      dataset/train/good
      dataset/class        (containing train/good)
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise ValueError(f"Dataset folder does not exist: {source_path}")

    if (source_path / "train" / "good").is_dir():
        # The existing train.py already supports dataset roots that directly
        # contain train/good, so keep this source untouched.
        return source_path, class_name

    if source_path.name.lower() == "good" and source_path.parent.name.lower() == "train":
        train_root = source_path.parent.parent
        return train_root.parent, train_root.name

    # Canonical AnomaVision layout: dataset_root/<class>/train/good.\n    if class_name and (source_path / class_name / "train" / "good").is_dir():\n        return source_path, class_name\n\n    if (source_path / "class_name_placeholder" / "train" / "good").is_dir():
        return source_path, class_name

    raise ValueError(
        "Training data must contain a 'train/good' folder. "
        "Select the dataset root or the class folder."
    )


def _load_base_config() -> dict[str, Any]:
    """Load AnomaVision's canonical repository config as the Studio template."""
    repo_config = Path(__file__).resolve().parents[3] / "config.yml"
    if not repo_config.is_file():
        raise FileNotFoundError(f"Canonical AnomaVision config not found: {repo_config}")

    config = yaml.safe_load(repo_config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Canonical config must contain a YAML mapping: {repo_config}")
    return config


def create_training_config(
    project_dir: Path | str,
    dataset_source: str,
    algorithm: str,
    class_name: str = "default",
    **overrides: Any,
) -> Path:
    """Create a Studio run config from AnomaVision's canonical config.

    Studio only overrides values selected by the UI. The core configuration
    schema stays owned by the main AnomaVision config.yml.
    """
    dataset_path, detected_class = normalize_dataset_source(dataset_source, class_name)
    effective_class = detected_class if detected_class != "default" else class_name

    run_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{algorithm}-{effective_class}").strip("-").lower()
    run_dir = Path(project_dir) / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _load_base_config()
    config.update(
        {
            "dataset_path": str(dataset_path),
            "class_name": effective_class,
            "algorithm": algorithm.lower(),
            "model_data_path": str(Path(project_dir) / "models"),
            "run_name": run_id,
        }
    )
    config.update(overrides)

    config_path = run_dir / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def train_project(
    project_dir: Path | str,
    dataset_source: str,
    algorithm: str,
    class_name: str = "default",
    **overrides: Any,
) -> dict[str, Any]:
    """Run the existing AnomaVision training implementation and register metadata."""
    project_dir = Path(project_dir)
    config_path = create_training_config(
        project_dir, dataset_source, algorithm, class_name, **overrides
    )

    args = Namespace(config=str(config_path))
    _, config, run_dir, _ = run_training(args)

    metadata = {
        "algorithm": str(config.algorithm),
        "class_name": str(config.class_name),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "model": str(Path(run_dir) / str(config.output_model)),
        "status": "completed",
    }
    metadata_path = project_dir / "models" / "latest_training.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
