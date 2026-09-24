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
    """Resolve any common MVTec folder selection to train.py's exact contract.

    train.py receives:
        dataset_path=<dataset root>
        class_name=<class>
    and reads:
        <dataset_path>/<class_name>/train/good

    Supported selections:
      <dataset>
      <dataset>/<class>
      <dataset>/<class>/train
      <dataset>/<class>/train/good
      <dataset>/train/good
    """
    selected = Path(source).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"Dataset folder does not exist: {selected}")

    parts = [p.lower() for p in selected.parts]

    # A complete class training folder: .../<class>/train/good
    if selected.name.lower() == "good" and selected.parent.name.lower() == "train":
        class_dir = selected.parent.parent
        return class_dir.parent, class_dir.name

    # A class training folder: .../<class>/train
    if selected.name.lower() == "train" and (selected / "good").is_dir():
        class_dir = selected.parent
        return class_dir.parent, class_dir.name

    # A class folder: .../<class>/train/good
    if (selected / "train" / "good").is_dir():
        return selected.parent, selected.name

    # Dataset root with a named class.
    requested_class = (class_name or "").strip()
    if requested_class and requested_class.lower() != "default":
        class_good = selected / requested_class / "train" / "good"
        if class_good.is_dir():
            return selected, requested_class

        # Case-insensitive class directory lookup.
        for child in selected.iterdir():
            if child.is_dir() and child.name.lower() == requested_class.lower():
                if (child / "train" / "good").is_dir():
                    return selected, child.name

    # Dataset root with exactly one discoverable class.
    candidates = [
        child for child in selected.iterdir()
        if child.is_dir() and (child / "train" / "good").is_dir()
    ]
    if len(candidates) == 1:
        return selected, candidates[0].name

    # Dataset root that directly contains train/good.
    if (selected / "train" / "good").is_dir():
        effective_class = requested_class if requested_class and requested_class.lower() != "default" else selected.name
        return selected, effective_class

    raise ValueError(
        "Invalid dataset layout. Select the dataset root, class folder, "
        "train folder, or train/good folder."
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
