# anodet/config.py
import json
from argparse import Namespace
from pathlib import Path

import yaml


def load_config(path: str = None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        config = yaml.safe_load(p.read_text()) or {}
    elif p.suffix.lower() == ".json":
        config = json.loads(p.read_text()) or {}
    else:
        raise ValueError("Config must be .yml/.yaml or .json")

    # Native model selector: allow either historical ``algorithm: padim``
    # or ``model: {name: padim}`` configuration.
    model_section = config.get("model")
    if isinstance(model_section, dict):
        if not config.get("algorithm") and model_section.get("name"):
            config["algorithm"] = model_section["name"]
        if model_section.get("file") and not config.get("model_path"):
            config["model_path"] = model_section["file"]
        # Existing detect/export code expects ``model`` to be the artifact name.
        config["model"] = config.get("model_path")
    return config


def to_dict(ns: Namespace) -> dict:
    return {k: v for k, v in vars(ns).items()}


def pick(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _shape(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        if len(v) == 1:
            return int(v[0])
        if len(v) == 2:
            return (int(v[0]), int(v[1]))
    raise ValueError("resize/crop_size must be int, [int], [h,w], or None")
