# anodet/config.py
import json, yaml
from pathlib import Path
from argparse import Namespace


def load_config(path: str = None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError("Config must be .yml/.yaml or .json")


def to_dict(ns: Namespace) -> dict:
    # turn argparse Namespace into a dict (ignores None later)
    return {k: v for k, v in vars(ns).items()}


def pick(*vals):
    # first non-None
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
