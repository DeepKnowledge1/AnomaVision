"""Compatibility entry point for the AnomaVision production dashboard.

Gradio only serves files from approved directories. Production image datasets
can live outside the repository, so this entry point adds the configured image
root to Gradio's allowed paths before starting the dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr


def _configured_image_root() -> Path | None:
    configured = os.getenv("ANOMAVISION_DRIFT_IMAGE_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.exists() else None

    config_path = Path(os.getenv("ANOMAVISION_DRIFT_CONFIG", "./config.yml"))
    if not config_path.exists():
        return None

    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        value = config.get("img_path") if isinstance(config, dict) else None
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            else:
                path = path.resolve()
            return path if path.exists() else None
    except Exception:
        return None
    return None


_original_launch = gr.Blocks.launch


def _launch_with_allowed_paths(self, *args, **kwargs):
    allowed = list(kwargs.pop("allowed_paths", None) or [])
    image_root = _configured_image_root()
    if image_root is not None:
        allowed.append(str(image_root if image_root.is_dir() else image_root.parent))
    # Remove duplicates while preserving order.
    kwargs["allowed_paths"] = list(dict.fromkeys(allowed))
    return _original_launch(self, *args, **kwargs)


graphics_launch = _launch_with_allowed_paths
gr.Blocks.launch = graphics_launch

from anomavision.drift_dashboard import main


if __name__ == "__main__":
    main()
