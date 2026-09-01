"""Persist inspection results and optional visual evidence locally."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from anomavision.actions.ActionBase import ActionBase


class EvidenceAction(ActionBase):
    """Save inspection metadata and optional image/heatmap evidence.

    The action accepts the normalized result dictionary and may also consume
    ``image`` and ``heatmap`` numpy/PIL values when supplied by a caller.
    Binary image data is never sent to industrial transports.
    """

    def __init__(
        self,
        directory: str = "./evidence",
        save_pass: bool = False,
        save_fail: bool = True,
        save_unknown: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.save_pass = bool(save_pass)
        self.save_fail = bool(save_fail)
        self.save_unknown = bool(save_unknown)
        self._connected = False

    def connect(self) -> bool:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._connected = True
        return True

    def execute(self, result: Dict[str, Any]) -> bool:
        if not self._connected:
            return False

        decision = str(result.get("decision") or "UNKNOWN").upper()
        should_save = {
            "PASS": self.save_pass,
            "FAIL": self.save_fail,
            "UNKNOWN": self.save_unknown,
        }.get(decision, self.save_unknown)
        if not should_save:
            return True

        event_id = str(result.get("event_id") or "event")
        event_dir = self.directory / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        metadata = dict(result)
        image = metadata.pop("image", None)
        heatmap = metadata.pop("heatmap", None)
        metadata["evidence_directory"] = str(event_dir)

        with (event_dir / "event.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, default=str)

        if image is not None:
            self._save_image(image, event_dir / "image.png")
        if heatmap is not None:
            self._save_image(heatmap, event_dir / "heatmap.png")

        return True

    @staticmethod
    def _save_image(image: Any, path: Path) -> None:
        if isinstance(image, Image.Image):
            image.save(path)
            return
        if isinstance(image, np.ndarray):
            array = image
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(path)
            return
        raise TypeError("Evidence image must be a PIL Image or numpy array")

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected
