"""AMD Kria K26/KV260 Vitis AI XModel inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import InferenceBackend, ScoresMaps


class KV260Backend(InferenceBackend):
    """Run an AMD Vitis AI XModel through VART on a KV260/K26 target."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "k260",
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        del device
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        try:
            import vart
            import xir
        except ImportError as exc:  # pragma: no cover - target-only dependency
            raise RuntimeError(
                "Vitis AI VART/XIR is required for KV260 XModel inference."
            ) from exc
        self._vart = vart
        self.input_size = tuple(int(value) for value in input_size)
        graph = xir.Graph.deserialize(str(self.model_path))
        runners = graph.get_root_subgraph().children_topological_sort()
        if not runners:
            raise RuntimeError(f"No runnable subgraph found in {self.model_path}")
        self.runner = vart.Runner.create_runner(runners[0], "run")
        self.input_tensor = self.runner.get_input_tensors()[0]
        self.output_tensors = self.runner.get_output_tensors()
        if len(self.output_tensors) < 2:
            raise ValueError(
                "KV260 XModel must expose image_scores and score_map outputs"
            )
        self.output_names = [tensor.name for tensor in self.output_tensors]
        self.input_shape = tuple(int(value) for value in self.input_tensor.dims)

    @staticmethod
    def _image_array(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must be an HxWx3 RGB image")
        return array

    def _preprocess(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        array = self._image_array(image)
        height, width = self.input_size
        resized = np.asarray(
            Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        batch = resized[None]
        if len(self.input_shape) == 4 and self.input_shape[1] == 3:
            batch = np.transpose(batch, (0, 3, 1, 2))
        if np.issubdtype(self.input_tensor.dtype, np.floating):
            return (batch.astype(np.float32) / 255.0).astype(self.input_tensor.dtype)
        return batch.astype(self.input_tensor.dtype)

    def predict(self, batch: Any) -> ScoresMaps:
        """Run one image and return ``(image_scores, score_maps)``."""
        input_data = self._preprocess(batch)
        outputs = [
            np.empty(tuple(int(value) for value in tensor.dims), dtype=tensor.dtype)
            for tensor in self.output_tensors
        ]
        job_id = self.runner.execute_async([input_data], outputs)
        self.runner.wait(job_id)
        image_index = next(
            (
                index
                for index, name in enumerate(self.output_names)
                if "image_score" in name
            ),
            0,
        )
        map_index = next(
            (
                index
                for index, name in enumerate(self.output_names)
                if "score_map" in name or "map" in name
            ),
            1,
        )
        return (
            np.asarray(outputs[image_index]).squeeze(),
            np.asarray(outputs[map_index]).squeeze(),
        )

    def close(self) -> None:
        """Release the VART runner."""
        self.runner = None


# Preserve the shorter name used by earlier experimental code.
K260Backend = KV260Backend
