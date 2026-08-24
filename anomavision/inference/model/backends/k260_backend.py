"""AMD Kria K26/KV260 Vitis AI XModel inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import InferenceBackend, ScoresMaps


class KV260Backend(InferenceBackend):
    """Run an AMD Vitis AI XModel through VART on a KV260/K26 target.

    Compiled AnomaVision XModels can contain both DPU and CPU subgraphs.  When
    the Vitis AI GraphRunner is available, use it so the complete graph is
    executed instead of accidentally running only the first child subgraph.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "k260",
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        """Load an AMD Vitis AI XModel through the VART runtime."""
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
        self.graph = xir.Graph.deserialize(str(self.model_path))
        self.runner = None
        self._graph_runner = False

        # A compiled XModel may be split into DPU + CPU subgraphs. GraphRunner
        # is the Vitis AI API intended for executing such complete graphs.
        try:
            from vitis_ai_library import GraphRunner

            self.runner = GraphRunner.create_graph_runner(self.graph)
            self._graph_runner = True
            self.input_tensors = [
                buffer.get_tensor() for buffer in self.runner.get_inputs()
            ]
            self.output_tensors = [
                buffer.get_tensor() for buffer in self.runner.get_outputs()
            ]
        except (ImportError, AttributeError, RuntimeError) as exc:
            # Keep a VART-only fallback for single-DPU XModels. This is useful
            # for minimal graphs, but cannot execute CPU post-processing nodes.
            children = self.graph.get_root_subgraph().toposort_child_subgraph()
            dpu_children = [
                child
                for child in children
                if child.has_attr("device") and child.get_attr("device") == "DPU"
            ]
            if len(dpu_children) != 1:
                raise RuntimeError(
                    "This XModel contains multiple subgraphs and requires "
                    "Vitis AI GraphRunner (vitis_ai_library)."
                ) from exc

            self.runner = vart.Runner.create_runner(dpu_children[0], "run")
            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()

        if not self.input_tensors:
            raise RuntimeError(f"No input tensor found in {self.model_path}")
        if len(self.output_tensors) < 2:
            raise ValueError(
                "KV260 XModel must expose image_scores and score_map outputs. "
                "A backbone-only XModel is not a complete PaDiM/PatchCore model."
            )

        self.input_tensor = self.input_tensors[0]
        self.output_names = [
            getattr(tensor, "name", "") for tensor in self.output_tensors
        ]
        self.input_shape = tuple(int(value) for value in self.input_tensor.dims)

    @staticmethod
    def _image_array(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        """Convert an image path, PIL image, or array into RGB HxWx3 data."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must be an HxWx3 RGB image")
        return array

    def _preprocess(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        """Resize and format an image for the XModel input tensor."""
        array = self._image_array(image)
        height, width = self.input_size
        resized = np.asarray(
            Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        batch = resized[None]
        if len(self.input_shape) == 4 and self.input_shape[-1] == 3:
            # Vitis AI DPU tensors are normally NHWC.
            pass
        elif len(self.input_shape) == 4 and self.input_shape[1] == 3:
            batch = np.transpose(batch, (0, 3, 1, 2))
        if np.issubdtype(np.dtype(self.input_tensor.dtype), np.floating):
            return (batch.astype(np.float32) / 255.0).astype(self.input_tensor.dtype)
        return batch.astype(self.input_tensor.dtype)

    @staticmethod
    def _copy_to_tensor_buffer(buffer: Any, data: np.ndarray) -> None:
        """Copy a NumPy array into a VART TensorBuffer."""
        target = np.asarray(buffer)
        if target.shape != data.shape:
            data = data.reshape(target.shape)
        np.copyto(target, data, casting="unsafe")

    def predict(self, batch: Any) -> ScoresMaps:
        """Run one image through the complete XModel graph."""
        input_data = self._preprocess(batch)

        if self._graph_runner:
            input_buffers = self.runner.get_inputs()
            output_buffers = self.runner.get_outputs()
            self._copy_to_tensor_buffer(input_buffers[0], input_data)
            for buffer in input_buffers:
                buffer.sync_for_write(
                    0,
                    buffer.get_tensor().get_data_size() // buffer.get_tensor().dims[0],
                )
            job_id, status = self.runner.execute_async(input_buffers, output_buffers)
            if status != 0:
                raise RuntimeError(f"KV260 GraphRunner execution failed: {status}")
            status = self.runner.wait(job_id)
            if status != 0:
                raise RuntimeError(f"KV260 GraphRunner wait failed: {status}")
            for buffer in output_buffers:
                buffer.sync_for_read(
                    0,
                    buffer.get_tensor().get_data_size() // buffer.get_tensor().dims[0],
                )
            outputs = [np.asarray(buffer).copy() for buffer in output_buffers]
        else:
            outputs = [
                np.empty(tuple(int(value) for value in tensor.dims), dtype=tensor.dtype)
                for tensor in self.output_tensors
            ]
            result = self.runner.execute_async([input_data], outputs)
            job_id = result[0] if isinstance(result, tuple) else result
            self.runner.wait(job_id)

        image_index = next(
            (
                index
                for index, name in enumerate(self.output_names)
                if "image_score" in name or "score" in name
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
        """Release the VART/GraphRunner resources."""
        self.runner = None


# Preserve the shorter name used by earlier experimental code.
K260Backend = KV260Backend
