# inference/model/backends/openvino_backend.py

"""
OpenVINO backend implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)

try:
    import openvino.runtime as ov

    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


class OpenVinoBackend(InferenceBackend):
    """Inference backend based on OpenVINO Runtime."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        *,
        num_threads: int | None = None,
    ):
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO is not installed. Install with: pip install openvino"
            )

        self.device = device.upper()
        self.core = ov.Core()

        if self.device == "CPU" and num_threads:
            self.core.set_property("CPU", {"NUM_STREAMS": str(num_threads)})

        logger.info("Initializing OpenVINO with device=%s", self.device)

        # Handle directory or .xml file
        model_path = Path(model_path)

        if model_path.is_dir():
            xml_files = list(model_path.glob("*.xml"))
            if not xml_files:
                raise FileNotFoundError(f"No .xml model file found in {model_path}")
            model_file = xml_files[0]
        else:
            model_file = model_path

        self.model = self.core.read_model(model_file)
        self.compiled_model = self.core.compile_model(self.model, self.device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layers: List = [
            self.compiled_model.output(i)
            for i in range(len(self.compiled_model.outputs))
        ]

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run OpenVINO inference on the input batch."""
        if isinstance(batch, np.ndarray):
            input_arr = batch
        else:
            input_arr = batch.detach().cpu().numpy()

        logger.debug(
            "OpenVINO input shape: %s dtype: %s", input_arr.shape, input_arr.dtype
        )

        outputs = self.compiled_model([input_arr])

        scores = outputs[self.output_layers[0]]
        maps = outputs[self.output_layers[1]]

        logger.debug("OpenVINO output shapes: %s, %s", scores.shape, maps.shape)
        return scores, maps

    def close(self) -> None:
        """Release OpenVINO resources."""
        self.compiled_model = None
        self.model = None
        self.core = None
