# inference/model/backends/openvino_backend.py

"""
OpenVINO backend implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from anomavision.utils import get_logger

from .base import Batch, InferenceBackend, ScoresMaps

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
        """Initialize OpenVINO backend for optimized CPU/GPU inference.

        Creates an OpenVINO runtime optimized for Intel hardware. Supports both
        directory-based models (with .xml/.bin files) and single .xml files.

        Args:
            model_path (str): Path to OpenVINO model. Can be:
                - Directory containing .xml and .bin files
                - Direct path to .xml file
            device (str, optional): OpenVINO device target. Common values:
                - "CPU" → Intel CPU optimization
                - "GPU" → Intel GPU acceleration
                - "AUTO" → Automatic device selection
                Defaults to "CPU".
            num_threads (int | None, optional): Number of CPU threads for inference.
                Only applicable when device="CPU".

        Raises:
            ImportError: If OpenVINO is not installed.
            FileNotFoundError: If no .xml file found in directory.

        Example:
            >>> backend = OpenVinoBackend("model.xml", "CPU", num_threads=4)
            >>> backend = OpenVinoBackend("model_dir/", "AUTO")
        """

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
        """Run OpenVINO inference on input batch.

        Executes the compiled OpenVINO model on the input batch with hardware-specific
        optimizations. Automatically handles tensor conversion for OpenVINO runtime.

        Args:
            batch (Batch): Input batch of images. Supports torch.Tensor and numpy.ndarray.

        Returns:
            ScoresMaps: Tuple containing:
                - scores (np.ndarray): Per-image anomaly scores
                - maps (np.ndarray): Pixel-level anomaly maps

        Example:
            >>> batch = torch.randn(4, 3, 224, 224)
            >>> scores, maps = backend.predict(batch)
        """

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
        """Release OpenVINO runtime resources.

        Cleans up compiled model, core runtime, and associated resources.
        Important for proper resource management in long-running applications.
        """

        self.compiled_model = None
        self.model = None
        self.core = None

    def warmup(self, batch=None, runs: int = 2) -> None:
        """Warm up OpenVINO runtime for consistent performance.

        Creates dedicated inference request and performs initial runs to optimize
        runtime performance. Particularly beneficial for Intel hardware acceleration.

        Args:
            batch: Input batch for warmup. Must be provided for OpenVINO warmup.
            runs (int, optional): Number of warmup iterations. Defaults to 2.

        Example:
            >>> dummy_input = np.random.randn(1, 3, 224, 224)
            >>> backend.warmup(dummy_input, runs=5)
        """

        if isinstance(batch, np.ndarray):
            input_arr = batch
        else:
            input_arr = batch.detach().cpu().numpy()

        infer_request = self.compiled_model.create_infer_request()

        for _ in range(max(1, runs)):
            _ = infer_request.infer({self.input_layer.any_name: input_arr})

        logger.info(
            "OpenVinoBackend warm-up completed (runs=%d, shape=%s).",
            runs,
            tuple(input_arr.shape),
        )
