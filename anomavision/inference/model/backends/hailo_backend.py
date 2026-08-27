"""HailoRT runtime for complete AnomaVision anomaly HEFs.

A complete HEF contains the full feature extraction, distance calculation,
score-map generation, and image-score calculation. The runtime accepts the
stable AnomaVision output names when preserved by Hailo, and otherwise maps the
first two output streams by their validated shapes. No anomaly calculation is
performed on the CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .base import InferenceBackend


class HailoAnomalyRuntime:
    """Run a complete PaDiM or PatchCore HEF through HailoRT."""

    def __init__(
        self,
        hef_path: str | Path,
        input_size: Tuple[int, int] = (224, 224),
        input_dtype: np.dtype = np.float32,
    ) -> None:
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HailoRT is not installed. Install the HailoRT Python package on "
                "the target device before loading a HEF."
            ) from exc

        self._api = {
            "ConfigureParams": ConfigureParams,
            "FormatType": FormatType,
            "HEF": HEF,
            "HailoStreamInterface": HailoStreamInterface,
            "InputVStreamParams": InputVStreamParams,
            "InferVStreams": InferVStreams,
            "OutputVStreamParams": OutputVStreamParams,
            "VDevice": VDevice,
        }
        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(self.hef_path)
        self.input_size = tuple(int(v) for v in input_size)
        self.input_dtype = input_dtype
        self.device = VDevice()
        self.hef = HEF(str(self.hef_path))
        self.network_groups = self.device.configure(self.hef)
        if not self.network_groups:
            raise RuntimeError(f"No network group found in {self.hef_path}")
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        input_infos = self.hef.get_input_vstream_infos()
        output_infos = self.hef.get_output_vstream_infos()
        if not input_infos:
            raise ValueError(f"The HEF has no input streams: {self.hef_path}")
        if len(output_infos) < 2:
            raise ValueError(
                f"The HEF must expose image score and score map outputs; found {len(output_infos)}"
            )
        self.input_name = input_infos[0].name
        self.input_shape = tuple(int(v) for v in input_infos[0].shape)
        self.output_names = [info.name for info in output_infos]
        self.output_shapes = {info.name: tuple(int(v) for v in info.shape) for info in output_infos}

        if "image_scores" in self.output_names and "score_map" in self.output_names:
            self.score_output_name = "image_scores"
            self.map_output_name = "score_map"
        else:
            # Hailo may rename ONNX outputs to internal end-node names during
            # parsing (for example MaxPool/Squeeze). Validate by shape rather
            # than relying on those generated names.
            score_candidates = []
            map_candidates = []
            for name, shape in self.output_shapes.items():
                dims = tuple(d for d in shape if d > 1)
                if len(dims) <= 1:
                    score_candidates.append(name)
                elif len(dims) == 2 and dims[-2:] == self.input_size:
                    map_candidates.append(name)
                elif len(shape) == 3 and shape[-2:] == self.input_size:
                    map_candidates.append(name)
                elif len(shape) == 4 and shape[-3:] == (1, *self.input_size):
                    map_candidates.append(name)

            if len(score_candidates) != 1 or len(map_candidates) != 1:
                raise ValueError(
                    "Could not identify the complete AnomaVision outputs in the HEF. "
                    f"Available streams: {self.output_shapes}. Expected one scalar "
                    "image-score stream and one 224x224 score-map stream."
                )
            self.score_output_name = score_candidates[0]
            self.map_output_name = map_candidates[0]

        print(
            f"Hailo outputs: score={self.score_output_name}, map={self.map_output_name}"
        )

    def _preprocess(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        """Convert an image to the HEF input layout without changing normalization."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        array = np.asarray(image)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("HailoAnomalyRuntime currently supports batch size 1")
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("image must be an HxWx3 RGB image or 1x3xHxW tensor")

        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32, copy=False)

        target_h, target_w = self.input_size
        if array.shape[:2] != (target_h, target_w):
            if np.nanmin(array) < 0.0 or np.nanmax(array) > 1.0:
                raise ValueError(
                    "Preprocessed float input has values outside [0, 1] and cannot "
                    "be safely resized. Use the detect preprocessing pipeline."
                )
            image_u8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            array = np.asarray(
                Image.fromarray(image_u8, mode="RGB").resize(
                    (target_w, target_h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0

        return array[None].astype(self.input_dtype, copy=False)

    def predict(self, image: Image.Image | np.ndarray | str | Path) -> Dict[str, np.ndarray]:
        """Run one image and return complete image and localization outputs."""
        api = self._api
        input_params = api["InputVStreamParams"].make(
            self.network_group, quantized=False, format_type=api["FormatType"].FLOAT32
        )
        output_params = api["OutputVStreamParams"].make(
            self.network_group, quantized=False, format_type=api["FormatType"].FLOAT32
        )
        tensor = self._preprocess(image)
        with self.network_group.activate(self.network_group_params):
            with api["InferVStreams"](
                self.network_group, input_params, output_params
            ) as infer_pipeline:
                outputs = infer_pipeline.infer({self.input_name: tensor})

        score = np.asarray(outputs[self.score_output_name]).squeeze()
        score_map = np.asarray(outputs[self.map_output_name]).squeeze()
        return {"image_scores": score, "score_map": score_map}

    def close(self) -> None:
        release = getattr(self.device, "release", None)
        if callable(release):
            release()

    def __enter__(self) -> "HailoAnomalyRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class HailoBackend(InferenceBackend):
    """AnomaVision inference backend for a complete Hailo-8 HEF."""

    def __init__(self, model_path: str | Path, device: str = "hailo", input_size: Tuple[int, int] = (224, 224)) -> None:
        del device
        self.runtime = HailoAnomalyRuntime(model_path, input_size=input_size)

    def predict(self, batch) -> Tuple[np.ndarray, np.ndarray]:
        """Run one preprocessed image through the common backend contract."""
        array = np.asarray(batch)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("HailoBackend currently supports batch size 1")
            array = array[0]
        if array.ndim != 3:
            raise ValueError("batch must be an HxWx3 or 1x3xHxW image")
        result = self.runtime.predict(array)
        return result["image_scores"], result["score_map"]

    def warmup(self, batch=None, runs: int = 2) -> None:
        if batch is None:
            raise ValueError("Hailo warmup requires a sample image batch")
        for _ in range(max(1, int(runs))):
            self.predict(batch)

    def close(self) -> None:
        self.runtime.close()
