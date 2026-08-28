"""HailoRT runtime for complete AnomaVision anomaly HEFs.

The HEF is expected to expose ``image_scores`` and ``score_map``. Feature
extraction and anomaly scoring are compiled into the HEF; this runtime only
adapts the common AnomaVision inference input contract to HailoRT.

The public backend contract matches the ONNX backend: input is a single
ImageNet-normalized RGB tensor in NCHW float32. Hailo receives the same values
in NHWC layout. No second resize or normalization is applied to tensors that
have already passed through the AnomaVision dataset preprocessing pipeline.
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
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
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
        except ImportError as exc:  # pragma: no cover - depends on Kria image
            raise RuntimeError(
                "HailoRT is not installed. Install the HailoRT Python package on "
                "the Kria K26 image before loading a HEF."
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
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.device = VDevice()
        self.hef = HEF(str(self.hef_path))
        self.network_groups = self.device.configure(self.hef)
        if not self.network_groups:
            raise RuntimeError(f"No network group found in {self.hef_path}")
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        input_infos = self.hef.get_input_vstream_infos()
        if not input_infos:
            raise ValueError(f"The HEF has no input stream: {self.hef_path}")
        self.input_name = input_infos[0].name
        output_names = [info.name for info in self.hef.get_output_vstream_infos()]
        required = {"image_scores", "score_map"}
        missing = sorted(required.difference(output_names))
        if missing:
            raise ValueError(
                "The HEF is not a complete AnomaVision anomaly graph; missing "
                f"outputs: {', '.join(missing)}"
            )
        self.output_names = output_names

    def _prepare_input(self, batch) -> np.ndarray:
        """Convert AnomaVision NCHW input to the HEF's NHWC input.

        ``detect.py`` already applies resize/crop and ImageNet normalization,
        exactly as it does for ONNX/PyTorch inference. Therefore a tensor coming
        from the normal detection pipeline is only transposed here. A raw PIL or
        uint8 HWC image is supported for direct backend use and is preprocessed
        once using the same resize and ImageNet normalization.
        """
        if isinstance(batch, Image.Image):
            array = np.asarray(batch.convert("RGB"), dtype=np.uint8)
            return self._preprocess_raw_hwc(array)[None]

        array = np.asarray(batch)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("HailoBackend currently supports batch size 1")
            array = array[0]

        if array.ndim != 3:
            raise ValueError("batch must be an HxWx3, 3xHxW, or single-image batch")

        if array.shape[0] == 3:
            if array.shape[1:] != self.input_size:
                raise ValueError(
                    f"Hailo input must be {self.input_size}, got {array.shape[1:]}"
                )
            # Already ImageNet-normalized NCHW from AnomaVision.
            return np.ascontiguousarray(np.transpose(array, (1, 2, 0)), dtype=self.input_dtype)

        if array.shape[2] == 3:
            # Raw HWC input is accepted only when it is clearly an image.
            if np.issubdtype(array.dtype, np.integer):
                return self._preprocess_raw_hwc(array)
            if array.shape[:2] != self.input_size:
                raise ValueError(
                    f"Hailo input must be {self.input_size}, got {array.shape[:2]}"
                )
            # Float HWC is assumed to already use the common normalized contract.
            return np.ascontiguousarray(array, dtype=self.input_dtype)

        raise ValueError("batch must be RGB with three channels")

    def _preprocess_raw_hwc(self, array: np.ndarray) -> np.ndarray:
        """Preprocess a raw uint8 HWC RGB image exactly once."""
        image = Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
            (self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR
        )
        normalized = np.asarray(image, dtype=np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        return np.ascontiguousarray(normalized, dtype=self.input_dtype)

    def predict(self, image) -> Dict[str, np.ndarray]:
        """Run one image and return the HEF's complete anomaly outputs."""
        api = self._api
        input_params = api["InputVStreamParams"].make(
            self.network_group, quantized=False, format_type=api["FormatType"].FLOAT32
        )
        output_params = api["OutputVStreamParams"].make(
            self.network_group, quantized=False, format_type=api["FormatType"].FLOAT32
        )
        tensor = self._prepare_input(image)
        with self.network_group.activate(self.network_group_params):
            with api["InferVStreams"](
                self.network_group, input_params, output_params
            ) as infer_pipeline:
                outputs = infer_pipeline.infer({self.input_name: tensor})
        return {
            "image_scores": np.asarray(outputs["image_scores"]).squeeze(),
            "score_map": np.asarray(outputs["score_map"]).squeeze(),
        }

    def close(self) -> None:
        """Release the Hailo virtual device."""
        release = getattr(self.device, "release", None)
        if callable(release):
            release()

    def __enter__(self) -> "HailoAnomalyRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class HailoBackend(InferenceBackend):
    """AnomaVision inference backend for a complete Hailo-8 HEF."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "hailo",
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        del device
        self.runtime = HailoAnomalyRuntime(model_path, input_size=input_size)

    def predict(self, batch) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single image through the HEF using the common backend contract."""
        result = self.runtime.predict(batch)
        return result["image_scores"], result["score_map"]

    def warmup(self, batch=None, runs: int = 2) -> None:
        """Warm up the device with a supplied preprocessed image batch."""
        if batch is None:
            raise ValueError("Hailo warmup requires a sample image batch")
        for _ in range(max(1, int(runs))):
            self.predict(batch)

    def close(self) -> None:
        """Release HailoRT resources through the shared backend lifecycle."""
        self.runtime.close()
