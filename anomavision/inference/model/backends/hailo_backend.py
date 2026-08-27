"""HailoRT inference backend for AnomaVision anomaly models.

This backend follows the common :class:`InferenceBackend` contract used by
ONNX, PyTorch, TorchScript, and other AnomaVision backends. The HEF is expected
to contain the complete anomaly model, including feature extraction, anomaly
scoring, and score-map generation.

The backend deliberately contains no PatchCore- or PaDiM-specific inference
logic. Both algorithms expose the same two logical outputs:
``image_scores`` and ``score_map``. This keeps the runtime generic so PaDiM can
use the same backend in the future without another Hailo runtime implementation.

Preprocessing is owned by the common AnomaVision detection pipeline. The Hailo
backend therefore accepts the already-preprocessed NCHW tensor and only performs
the layout conversion required by Hailo (NCHW -> NHWC). No normalization,
resizing, scoring, or heatmap generation is performed on the CPU backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .base import Batch, InferenceBackend, ScoresMaps


class HailoAnomalyRuntime:
    """Run a complete AnomaVision anomaly HEF through HailoRT.

    The runtime is algorithm-agnostic. PatchCore and PaDiM HEFs are supported
    as long as they expose one image-score output and one score-map output.
    """

    def __init__(
        self,
        hef_path: str | Path,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        try:
            from hailo_platform import (
                HEF,
                FormatType,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HailoRT is not installed. Install the HailoRT Python package "
                "on the target device before loading a HEF."
            ) from exc

        self._api = {
            "FormatType": FormatType,
            "HEF": HEF,
            "InferVStreams": InferVStreams,
            "InputVStreamParams": InputVStreamParams,
            "OutputVStreamParams": OutputVStreamParams,
            "VDevice": VDevice,
        }
        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(self.hef_path)

        self.input_size = tuple(int(v) for v in input_size)
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
                "The HEF must expose image score and score map outputs; "
                f"found {len(output_infos)}"
            )

        self.input_name = input_infos[0].name
        self.input_shape = tuple(int(v) for v in input_infos[0].shape)
        self.output_names = [info.name for info in output_infos]
        self.output_shapes = {
            info.name: tuple(int(v) for v in info.shape) for info in output_infos
        }

        self.score_output_name, self.map_output_name = self._resolve_outputs()

    def _resolve_outputs(self) -> Tuple[str, str]:
        """Resolve the logical score and score-map streams from HEF metadata.

        Stable ``image_scores``/``score_map`` names are preferred. Hailo may
        rename ONNX end nodes during parsing, so shape-based resolution is used
        as a fallback. The fallback intentionally accepts common singleton
        dimensions produced by Hailo DFC.
        """
        if "image_scores" in self.output_names and "score_map" in self.output_names:
            return "image_scores", "score_map"

        score_candidates = []
        map_candidates = []
        target_h, target_w = self.input_size

        for name, shape in self.output_shapes.items():
            non_singleton = tuple(dim for dim in shape if dim > 1)

            # Image score: scalar per batch, commonly [B], [B,1], or [B,1,1,1].
            if len(non_singleton) <= 1:
                score_candidates.append(name)

            # Score map: [B,H,W], [B,1,H,W], or [B,H,W,1].
            if (target_h in shape and target_w in shape) and len(non_singleton) >= 2:
                map_candidates.append(name)

        # Prefer the output with the expected spatial resolution if several
        # candidates are present.
        exact_maps = [
            name
            for name in map_candidates
            if target_h in self.output_shapes[name]
            and target_w in self.output_shapes[name]
        ]
        if exact_maps:
            map_candidates = exact_maps

        if len(score_candidates) != 1 or len(map_candidates) != 1:
            raise ValueError(
                "Could not identify the complete AnomaVision outputs in the HEF. "
                f"Available streams: {self.output_shapes}. Expected one scalar "
                f"image-score stream and one {target_h}x{target_w} score-map stream."
            )

        return score_candidates[0], map_candidates[0]

    def _prepare_input(self, batch: Batch) -> np.ndarray:
        """Convert the common NCHW input tensor to the HEF input layout.

        The tensor is assumed to have already gone through AnomaVision's common
        preprocessing pipeline. In particular, this method never divides by 255,
        applies normalization statistics, or resizes the image.
        """
        if isinstance(batch, torch.Tensor):
            array = batch.detach().cpu().numpy()
        else:
            array = np.asarray(batch)

        if array.ndim != 4:
            raise ValueError(
                f"HailoBackend expects a 4-D NCHW batch, got shape {array.shape}"
            )
        if array.shape[0] != 1:
            raise ValueError("HailoBackend currently supports batch size 1")
        if array.shape[1] != 3:
            raise ValueError(
                f"HailoBackend expects 3 input channels, got shape {array.shape}"
            )

        # Common AnomaVision input is NCHW; Hailo HEFs normally use NHWC.
        return np.ascontiguousarray(array.transpose(0, 2, 3, 1), dtype=np.float32)

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run one preprocessed batch and return score and score-map arrays."""
        api = self._api
        tensor = self._prepare_input(batch)

        input_params = api["InputVStreamParams"].make(
            self.network_group,
            quantized=False,
            format_type=api["FormatType"].FLOAT32,
        )
        output_params = api["OutputVStreamParams"].make(
            self.network_group,
            quantized=False,
            format_type=api["FormatType"].FLOAT32,
        )

        with self.network_group.activate(self.network_group_params):
            with api["InferVStreams"](
                self.network_group, input_params, output_params
            ) as infer_pipeline:
                outputs = infer_pipeline.infer({self.input_name: tensor})

        scores = np.asarray(outputs[self.score_output_name])
        maps = np.asarray(outputs[self.map_output_name])

        return self._normalize_outputs(scores, maps)

    def _normalize_outputs(
        self, scores: np.ndarray, maps: np.ndarray
    ) -> ScoresMaps:
        """Normalize Hailo output singleton dimensions to the common contract."""
        scores = np.asarray(scores).squeeze()
        maps = np.asarray(maps)

        if maps.ndim == 4:
            if maps.shape[1] == 1:
                maps = maps[:, 0]
            elif maps.shape[-1] == 1:
                maps = maps[..., 0]
        elif maps.ndim == 2:
            maps = maps[None, ...]

        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim > 1:
            scores = scores.reshape(scores.shape[0], -1).max(axis=1)

        return scores.astype(np.float32, copy=False), maps.astype(np.float32, copy=False)

    def close(self) -> None:
        """Release Hailo device resources."""
        release = getattr(self.device, "release", None)
        if callable(release):
            release()

    def __enter__(self) -> "HailoAnomalyRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class HailoBackend(InferenceBackend):
    """AnomaVision inference backend for complete Hailo anomaly HEFs.

    The implementation is intentionally algorithm-independent so the same
    backend can serve PatchCore now and PaDiM later. Algorithm-specific logic
    belongs in the exported HEF graph, not in this runtime adapter.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "hailo",
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        del device
        self.runtime = HailoAnomalyRuntime(model_path, input_size=input_size)

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run inference using the common AnomaVision backend contract."""
        return self.runtime.predict(batch)

    def warmup(self, batch: Batch, runs: int = 2) -> None:
        """Warm up the Hailo network group with a representative batch."""
        if batch is None:
            raise ValueError("Hailo warmup requires a representative input batch")
        for _ in range(max(1, int(runs))):
            self.predict(batch)

    def close(self) -> None:
        """Release Hailo runtime resources."""
        self.runtime.close()
