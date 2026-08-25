"""Hailo runtime wrapper for PaDiM feature extraction.

This module only replaces the backbone inference step.
The PaDiM scoring pipeline remains unchanged.
"""

from pathlib import Path

import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    InputVStreams,
    OutputVStreams,
)


class HailoPadimExtractor:
    def __init__(self, hef_path: str):
        self.hef_path = str(Path(hef_path))
        self.device = VDevice()

        hef = HEF(self.hef_path)
        params = ConfigureParams.create_from_hef(hef)
        self.network_group = self.device.configure(hef, params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8,
        )
        self.output_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8,
        )

    def extract(self, image: np.ndarray):
        with InputVStreams(self.network_group, self.input_params) as input_streams:
            with OutputVStreams(self.network_group, self.output_params) as output_streams:
                input_streams[0].send(image)
                features = output_streams[0].recv()

        return features
