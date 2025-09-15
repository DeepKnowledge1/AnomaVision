"""
Dataset factory for anomaly detection.

Selects an appropriate dataset based on source type and delegates
all data loading calls to that dataset implementation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from anodet.utils import get_logger

from .base import BaseRealtimeDataset
from .cam_dataset import WebcamDataset
from .dataset import AnodetDataset
from .mqtt_dataset import MQTTDataset
from .mvtec_dataset import MVTecDataset
from .tcp_dataset import TCPDataset
from .video_dataset import VideoDataset

logger = get_logger(__name__)


# Factory function for easy dataset creation
def create_realtime_dataset(source_type: str, **kwargs) -> BaseRealtimeDataset:
    """Factory function to create appropriate dataset"""
    datasets = {
        "webcam": WebcamDataset,
        "video": VideoDataset,
        "mqtt": MQTTDataset,
        "tcp": TCPDataset,
    }

    if source_type not in datasets:
        raise ValueError(f"Unknown source type: {source_type}")

    return datasets[source_type](**kwargs)


class DatasetType:
    """Enumeration of supported dataset types."""

    STATIC = "static"
    MVTEC = "mvtec"
    WEBCAM = "webcam"
    VIDEO = "video"
    MQTT = "mqtt"
    TCP = "tcp"

    @classmethod
    def from_source(cls, source: str) -> str:
        """Determine dataset type from source identifier."""
        source = source.lower()

        # Real-time sources
        if source in ["webcam", "camera", "cam"]:
            return cls.WEBCAM
        elif source in ["video", "vid", "mp4", "avi", "mov"]:
            return cls.VIDEO
        elif source in ["mqtt", "message"]:
            return cls.MQTT
        elif source in ["tcp", "socket", "network"]:
            return cls.TCP

        # Static dataset sources
        elif source in ["mvtec", "mvtec-ad"]:
            return cls.MVTEC
        elif source in ["static", "folder", "directory", "images"]:
            return cls.STATIC

        # Try to infer from path/extension
        if os.path.exists(source):
            if os.path.isdir(source):
                # Check if it's MVTec structure
                if cls._is_mvtec_structure(source):
                    return cls.MVTEC
                else:
                    return cls.STATIC
            elif os.path.isfile(source):
                ext = Path(source).suffix.lower()
                if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    return cls.VIDEO

        # Default fallback
        logger.warning(
            f"Could not determine dataset type for '{source}', defaulting to static"
        )
        return cls.STATIC

    @staticmethod
    def _is_mvtec_structure(path: str) -> bool:
        """Check if directory has MVTec-AD structure."""
        try:
            # MVTec has train/test folders and ground_truth
            required_folders = ["train", "test"]
            path_obj = Path(path)

            if not path_obj.is_dir():
                return False

            # Check for train/test in immediate subdirectories
            subdirs = [d.name for d in path_obj.iterdir() if d.is_dir()]
            has_train_test = all(folder in subdirs for folder in required_folders)

            if has_train_test:
                return True

            # Check if any subdirectory has train/test structure
            for subdir in path_obj.iterdir():
                if subdir.is_dir():
                    sub_subdirs = [d.name for d in subdir.iterdir() if d.is_dir()]
                    if all(folder in sub_subdirs for folder in required_folders):
                        return True

            return False
        except Exception:
            return False


def make_dataset(
    source: str, dataset_type: Optional[str] = None, **kwargs
) -> Union[AnodetDataset, MVTecDataset, BaseRealtimeDataset]:
    """Factory function to create appropriate dataset based on source type.

    Automatically detects the dataset format from source identifier and instantiates
    the corresponding dataset implementation. Supports both static and real-time
    data sources for flexible anomaly detection workflows.

    Args:
        source (str): Source identifier. Can be:
            - Directory path for static datasets
            - "webcam" or camera ID for webcam input
            - Video file path for video datasets
            - MQTT broker details for MQTT streams
            - TCP connection details for TCP streams

        dataset_type (Optional[str]): Explicit dataset type override.
            If None, will auto-detect from source.

        **kwargs: Additional arguments passed to dataset constructor:
            - For static/MVTec: image_transforms, resize, crop_size, etc.
            - For real-time: buffer_size, timeout, fps_limit, etc.
            - Source-specific params: camera_id, broker_host, etc.

    Returns:
        Dataset: Initialized dataset instance ready for data loading.

    Raises:
        NotImplementedError: If the detected dataset type is not supported.
        FileNotFoundError: If source path does not exist (for file-based sources).
        ValueError: If required parameters are missing for specific dataset types.

    Examples:
        >>> # Static image folder
        >>> dataset = make_dataset("/path/to/images")

        >>> # MVTec dataset
        >>> dataset = make_dataset("/path/to/mvtec", class_name="bottle")

        >>> # Webcam
        >>> dataset = make_dataset("webcam", camera_id=0, fps_limit=10)

        >>> # Video file
        >>> dataset = make_dataset("/path/to/video.mp4", loop=True)

        >>> # MQTT stream
        >>> dataset = make_dataset("mqtt", broker_host="localhost", topic="camera/feed")

        >>> # TCP stream
        >>> dataset = make_dataset("tcp", host="192.168.1.100", port=8080)
    """

    logger.info(f"Creating dataset for source: {source}")

    # Determine dataset type
    if dataset_type is None:
        dataset_type = DatasetType.from_source(source)

    logger.info(f"Detected dataset type: {dataset_type}")

    # Extract common parameters
    common_params = {
        "image_transforms": kwargs.pop("image_transforms", None),
        "resize": kwargs.pop("resize", 224),
        "crop_size": kwargs.pop("crop_size", 224),
        "normalize": kwargs.pop("normalize", True),
        "mean": kwargs.pop("mean", [0.485, 0.456, 0.406]),
        "std": kwargs.pop("std", [0.229, 0.224, 0.225]),
    }

    # Create appropriate dataset
    if dataset_type == DatasetType.STATIC:
        logger.info("Loading static image dataset...")
        # For static datasets, source is image directory path
        mask_directory_path = kwargs.pop("mask_directory_path", None)

        logger.debug(f"Selected static dataset for {source}")
        dataset = AnodetDataset(
            image_directory_path=source,
            mask_directory_path=mask_directory_path,
            **common_params,
            **kwargs,
        )
        logger.info("Static dataset created successfully")
        return dataset

    elif dataset_type == DatasetType.MVTEC:
        logger.info("Loading MVTec dataset...")
        # MVTec requires additional parameters
        class_name = kwargs.pop("class_name", None)
        if class_name is None:
            raise ValueError("class_name is required for MVTec dataset")

        is_train = kwargs.pop("is_train", False)

        logger.debug(f"Selected MVTec dataset for {source}, class: {class_name}")
        dataset = MVTecDataset(
            dataset_path=source,
            class_name=class_name,
            is_train=is_train,
            **common_params,
            **kwargs,
        )
        logger.info("MVTec dataset created successfully")
        return dataset

    elif dataset_type == DatasetType.WEBCAM:
        logger.info("Loading webcam dataset...")
        # Extract webcam-specific parameters
        camera_id = kwargs.pop("camera_id", 0)
        if source.isdigit():
            camera_id = int(source)

        fps_limit = kwargs.pop("fps_limit", None)
        buffer_size = kwargs.pop("buffer_size", 100)
        timeout = kwargs.pop("timeout", 1.0)

        logger.debug(f"Selected webcam dataset for camera {camera_id}")
        dataset = WebcamDataset(
            camera_id=camera_id,
            fps_limit=fps_limit,
            buffer_size=buffer_size,
            timeout=timeout,
            **common_params,
            **kwargs,
        )
        logger.info("Webcam dataset created successfully")
        return dataset

    elif dataset_type == DatasetType.VIDEO:
        logger.info("Loading video dataset...")
        # For video, source can be path or we need video_path parameter
        video_path = kwargs.pop("video_path", source)
        loop = kwargs.pop("loop", True)
        buffer_size = kwargs.pop("buffer_size", 100)
        timeout = kwargs.pop("timeout", 1.0)

        logger.debug(f"Selected video dataset for {video_path}")
        dataset = VideoDataset(
            video_path=video_path,
            loop=loop,
            buffer_size=buffer_size,
            timeout=timeout,
            **common_params,
            **kwargs,
        )
        logger.info("Video dataset created successfully")
        return dataset

    elif dataset_type == DatasetType.MQTT:
        logger.info("Loading MQTT dataset...")
        # MQTT requires broker connection details
        broker_host = kwargs.pop("broker_host", None)
        if broker_host is None:
            raise ValueError("broker_host is required for MQTT dataset")

        broker_port = kwargs.pop("broker_port", 1883)
        topic = kwargs.pop("topic", "camera/image")
        username = kwargs.pop("username", None)
        password = kwargs.pop("password", None)
        buffer_size = kwargs.pop("buffer_size", 100)
        timeout = kwargs.pop("timeout", 1.0)

        logger.debug(f"Selected MQTT dataset for {broker_host}:{broker_port}")
        dataset = MQTTDataset(
            broker_host=broker_host,
            broker_port=broker_port,
            topic=topic,
            username=username,
            password=password,
            buffer_size=buffer_size,
            timeout=timeout,
            **common_params,
            **kwargs,
        )
        logger.info("MQTT dataset created successfully")
        return dataset

    elif dataset_type == DatasetType.TCP:
        logger.info("Loading TCP dataset...")
        # TCP requires connection details
        host = kwargs.pop("host", None)
        port = kwargs.pop("port", None)

        if host is None or port is None:
            raise ValueError("host and port are required for TCP dataset")

        buffer_size_tcp = kwargs.pop("buffer_size_tcp", 4096)
        buffer_size = kwargs.pop("buffer_size", 100)
        timeout = kwargs.pop("timeout", 1.0)

        logger.debug(f"Selected TCP dataset for {host}:{port}")
        dataset = TCPDataset(
            host=host,
            port=port,
            buffer_size_tcp=buffer_size_tcp,
            buffer_size=buffer_size,
            timeout=timeout,
            **common_params,
            **kwargs,
        )
        logger.info("TCP dataset created successfully")
        return dataset

    else:
        raise NotImplementedError(f"Dataset type {dataset_type} is not supported.")


class DatasetWrapper:
    """
    Thin wrapper around dataset implementations. Clients use this class to
    abstract away the dataset-specific initialization and data loading API.
    """

    def __init__(
        self,
        source: str,
        dataset_type: Optional[str] = None,
        auto_start: bool = True,
        **kwargs,
    ):
        """Initialize DatasetWrapper.

        Args:
            source: Source identifier for the dataset
            dataset_type: Optional explicit dataset type
            auto_start: For real-time datasets, automatically start capture
            **kwargs: Additional arguments passed to dataset constructor
        """
        logger.info(f"Initializing DatasetWrapper with {source}")

        self.source = source
        self.dataset_type = dataset_type or DatasetType.from_source(source)
        self.auto_start = auto_start

        # Create the dataset
        self.dataset = make_dataset(source, dataset_type, **kwargs)

        # Auto-start real-time datasets
        if isinstance(self.dataset, BaseRealtimeDataset) and auto_start:
            logger.info("Auto-starting real-time dataset")
            self.dataset.start()

        logger.info("DatasetWrapper initialization completed successfully")

    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.dataset[idx]

    def __iter__(self):
        """Make dataset iterable."""
        return iter(self.dataset)

    def start(self) -> None:
        """Start real-time dataset capture (if applicable)."""
        if isinstance(self.dataset, BaseRealtimeDataset):
            logger.info("Starting real-time dataset capture")
            self.dataset.start()
        else:
            logger.info("Dataset is not real-time, start() has no effect")

    def stop(self) -> None:
        """Stop real-time dataset capture (if applicable)."""
        if isinstance(self.dataset, BaseRealtimeDataset):
            logger.info("Stopping real-time dataset capture")
            self.dataset.stop()
        else:
            logger.debug("Dataset is not real-time, stop() has no effect")

    def get_frame(self):
        """Get latest frame (real-time datasets only)."""
        if isinstance(self.dataset, BaseRealtimeDataset):
            return self.dataset.get_frame()
        else:
            logger.warning("get_frame() only available for real-time datasets")
            return None

    def close(self) -> None:
        """Release resources associated with the dataset."""
        logger.info("Closing DatasetWrapper and releasing resources")
        self.stop()  # Stop real-time capture if applicable

        # Additional cleanup if dataset has close method
        if hasattr(self.dataset, "close"):
            self.dataset.close()

    def is_realtime(self) -> bool:
        """Check if this is a real-time dataset."""
        return isinstance(self.dataset, BaseRealtimeDataset)

    def get_metadata(self) -> dict:
        """Get dataset metadata."""
        metadata = {
            "source": self.source,
            "type": self.dataset_type,
            "length": len(self.dataset),
            "is_realtime": self.is_realtime(),
            "class_name": self.dataset.__class__.__name__,
        }

        # Add dataset-specific metadata
        if hasattr(self.dataset, "get_metadata"):
            metadata.update(self.dataset.get_metadata())

        return metadata


# Convenience functions for common use cases
def create_static_dataset(
    image_path: str, mask_path: str = None, **kwargs
) -> DatasetWrapper:
    """Create a static image dataset."""
    return DatasetWrapper(
        image_path, DatasetType.STATIC, mask_directory_path=mask_path, **kwargs
    )


def create_mvtec_dataset(
    dataset_path: str, class_name: str, is_train: bool = False, **kwargs
) -> DatasetWrapper:
    """Create an MVTec dataset."""
    return DatasetWrapper(
        dataset_path,
        DatasetType.MVTEC,
        class_name=class_name,
        is_train=is_train,
        **kwargs,
    )


def create_webcam_dataset(
    camera_id: int = 0, fps_limit: int = None, **kwargs
) -> DatasetWrapper:
    """Create a webcam dataset."""
    return DatasetWrapper(
        "webcam", DatasetType.WEBCAM, camera_id=camera_id, fps_limit=fps_limit, **kwargs
    )


def create_video_dataset(
    video_path: str, loop: bool = True, **kwargs
) -> DatasetWrapper:
    """Create a video dataset."""
    return DatasetWrapper(video_path, DatasetType.VIDEO, loop=loop, **kwargs)


def create_mqtt_dataset(
    broker_host: str, topic: str = "camera/image", **kwargs
) -> DatasetWrapper:
    """Create an MQTT dataset."""
    return DatasetWrapper(
        "mqtt", DatasetType.MQTT, broker_host=broker_host, topic=topic, **kwargs
    )


def create_tcp_dataset(host: str, port: int, **kwargs) -> DatasetWrapper:
    """Create a TCP dataset."""
    return DatasetWrapper("tcp", DatasetType.TCP, host=host, port=port, **kwargs)


# # Auto-detection (like your ModelWrapper)
# dataset = DatasetWrapper("/path/to/images")  # → Static dataset
# dataset = DatasetWrapper("webcam", camera_id=0)  # → Webcam dataset
# dataset = DatasetWrapper("/path/to/video.mp4")  # → Video dataset

# # Explicit type specification
# dataset = DatasetWrapper("/path/to/mvtec", "mvtec", class_name="bottle")

# # Use with your existing inference loop
# dataloader = DataLoader(dataset.dataset, batch_size=8)
# for batch in dataloader:
#     # Your inference code
#     pass

# # Resource cleanup
# dataset.close()
