from typing import Dict, Any

from anomavision.datasets.StreamDataset import StreamDataset
from anomavision.datasets.StreamSource import StreamSource

from torch.utils.data import IterableDataset

from anomavision.datasets.MQTTSource import MQTTSource
from anomavision.datasets.TCPsource import TCPSource
from anomavision.datasets.VideoSource import VideoSource
from anomavision.datasets.WebcamSource import WebcamSource


class StreamSourceFactory:
    """
    Build concrete StreamSource implementations from a simple config dict.

    Expected minimal configs:
        {"type": "webcam", "camera_id": 0}

        {"type": "video", "video_path": "path/to/video.mp4", "loop": False}

        {"type": "mqtt", "broker": "localhost", "port": 1883,
         "topic": "camera/frames", "client_id": "cam1"}

        {"type": "tcp", "host": "192.168.1.100", "port": 8080}
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> StreamSource:
        source_type = config.get("type")
        if source_type is None:
            raise ValueError("StreamSource config requires a 'type' field")

        if source_type == "webcam":
            return WebcamSource(
                camera_id=config.get("camera_id", 0),
            )

        if source_type == "video":
            return VideoSource(
                video_path=config["video_path"],
                loop=config.get("loop", False),
            )

        if source_type == "mqtt":
            return MQTTSource(
                broker=config["broker"],
                port=config.get("port", 1883),
                topic=config["topic"],
                client_id=config.get("client_id"),
                keepalive=config.get("keepalive", 60),
                qos=config.get("qos", 0),
                max_queue_size=config.get("max_queue_size", 10),
                read_timeout=config.get("read_timeout", 1.0),
            )

        if source_type == "tcp":
            return TCPSource(
                host=config["host"],
                port=config["port"],
                recv_timeout=config.get("recv_timeout", 1.0),
                header_size=config.get("header_size", 4),
                max_message_size=config.get("max_message_size"),
            )

        raise ValueError(f"Unknown StreamSource type: {source_type}")


# Optional: convenience ctor on StreamDataset
class StreamDataset(IterableDataset):
    # your existing __init__ here...

    @classmethod
    def from_config(
        cls,
        source_config: Dict[str, Any],
        **dataset_kwargs: Any,
    ) -> "StreamDataset":
        """
        Build StreamDataset from a source config dict + dataset kwargs.

        Example:
            source_cfg = {"type": "webcam", "camera_id": 0}
            ds = StreamDataset.from_config(source_cfg, max_frames=100)
        """
        source = StreamSourceFactory.create(source_config)
        return cls(source=source, **dataset_kwargs)
