from abc import ABC, abstractmethod


# ========== BASE STRATEGY INTERFACE ==========

class StreamSource(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the source."""
        pass

    @abstractmethod
    def read_frame(self):
        """
        Read a single frame/message.
        Returns:
            np.ndarray (H, W, C) in RGB format, or None to signal end/failed read.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if source is connected."""
        pass

