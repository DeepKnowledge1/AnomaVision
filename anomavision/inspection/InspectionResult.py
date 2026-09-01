from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class InspectionResult:
    """Normalized result passed from inference to industrial actions.

    The object deliberately contains only transport-safe metadata. Images and
    heatmaps can be handled by a future evidence action without forcing every
    industrial output to serialize large numpy arrays.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_id: str = "unknown"
    frame_id: Optional[int] = None
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    model_name: str = "unknown"
    model_version: Optional[str] = None
    inference_time_ms: Optional[float] = None
    decision: Optional[str] = None
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the result."""
        return asdict(self)
