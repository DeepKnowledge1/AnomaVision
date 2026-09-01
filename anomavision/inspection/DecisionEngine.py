from typing import Any


class DecisionEngine:
    """Convert an anomaly score into a production decision.

    Keeping this outside the model and action layers allows thresholds and
    future production rules to change without modifying inference backends.
    """

    def __init__(self, threshold: float):
        threshold = float(threshold)
        if not 0.0 <= threshold:
            raise ValueError("threshold must be greater than or equal to 0")
        self.threshold = threshold

    def evaluate(self, anomaly_score: Any) -> str:
        """Return ``FAIL`` when the score reaches the configured threshold."""
        score = float(anomaly_score)
        return "FAIL" if score >= self.threshold else "PASS"

    def is_anomaly(self, anomaly_score: Any) -> bool:
        """Return the boolean anomaly decision for an anomaly score."""
        return self.evaluate(anomaly_score) == "FAIL"
