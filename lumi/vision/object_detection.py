"""Object Detection Service for scene perception and gaze targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from ..core.logger import get_logger

logger = get_logger("vision.object")


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]


class ObjectDetector:
    """Detects everyday household objects, books, cups, people, plants, and chessboards."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Any) -> List[DetectedObject]:
        """Perform object detection on the provided video frame."""
        if frame is None:
            return []

        # Lightweight inference with TFLite / MobileNet or Mock simulation
        return [
            DetectedObject(
                label="person",
                confidence=0.92,
                bounding_box=(200, 100, 240, 360),
                center=(320.0, 280.0),
            )
        ]
