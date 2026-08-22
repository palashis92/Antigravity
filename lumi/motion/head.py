"""Head Movement Controller for Pan/Tilt tracking and expressive glances."""

from __future__ import annotations

import random
import time
from typing import Tuple

from ..core.logger import get_logger
from .servo_controller import ServoController

logger = get_logger("motion.head")


class HeadController:
    """Controls robot head orientation, face tracking, and idle gestures."""

    def __init__(self, controller: ServoController) -> None:
        self.controller = controller
        self.current_pan = 0.0
        self.current_tilt = 0.0

    def look_at(self, pan_deg: float, tilt_deg: float, duration_s: float = 0.4) -> None:
        """Orient head to specific pan/tilt angles."""
        self.current_pan = pan_deg
        self.current_tilt = tilt_deg
        self.controller.move_multiple(
            {"head_pan": pan_deg, "head_tilt": tilt_deg},
            duration_s=duration_s,
        )

    def look_center(self, duration_s: float = 0.3) -> None:
        """Return head to center home position (0, 0)."""
        self.look_at(0.0, 0.0, duration_s=duration_s)

    def look_left(self, deg: float = 35.0, duration_s: float = 0.35) -> None:
        """Turn head left."""
        self.look_at(-abs(deg), self.current_tilt, duration_s=duration_s)

    def look_right(self, deg: float = 35.0, duration_s: float = 0.35) -> None:
        """Turn head right."""
        self.look_at(abs(deg), self.current_tilt, duration_s=duration_s)

    def look_up(self, deg: float = 25.0, duration_s: float = 0.35) -> None:
        """Tilt head up."""
        self.look_at(self.current_pan, abs(deg), duration_s=duration_s)

    def look_down(self, deg: float = 20.0, duration_s: float = 0.35) -> None:
        """Tilt head down."""
        self.look_at(self.current_pan, -abs(deg), duration_s=duration_s)

    def nod(self, count: int = 2, amplitude_deg: float = 15.0) -> None:
        """Expressive nod gesture (yes / agreement)."""
        for _ in range(count):
            self.look_at(self.current_pan, amplitude_deg, duration_s=0.18)
            self.look_at(self.current_pan, -amplitude_deg * 0.6, duration_s=0.18)
        self.look_center(duration_s=0.2)

    def shake(self, count: int = 2, amplitude_deg: float = 20.0) -> None:
        """Expressive shake gesture (no / disagreement)."""
        for _ in range(count):
            self.look_at(-amplitude_deg, self.current_tilt, duration_s=0.18)
            self.look_at(amplitude_deg, self.current_tilt, duration_s=0.18)
        self.look_center(duration_s=0.2)

    def track_bounding_box(
        self,
        center_x: float,
        center_y: float,
        frame_w: float = 640.0,
        frame_h: float = 480.0,
        gain: float = 0.05,
    ) -> Tuple[float, float]:
        """Convert a detected 2D bounding box center to incremental pan/tilt adjustments."""
        err_x = (center_x - frame_w / 2.0) / (frame_w / 2.0)  # -1.0 to +1.0
        err_y = (center_y - frame_h / 2.0) / (frame_h / 2.0)  # -1.0 to +1.0

        # Adjust pan and tilt proportionally
        delta_pan = err_x * 40.0 * gain
        delta_tilt = -err_y * 30.0 * gain

        target_pan = max(-70.0, min(70.0, self.current_pan + delta_pan))
        target_tilt = max(-25.0, min(35.0, self.current_tilt + delta_tilt))

        self.look_at(target_pan, target_tilt, duration_s=0.1)
        return target_pan, target_tilt

    def subtle_idle_wander(self) -> None:
        """Generate subtle organic micro-movements to simulate breathing/lifelike idle."""
        pan_offset = random.uniform(-8.0, 8.0)
        tilt_offset = random.uniform(-5.0, 8.0)
        self.look_at(pan_offset, tilt_offset, duration_s=random.uniform(0.8, 1.4))
