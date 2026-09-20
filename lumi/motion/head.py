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

    # Safe kinematic limits ensuring servo never hits physical stops or exceeds rotation limits
    SAFE_MIN_PAN: float = -70.0   # Turn right max
    SAFE_MAX_PAN: float = 70.0    # Turn left max
    SAFE_MIN_TILT: float = -15.0  # Tilt up max
    SAFE_MAX_TILT: float = 15.0   # Tilt down max

    def __init__(self, controller: ServoController) -> None:
        self.controller = controller
        self.current_pan = 0.0
        self.current_tilt = 0.0

    def look_at(self, pan_deg: float, tilt_deg: float, duration_s: float = 0.22) -> None:
        """Orient head to specific pan/tilt angles with responsive easing and safety limits."""
        pan_deg = max(self.SAFE_MIN_PAN, min(self.SAFE_MAX_PAN, float(pan_deg)))
        tilt_deg = max(self.SAFE_MIN_TILT, min(self.SAFE_MAX_TILT, float(tilt_deg)))
        self.current_pan = pan_deg
        self.current_tilt = tilt_deg
        self.controller.move_multiple(
            {"head_pan": pan_deg, "head_tilt": tilt_deg},
            duration_s=duration_s,
        )

    def look_center(self, duration_s: float = 0.2) -> None:
        """Return head to center home position (0, 0)."""
        self.look_at(0.0, 0.0, duration_s=duration_s)

    def look_left(self, deg: float = 35.0, duration_s: float = 0.22) -> None:
        """Turn head/body left (+70° = left max)."""
        self.look_at(abs(deg), self.current_tilt, duration_s=duration_s)

    def look_right(self, deg: float = 35.0, duration_s: float = 0.22) -> None:
        """Turn head/body right (-70° = right max)."""
        self.look_at(-abs(deg), self.current_tilt, duration_s=duration_s)

    def look_up(self, deg: float = 12.0, duration_s: float = 0.2) -> None:
        """Tilt head up (-12° = up)."""
        self.look_at(self.current_pan, -abs(deg), duration_s=duration_s)

    def look_down(self, deg: float = 12.0, duration_s: float = 0.2) -> None:
        """Tilt head down (+12° = down)."""
        self.look_at(self.current_pan, abs(deg), duration_s=duration_s)

    def pan(self, deg: float, duration_s: float = 0.22) -> None:
        """Set head/body pan angle directly (+70° = left, -70° = right)."""
        clamped = max(self.SAFE_MIN_PAN, min(self.SAFE_MAX_PAN, deg))
        self.look_at(clamped, self.current_tilt, duration_s=duration_s)

    def tilt(self, deg: float, duration_s: float = 0.2) -> None:
        """Set head tilt angle directly (-12° = up, +12° = down)."""
        clamped = max(self.SAFE_MIN_TILT, min(self.SAFE_MAX_TILT, deg))
        self.look_at(self.current_pan, clamped, duration_s=duration_s)

    def nod(self, count: int = 2, amplitude_deg: float = 12.0) -> None:
        """Expressive nod gesture (yes / agreement). Down is +, Up is -."""
        amp = min(15.0, abs(amplitude_deg))
        for _ in range(count):
            self.look_at(self.current_pan, amp, duration_s=0.14)        # Down (+)
            self.look_at(self.current_pan, -amp * 0.7, duration_s=0.14) # Up (-)
        self.look_center(duration_s=0.16)

    def shake(self, count: int = 2, amplitude_deg: float = 20.0) -> None:
        """Expressive shake gesture (no / disagreement). Left is +, Right is -."""
        amp = min(90.0, abs(amplitude_deg))
        for _ in range(count):
            self.look_at(amp, self.current_tilt, duration_s=0.14)   # Left (+)
            self.look_at(-amp, self.current_tilt, duration_s=0.14)  # Right (-)
        self.look_center(duration_s=0.16)

    def track_bounding_box(
        self,
        center_x: float,
        center_y: float,
        frame_w: float = 640.0,
        frame_h: float = 480.0,
        gain: float = 0.22,
    ) -> Tuple[float, float]:
        """Convert a detected 2D bounding box center to responsive, smooth pan/tilt adjustments.
        
        Uses responsive proportional tracking with a deadband to follow moving people smoothly
        without high-frequency motor jitter.
        """
        err_x = (center_x - frame_w / 2.0) / (frame_w / 2.0)  # -1.0 to +1.0
        err_y = (center_y - frame_h / 2.0) / (frame_h / 2.0)  # -1.0 to +1.0

        # Deadband: ignore tiny jitter (< 4% of frame)
        if abs(err_x) < 0.04:
            delta_pan = 0.0
        else:
            delta_pan = -err_x * 55.0 * gain

        if abs(err_y) < 0.04:
            delta_tilt = 0.0
        else:
            delta_tilt = err_y * 20.0 * gain

        target_pan = max(self.SAFE_MIN_PAN, min(self.SAFE_MAX_PAN, self.current_pan + delta_pan))
        target_tilt = max(self.SAFE_MIN_TILT, min(self.SAFE_MAX_TILT, self.current_tilt + delta_tilt))

        if abs(delta_pan) > 0.4 or abs(delta_tilt) > 0.4:
            self.look_at(target_pan, target_tilt, duration_s=0.12)
        return target_pan, target_tilt

    def subtle_idle_wander(self) -> None:
        """Generate subtle organic micro-movements to simulate breathing/lifelike idle."""
        pan_offset = random.uniform(-15.0, 15.0)
        tilt_offset = random.uniform(-10.0, 10.0)  # safely inside [-15°, +15°]
        self.look_at(pan_offset, tilt_offset, duration_s=random.uniform(0.6, 1.0))
