"""Kinematic Servo Controller with S-Curve / Cubic Easing and Limit Enforcement."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..config.settings import HardwareConfig
from ..core.logger import get_logger
from ..hardware.base import ServoDriverBase

logger = get_logger("motion.servo_controller")


def ease_in_out_cubic(t: float) -> float:
    """Cubic easing function for natural acceleration and deceleration."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 3.0) / 2.0


@dataclass
class ChannelCalibration:
    channel: int
    min_angle: float = -90.0
    max_angle: float = 90.0
    home_angle: float = 0.0
    min_pulse_us: int = 500
    max_pulse_us: int = 2500
    inverted: bool = False


class ServoController:
    """High-level kinematic controller managing multi-joint smooth trajectories."""

    def __init__(
        self,
        driver: ServoDriverBase,
        hw_config: Optional[HardwareConfig] = None,
        auto_relax_delay_s: float = 5.0,
    ) -> None:
        self.driver = driver
        self.auto_relax_delay_s = auto_relax_delay_s
        self.channels: Dict[str, ChannelCalibration] = {}
        self.current_angles: Dict[str, float] = {}
        self.target_angles: Dict[str, float] = {}
        self._last_move_time: float = time.time()
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._load_calibrations(hw_config)

    def _load_calibrations(self, hw_config: Optional[HardwareConfig]) -> None:
        raw_channels = hw_config.channels if hw_config else {}
        defaults = {
            "head_pan": {"channel": 0, "min_angle": -80.0, "max_angle": 80.0, "home_angle": 0.0},
            "head_tilt": {"channel": 1, "min_angle": -30.0, "max_angle": 45.0, "home_angle": 0.0},
            "left_arm": {"channel": 2, "min_angle": -45.0, "max_angle": 90.0, "home_angle": 0.0},
            "right_arm": {"channel": 3, "min_angle": -45.0, "max_angle": 90.0, "home_angle": 0.0, "inverted": True},
        }

        for name, def_vals in defaults.items():
            cfg = raw_channels.get(name, def_vals)
            cal = ChannelCalibration(
                channel=int(cfg.get("channel", def_vals.get("channel", 0))),
                min_angle=float(cfg.get("min_angle", def_vals.get("min_angle", -90.0))),
                max_angle=float(cfg.get("max_angle", def_vals.get("max_angle", 90.0))),
                home_angle=float(cfg.get("home_angle", def_vals.get("home_angle", 0.0))),
                min_pulse_us=int(cfg.get("min_pulse_us", 500)),
                max_pulse_us=int(cfg.get("max_pulse_us", 2500)),
                inverted=bool(cfg.get("inverted", def_vals.get("inverted", False))),
            )
            self.channels[name] = cal
            self.current_angles[name] = cal.home_angle
            self.target_angles[name] = cal.home_angle

    def initialize(self) -> bool:
        """Initialize driver and move all channels to home position."""
        if not self.driver.initialize():
            logger.error("Failed to initialize underlying servo driver.")
            return False

        self.home_all()
        self._running = True
        logger.info("ServoController online with calibrated channels: " + ", ".join(self.channels.keys()))
        return True

    def home_all(self, duration_s: float = 0.5) -> None:
        """Move all calibrated channels to their safe home positions."""
        targets = {name: cal.home_angle for name, cal in self.channels.items()}
        self.move_multiple(targets, duration_s=duration_s)

    def angle_to_pulse_us(self, name: str, angle_deg: float) -> int:
        """Translate physical angle in degrees to pulse width microseconds."""
        cal = self.channels[name]
        # Clamp to safe limits
        clamped = max(cal.min_angle, min(cal.max_angle, angle_deg))
        if cal.inverted:
            clamped = -clamped

        # Normalize 0.0 - 1.0 across range
        range_deg = cal.max_angle - cal.min_angle or 1.0
        norm = (clamped - cal.min_angle) / range_deg
        pulse = cal.min_pulse_us + norm * (cal.max_pulse_us - cal.min_pulse_us)
        return int(pulse)

    def set_angle_immediate(self, name: str, angle_deg: float) -> None:
        """Set joint angle immediately without interpolation."""
        with self._lock:
            if name not in self.channels:
                logger.warning(f"Unknown servo channel: '{name}'")
                return
            cal = self.channels[name]
            clamped = max(cal.min_angle, min(cal.max_angle, angle_deg))
            self.current_angles[name] = clamped
            self.target_angles[name] = clamped
            pulse = self.angle_to_pulse_us(name, clamped)
            self.driver.set_pwm_us(cal.channel, pulse)
            self._last_move_time = time.time()

    def move_joint(self, name: str, target_angle_deg: float, duration_s: float = 0.4) -> None:
        """Move a single joint smoothly to target angle over duration."""
        self.move_multiple({name: target_angle_deg}, duration_s=duration_s)

    def move_multiple(self, targets: Dict[str, float], duration_s: float = 0.4) -> None:
        """Interpolate multiple joints simultaneously using eased trajectory."""
        with self._lock:
            start_angles = {k: self.current_angles.get(k, 0.0) for k in targets.keys()}
            valid_targets = {}
            for k, tgt in targets.items():
                if k in self.channels:
                    cal = self.channels[k]
                    valid_targets[k] = max(cal.min_angle, min(cal.max_angle, tgt))

        if duration_s <= 0.05:
            for k, tgt in valid_targets.items():
                self.set_angle_immediate(k, tgt)
            return

        steps = max(5, int(duration_s * 50))  # 50 Hz interpolation loop
        dt = duration_s / steps

        for step in range(1, steps + 1):
            t = step / steps
            eased_t = ease_in_out_cubic(t)
            with self._lock:
                for k, end_val in valid_targets.items():
                    start_val = start_angles[k]
                    current_interp = start_val + (end_val - start_val) * eased_t
                    self.current_angles[k] = current_interp
                    pulse = self.angle_to_pulse_us(k, current_interp)
                    self.driver.set_pwm_us(self.channels[k].channel, pulse)
            time.sleep(dt)

        with self._lock:
            for k, end_val in valid_targets.items():
                self.current_angles[k] = end_val
            self._last_move_time = time.time()

    def relax_all(self) -> None:
        """De-energize all servo channels to prevent humming and heating."""
        with self._lock:
            for name, cal in self.channels.items():
                self.driver.release_channel(cal.channel)
            logger.debug("All servo channels relaxed.")

    def shutdown(self) -> None:
        """Stop controller and park servos."""
        self._running = False
        self.home_all(duration_s=0.3)
        time.sleep(0.3)
        self.relax_all()
        self.driver.shutdown()
        logger.info("ServoController shut down cleanly.")
