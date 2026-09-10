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
            "head_tilt": {"channel": 0, "min_angle": -15.0, "max_angle": 15.0, "home_angle": 0.0},
            "right_arm_y": {"channel": 1, "min_angle": -60.0, "max_angle": 25.0, "home_angle": 0.0},
            "left_arm_y": {"channel": 2, "min_angle": -60.0, "max_angle": 25.0, "home_angle": 0.0},
            "right_arm_x": {"channel": 3, "min_angle": -25.0, "max_angle": 5.0, "home_angle": 0.0},
            "left_arm_x": {"channel": 4, "min_angle": -25.0, "max_angle": 5.0, "home_angle": 0.0},
            "head_pan": {"channel": 5, "min_angle": -90.0, "max_angle": 90.0, "home_angle": 0.0},
        }

        all_names = set(defaults.keys()) | set(raw_channels.keys())
        for name in all_names:
            def_vals = defaults.get(name, {"channel": 0, "min_angle": -90.0, "max_angle": 90.0, "home_angle": 0.0})
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

        # Aliases for convenience, backward compatibility, and direct channel addressing
        alias_map = {
            "right_arm": "right_arm_x",
            "left_arm": "left_arm_x",
            "right_arm_pitch": "right_arm_y",
            "left_arm_pitch": "left_arm_y",
            "body": "head_pan",
            "waist": "head_pan",
            "body_pan": "head_pan",
            "0": "head_tilt",
            "1": "right_arm_y",
            "2": "left_arm_y",
            "3": "right_arm_x",
            "4": "left_arm_x",
            "5": "head_pan",
            "channel_0": "head_tilt",
            "channel_1": "right_arm_y",
            "channel_2": "left_arm_y",
            "channel_3": "right_arm_x",
            "channel_4": "left_arm_x",
            "channel_5": "head_pan",
        }
        for alias, target in alias_map.items():
            if target in self.channels and alias not in self.channels:
                self.channels[alias] = self.channels[target]
                self.current_angles[alias] = self.channels[target].home_angle
                self.target_angles[alias] = self.channels[target].home_angle

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
        targets = {}
        seen_channels = set()
        for name, cal in self.channels.items():
            if cal.channel not in seen_channels:
                seen_channels.add(cal.channel)
                targets[name] = cal.home_angle
        self.move_multiple(targets, duration_s=duration_s)

    def angle_to_pulse_us(self, name: str, angle_deg: float) -> int:
        """Translate physical angle in degrees to pulse width microseconds.
        
        Standard PCA9685/servo mapping:
        -90° to +90° corresponds to min_pulse_us to max_pulse_us (500us - 2500us).
        0° is center (1500us).
        Angles are clamped strictly within [cal.min_angle, cal.max_angle].
        """
        cal = self.channels[name]
        # Clamp to safe limits
        clamped = max(cal.min_angle, min(cal.max_angle, angle_deg))
        if cal.inverted:
            clamped = -clamped

        norm = (clamped + 90.0) / 180.0
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

            # Sync aliases sharing the same physical channel
            for other_name, other_cal in self.channels.items():
                if other_cal.channel == cal.channel:
                    self.current_angles[other_name] = clamped
                    self.target_angles[other_name] = clamped

            pulse = self.angle_to_pulse_us(name, clamped)
            try:
                self.driver.set_pwm_us(cal.channel, pulse)
            except OSError as e:
                logger.warning(f"I2C error setting servo {name} (ch{cal.channel}): {e}")
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
                written_channels = set()
                for k, end_val in valid_targets.items():
                    start_val = start_angles[k]
                    current_interp = start_val + (end_val - start_val) * eased_t
                    self.current_angles[k] = current_interp
                    ch_num = self.channels[k].channel
                    if ch_num not in written_channels:
                        written_channels.add(ch_num)
                        pulse = self.angle_to_pulse_us(k, current_interp)
                        try:
                            self.driver.set_pwm_us(ch_num, pulse)
                        except OSError as e:
                            logger.warning(f"I2C error during interpolation (ch{ch_num}): {e}")
            time.sleep(dt)

        with self._lock:
            for k, end_val in valid_targets.items():
                self.current_angles[k] = end_val
                ch_num = self.channels[k].channel
                for other_name, other_cal in self.channels.items():
                    if other_cal.channel == ch_num:
                        self.current_angles[other_name] = end_val
            self._last_move_time = time.time()

    def relax_all(self) -> None:
        """De-energize all servo channels to prevent humming and heating."""
        with self._lock:
            for name, cal in self.channels.items():
                try:
                    self.driver.release_channel(cal.channel)
                except OSError as e:
                    logger.warning(f"I2C error relaxing servo {name} (ch{cal.channel}): {e}")
            logger.debug("All servo channels relaxed.")

    def shutdown(self) -> None:
        """Stop controller and park servos."""
        self._running = False
        self.home_all(duration_s=0.3)
        time.sleep(0.3)
        self.relax_all()
        self.driver.shutdown()
        logger.info("ServoController shut down cleanly.")
