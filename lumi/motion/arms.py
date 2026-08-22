"""Arm Movement Controller for expressive Moxie-style dual arms."""

from __future__ import annotations

from ..core.logger import get_logger
from .servo_controller import ServoController

logger = get_logger("motion.arms")


class ArmController:
    """Controls left and right expressive arm kinematics."""

    def __init__(self, controller: ServoController) -> None:
        self.controller = controller
        self.current_left = 0.0
        self.current_right = 0.0

    def set_left_arm(self, angle_deg: float, duration_s: float = 0.35) -> None:
        self.current_left = angle_deg
        self.controller.move_joint("left_arm", angle_deg, duration_s=duration_s)

    def set_right_arm(self, angle_deg: float, duration_s: float = 0.35) -> None:
        self.current_right = angle_deg
        self.controller.move_joint("right_arm", angle_deg, duration_s=duration_s)

    def set_both_arms(self, left_deg: float, right_deg: float, duration_s: float = 0.35) -> None:
        self.current_left = left_deg
        self.current_right = right_deg
        self.controller.move_multiple(
            {"left_arm": left_deg, "right_arm": right_deg},
            duration_s=duration_s,
        )

    def arms_home(self, duration_s: float = 0.3) -> None:
        """Lower arms to resting idle position (0°, 0°)."""
        self.set_both_arms(0.0, 0.0, duration_s=duration_s)

    def raise_both(self, angle_deg: float = 75.0, duration_s: float = 0.4) -> None:
        """Raise both arms enthusiastically (e.g. celebration/excitement)."""
        self.set_both_arms(angle_deg, angle_deg, duration_s=duration_s)

    def wave_right(self, count: int = 3) -> None:
        """Perform a friendly right-hand wave gesture."""
        for _ in range(count):
            self.set_right_arm(75.0, duration_s=0.2)
            self.set_right_arm(35.0, duration_s=0.2)
        self.set_right_arm(0.0, duration_s=0.25)

    def wave_left(self, count: int = 3) -> None:
        """Perform a friendly left-hand wave gesture."""
        for _ in range(count):
            self.set_left_arm(75.0, duration_s=0.2)
            self.set_left_arm(35.0, duration_s=0.2)
        self.set_left_arm(0.0, duration_s=0.25)
