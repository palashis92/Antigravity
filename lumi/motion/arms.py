"""Arm Movement Controller for expressive Moxie-style dual arms (2-DOF per arm).

Channel Mapping:
- Channel 1: Right hand, Y-axis (front/back): +25° = back, -60° = front
- Channel 2: Left hand, Y-axis (front/back): -25° = back, +60° = front
- Channel 3: Right hand, X-axis (up/down): -25° = up, +5° = down
- Channel 4: Left hand, X-axis (up/down): +25° = up, -5° = down
"""

from __future__ import annotations

from ..core.logger import get_logger
from .servo_controller import ServoController

logger = get_logger("motion.arms")


class ArmController:
    """Controls left and right expressive arm kinematics with dual-axis support."""

    def __init__(self, controller: ServoController) -> None:
        self.controller = controller
        self.current_left_x = 0.0
        self.current_left_y = 0.0
        self.current_right_x = 0.0
        self.current_right_y = 0.0

    @property
    def current_left(self) -> float:
        return self.current_left_x

    @property
    def current_right(self) -> float:
        return self.current_right_x

    def set_left_arm(self, angle_deg: float, duration_s: float = 0.25) -> None:
        """Set Left arm X-axis elevation (+25° = up, -5° = down)."""
        self.current_left_x = angle_deg
        self.controller.move_joint("left_arm_x", angle_deg, duration_s=duration_s)

    def set_right_arm(self, angle_deg: float, duration_s: float = 0.25) -> None:
        """Set Right arm X-axis elevation (-25° = up, +5° = down)."""
        self.current_right_x = angle_deg
        self.controller.move_joint("right_arm_x", angle_deg, duration_s=duration_s)

    def set_left_arm_y(self, angle_deg: float, duration_s: float = 0.25) -> None:
        """Set Left arm Y-axis reach (+60° = front, -25° = back)."""
        self.current_left_y = angle_deg
        self.controller.move_joint("left_arm_y", angle_deg, duration_s=duration_s)

    def set_right_arm_y(self, angle_deg: float, duration_s: float = 0.25) -> None:
        """Set Right arm Y-axis reach (-60° = front, +25° = back)."""
        self.current_right_y = angle_deg
        self.controller.move_joint("right_arm_y", angle_deg, duration_s=duration_s)

    def set_both_arms(self, left_deg: float, right_deg: float, duration_s: float = 0.25) -> None:
        """Set elevation (X-axis) for both arms simultaneously."""
        self.current_left_x = left_deg
        self.current_right_x = right_deg
        self.controller.move_multiple(
            {"left_arm_x": left_deg, "right_arm_x": right_deg},
            duration_s=duration_s,
        )

    def arms_home(self, duration_s: float = 0.25) -> None:
        """Lower arms to resting idle position (0°, 0° on all axes)."""
        self.current_left_x = 0.0
        self.current_left_y = 0.0
        self.current_right_x = 0.0
        self.current_right_y = 0.0
        self.controller.move_multiple(
            {
                "left_arm_x": 0.0,
                "right_arm_x": 0.0,
                "left_arm_y": 0.0,
                "right_arm_y": 0.0,
            },
            duration_s=duration_s,
        )

    def lower_both(self, duration_s: float = 0.25) -> None:
        """Lower both arms down to resting position."""
        self.arms_home(duration_s=duration_s)

    def raise_right(self, duration_s: float = 0.25) -> None:
        """Raise right arm up and slightly forward (-25° X, -20° Y)."""
        self.current_right_x = -25.0
        self.current_right_y = -20.0
        self.controller.move_multiple(
            {"right_arm_x": -25.0, "right_arm_y": -20.0},
            duration_s=duration_s,
        )

    def raise_left(self, duration_s: float = 0.25) -> None:
        """Raise left arm up and slightly forward (+25° X, +20° Y)."""
        self.current_left_x = 25.0
        self.current_left_y = 20.0
        self.controller.move_multiple(
            {"left_arm_x": 25.0, "left_arm_y": 20.0},
            duration_s=duration_s,
        )

    def raise_both(self, duration_s: float = 0.28) -> None:
        """Raise both arms up enthusiastically (Left X: +25°, Right X: -25°)."""
        self.current_left_x = 25.0
        self.current_right_x = -25.0
        self.controller.move_multiple(
            {
                "left_arm_x": 25.0,
                "right_arm_x": -25.0,
                "left_arm_y": 20.0,
                "right_arm_y": -20.0,
            },
            duration_s=duration_s,
        )

    def point_right(self, duration_s: float = 0.25) -> None:
        """Point right arm forward (-50° Y, -15° X)."""
        self.controller.move_multiple(
            {"right_arm_x": -15.0, "right_arm_y": -50.0},
            duration_s=duration_s,
        )

    def point_left(self, duration_s: float = 0.25) -> None:
        """Point left arm forward (+50° Y, +15° X)."""
        self.controller.move_multiple(
            {"left_arm_x": 15.0, "left_arm_y": 50.0},
            duration_s=duration_s,
        )

    def set_arm_pose(
        self,
        left_x: float = 0.0,
        left_y: float = 0.0,
        right_x: float = 0.0,
        right_y: float = 0.0,
        duration_s: float = 0.25,
    ) -> None:
        """Simultaneously coordinate all 4 arm servos (2-DOF per arm)."""
        self.current_left_x = left_x
        self.current_left_y = left_y
        self.current_right_x = right_x
        self.current_right_y = right_y
        self.controller.move_multiple(
            {
                "left_arm_x": left_x,
                "left_arm_y": left_y,
                "right_arm_x": right_x,
                "right_arm_y": right_y,
            },
            duration_s=duration_s,
        )

    def reach_forward_left(self, y_deg: float = 35.0, x_deg: float = 15.0, duration_s: float = 0.28) -> None:
        """Reach left hand forward prominently on the Y-axis."""
        self.current_left_x = x_deg
        self.current_left_y = y_deg
        self.controller.move_multiple(
            {"left_arm_x": x_deg, "left_arm_y": y_deg}, duration_s=duration_s
        )

    def reach_forward_right(self, y_deg: float = -35.0, x_deg: float = -15.0, duration_s: float = 0.28) -> None:
        """Reach right hand forward prominently on the Y-axis."""
        self.current_right_x = x_deg
        self.current_right_y = y_deg
        self.controller.move_multiple(
            {"right_arm_x": x_deg, "right_arm_y": y_deg}, duration_s=duration_s
        )

    def open_hands(self, forward_deg: float = 28.0, lift_deg: float = 12.0, duration_s: float = 0.3) -> None:
        """Open both hands forward towards user in welcoming conversational posture."""
        self.set_arm_pose(
            left_x=lift_deg,
            left_y=forward_deg,
            right_x=-lift_deg,
            right_y=-forward_deg,
            duration_s=duration_s,
        )

    def shrug(self, lift_deg: float = 15.0, forward_deg: float = 18.0, duration_s: float = 0.28) -> None:
        """Expressive human-like shrug moving both elevation and reach axes."""
        self.set_arm_pose(
            left_x=lift_deg,
            left_y=forward_deg,
            right_x=-lift_deg,
            right_y=-forward_deg,
            duration_s=duration_s,
        )

    def wave_right(self, count: int = 3) -> None:
        """Perform a friendly right-hand wave gesture with dynamic Y-axis reach."""
        for _ in range(count):
            self.controller.move_multiple(
                {"right_arm_x": -25.0, "right_arm_y": -45.0}, duration_s=0.2
            )
            self.controller.move_multiple(
                {"right_arm_x": -15.0, "right_arm_y": -15.0}, duration_s=0.2
            )
        self.arms_home(duration_s=0.25)

    def wave_left(self, count: int = 3) -> None:
        """Perform a friendly left-hand wave gesture with dynamic Y-axis reach."""
        for _ in range(count):
            self.controller.move_multiple(
                {"left_arm_x": 25.0, "left_arm_y": 45.0}, duration_s=0.2
            )
            self.controller.move_multiple(
                {"left_arm_x": 15.0, "left_arm_y": 15.0}, duration_s=0.2
            )
        self.arms_home(duration_s=0.25)
