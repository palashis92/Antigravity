"""Unit tests for kinematic motion and gestures."""

from lumi.hardware.mocks import MockServoDriver
from lumi.motion.arms import ArmController
from lumi.motion.gestures import GestureManager
from lumi.motion.head import HeadController
from lumi.motion.servo_controller import ServoController, ease_in_out_cubic


def test_cubic_easing() -> None:
    assert ease_in_out_cubic(0.0) == 0.0
    assert ease_in_out_cubic(1.0) == 1.0
    assert 0.45 < ease_in_out_cubic(0.5) < 0.55


def test_servo_controller_limits_and_interpolation() -> None:
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()

    # Move beyond limit, ensure clamped safely
    ctrl.move_joint("head_pan", 120.0, duration_s=0.05)
    # Head pan max is 80.0
    assert ctrl.current_angles["head_pan"] == 80.0


def test_head_and_arms() -> None:
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()

    head = HeadController(ctrl)
    arms = ArmController(ctrl)

    head.look_left(30.0, duration_s=0.02)
    assert ctrl.current_angles["head_pan"] == -30.0

    arms.raise_both(60.0, duration_s=0.02)
    assert ctrl.current_angles["left_arm"] == 60.0
    assert ctrl.current_angles["right_arm"] == 60.0


def test_gestures_execution() -> None:
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()
    head = HeadController(ctrl)
    arms = ArmController(ctrl)
    gestures = GestureManager(ctrl, head, arms)

    # Execute synchronous test of greet
    gestures.greet()
    assert ctrl.current_angles["head_pan"] == 0.0
    assert ctrl.current_angles["head_tilt"] == 0.0
