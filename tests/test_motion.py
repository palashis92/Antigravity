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

    # Head pan max is 90.0
    ctrl.move_joint("head_pan", 120.0, duration_s=0.05)
    assert ctrl.current_angles["head_pan"] == 90.0
    ctrl.move_joint("head_pan", -120.0, duration_s=0.05)
    assert ctrl.current_angles["head_pan"] == -90.0

    # Head tilt clamp [-15.0, 15.0]
    ctrl.move_joint("head_tilt", 50.0, duration_s=0.05)
    assert ctrl.current_angles["head_tilt"] == 15.0
    ctrl.move_joint("head_tilt", -50.0, duration_s=0.05)
    assert ctrl.current_angles["head_tilt"] == -15.0

    # Right hand Y clamp [-60.0, 25.0]
    ctrl.move_joint("right_arm_y", -100.0, duration_s=0.05)
    assert ctrl.current_angles["right_arm_y"] == -60.0
    ctrl.move_joint("right_arm_y", 50.0, duration_s=0.05)
    assert ctrl.current_angles["right_arm_y"] == 25.0

    # Left hand Y clamp [-60.0, 25.0]
    ctrl.move_joint("left_arm_y", -100.0, duration_s=0.05)
    assert ctrl.current_angles["left_arm_y"] == -60.0
    ctrl.move_joint("left_arm_y", 50.0, duration_s=0.05)
    assert ctrl.current_angles["left_arm_y"] == 25.0

    # Right hand X clamp [-25.0, 5.0]
    ctrl.move_joint("right_arm_x", -50.0, duration_s=0.05)
    assert ctrl.current_angles["right_arm_x"] == -25.0
    ctrl.move_joint("right_arm_x", 50.0, duration_s=0.05)
    assert ctrl.current_angles["right_arm_x"] == 5.0

    # Left hand X clamp [-25.0, 5.0]
    ctrl.move_joint("left_arm_x", -50.0, duration_s=0.05)
    assert ctrl.current_angles["left_arm_x"] == -25.0
    ctrl.move_joint("left_arm_x", 50.0, duration_s=0.05)
    assert ctrl.current_angles["left_arm_x"] == 5.0


def test_head_and_arms() -> None:
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()

    head = HeadController(ctrl)
    arms = ArmController(ctrl)

    # +90° = left, -90° = right
    head.look_left(30.0, duration_s=0.02)
    assert ctrl.current_angles["head_pan"] == 30.0
    head.look_right(30.0, duration_s=0.02)
    assert ctrl.current_angles["head_pan"] == -30.0

    # -15° = up, +15° = down
    head.look_up(15.0, duration_s=0.02)
    assert ctrl.current_angles["head_tilt"] == -15.0
    head.look_down(15.0, duration_s=0.02)
    assert ctrl.current_angles["head_tilt"] == 15.0

    # Raise both: Left X: -25° (up), Right X: -25° (up)
    arms.raise_both(duration_s=0.02)
    assert ctrl.current_angles["left_arm_x"] == -25.0
    assert ctrl.current_angles["right_arm_x"] == -25.0


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
    assert ctrl.current_angles["right_arm_x"] == 0.0
    assert ctrl.current_angles["right_arm_y"] == 0.0


def test_ground_truth_channel_mappings_and_clamps() -> None:
    """Explicit verification of the 6 ground-truth tested hardware channels."""
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()

    # Channel 0: Head tilt [-15°, +15°]
    assert ctrl.channels["head_tilt"].channel == 0
    assert ctrl.channels["head_tilt"].min_angle == -15.0
    assert ctrl.channels["head_tilt"].max_angle == 15.0

    # Channel 1: Right hand Y [-60°, +25°]
    assert ctrl.channels["right_arm_y"].channel == 1
    assert ctrl.channels["right_arm_y"].min_angle == -60.0
    assert ctrl.channels["right_arm_y"].max_angle == 25.0

    # Channel 2: Left hand Y [-60°, +25°]
    assert ctrl.channels["left_arm_y"].channel == 2
    assert ctrl.channels["left_arm_y"].min_angle == -60.0
    assert ctrl.channels["left_arm_y"].max_angle == 25.0

    # Channel 3: Right hand X [-25°, +5°]
    assert ctrl.channels["right_arm_x"].channel == 3
    assert ctrl.channels["right_arm_x"].min_angle == -25.0
    assert ctrl.channels["right_arm_x"].max_angle == 5.0

    # Channel 4: Left hand X [-25°, +5°]
    assert ctrl.channels["left_arm_x"].channel == 4
    assert ctrl.channels["left_arm_x"].min_angle == -25.0
    assert ctrl.channels["left_arm_x"].max_angle == 5.0

    # Channel 5: Body waist rotation [-90°, +90°]
    assert ctrl.channels["head_pan"].channel == 5
    assert ctrl.channels["head_pan"].min_angle == -90.0
    assert ctrl.channels["head_pan"].max_angle == 90.0

    # Pulse width calculations matching test_servo.py exactly
    assert ctrl.angle_to_pulse_us("head_pan", 0.0) == 1500
    assert ctrl.angle_to_pulse_us("head_pan", -90.0) == 500
    assert ctrl.angle_to_pulse_us("head_pan", 90.0) == 2500
    assert ctrl.angle_to_pulse_us("head_tilt", -15.0) == 1333
    assert ctrl.angle_to_pulse_us("head_tilt", 15.0) == 1666
