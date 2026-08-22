"""Reusable High-Level Gestures Library."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from ..core.logger import get_logger
from .arms import ArmController
from .head import HeadController
from .servo_controller import ServoController

logger = get_logger("motion.gestures")


class GestureManager:
    """Orchestrates combined multi-joint expressive bodily gestures."""

    def __init__(
        self,
        servo_controller: ServoController,
        head: HeadController,
        arms: ArmController,
    ) -> None:
        self.servo = servo_controller
        self.head = head
        self.arms = arms
        self._current_gesture_thread: Optional[threading.Thread] = None

    def play_async(self, gesture_fn: Callable[[], None], name: str = "gesture") -> None:
        """Run a gesture choreography in a dedicated background thread."""
        if self._current_gesture_thread and self._current_gesture_thread.is_alive():
            logger.debug(f"Interrupting/ignoring overlapping gesture request for '{name}'.")
            return

        t = threading.Thread(target=gesture_fn, name=f"LumiGesture-{name}", daemon=True)
        self._current_gesture_thread = t
        t.start()

    # -------------------------------------------------------------------------
    # Core Gesture Choreographies
    # -------------------------------------------------------------------------

    def greet(self) -> None:
        """Friendly greeting: tilts head up slightly, waves right arm, returns home."""
        logger.info("Executing gesture: GREET")
        self.head.look_at(0.0, 15.0, duration_s=0.3)
        self.arms.wave_right(count=2)
        self.head.look_center(duration_s=0.25)
        self.arms.arms_home(duration_s=0.25)

    def wave(self) -> None:
        """Simple arm wave gesture."""
        logger.info("Executing gesture: WAVE")
        self.arms.wave_right(count=3)

    def happy(self) -> None:
        """Joyful expression: nods head enthusiastically and raises both arms."""
        logger.info("Executing gesture: HAPPY")
        self.arms.raise_both(60.0, duration_s=0.3)
        self.head.nod(count=2, amplitude_deg=18.0)
        self.arms.arms_home(duration_s=0.3)
        self.head.look_center(duration_s=0.2)

    def thinking(self) -> None:
        """Contemplative gesture: tilts head to side and slightly up."""
        logger.info("Executing gesture: THINKING")
        self.head.look_at(18.0, 20.0, duration_s=0.5)
        self.arms.set_left_arm(25.0, duration_s=0.4)

    def curious(self) -> None:
        """Inquisitive head tilt."""
        logger.info("Executing gesture: CURIOUS")
        self.head.look_at(-15.0, 10.0, duration_s=0.4)
        time.sleep(0.3)
        self.head.look_at(15.0, 10.0, duration_s=0.4)
        self.head.look_center(duration_s=0.3)

    def excited(self) -> None:
        """Energetic bounce with arms up."""
        logger.info("Executing gesture: EXCITED")
        self.arms.raise_both(75.0, duration_s=0.25)
        for _ in range(2):
            self.head.look_at(0.0, 25.0, duration_s=0.15)
            self.head.look_at(0.0, -10.0, duration_s=0.15)
        self.head.look_center(duration_s=0.2)
        self.arms.arms_home(duration_s=0.3)

    def sleep(self) -> None:
        """Head drops down, arms rest down."""
        logger.info("Executing gesture: SLEEP")
        self.arms.arms_home(duration_s=0.4)
        self.head.look_down(25.0, duration_s=0.8)

    def bored(self) -> None:
        """Bored gesture: subtle head tilt, slight arm shift, look around."""
        logger.info("Executing gesture: BORED")
        self.head.look_at(-12.0, 15.0, duration_s=0.6)
        self.arms.set_left_arm(18.0, duration_s=0.4)
        time.sleep(0.4)
        self.arms.set_left_arm(0.0, duration_s=0.4)
        self.head.look_at(12.0, 10.0, duration_s=0.6)
        time.sleep(0.3)
        self.head.look_center(duration_s=0.5)

    def idle_alive_motion(self) -> None:
        """Lifelike organic micro-movements when idle (head wander, small arm twitch)."""
        import random
        action = random.choice(["look_around", "bored_shrug", "head_glance", "arm_stretch"])
        logger.debug(f"Executing idle alive motion: {action}")

        if action == "look_around":
            pan = random.choice([-25.0, -15.0, 15.0, 25.0])
            tilt = random.uniform(-8.0, 12.0)
            self.head.look_at(pan, tilt, duration_s=0.6)
            time.sleep(random.uniform(0.4, 0.8))
            self.head.look_center(duration_s=0.5)

        elif action == "bored_shrug":
            self.head.look_at(random.uniform(-10.0, 10.0), -12.0, duration_s=0.4)
            self.arms.set_both_arms(15.0, 15.0, duration_s=0.35)
            time.sleep(0.3)
            self.arms.arms_home(duration_s=0.4)
            self.head.look_center(duration_s=0.4)

        elif action == "head_glance":
            self.head.look_at(random.uniform(-20.0, 20.0), random.uniform(5.0, 18.0), duration_s=0.5)
            time.sleep(0.3)
            self.head.look_center(duration_s=0.4)

        elif action == "arm_stretch":
            arm = random.choice(["left", "right"])
            if arm == "left":
                self.arms.set_left_arm(20.0, duration_s=0.4)
                time.sleep(0.3)
                self.arms.set_left_arm(0.0, duration_s=0.4)
            else:
                self.arms.set_right_arm(20.0, duration_s=0.4)
                time.sleep(0.3)
                self.arms.set_right_arm(0.0, duration_s=0.4)

    def idle_pose(self) -> None:
        """Return all joints to neutral rest position."""
        self.head.look_center(duration_s=0.3)
        self.arms.arms_home(duration_s=0.3)
