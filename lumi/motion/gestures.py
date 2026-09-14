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
        self._last_gesture_times: Dict[str, float] = {}
        self._last_speech_anim_time: float = 0.0

    def play_async(self, gesture_fn: Callable[[], None], name: str = "gesture") -> None:
        """Run a gesture choreography in a dedicated background thread."""
        if self._current_gesture_thread and self._current_gesture_thread.is_alive():
            logger.debug(f"Interrupting/ignoring overlapping gesture request for '{name}'.")
            return

        t = threading.Thread(target=gesture_fn, name=f"LumiGesture-{name}", daemon=True)
        self._current_gesture_thread = t
        t.start()

    @property
    def is_playing(self) -> bool:
        """Return True if a gesture choreography is actively executing."""
        return bool(self._current_gesture_thread and self._current_gesture_thread.is_alive())

    def _should_dampen(self, gesture_name: str, cooldown_s: float = 15.0) -> bool:
        """Check if gesture was recently played to prevent annoying repetitive loops."""
        now = time.time()
        last = self._last_gesture_times.get(gesture_name, 0.0)
        self._last_gesture_times[gesture_name] = now
        return (now - last) < cooldown_s

    # -------------------------------------------------------------------------
    # Core Gesture Choreographies (Full 2-DOF X + Y Axis Utilization)
    # -------------------------------------------------------------------------

    def greet(self) -> None:
        """Friendly greeting: reaches forward on Y, waves with head nod."""
        logger.info("Executing gesture: GREET")
        if self._should_dampen("greet", cooldown_s=12.0):
            # Subtle conversational nod & soft hand gesture if called repeatedly
            self.head.nod(count=1, amplitude_deg=8.0)
            self.arms.reach_forward_right(y_deg=-25.0, x_deg=-12.0, duration_s=0.25)
            time.sleep(0.3)
            self.arms.arms_home(duration_s=0.25)
            return

        self.head.look_at(0.0, -8.0, duration_s=0.25)
        self.arms.wave_right(count=2)
        self.head.look_center(duration_s=0.25)
        self.arms.arms_home(duration_s=0.25)

    def wave(self) -> None:
        """Friendly arm wave gesture with Y-axis reach."""
        logger.info("Executing gesture: WAVE")
        self.arms.wave_right(count=2)

    def happy(self) -> None:
        """Joyful expression: coordinates Y reach and head nod without repetitive flapping."""
        logger.info("Executing gesture: HAPPY")
        if self._should_dampen("happy", cooldown_s=12.0):
            # Soft pleasant nod and gentle open hands on repeated happy triggers
            self.head.nod(count=1, amplitude_deg=10.0)
            self.arms.open_hands(forward_deg=22.0, lift_deg=10.0, duration_s=0.25)
            time.sleep(0.3)
            self.arms.arms_home(duration_s=0.25)
            return

        # Full celebratory happy gesture (reaches forward on Y and up on X)
        self.arms.set_arm_pose(
            left_x=22.0, left_y=35.0,
            right_x=-22.0, right_y=-35.0,
            duration_s=0.28,
        )
        self.head.nod(count=2, amplitude_deg=12.0)
        time.sleep(0.2)
        self.arms.arms_home(duration_s=0.3)
        self.head.look_center(duration_s=0.2)

    def thinking(self) -> None:
        """Contemplative gesture: tilts head up and brings left hand forward on Y."""
        logger.info("Executing gesture: THINKING")
        self.head.look_at(15.0, -8.0, duration_s=0.4)
        # Prominently move left arm forward on Y (+35°) and slight lift (+15°)
        self.arms.reach_forward_left(y_deg=35.0, x_deg=16.0, duration_s=0.35)
        time.sleep(0.5)
        self.arms.arms_home(duration_s=0.35)
        self.head.look_center(duration_s=0.3)

    def curious(self) -> None:
        """Inquisitive head tilt with subtle conversational hand reach."""
        logger.info("Executing gesture: CURIOUS")
        self.head.look_at(-14.0, -6.0, duration_s=0.35)
        self.arms.reach_forward_left(y_deg=25.0, x_deg=10.0, duration_s=0.3)
        time.sleep(0.3)
        self.head.look_at(14.0, -6.0, duration_s=0.35)
        self.arms.arms_home(duration_s=0.3)
        self.head.look_center(duration_s=0.25)

    def excited(self) -> None:
        """Energetic bounce with forward arm reach and head nods."""
        logger.info("Executing gesture: EXCITED")
        self.arms.set_arm_pose(
            left_x=25.0, left_y=40.0,
            right_x=-25.0, right_y=-40.0,
            duration_s=0.22,
        )
        for _ in range(2):
            self.head.look_at(0.0, -12.0, duration_s=0.14)
            self.head.look_at(0.0, 8.0, duration_s=0.14)
        self.head.look_center(duration_s=0.2)
        self.arms.arms_home(duration_s=0.25)

    def celebrate(self) -> None:
        """Triumphant celebration: raises and extends arms forward on Y."""
        logger.info("Executing gesture: CELEBRATE")
        self.arms.set_arm_pose(
            left_x=25.0, left_y=45.0,
            right_x=-25.0, right_y=-45.0,
            duration_s=0.25,
        )
        self.head.nod(count=3, amplitude_deg=12.0)
        time.sleep(0.2)
        self.arms.arms_home(duration_s=0.3)
        self.head.look_center(duration_s=0.2)

    def dance(self) -> None:
        """Playful rhythmic dance: sways waist left/right with alternating Y-reach pulses."""
        logger.info("Executing gesture: DANCE")
        for _ in range(2):
            # Sway left with left arm reaching forward on Y
            self.head.look_at(22.0, -5.0, duration_s=0.22)
            self.arms.set_arm_pose(left_x=20.0, left_y=40.0, right_x=0.0, right_y=10.0, duration_s=0.2)
            time.sleep(0.18)
            # Sway right with right arm reaching forward on Y
            self.head.look_at(-22.0, -5.0, duration_s=0.22)
            self.arms.set_arm_pose(left_x=0.0, left_y=-10.0, right_x=-20.0, right_y=-40.0, duration_s=0.2)
            time.sleep(0.18)
        self.head.look_center(duration_s=0.25)
        self.arms.arms_home(duration_s=0.25)

    def sleep(self) -> None:
        """Head drops down (+15°), arms relax completely."""
        logger.info("Executing gesture: SLEEP")
        self.arms.arms_home(duration_s=0.4)
        self.head.look_down(14.0, duration_s=0.8)

    def bored(self) -> None:
        """Bored gesture: subtle head tilt with asymmetrical forward arm stretch."""
        logger.info("Executing gesture: BORED")
        self.head.look_at(-12.0, 8.0, duration_s=0.5)
        self.arms.reach_forward_left(y_deg=30.0, x_deg=10.0, duration_s=0.4)
        time.sleep(0.4)
        self.arms.arms_home(duration_s=0.4)
        self.head.look_center(duration_s=0.4)

    # -------------------------------------------------------------------------
    # Conversational Speech Gesticulation (Co-Verbal Micro-Movements)
    # -------------------------------------------------------------------------

    def play_conversational_step(self) -> None:
        """Execute a single organic, natural micro-gesture while robot is speaking."""
        import random

        actions = [
            "explain_left",
            "explain_right",
            "open_hands",
            "thoughtful_nod",
            "subtle_shrug",
            "head_tilt_talk",
        ]
        chosen = random.choice(actions)

        if chosen == "explain_left":
            # Left hand reaches forward on Y (+30°) with subtle head angle
            self.head.look_at(random.uniform(-8.0, 8.0), random.uniform(-6.0, 4.0), duration_s=0.3)
            self.arms.reach_forward_left(y_deg=random.uniform(22.0, 36.0), x_deg=random.uniform(8.0, 15.0), duration_s=0.3)
            time.sleep(0.35)
            self.arms.arms_home(duration_s=0.35)

        elif chosen == "explain_right":
            # Right hand reaches forward on Y (-30°) with subtle head angle
            self.head.look_at(random.uniform(-8.0, 8.0), random.uniform(-6.0, 4.0), duration_s=0.3)
            self.arms.reach_forward_right(y_deg=random.uniform(-36.0, -22.0), x_deg=random.uniform(-15.0, -8.0), duration_s=0.3)
            time.sleep(0.35)
            self.arms.arms_home(duration_s=0.35)

        elif chosen == "open_hands":
            # Both hands reach forward slightly in open conversational posture
            self.head.look_at(0.0, random.uniform(-5.0, 5.0), duration_s=0.28)
            self.arms.open_hands(forward_deg=random.uniform(18.0, 28.0), lift_deg=random.uniform(8.0, 14.0), duration_s=0.3)
            time.sleep(0.4)
            self.arms.arms_home(duration_s=0.35)

        elif chosen == "thoughtful_nod":
            # Gentle nod while speaking
            self.head.look_at(random.uniform(-6.0, 6.0), 8.0, duration_s=0.2)
            self.head.look_at(0.0, -4.0, duration_s=0.2)
            time.sleep(0.2)
            self.head.look_center(duration_s=0.2)

        elif chosen == "subtle_shrug":
            # Small communicative shrug using X and Y axes
            self.arms.shrug(lift_deg=12.0, forward_deg=18.0, duration_s=0.25)
            time.sleep(0.3)
            self.arms.arms_home(duration_s=0.3)

        elif chosen == "head_tilt_talk":
            # Subtle expressive tilt
            tilt_dir = random.choice([-10.0, 10.0])
            self.head.look_at(tilt_dir, random.uniform(-4.0, 4.0), duration_s=0.35)
            time.sleep(0.3)
            self.head.look_center(duration_s=0.3)

    def idle_alive_motion(self) -> None:
        """Lifelike organic micro-movements when idle (actively moves Y-axis hand)."""
        import random

        action = random.choice([
            "look_around",
            "bored_shrug",
            "reach_and_look",
            "arm_stretch_y",
            "two_handed_open",
        ])
        logger.debug(f"Executing idle alive motion: {action}")

        if action == "look_around":
            pan = random.choice([-22.0, -12.0, 12.0, 22.0])
            tilt = random.uniform(-8.0, 8.0)
            self.head.look_at(pan, tilt, duration_s=0.5)
            time.sleep(random.uniform(0.4, 0.7))
            self.head.look_center(duration_s=0.4)

        elif action == "bored_shrug":
            self.head.look_at(random.uniform(-8.0, 8.0), 8.0, duration_s=0.35)
            self.arms.shrug(lift_deg=12.0, forward_deg=18.0, duration_s=0.3)
            time.sleep(0.3)
            self.arms.arms_home(duration_s=0.35)
            self.head.look_center(duration_s=0.3)

        elif action == "reach_and_look":
            # Look right and extend right hand forward on Y
            self.head.look_at(-18.0, -4.0, duration_s=0.4)
            self.arms.reach_forward_right(y_deg=-30.0, x_deg=-12.0, duration_s=0.35)
            time.sleep(0.4)
            self.arms.arms_home(duration_s=0.35)
            self.head.look_center(duration_s=0.3)

        elif action == "arm_stretch_y":
            # Reach left hand forward on Y
            self.head.look_at(18.0, -4.0, duration_s=0.4)
            self.arms.reach_forward_left(y_deg=30.0, x_deg=12.0, duration_s=0.35)
            time.sleep(0.4)
            self.arms.arms_home(duration_s=0.35)
            self.head.look_center(duration_s=0.3)

        elif action == "two_handed_open":
            # Subtle open posture
            self.arms.open_hands(forward_deg=20.0, lift_deg=10.0, duration_s=0.3)
            time.sleep(0.35)
            self.arms.arms_home(duration_s=0.3)

    def idle_pose(self) -> None:
        """Return all joints to neutral rest position."""
        self.head.look_center(duration_s=0.3)
        self.arms.arms_home(duration_s=0.3)
