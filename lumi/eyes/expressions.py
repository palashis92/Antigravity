"""Procedural eye expression definitions and emotion states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class EyeState(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CURIOUS = "curious"
    SURPRISED = "surprised"
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"
    SLEEPY = "sleepy"
    EXCITED = "excited"
    SAD = "sad"
    BLINK = "blink"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"


@dataclass
class ExpressionConfig:
    """Procedural geometry parameters for rendering round eyes on 240x240 LCDs."""
    name: str
    pupil_radius: float = 48.0
    pupil_scale_x: float = 1.0
    pupil_scale_y: float = 1.0
    upper_lid_cover: float = 0.0   # 0.0 (fully open) to 1.0 (fully closed)
    lower_lid_cover: float = 0.0   # 0.0 (fully open) to 1.0 (fully closed)
    eye_squint: float = 0.0        # Corner pinching
    eye_tilt_deg: float = 0.0      # Angular tilt
    iris_color_rgb: Tuple[int, int, int] = (0, 210, 255)  # Glowing cyan/blue
    pupil_color_rgb: Tuple[int, int, int] = (10, 20, 40)
    highlight_offset: Tuple[float, float] = (-12.0, -12.0)
    gaze_offset: Tuple[float, float] = (0.0, 0.0)


EXPRESSIONS: Dict[str, ExpressionConfig] = {
    "neutral": ExpressionConfig(
        name="neutral",
        pupil_radius=52.0,
        upper_lid_cover=0.08,
        lower_lid_cover=0.05,
        iris_color_rgb=(0, 200, 255),
    ),
    "happy": ExpressionConfig(
        name="happy",
        pupil_radius=50.0,
        upper_lid_cover=0.0,
        lower_lid_cover=0.55,  # Curved smiling crescent
        eye_tilt_deg=5.0,
        iris_color_rgb=(0, 240, 200),
    ),
    "curious": ExpressionConfig(
        name="curious",
        pupil_radius=56.0,
        upper_lid_cover=0.0,
        lower_lid_cover=0.0,
        eye_tilt_deg=-8.0,
        iris_color_rgb=(0, 220, 255),
    ),
    "surprised": ExpressionConfig(
        name="surprised",
        pupil_radius=42.0,
        pupil_scale_x=1.1,
        pupil_scale_y=1.1,
        upper_lid_cover=0.0,
        lower_lid_cover=0.0,
        iris_color_rgb=(100, 240, 255),
    ),
    "thinking": ExpressionConfig(
        name="thinking",
        pupil_radius=46.0,
        upper_lid_cover=0.25,
        lower_lid_cover=0.15,
        gaze_offset=(18.0, -22.0),  # Looking up and away
        iris_color_rgb=(180, 140, 255),
    ),
    "listening": ExpressionConfig(
        name="listening",
        pupil_radius=54.0,
        upper_lid_cover=0.05,
        lower_lid_cover=0.05,
        iris_color_rgb=(0, 255, 170),
    ),
    "speaking": ExpressionConfig(
        name="speaking",
        pupil_radius=52.0,
        upper_lid_cover=0.1,
        lower_lid_cover=0.1,
        iris_color_rgb=(0, 215, 255),
    ),
    "sleepy": ExpressionConfig(
        name="sleepy",
        pupil_radius=40.0,
        upper_lid_cover=0.65,
        lower_lid_cover=0.25,
        iris_color_rgb=(120, 160, 200),
    ),
    "excited": ExpressionConfig(
        name="excited",
        pupil_radius=58.0,
        upper_lid_cover=0.0,
        lower_lid_cover=0.3,
        iris_color_rgb=(255, 210, 0),
    ),
    "sad": ExpressionConfig(
        name="sad",
        pupil_radius=44.0,
        upper_lid_cover=0.4,
        lower_lid_cover=0.0,
        eye_tilt_deg=10.0,
        iris_color_rgb=(80, 140, 220),
    ),
    "blink": ExpressionConfig(
        name="blink",
        pupil_radius=40.0,
        upper_lid_cover=1.0,
        lower_lid_cover=1.0,
    ),
    "look_left": ExpressionConfig(
        name="look_left",
        pupil_radius=50.0,
        gaze_offset=(-32.0, 0.0),
    ),
    "look_right": ExpressionConfig(
        name="look_right",
        pupil_radius=50.0,
        gaze_offset=(32.0, 0.0),
    ),
    "look_up": ExpressionConfig(
        name="look_up",
        pupil_radius=50.0,
        gaze_offset=(0.0, -28.0),
    ),
    "look_down": ExpressionConfig(
        name="look_down",
        pupil_radius=50.0,
        gaze_offset=(0.0, 28.0),
    ),
}
