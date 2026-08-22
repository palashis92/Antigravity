"""Local Command & Skill Router for LUMI.

Prioritizes local skill execution over cloud LLM calls.
If a spoken sentence matches a physical gesture, camera snapshot inspection,
chess evaluation, plant diagnosis, weather, time, or system control command,
it executes locally and returns the response immediately without hitting the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..core.logger import get_logger

logger = get_logger("core.command_router")


class CommandRouter:
    """Matches user speech against local hardware skills and commands before LLM fallback."""

    def __init__(self, brain: Any) -> None:
        self.brain = brain
        self._commands: List[Tuple[List[str], Callable[[str], str], str]] = []
        self._register_default_commands()

    def register(self, keywords: List[str], handler: Callable[[str], str], description: str) -> None:
        """Register a list of trigger keywords/phrases to a local handler function."""
        self._commands.append(([k.lower() for k in keywords], handler, description))

    def _register_default_commands(self) -> None:
        # 1. Physical Gestures & Movements
        self.register(
            ["হাত নাড়াও", "হাত তোলো", "হাই বলো", "wave", "wave hand", "say hi", "greet"],
            self._cmd_wave,
            "Wave hands / Greet gesture"
        )
        self.register(
            ["মাথা নাড়াও", "সম্মতি জানাও", "nod head", "nod"],
            self._cmd_nod,
            "Nod head"
        )
        self.register(
            ["মাথা ডানে বামে করো", "না বলো", "shake head"],
            self._cmd_shake,
            "Shake head"
        )
        self.register(
            ["ডানে তাকাও", "ডান দিকে তাকাও", "look right", "turn right"],
            self._cmd_look_right,
            "Look right"
        )
        self.register(
            ["বামে তাকাও", "বাম দিকে তাকাও", "look left", "turn left"],
            self._cmd_look_left,
            "Look left"
        )
        self.register(
            ["উপরে তাকাও", "look up"],
            self._cmd_look_up,
            "Look up"
        )
        self.register(
            ["নিচে তাকাও", "look down"],
            self._cmd_look_down,
            "Look down"
        )
        self.register(
            ["সামনে তাকাও", "সোজা তাকাও", "মাথা সোজা করো", "look center", "center"],
            self._cmd_look_center,
            "Look center"
        )
        self.register(
            ["আনন্দ প্রকাশ করো", "খুশি হও", "be happy", "happy gesture"],
            self._cmd_happy,
            "Happy gesture"
        )
        self.register(
            ["ঘুমিয়ে পড়ো", "ঘুমাও", "বিশ্রাম নাও", "go to sleep", "sleep mode"],
            self._cmd_sleep,
            "Sleep gesture & mode"
        )
        self.register(
            ["জেগে ওঠো", "উঠো", "wake up"],
            self._cmd_wake_up,
            "Wake up"
        )

        # 2. Chess Analysis Command
        self.register(
            ["দাবা বোর্ড দেখো", "দাবার চাল বলো", "চেসবোর্ড বিশ্লেষণ", "দাবা খেলো", "chess move", "best move", "analyze chess"],
            self._cmd_chess,
            "Chessboard snapshot and Stockfish analysis"
        )

        # 3. Plant Disease Diagnosis
        self.register(
            ["গাছের পাতা দেখো", "পাতার রোগ", "গাছের রোগ কি", "গাছ পরীক্ষা করো", "plant disease", "leaf disease", "check plant"],
            self._cmd_plant,
            "Plant leaf disease detection"
        )

        # 4. On-Demand Visual Description (When camera is active)
        self.register(
            ["তুমি কি দেখতে পাচ্ছো", "সামনে কি আছে", "পরিবেশ বর্ণনা করো", "ক্যামেরা দিয়ে দেখো", "what do you see", "describe what you see"],
            self._cmd_describe_vision,
            "On-demand visual snapshot analysis"
        )

        # 5. Date and Time
        self.register(
            ["কয়টা বাজে", "সময় কত", "আজকের তারিখ", "আজকে কি বার", "what time is it", "current time", "date today"],
            self._cmd_datetime,
            "Current date and time"
        )

        # 6. Weather Query
        self.register(
            ["আজকের আবহাওয়া কেমন", "আবহাওয়া কি", "বৃষ্টি হবে কি", "weather today", "how is the weather"],
            self._cmd_weather,
            "Weather check"
        )

        # 7. Volume & System Control
        self.register(
            ["ভলিউম বাড়াও", "আওয়াজ বাড়াও", "volume up", "increase volume"],
            self._cmd_volume_up,
            "Increase speaker volume"
        )
        self.register(
            ["ভলিউম কমাও", "আওয়াজ কমাও", "volume down", "decrease volume"],
            self._cmd_volume_down,
            "Decrease speaker volume"
        )
        self.register(
            ["চুপ করো", "থেমে যাও", "stop talking", "be quiet", "shut up"],
            self._cmd_stop_speaking,
            "Stop speaking immediately"
        )

    def route_speech(self, text: str) -> Optional[str]:
        """Check if speech text matches any local command.
        
        Returns response text if handled locally, or None if it should be sent to LLM.
        """
        if not text:
            return None

        cleaned = text.lower().strip()
        cleaned_no_punct = re.sub(r'[^\w\s]', '', cleaned)

        for keywords, handler, desc in self._commands:
            for kw in keywords:
                if kw in cleaned or kw in cleaned_no_punct:
                    logger.info(f"Local command triggered: '{desc}' from input: '{text}'")
                    try:
                        return handler(text)
                    except Exception as e:
                        logger.error(f"Error executing local command '{desc}': {e}", exc_info=True)
                        return f"কমান্ডটি সম্পন্ন করতে ত্রুটি হয়েছে: {e}"

        return None

    # -------------------------------------------------------------------------
    # Command Handler Implementations
    # -------------------------------------------------------------------------
    def _cmd_wave(self, text: str) -> str:
        self.brain.eyes.set_expression("happy")
        self.brain.gestures.play_async(self.brain.gestures.greet, name="greet")
        return "হ্যালো! আমি আপনার সাথেই আছি।"

    def _cmd_nod(self, text: str) -> str:
        self.brain.eyes.set_expression("happy")
        self.brain.head.nod(count=2)
        return "জ্বী, আমি বুঝতে পেরেছি।"

    def _cmd_shake(self, text: str) -> str:
        self.brain.eyes.set_expression("thinking")
        self.brain.head.shake(count=2)
        return "না, এটি সম্ভব নয়।"

    def _cmd_look_right(self, text: str) -> str:
        self.brain.head.look_right(deg=40.0)
        self.brain.eyes.set_gaze(0.8, 0.0)
        return "ডান দিকে তাকালাম।"

    def _cmd_look_left(self, text: str) -> str:
        self.brain.head.look_left(deg=40.0)
        self.brain.eyes.set_gaze(-0.8, 0.0)
        return "বাম দিকে তাকালাম।"

    def _cmd_look_up(self, text: str) -> str:
        self.brain.head.look_up(deg=25.0)
        self.brain.eyes.set_gaze(0.0, -0.8)
        return "উপরে তাকালাম।"

    def _cmd_look_down(self, text: str) -> str:
        self.brain.head.look_down(deg=20.0)
        self.brain.eyes.set_gaze(0.0, 0.8)
        return "নিচে তাকালাম।"

    def _cmd_look_center(self, text: str) -> str:
        self.brain.head.look_center()
        self.brain.eyes.set_gaze(0.0, 0.0)
        return "সামনে তাকালাম।"

    def _cmd_happy(self, text: str) -> str:
        self.brain.eyes.set_expression("happy")
        self.brain.gestures.play_async(self.brain.gestures.happy, name="happy")
        return "আমি খুব আনন্দিত!"

    def _cmd_sleep(self, text: str) -> str:
        self.brain.eyes.set_expression("sleepy")
        self.brain.gestures.play_async(self.brain.gestures.sleep, name="sleep")
        return "আমি এখন স্লিপ মোডে যাচ্ছি।"

    def _cmd_wake_up(self, text: str) -> str:
        self.brain.eyes.set_expression("neutral")
        self.brain.head.look_center()
        self.brain.arms.arms_home()
        return "আমি জেগে উঠেছি। বলুন কীভাবে সাহায্য করতে পারি?"

    def _cmd_chess(self, text: str) -> str:
        frame = self.brain.camera.get_frame()
        if frame is None:
            return "বর্তমানে ক্যামেরা সংযুক্ত নেই, তাই দাবাবোর্ড দেখা যাচ্ছে না।"
        return self.brain.analyze_chessboard(frame)

    def _cmd_plant(self, text: str) -> str:
        frame = self.brain.camera.get_frame()
        if frame is None:
            return "বর্তমানে ক্যামেরা সংযুক্ত নেই, তাই গাছের পাতা দেখা যাচ্ছে না।"
        return self.brain.analyze_plant_leaf(frame)

    def _cmd_describe_vision(self, text: str) -> str:
        frame = self.brain.camera.get_frame()
        if frame is None:
            return "বর্তমানে ক্যামেরা অফলাইনে রয়েছে।"
        return self.brain._tool_describe_vision()

    def _cmd_datetime(self, text: str) -> str:
        return self.brain.tools.get_current_datetime()

    def _cmd_weather(self, text: str) -> str:
        return self.brain.tools.get_weather("Dhaka")

    def _cmd_volume_up(self, text: str) -> str:
        current = self.brain.speaker.backend.volume if hasattr(self.brain.speaker.backend, "volume") else 80
        new_vol = min(100, current + 15)
        self.brain.speaker.set_volume(new_vol)
        return f"ভলিউম বাড়িয়ে {new_vol}% করা হয়েছে।"

    def _cmd_volume_down(self, text: str) -> str:
        current = self.brain.speaker.backend.volume if hasattr(self.brain.speaker.backend, "volume") else 80
        new_vol = max(10, current - 15)
        self.brain.speaker.set_volume(new_vol)
        return f"ভলিউম কমিয়ে {new_vol}% করা হয়েছে।"

    def _cmd_stop_speaking(self, text: str) -> str:
        self.brain.speaker.stop()
        return ""
