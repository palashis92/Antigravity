"""Specialized Speech-Therapy and Companion Engine for 5-year-old Anjum.

Designed to address speech delay, screen-time withdrawal, and hyperactivity:
- High energy, enthusiastic vocal stimuli
- "Reverse Teaching" counting game (Anjum teaches LUMI numbers 1 to 10)
- Animal sound mimicking and nursery rhymes
- Self-identity and movement prompts
- Automatic 10-second absence timeout
"""

from __future__ import annotations

import random
import re
import time
from typing import Optional, Tuple

from ..core.logger import get_logger

logger = get_logger("companion.anjum")

# Bengali and English number mappings for interactive counting
_NUMBER_MAP = {
    "১": 1, "এক": 1, "one": 1, "1": 1,
    "২": 2, "দুই": 2, "two": 2, "2": 2,
    "৩": 3, "তিন": 3, "three": 3, "3": 3,
    "৪": 4, "চার": 4, "four": 4, "4": 4,
    "৫": 5, "পাঁচ": 5, "five": 5, "5": 5,
    "৬": 6, "ছয়": 6, "six": 6, "6": 6,
    "৭": 7, "সাত": 7, "seven": 7, "7": 7,
    "৮": 8, "আট": 8, "eight": 8, "8": 8,
    "৯": 9, "নয়": 9, "nine": 9, "9": 9,
    "১০": 10, "দশ": 10, "ten": 10, "10": 10,
}

_BANGLA_DIGITS = ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "১০"]


class AnjumCompanionEngine:
    """Manages therapeutic interactions, games, and proactive prompts for Anjum."""

    def __init__(self, stimulus_cooldown: float = 7.0) -> None:
        self.is_active = False
        self.last_seen_time = 0.0
        self.last_stimulus_time = 0.0
        self.stimulus_cooldown = stimulus_cooldown
        self.counting_target = 1  # Next number LUMI expects or prompts
        self._stimulus_index = 0

        # Stimuli pools
        self._counting_prompts = [
            "আঞ্জুম আপু! আমাকে একটু গণনা শেখাও না! বলো তো ১!",
            "আঞ্জুম সোনা, তুমি তো অনেক ভালো পড়াতে পারো! বলো তো ১!",
            "চলো আমরা গুনি! বলো তো ১! ১ এর পর কি?",
        ]

        self._animal_prompts = [
            "আঞ্জুম! বিড়াল কেমন করে ডাকে বলতো? মিউ মিউ! তুমি একবার ডাকো তো!",
            "আঞ্জুম সোনা! কুকুর কেমন করে ডাকে বলতো? ঘেউ ঘেউ! তুমি ডাকো তো!",
            "পাখি কেমন করে গান গায় বলতো আঞ্জুম? কিচিরমিচির!",
            "গরু কেমন করে ডাকে বলতো? হাম্বা হাম্বা!",
        ]

        self._rhyme_prompts = [
            "আঞ্জুম! আয় আয় চাঁদ মামা, টিপ দিয়ে যা... এরপর কি বলতো?",
            "আঞ্জুম সোনা! তাই তাই তাই, মামার বাড়ি যাই... বলো তো পরের লাইন!",
            "খোকন খোকন ডাক দিয়ে যাই, খোকন মডার ঘরে নাই! আঞ্জুম বলো তো!",
            "বৃষ্টি পড়ে টাপুর টুপুর, নদে এলো বান! কি মজা!",
        ]

        self._identity_and_action_prompts = [
            "আঞ্জুম কে বলতো? এই তো আমাদের লক্ষ্মী মেয়ে! হাত তুলে নিজেকে দেখাও তো সোনা!",
            "আঞ্জুম! তোমার মিষ্টি নামটা একবার বলো তো শুনি!",
            "আঞ্জুম! আমার দিকে তাকিয়ে একটু হাত তুলে টা-টা দাও তো!",
            "আঞ্জুম সোনা! একটু মিষ্টি করে হাসো তো! এই তো কী সুন্দর হাসি!",
            "এই যে আঞ্জুম! আমি তোমার বন্ধু লুমি! হাত নাড়াও তো একবার!",
        ]

    def activate(self) -> str:
        """Called when Anjum is first spotted."""
        self.is_active = True
        now = time.time()
        self.last_seen_time = now
        self.last_stimulus_time = now + 1.0  # slight delay before first prompt
        self.counting_target = 1
        logger.info("AnjumCompanionEngine ACTIVATED.")
        return "আরেহ্! আমাদের মিষ্টি আঞ্জুম চলে এসেছে! কেমন আছো আঞ্জুম সোনা?"

    def deactivate(self) -> str:
        """Called when Anjum leaves or mode is exited."""
        self.is_active = False
        self.counting_target = 1
        logger.info("AnjumCompanionEngine DEACTIVATED.")
        return "আঞ্জুম আবার এসো কিন্তু! আমি তোমার জন্য এখানেই অপেক্ষা করব।"

    def mark_seen(self, now: Optional[float] = None) -> None:
        """Mark that Anjum's face is currently visible in camera."""
        self.last_seen_time = now or time.time()

    def is_timeout(self, now: Optional[float] = None, timeout_sec: float = 10.0) -> bool:
        """Check if Anjum has not been seen for longer than timeout_sec."""
        current = now or time.time()
        return (current - self.last_seen_time) > timeout_sec

    def check_proactive_stimulus(self, now: Optional[float] = None) -> Optional[str]:
        """Check if enough silence has passed to speak proactively to Anjum."""
        if not self.is_active:
            return None

        current = now or time.time()
        if (current - self.last_stimulus_time) < self.stimulus_cooldown:
            return None

        self.last_stimulus_time = current
        return self._get_next_stimulus()

    def _get_next_stimulus(self) -> str:
        """Rotate through engaging categories to keep her auditory interest high."""
        self._stimulus_index = (self._stimulus_index + 1) % 4
        if self._stimulus_index == 0:
            return random.choice(self._counting_prompts)
        elif self._stimulus_index == 1:
            return random.choice(self._animal_prompts)
        elif self._stimulus_index == 2:
            return random.choice(self._identity_and_action_prompts)
        else:
            return random.choice(self._rhyme_prompts)

    def handle_anjum_speech(self, text: str) -> Optional[str]:
        """Evaluate Anjum's speech utterance and generate an enthusiastic response.

        Specifically recognizes:
        - Numbers (for the counting game)
        - Name / identity responses
        - Generic vocalizations to heavily praise
        """
        if not self.is_active or not text.strip():
            return None

        self.last_stimulus_time = time.time()
        clean = text.lower().strip()
        # Strip punctuation to handle inputs like "এক!" or "2?"
        clean = re.sub(r'[।!?.,;:\'"()\[\]{}]', '', clean).strip()

        # 1. Number Detection (Counting game)
        for word, num in _NUMBER_MAP.items():
            if re.search(r'\b' + re.escape(word) + r'\b', clean) or clean == word:
                next_num = num + 1
                if next_num <= 10:
                    bn_next = _BANGLA_DIGITS[next_num]
                    bn_current = _BANGLA_DIGITS[num]
                    responses = [
                        f"বাহ্! {bn_current}! অনেক সুন্দর হয়েছে! এবার বলো তো {bn_next}!",
                        f"সাবাশ আঞ্জুম! {bn_current}! তারপর কি? বলো তো {bn_next}!",
                        f"ওয়াও! {bn_current}! তুমি তো দারুণ পড়াতে পারো! এবার বলো {bn_next}!",
                    ]
                    return random.choice(responses)
                else:
                    return "ওয়াও আঞ্জুম! তুমি তো পুরো ১০ পর্যন্ত গুনে ফেলেছো! তুমি অনেক জিনিয়াস মেয়ে! সাবাশ!"

        # 2. Name / Identity responses
        if "আঞ্জুম" in clean or "anjum" in clean:
            return "হ্যাঁ! তুমি আমাদের সোনা আঞ্জুম! কত লক্ষ্মী মেয়ে তুমি! অনেক আদর তোমার জন্য!"

        # 3. High-praise for any spoken words
        generic_praises = [
            "বাহ্ আঞ্জুম সোনা! কী সুন্দর করে কথা বলেছো! খুব ভালো লেগেছে!",
            "সাবাশ! তুমি এত সুন্দর কথা বলতে পারো! আরেকবার বলো তো সোনা!",
            "ওয়াও! কত সুন্দর কথা! তুমি তো অনেক লক্ষ্মী মেয়ে!",
        ]
        return random.choice(generic_praises)
