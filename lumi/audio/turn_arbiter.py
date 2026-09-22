"""Centralized Turn-Taking Arbiter and Audio Gating for LUMI.

Consolidates all silence commands, speaker AEC ducking, ambient gating,
and dialogue turn-window arbitration into a single unified authority.
Prevents uncoordinated gating conflicts between audio, perception, and cloud streams.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..core.logger import get_logger
from ..core.telemetry import get_telemetry

logger = get_logger("audio.turn_arbiter")


class AudioTurnArbiter:
    """Unified turn-taking and audio streaming arbiter."""

    def __init__(
        self,
        echo_tail_s: float = 0.35,
        turn_window_s: float = 25.0,
        energy_threshold: float = 100.0,
    ) -> None:
        self.echo_tail_s = echo_tail_s
        self.turn_window_s = turn_window_s
        self.energy_threshold = energy_threshold

        self._lock = threading.Lock()
        self._silent_until: float = 0.0
        self._speaker_active_until: float = 0.0
        self._dialogue_active_until: float = 0.0
        self._is_speaker_playing: bool = False
        self._is_presenting: bool = False
        self._last_barge_in_time: float = 0.0

    # ------------------------------------------------------------------
    # Presentation Mode Management
    # ------------------------------------------------------------------
    def set_presenting(self, presenting: bool) -> None:
        with self._lock:
            self._is_presenting = presenting
        logger.info(f"Presentation mode arbiter state: {presenting}")

    def is_presenting(self) -> bool:
        with self._lock:
            return self._is_presenting

    # ------------------------------------------------------------------
    # Silence Mode Management
    # ------------------------------------------------------------------
    def set_silent_until(self, timestamp: float) -> None:
        with self._lock:
            self._silent_until = timestamp
        dur = max(0.0, timestamp - time.time())
        logger.info(f"Silent mode set for {dur:.1f}s.")

    def is_silent(self) -> bool:
        with self._lock:
            return time.time() < self._silent_until

    def cancel_silence(self) -> None:
        with self._lock:
            self._silent_until = 0.0
        logger.info("Silent mode cancelled.")

    # ------------------------------------------------------------------
    # Speaker Playback & Echo Ducking Guard
    # ------------------------------------------------------------------
    def notify_speaker_started(self, duration_s: float) -> None:
        """Called when robot starts vocalizing to duck mic and prevent echo loops."""
        with self._lock:
            now = time.time()
            self._is_speaker_playing = True
            self._speaker_active_until = max(self._speaker_active_until, now + duration_s)
            # Keep dialogue open during robot speech
            self._dialogue_active_until = max(
                self._dialogue_active_until, now + duration_s + self.turn_window_s
            )

    def notify_speaker_stopped(self) -> None:
        """Called when robot speaker completes playback or is interrupted."""
        with self._lock:
            self._is_speaker_playing = False
            # Allow echo tail to decay naturally
            self._speaker_active_until = min(self._speaker_active_until, time.time())

    def is_speaker_active(self) -> bool:
        """True if the robot speaker is playing or room reverberation is still decaying."""
        with self._lock:
            now = time.time()
            if now >= (self._speaker_active_until + self.echo_tail_s):
                self._is_speaker_playing = False
            return self._is_speaker_playing or (now < (self._speaker_active_until + self.echo_tail_s))

    # ------------------------------------------------------------------
    # Dialogue & Addressee Turn Windows
    # ------------------------------------------------------------------
    def wake_up(self, duration_s: Optional[float] = None) -> None:
        """Open the active conversation window (e.g. on greeting or wake word)."""
        dur = duration_s if duration_s is not None else self.turn_window_s
        with self._lock:
            self._dialogue_active_until = max(self._dialogue_active_until, time.time() + dur)
        logger.debug(f"Dialogue turn window opened for {dur:.1f}s.")

    def is_in_dialogue(self) -> bool:
        """True if the robot is in an active turn-taking conversation."""
        with self._lock:
            return time.time() < self._dialogue_active_until

    def reset_dialogue(self) -> None:
        """Return to standby mode (stop streaming ambient sound)."""
        with self._lock:
            self._dialogue_active_until = 0.0

    # ------------------------------------------------------------------
    # Atomic Decision: Should Chunk Be Streamed to Cloud?
    # ------------------------------------------------------------------
    def should_stream_mic(
        self,
        energy: float,
        is_overlap: bool = False,
        allow_wake_detection: bool = True,
    ) -> bool:
        """Central arbitration function deciding if an audio chunk should be streamed.

        Prevents:
        1. Streaming during user-commanded silence.
        2. Robot self-talk from room echo during vocalization.
        3. Streaming background ambient room noise / fan hum when no conversation is active.
        """
        now = time.time()
        with self._lock:
            # 1. Check silence mode
            if now < self._silent_until:
                return False

            # 2. Check presentation mode (suppress mic streaming to cloud during monologue)
            if self._is_presenting:
                return False

            # 3. Check speaker echo ducking window
            if self._is_speaker_playing or (now < (self._speaker_active_until + self.echo_tail_s)):
                return False

            in_dialogue = now < self._dialogue_active_until

            # 3. If in active dialogue, stream ALL chunks (including silence
            #    that Gemini needs for server-side end-of-turn detection)
            if in_dialogue:
                return True

            # 4. Not in dialogue — apply energy floor to reject ambient noise
            if energy < self.energy_threshold:
                return False

            # 5. High energy speech can auto-wake dialogue window
            if not allow_wake_detection:
                return False
            if energy >= (self.energy_threshold * 1.5):
                self._dialogue_active_until = now + self.turn_window_s
                get_telemetry().record_event("dialogue_wake_on_speech", context="turn_arbiter")
                return True
            return False

    def record_barge_in(self, latency_ms: float) -> None:
        """Record telemetry for user interruption latency."""
        self._last_barge_in_time = time.time()
        get_telemetry().record_latency("TEL-01", latency_ms, context="barge_in")
        get_telemetry().record_event("TEL-02", context="barge_in_success")
