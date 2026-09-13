"""Audio Proximity Filter for LUMI.

RMS energy-based near-field audio priority filter.
When multiple speakers are detected simultaneously (overlap),
only the loudest (closest) audio chunks are passed through to Gemini.
When a single speaker is talking, all audio passes through regardless of volume.

This is a signal-processing solution — NOT a system prompt hack.
It works at the audio pipeline level before data reaches the AI model.

Hardware: ReSpeaker 2-Mic Pi HAT on Raspberry Pi 5
"""

from __future__ import annotations

import collections
import math
import struct
import threading
from typing import Optional

from ..core.logger import get_logger

logger = get_logger("audio.proximity")


class ProximityAudioFilter:
    """RMS energy-based near-field audio priority filter.

    Behavior:
        - Single speaker (no overlap): ALL audio passes through, even quiet/distant.
        - Multiple speakers (overlap detected): Only chunks whose RMS energy
          exceeds a dynamic threshold (based on recent peak energy) are passed.
          This ensures the closest/loudest speaker is prioritized.

    The filter maintains a sliding window of recent RMS values to compute
    an adaptive noise floor and peak tracker.

    Args:
        silence_rms: RMS below this is considered silence (no speech at all).
        overlap_ratio: During overlap, chunks must be at least this fraction
                       of the recent peak RMS to pass through. Higher = stricter.
                       0.6 means chunk must be at least 60% of the loudest recent audio.
        window_size: Number of recent RMS values to track for adaptive thresholds.
        cooldown_chunks: After overlap ends, keep filtering for this many extra
                         chunks to avoid letting trailing quiet audio through.
    """

    def __init__(
        self,
        silence_rms: float = 120.0,
        overlap_ratio: float = 0.55,
        window_size: int = 30,
        cooldown_chunks: int = 10,
    ) -> None:
        self.silence_rms = silence_rms
        self.overlap_ratio = overlap_ratio
        self.window_size = window_size
        self.cooldown_chunks = cooldown_chunks

        self._rms_history: collections.deque = collections.deque(maxlen=window_size)
        self._peak_rms: float = 0.0
        self._noise_floor: float = 0.0
        self._overlap_active: bool = False
        self._cooldown_remaining: int = 0
        self._lock = threading.Lock()

        # Statistics for debugging
        self._total_chunks: int = 0
        self._passed_chunks: int = 0
        self._dropped_chunks: int = 0

    def compute_rms(self, pcm_chunk: bytes) -> float:
        """Compute RMS energy of a 16-bit signed LE PCM chunk.

        Args:
            pcm_chunk: Raw PCM bytes (16-bit signed little-endian mono).

        Returns:
            RMS energy value (0.0 = silence, ~32768.0 = max).
        """
        count = len(pcm_chunk) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", pcm_chunk[:count * 2])
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count)

    def should_pass(self, chunk: bytes, is_overlap: bool) -> bool:
        """Decide whether this audio chunk should be sent to the AI model.

        Args:
            chunk: Raw PCM audio bytes (16-bit signed LE mono).
            is_overlap: True if multiple speakers/faces are currently detected.

        Returns:
            True if this chunk should be forwarded to Gemini Live.
            False if it should be dropped (quiet distant speaker during overlap).
        """
        with self._lock:
            self._total_chunks += 1
            rms = self.compute_rms(chunk)
            self._rms_history.append(rms)

            # Update peak tracker (exponential decay)
            if rms > self._peak_rms:
                self._peak_rms = rms
            else:
                # Slow decay so peak tracks the loudest recent speaker
                self._peak_rms = self._peak_rms * 0.97 + rms * 0.03

            # Update noise floor only during quiet periods to avoid treating loud speech as noise
            if rms < self.silence_rms * 2.5:
                self._noise_floor = self._noise_floor * 0.9 + rms * 0.1

            # --- Decision Logic ---

            # Pure silence: always pass (Gemini needs silence for turn detection)
            if rms < self.silence_rms:
                self._passed_chunks += 1
                if is_overlap:
                    self._cooldown_remaining = self.cooldown_chunks
                return True

            # Track overlap state transitions
            if is_overlap:
                self._overlap_active = True
                self._cooldown_remaining = self.cooldown_chunks
            elif self._overlap_active:
                # Overlap just ended — continue filtering briefly
                self._cooldown_remaining -= 1
                if self._cooldown_remaining <= 0:
                    self._overlap_active = False

            # No overlap (single speaker): pass everything
            if not self._overlap_active:
                self._passed_chunks += 1
                return True

            # --- Overlap Mode: Only pass loud (close) audio ---
            # Dynamic threshold: overlap_ratio × peak_rms
            threshold = self._peak_rms * self.overlap_ratio

            # Also ensure it's above noise floor, but never higher than 85% of peak
            # so the primary (loudest) speaker is always guaranteed to pass
            effective_threshold = max(threshold, self._noise_floor * 1.2)
            effective_threshold = min(effective_threshold, self._peak_rms * 0.85)

            if rms >= effective_threshold:
                self._passed_chunks += 1
                return True
            else:
                self._dropped_chunks += 1
                if self._dropped_chunks % 50 == 1:
                    logger.debug(
                        f"Proximity filter dropped chunk: rms={rms:.0f}, "
                        f"threshold={effective_threshold:.0f}, "
                        f"peak={self._peak_rms:.0f}, "
                        f"dropped={self._dropped_chunks}/{self._total_chunks}"
                    )
                return False

    def get_stats(self) -> dict:
        """Return filter statistics for debugging."""
        with self._lock:
            return {
                "total_chunks": self._total_chunks,
                "passed_chunks": self._passed_chunks,
                "dropped_chunks": self._dropped_chunks,
                "drop_rate": (
                    f"{self._dropped_chunks / self._total_chunks * 100:.1f}%"
                    if self._total_chunks > 0
                    else "0%"
                ),
                "current_peak_rms": round(self._peak_rms, 1),
                "noise_floor": round(self._noise_floor, 1),
                "overlap_active": self._overlap_active,
            }

    def reset(self) -> None:
        """Reset all internal state."""
        with self._lock:
            self._rms_history.clear()
            self._peak_rms = 0.0
            self._noise_floor = 0.0
            self._overlap_active = False
            self._cooldown_remaining = 0
            self._total_chunks = 0
            self._passed_chunks = 0
            self._dropped_chunks = 0
