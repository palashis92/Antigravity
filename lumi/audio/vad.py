"""Voice Activity Detection (VAD) for LUMI.

Uses webrtcvad to detect speech segments in real-time audio,
enabling turn-based speaker identification and overlap detection.

Designed for Raspberry Pi 5 with ReSpeaker 2-Mic Hat (16kHz mono PCM).
"""

from __future__ import annotations

import collections
import time
import threading
from enum import Enum, auto
from typing import Callable, List, Optional

from ..core.logger import get_logger

logger = get_logger("audio.vad")


class SpeechEvent(Enum):
    """Events emitted by the VAD."""
    SPEECH_START = auto()
    SPEECH_CONTINUE = auto()
    SPEECH_END = auto()
    SILENCE = auto()


class VoiceActivityDetector:
    """Lightweight VAD using webrtcvad for real-time speech detection.

    Collects speech segments and emits events when a person starts/stops
    talking. Works with 16kHz 16-bit mono PCM audio.

    Args:
        aggressiveness: 0-3, higher = more aggressive noise filtering.
                        2 is recommended for indoor environments.
        sample_rate: Audio sample rate in Hz (must be 8000, 16000, or 32000).
        frame_duration_ms: Frame size in ms (must be 10, 20, or 30).
        speech_pad_frames: Number of voiced frames to trigger speech start.
        silence_pad_frames: Number of unvoiced frames to trigger speech end.
        min_utterance_sec: Minimum utterance length to emit (seconds).
        max_utterance_sec: Maximum utterance length before forced cut (seconds).
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        speech_pad_frames: int = 8,
        silence_pad_frames: int = 20,
        min_utterance_sec: float = 1.5,
        max_utterance_sec: float = 8.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.speech_pad_frames = speech_pad_frames
        self.silence_pad_frames = silence_pad_frames
        self.min_utterance_bytes = int(min_utterance_sec * sample_rate * 2)  # 16-bit = 2 bytes/sample
        self.max_utterance_bytes = int(max_utterance_sec * sample_rate * 2)

        # Frame size in bytes: (sample_rate * frame_duration_ms / 1000) * 2 bytes
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 960 bytes for 30ms@16kHz

        self._vad = None
        self._vad_available = False
        self._init_vad(aggressiveness)

        # State tracking
        self._is_speaking = False
        self._ring_buffer = collections.deque(maxlen=silence_pad_frames)
        self._voiced_frames_count = 0
        self._utterance_buffer: bytearray = bytearray()
        self._residual_buffer: bytearray = bytearray()
        self._speech_start_time: float = 0.0

        # Overlap detection
        self._recent_energy_history: List[float] = []
        self._overlap_cooldown_until: float = 0.0

        # Callbacks
        self._on_utterance_complete: Optional[Callable[[bytes, float], None]] = None
        self._on_overlap_detected: Optional[Callable[[], None]] = None

        logger.info(f"VAD initialized (aggressiveness={aggressiveness}, frame={frame_duration_ms}ms)")

    def _init_vad(self, aggressiveness: int) -> None:
        """Initialize webrtcvad, gracefully handle if not installed."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            self._vad_available = True
        except ImportError:
            logger.warning("webrtcvad not installed. VAD disabled. Install with: pip install webrtcvad")
            self._vad_available = False

    def set_on_utterance_complete(self, callback: Callable[[bytes, float], None]) -> None:
        """Set callback for when a complete speech utterance is ready.

        Args:
            callback: function(audio_bytes, duration_sec)
        """
        self._on_utterance_complete = callback

    def set_on_overlap_detected(self, callback: Callable[[], None]) -> None:
        """Set callback for when overlapping speech is detected."""
        self._on_overlap_detected = callback

    def process_chunk(self, chunk: bytes) -> SpeechEvent:
        """Process an audio chunk through the VAD pipeline.

        Accepts arbitrary-length chunks and internally buffers/frames them
        to the required frame size for webrtcvad.

        Returns the current speech event state.
        """
        if not self._vad_available:
            return SpeechEvent.SILENCE

        # Add to residual buffer
        self._residual_buffer.extend(chunk)

        event = SpeechEvent.SILENCE

        # Process all complete frames
        while len(self._residual_buffer) >= self.frame_size:
            frame = bytes(self._residual_buffer[:self.frame_size])
            del self._residual_buffer[:self.frame_size]
            event = self._process_frame(frame)

        return event

    def _process_frame(self, frame: bytes) -> SpeechEvent:
        """Process a single VAD frame."""
        try:
            is_speech = self._vad.is_speech(frame, self.sample_rate)
        except Exception:
            return SpeechEvent.SILENCE

        if not self._is_speaking:
            # Looking for speech start
            if is_speech:
                self._voiced_frames_count += 1
                self._utterance_buffer.extend(frame)

                if self._voiced_frames_count >= self.speech_pad_frames:
                    # Confirmed speech start!
                    self._is_speaking = True
                    self._speech_start_time = time.time()
                    self._ring_buffer.clear()
                    logger.debug("🎤 Speech started")
                    return SpeechEvent.SPEECH_START
            else:
                self._voiced_frames_count = 0
                self._utterance_buffer.clear()
            return SpeechEvent.SILENCE

        else:
            # Currently in speech, looking for end
            self._utterance_buffer.extend(frame)
            self._ring_buffer.append(is_speech)

            # Track energy for overlap detection
            energy = self._compute_frame_energy(frame)
            self._recent_energy_history.append(energy)
            if len(self._recent_energy_history) > 30:
                self._recent_energy_history.pop(0)

            # Check for forced cut (too long)
            if len(self._utterance_buffer) >= self.max_utterance_bytes:
                self._emit_utterance()
                return SpeechEvent.SPEECH_END

            # Count unvoiced frames in ring buffer
            num_unvoiced = sum(1 for v in self._ring_buffer if not v)
            if num_unvoiced >= self.silence_pad_frames * 0.8:
                # Speech ended
                self._emit_utterance()
                return SpeechEvent.SPEECH_END

            return SpeechEvent.SPEECH_CONTINUE

    def _emit_utterance(self) -> None:
        """Finalize and emit a complete utterance."""
        self._is_speaking = False
        self._voiced_frames_count = 0
        audio_bytes = bytes(self._utterance_buffer)
        self._utterance_buffer.clear()
        self._ring_buffer.clear()
        self._recent_energy_history.clear()

        duration = time.time() - self._speech_start_time

        if len(audio_bytes) >= self.min_utterance_bytes:
            logger.debug(f"🎤 Speech ended. Duration: {duration:.1f}s, Size: {len(audio_bytes)} bytes")
            if self._on_utterance_complete:
                self._on_utterance_complete(audio_bytes, duration)
        else:
            logger.debug(f"🎤 Speech too short ({duration:.1f}s), discarding")

    def detect_overlap(self, num_visible_faces: int) -> bool:
        """Check if overlapping speech is likely happening.

        Heuristic: sustained high energy + multiple visible faces.
        """
        now = time.time()
        if now < self._overlap_cooldown_until:
            return False

        if num_visible_faces < 2:
            return False

        if not self._is_speaking or len(self._recent_energy_history) < 10:
            return False

        avg_energy = sum(self._recent_energy_history) / len(self._recent_energy_history)
        # High sustained energy with multiple faces suggests overlap
        if avg_energy > 2000:
            self._overlap_cooldown_until = now + 30.0  # 30s cooldown
            logger.info(f"🔊 Overlap detected! Energy={avg_energy:.0f}, Faces={num_visible_faces}")
            if self._on_overlap_detected:
                self._on_overlap_detected()
            return True

        return False

    @staticmethod
    def _compute_frame_energy(frame: bytes) -> float:
        """Compute RMS energy of a PCM frame."""
        import struct
        n = len(frame) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", frame)
        return (sum(s * s for s in samples) / n) ** 0.5

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
