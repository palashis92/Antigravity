"""Mock implementations of hardware drivers for development and testing."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..core.logger import get_logger
from .base import (
    CameraBackendBase,
    DisplayBackendBase,
    MicBackendBase,
    ServoDriverBase,
    SpeakerBackendBase,
)

logger = get_logger("hardware.mock")


class MockServoDriver(ServoDriverBase):
    """Mock servo driver that tracks virtual positions and logs actuation."""

    def __init__(self) -> None:
        self.is_initialized = False
        self.channel_angles: Dict[int, float] = {}
        self.channel_pulses: Dict[int, int] = {}

    def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("[MOCK] MockServoDriver initialized.")
        return True

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        self.channel_pulses[channel] = pulse_us
        logger.debug(f"[MOCK] Servo channel {channel} pulse set to {pulse_us} us")

    def set_angle(self, channel: int, angle_deg: float) -> None:
        self.channel_angles[channel] = angle_deg
        logger.debug(f"[MOCK] Servo channel {channel} angle set to {angle_deg:.1f}°")

    def release_channel(self, channel: int) -> None:
        if channel in self.channel_angles:
            del self.channel_angles[channel]
        logger.debug(f"[MOCK] Servo channel {channel} released (de-energized).")

    def shutdown(self) -> None:
        self.channel_angles.clear()
        self.channel_pulses.clear()
        self.is_initialized = False
        logger.info("[MOCK] MockServoDriver shut down.")


class MockCameraBackend(CameraBackendBase):
    """Mock camera generating synthetic frames."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._running = False
        self._frame_count = 0

    def start(self) -> bool:
        self._running = True
        logger.info("[MOCK] MockCameraBackend started.")
        return True

    def stop(self) -> None:
        self._running = False
        logger.info("[MOCK] MockCameraBackend stopped.")

    def is_available(self) -> bool:
        return self._running

    def get_frame(self) -> Optional[Dict[str, Any]]:
        if not self._running:
            return None
        self._frame_count += 1
        # Returns a mock frame representation
        return {
            "width": self.width,
            "height": self.height,
            "frame_id": self._frame_count,
            "timestamp": time.time(),
            "channels": 3,
        }


class MockMicBackend(MicBackendBase):
    """Mock microphone returning empty or simulated audio buffers."""

    def __init__(self) -> None:
        self._recording = False

    def start_recording(self) -> bool:
        self._recording = True
        logger.info("[MOCK] MockMicBackend started recording.")
        return True

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        if not self._recording:
            return None
        # Return silent PCM buffer of requested size
        return bytes(chunk_size)

    def stop_recording(self) -> None:
        self._recording = False
        logger.info("[MOCK] MockMicBackend stopped recording.")


class MockSpeakerBackend(SpeakerBackendBase):
    """Mock speaker that logs playback and simulates audio duration."""

    def __init__(self, volume: int = 80) -> None:
        self._volume = volume
        self._playing = False

    def play_audio_file(self, file_path: str, block: bool = True) -> bool:
        logger.info(f"[MOCK] Playing audio file: '{file_path}' (vol: {self._volume}%)")
        if block:
            time.sleep(0.05)
        return True

    def play_audio_stream(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        duration = len(audio_bytes) / (sample_rate * 2) if sample_rate else 0.1
        logger.info(f"[MOCK] Playing audio stream ({len(audio_bytes)} bytes, est: {duration:.2f}s)")
        return True

    def stop(self) -> None:
        self._playing = False
        logger.info("[MOCK] Speaker playback stopped.")

    def set_volume(self, volume_percent: int) -> None:
        self._volume = max(0, min(100, volume_percent))
        logger.info(f"[MOCK] Speaker volume set to {self._volume}%")


class MockDisplayBackend(DisplayBackendBase):
    """Mock display backend tracking rendered eye frames."""

    def __init__(self, width: int = 240, height: int = 240) -> None:
        self.width = width
        self.height = height
        self.is_initialized = False
        self.brightness = 100
        self.last_left_frame: Any = None
        self.last_right_frame: Any = None

    def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("[MOCK] MockDisplayBackend (Dual GC9A01 240x240) initialized.")
        return True

    def draw_eyes(self, left_image: Any, right_image: Any) -> None:
        self.last_left_frame = left_image
        self.last_right_frame = right_image
        logger.debug("[MOCK] Rendered frames sent to Dual GC9A01 displays.")

    def set_brightness(self, level_percent: int) -> None:
        self.brightness = max(0, min(100, level_percent))
        logger.info(f"[MOCK] Display brightness set to {self.brightness}%")

    def clear(self) -> None:
        self.last_left_frame = None
        self.last_right_frame = None
        logger.debug("[MOCK] Displays cleared to black.")

    def shutdown(self) -> None:
        self.is_initialized = False
        logger.info("[MOCK] MockDisplayBackend shut down.")
