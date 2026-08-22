"""Abstract Hardware Abstraction Layer (HAL) base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class ServoDriverBase(ABC):
    """Abstract interface for multi-channel servo controllers."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the servo driver bus/controller."""
        pass

    @abstractmethod
    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Set raw pulse width in microseconds on a specific channel."""
        pass

    @abstractmethod
    def set_angle(self, channel: int, angle_deg: float) -> None:
        """Set normalized physical angle in degrees on a channel."""
        pass

    @abstractmethod
    def release_channel(self, channel: int) -> None:
        """De-energize a servo channel to prevent overheating when idle."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Safely de-energize all channels and close driver."""
        pass


class CameraBackendBase(ABC):
    """Abstract interface for video capture sources."""

    @abstractmethod
    def start(self) -> bool:
        """Start the camera stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera stream."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if camera is online and providing valid frames."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[Any]:
        """Capture and return the latest RGB image frame (e.g. numpy array / PIL Image)."""
        pass


class MicBackendBase(ABC):
    """Abstract interface for audio capture sources."""

    @abstractmethod
    def start_recording(self) -> bool:
        """Begin audio capture."""
        pass

    @abstractmethod
    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        """Read a PCM byte buffer chunk."""
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop capturing audio."""
        pass


class SpeakerBackendBase(ABC):
    """Abstract interface for audio playback (MAX98357A I2S or local audio)."""

    @abstractmethod
    def play_audio_file(self, file_path: str, block: bool = True) -> bool:
        """Play a WAV or MP3 audio file."""
        pass

    @abstractmethod
    def play_audio_stream(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        """Play raw PCM audio stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Interrupt active playback immediately."""
        pass

    @abstractmethod
    def set_volume(self, volume_percent: int) -> None:
        """Set output volume (0 - 100)."""
        pass


class DisplayBackendBase(ABC):
    """Abstract interface for eye displays (GC9A01 dual SPI or virtual UI)."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize SPI bus and display controllers."""
        pass

    @abstractmethod
    def draw_eyes(self, left_image: Any, right_image: Any) -> None:
        """Send rendered frame buffers to left and right eye displays."""
        pass

    @abstractmethod
    def set_brightness(self, level_percent: int) -> None:
        """Adjust display backlight brightness."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear displays to black."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Turn off displays and release SPI pins."""
        pass
