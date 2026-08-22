"""LUMI Hardware Abstraction Layer (HAL)."""

from .base import (
    CameraBackendBase,
    DisplayBackendBase,
    MicBackendBase,
    ServoDriverBase,
    SpeakerBackendBase,
)
from .mocks import (
    MockCameraBackend,
    MockDisplayBackend,
    MockMicBackend,
    MockServoDriver,
    MockSpeakerBackend,
)
from .hardware_manager import HardwareManager

__all__ = [
    "CameraBackendBase",
    "DisplayBackendBase",
    "MicBackendBase",
    "MockCameraBackend",
    "MockDisplayBackend",
    "MockMicBackend",
    "MockServoDriver",
    "MockSpeakerBackend",
    "ServoDriverBase",
    "SpeakerBackendBase",
    "HardwareManager",
]
