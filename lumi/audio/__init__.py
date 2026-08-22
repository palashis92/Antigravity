"""Audio input and output subsystem for LUMI."""

from .mic import MicInterface
from .speaker import SpeakerInterface

__all__ = [
    "MicInterface",
    "SpeakerInterface",
]
