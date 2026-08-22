"""Speech processing subsystem for Bangla STT and TTS."""

from .stt import BanglaSTT
from .tts import BanglaTTS

__all__ = [
    "BanglaSTT",
    "BanglaTTS",
]
