"""Bangla Speech-to-Text (STT) — Converts spoken Bangla audio into text.

Primary provider: OpenAI Whisper API (via OpenAI SDK)
Uses the same OPENAI_API_KEY that can be set in .env

Pipeline:
    Microphone audio (WAV/PCM) → OpenAI Whisper API → Bengali text → LumiBrain
"""

from __future__ import annotations

import io
import os
import tempfile
import wave
from typing import Optional

from ..core.logger import get_logger

logger = get_logger("speech.stt")

# Import OpenAI SDK (already installed for Inworld.ai conversation)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI SDK not installed. Run: pip install openai")


class BanglaSTT:
    """Transcribes spoken Bangla audio into UTF-8 Bengali text using OpenAI Whisper."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None

        if OpenAI and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Whisper STT initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        elif not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not set. STT will be unavailable. "
                "Set it in .env or environment variables."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_audio_file(self, file_path: str, language: str = "bn") -> str:
        """Transcribe a WAV/MP3/M4A audio file to Bangla text.

        Args:
            file_path: Path to the audio file.
            language: ISO 639-1 language code. Default "bn" for Bangla.

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file '{file_path}' does not exist.")
            return ""

        if not self._client:
            logger.warning("STT unavailable — no OpenAI client initialized.")
            return ""

        try:
            with open(file_path, "rb") as audio_file:
                # Omit language param so Whisper auto-detects Bangla without 400 unsupported_language error
                transcription = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            result = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
            logger.info(f"Whisper transcription: '{result}'")
            return result

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""

    def transcribe_pcm_bytes(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        language: str = "bn",
    ) -> str:
        """Transcribe raw PCM audio bytes (from microphone buffer) to Bangla text.

        Wraps raw PCM data in a WAV header, saves to a temp file,
        then sends to OpenAI Whisper API.

        Args:
            pcm_bytes: Raw PCM audio byte buffer (signed 16-bit little-endian).
            sample_rate: Audio sample rate in Hz (default 16000).
            channels: Number of audio channels (default 1 = mono).
            sample_width: Bytes per sample (default 2 = 16-bit).
            language: ISO 639-1 language code (default "bn" for Bangla).

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not pcm_bytes or len(pcm_bytes) < 1600:
            # Too short to be meaningful speech (< 0.05s at 16kHz)
            return ""

        if not self._client:
            logger.warning("STT unavailable — no OpenAI client initialized.")
            return ""

        # Wrap raw PCM bytes in a proper WAV container
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_bytes)

            # Now transcribe the temporary WAV file
            return self.transcribe_audio_file(tmp_path, language=language)

        except Exception as e:
            logger.error(f"Error wrapping PCM bytes for Whisper: {e}")
            return ""
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
