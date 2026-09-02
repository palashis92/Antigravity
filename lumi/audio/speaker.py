"""Speaker Abstraction Layer (MAX98357A I2S Amplifier, Local Audio, Mock)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

from ..core.logger import get_logger
from ..hardware.base import SpeakerBackendBase
from ..hardware.mocks import MockSpeakerBackend

logger = get_logger("audio.speaker")


class I2SSpeakerBackend(SpeakerBackendBase):
    """Plays audio via MAX98357A I2S Mono DAC/Amp on Raspberry Pi 5 with auto format conversion."""

    def __init__(self, alsa_device: str = "default", volume: int = 85) -> None:
        import queue
        self.alsa_device = alsa_device
        self.volume = volume
        self._current_process: Optional[subprocess.Popen] = None
        self._stream_queue: queue.Queue = queue.Queue()
        self._stream_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_worker_loop, daemon=True, name="I2SStreamWorker"
        )
        self._stream_thread.start()
        logger.info(f"Speaker initialized on ALSA device {self.alsa_device}")

    def _stream_worker_loop(self) -> None:
        """Background worker thread feeding streaming audio chunks to a persistent aplay process."""
        import queue
        proc = None
        current_sample_rate = 24000
        channels = "1"
        
        while self._stream_running:
            try:
                try:
                    audio_bytes, sample_rate = self._stream_queue.get(timeout=2.0)
                except queue.Empty:
                    if proc is not None:
                        try:
                            if proc.stdin: proc.stdin.close()
                            proc.wait(timeout=0.1)
                        except Exception: pass
                        proc = None
                    continue

                if proc is None or proc.poll() is not None or current_sample_rate != sample_rate:
                    if proc is not None:
                        if proc.poll() is not None and proc.stderr:
                            err = proc.stderr.read().decode('utf-8', errors='ignore')
                            logger.error(f"aplay failed: {err.strip()}")
                        try:
                            if proc.stdin: proc.stdin.close()
                            proc.terminate()
                        except Exception: pass
                        
                    current_sample_rate = sample_rate
                    device = "plug:default" if self.alsa_device == "default" else self.alsa_device
                    if shutil.which("aplay"):
                        proc = subprocess.Popen(
                            ["aplay", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", channels],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=False
                        )

                if proc and proc.stdin:
                    proc.stdin.write(audio_bytes)
                    proc.stdin.flush()
            except Exception as e:
                logger.warning(f"Audio stream worker error: {e}")
                proc = None

    def _detect_alsa_device(self) -> None:
        """Find the MAX98357A card index automatically if available."""
        try:
            res = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "MAX98357A" in line or "max98357a" in line or "i2s" in line.lower():
                    # Extract card number, e.g. "card 1: MAX98357A"
                    parts = line.split(":")
                    if parts and "card" in parts[0]:
                        card_num = parts[0].replace("card", "").strip()
                        self.alsa_device = f"plughw:{card_num},0"
                        logger.info(f"Auto-detected MAX98357A at ALSA device '{self.alsa_device}'.")
                        return
        except Exception:
            pass

    def _convert_to_clean_wav(self, input_path: str) -> str:
        """Convert MP3/compressed audio to 16-bit 44.1kHz Stereo PCM WAV for clean I2S DAC output."""
        output_wav = tempfile.mktemp(suffix="_clean.wav")

        # 1. Try sox (installed by default)
        if shutil.which("sox"):
            try:
                subprocess.run(
                    ["sox", input_path, "-r", "44100", "-c", "2", "-b", "16", output_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                return output_wav
            except Exception:
                pass

        # 2. Try ffmpeg
        if shutil.which("ffmpeg"):
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "2", "-f", "wav", output_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                return output_wav
            except Exception:
                pass

        return input_path

    def play_audio_file(self, file_path: str, block: bool = True) -> bool:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: '{file_path}'")
            return False

        # Convert to clean PCM WAV so MAX98357A doesn't emit white noise
        clean_path = self._convert_to_clean_wav(file_path)
        is_temp = clean_path != file_path

        try:
            # Try mpg123 if installed
            if file_path.endswith(".mp3") and shutil.which("mpg123"):
                cmd = ["mpg123", "-q", "-a", self.alsa_device, file_path]
            # Try aplay on clean PCM WAV
            elif shutil.which("aplay"):
                cmd = ["aplay", "-D", self.alsa_device, clean_path]
            # Fallback to play (sox)
            elif shutil.which("play"):
                cmd = ["play", "-q", clean_path]
            else:
                logger.warning("No audio player found (aplay/mpg123/play).")
                return False

            if block:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if is_temp and os.path.exists(clean_path):
                    try:
                        os.remove(clean_path)
                    except Exception:
                        pass
            else:
                self._current_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if is_temp:
                    # Clean up the temp file after a delay to allow async playback to finish
                    def delayed_cleanup(path, proc):
                        try:
                            proc.wait(timeout=30.0)
                        except Exception:
                            pass
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                    threading.Thread(target=delayed_cleanup, args=(clean_path, self._current_process), daemon=True).start()
            return True

        except Exception as e:
            logger.warning(f"I2S playback error: {e}")
            if is_temp and os.path.exists(clean_path):
                try:
                    os.remove(clean_path)
                except Exception:
                    pass
            return False

    def play_audio_stream(self, audio_bytes: bytes, sample_rate: int = 24000) -> bool:
        """Stream raw 16-bit PCM audio directly to MAX98357A I2S DAC (Non-blocking)."""
        self._stream_queue.put((audio_bytes, sample_rate))
        return True

    def stop(self) -> None:
        if self._current_process is not None:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None

    def set_volume(self, volume_percent: int) -> None:
        self.volume = max(0, min(100, volume_percent))
        logger.info(f"I2S Speaker volume: {self.volume}%")


class SpeakerInterface:
    """Unified Speaker Manager for LUMI."""

    def __init__(self, backend: Optional[SpeakerBackendBase] = None) -> None:
        self.backend: SpeakerBackendBase = backend or MockSpeakerBackend()

    def set_backend(self, backend: SpeakerBackendBase) -> None:
        self.backend = backend

    def play_file(self, file_path: str, block: bool = True) -> bool:
        return self.backend.play_audio_file(file_path, block=block)

    def play_stream(self, audio_bytes: bytes, sample_rate: int = 24000) -> bool:
        return self.backend.play_audio_stream(audio_bytes, sample_rate=sample_rate)

    def stop(self) -> None:
        self.backend.stop()

    def set_volume(self, volume_percent: int) -> None:
        self.backend.set_volume(volume_percent)
