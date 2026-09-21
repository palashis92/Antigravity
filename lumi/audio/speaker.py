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


def _mono_to_stereo(mono_bytes: bytes) -> bytes:
    """Convert 16-bit mono PCM bytes to 16-bit stereo PCM bytes (interleaved L=R)."""
    try:
        import numpy as np
        samples = np.frombuffer(mono_bytes, dtype=np.int16)
        stereo_samples = np.column_stack((samples, samples))
        return stereo_samples.tobytes()
    except Exception:
        # Pure Python fallback
        n_samples = len(mono_bytes) // 2
        stereo = bytearray(n_samples * 4)
        for i in range(n_samples):
            b0 = mono_bytes[i * 2]
            b1 = mono_bytes[i * 2 + 1]
            idx = i * 4
            stereo[idx] = b0
            stereo[idx + 1] = b1
            stereo[idx + 2] = b0
            stereo[idx + 3] = b1
        return bytes(stereo)


class I2SSpeakerBackend(SpeakerBackendBase):
    """Plays audio via MAX98357A I2S Mono DAC/Amp on Raspberry Pi 5 with auto format conversion."""

    def __init__(self, alsa_device: str = "default", volume: int = 85) -> None:
        import queue
        self.alsa_device = alsa_device
        self.volume = volume
        self._muted = False
        self._current_process: Optional[subprocess.Popen] = None
        self._stream_proc: Optional[subprocess.Popen] = None
        self._stream_queue: queue.Queue = queue.Queue()
        self._stream_running = True
        self._detect_alsa_device()
        self._stream_thread = threading.Thread(
            target=self._stream_worker_loop, daemon=True, name="I2SStreamWorker"
        )
        self._stream_thread.start()
        self._unmute_and_max_alsa_mixer()
        logger.info(f"Speaker initialized on ALSA device {self.alsa_device}")

    def _unmute_and_max_alsa_mixer(self) -> None:
        """Force ALSA mixer controls on Raspberry Pi (WM8960 / ReSpeaker 2-Mics) to 100% and unmuted."""
        import re
        card_id = None
        if "hw:" in self.alsa_device or "plughw:" in self.alsa_device:
            m = re.search(r'(?:plug)?hw:(\w+)', self.alsa_device)
            if m:
                card_id = m.group(1)

        candidate_cards = []
        if card_id:
            candidate_cards.append(card_id)
        candidate_cards.extend(["1", "0", "seeed-2mic-voicecard", "seeed2micvoicec", "wm8960-soundcard", "wm8960soundcard", "default"])
        cards = list(dict.fromkeys(candidate_cards))

        controls = ["Playback", "Speaker", "Headphone", "PCM", "Line", "HP DAC", "Line DAC", "Master"]
        mixer_switches = [
            "Left Output Mixer PCM",
            "Right Output Mixer PCM",
        ]

        for card in cards:
            for ctrl in controls:
                for val in ["100%", "127", "unmute"]:
                    try:
                        subprocess.run(
                            ["amixer", "-c", str(card), "sset", ctrl, val],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    except Exception:
                        pass
            for sw in mixer_switches:
                try:
                    subprocess.run(
                        ["amixer", "-c", str(card), "sset", sw, "on"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:
                    pass
                try:
                    subprocess.run(
                        ["amixer", "-c", str(card), "cset", f"name='{sw} Playback Switch'", "1"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:
                    pass
        logger.info(f"Unmuted and configured ALSA mixer controls for candidate card(s): {cards[:3]}")

    def _stream_worker_loop(self) -> None:
        """Background worker thread feeding streaming audio chunks to a persistent aplay process."""
        import queue
        proc = None
        current_sample_rate = 24000
        channels = "2"  # WM8960 requires stereo (2 channels)
        
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
                        self._stream_proc = None
                    continue

                stereo_bytes = _mono_to_stereo(audio_bytes)

                if proc is None or proc.poll() is not None or current_sample_rate != sample_rate:
                    if proc is not None:
                        if proc.poll() is not None and proc.stderr:
                            err = proc.stderr.read().decode('utf-8', errors='ignore')
                            if err.strip():
                                logger.error(f"aplay exited unexpectedly: {err.strip()}")
                        try:
                            if proc.stdin: proc.stdin.close()
                            proc.terminate()
                        except Exception: pass
                        proc = None
                        self._stream_proc = None
                        
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
                        self._stream_proc = proc

                if proc and proc.stdin:
                    proc.stdin.write(stereo_bytes)
                    proc.stdin.flush()
            except Exception as e:
                err_text = ""
                if proc is not None and proc.stderr:
                    try:
                        err_text = proc.stderr.read().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        pass
                if err_text:
                    logger.error(f"Audio stream worker aplay error: {err_text} (exception: {e})")
                elif isinstance(e, (BrokenPipeError, OSError, ValueError)):
                    logger.warning(f"Audio stream worker pipe closed: {e}")
                else:
                    logger.warning(f"Audio stream worker error: {e}")
                if proc is not None:
                    try:
                        if proc.stdin: proc.stdin.close()
                    except Exception: pass
                    try:
                        if proc.stderr: proc.stderr.close()
                    except Exception: pass
                    try:
                        proc.terminate()
                        proc.wait(timeout=0.2)
                    except Exception: pass
                proc = None
                self._stream_proc = None

        if proc is not None:
            try:
                if proc.stdin: proc.stdin.close()
            except Exception: pass
            try:
                if proc.stderr: proc.stderr.close()
            except Exception: pass
            try:
                proc.terminate()
                proc.wait(timeout=0.2)
            except Exception: pass
            proc = None
            self._stream_proc = None

    def _detect_alsa_device(self) -> None:
        """Find the ReSpeaker 2-Mics (WM8960) or MAX98357A card index automatically if available."""
        if self.alsa_device != "default":
            return
        try:
            res = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
            lines = res.stdout.splitlines()

            # Priority 1: Check for ReSpeaker 2-Mics Pi HAT (WM8960 / seeed)
            for line in lines:
                lower = line.lower()
                if "seeed" in lower or "wm8960" in lower or "voicecard" in lower:
                    parts = line.split(":")
                    if parts and "card" in parts[0].lower():
                        card_num = parts[0].lower().replace("card", "").strip()
                        self.alsa_device = f"plughw:{card_num},0"
                        logger.info(f"Auto-detected ReSpeaker 2-Mics (WM8960) at ALSA device '{self.alsa_device}'.")
                        return

            # Priority 2: Check for MAX98357A I2S DAC
            for line in lines:
                lower = line.lower()
                if "max98357a" in lower or "i2s" in lower:
                    parts = line.split(":")
                    if parts and "card" in parts[0].lower():
                        card_num = parts[0].lower().replace("card", "").strip()
                        self.alsa_device = f"plughw:{card_num},0"
                        logger.info(f"Auto-detected MAX98357A at ALSA device '{self.alsa_device}'.")
                        return
        except Exception as e:
            logger.debug(f"ALSA device auto-detection error: {e}")

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

    def set_muted(self, muted: bool) -> None:
        """Hardware/software mute control for the speaker."""
        self._muted = muted
        if muted:
            self.stop()
            # Drain any pending stream chunks
            import queue
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                except queue.Empty:
                    break
        logger.info(f"Speaker mute state changed to: {muted}")

    @property
    def is_muted(self) -> bool:
        return self._muted

    def play_audio_file(self, file_path: str, block: bool = True) -> bool:
        if self._muted:
            logger.debug("Speaker is muted, dropping play_audio_file.")
            return False

        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: '{file_path}'")
            return False

        # Convert to clean PCM WAV so MAX98357A doesn't emit white noise
        clean_path = self._convert_to_clean_wav(file_path)
        is_temp = clean_path != file_path

        try:
            is_mp3 = file_path.endswith(".mp3")
            if not is_mp3:
                try:
                    with open(file_path, "rb") as f:
                        hdr = f.read(4)
                        if hdr.startswith(b"ID3") or (len(hdr) >= 2 and hdr[0] == 0xFF and (hdr[1] & 0xE0) == 0xE0):
                            is_mp3 = True
                except Exception:
                    pass

            # Try mpg123 if installed
            if is_mp3 and shutil.which("mpg123"):
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
        if self._muted:
            return False
        self._stream_queue.put((audio_bytes, sample_rate))
        return True

    def stop_stream(self) -> None:
        """Immediately stop currently streaming audio and clear the stream queue without stopping the worker thread."""
        import queue
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break

        proc = self._stream_proc
        if proc is not None:
            try:
                if proc.stdin: proc.stdin.close()
            except Exception: pass
            try:
                if proc.stderr: proc.stderr.close()
            except Exception: pass
            try:
                proc.terminate()
                proc.wait(timeout=0.1)
            except Exception: pass
            self._stream_proc = None
        logger.debug("I2S audio stream stopped and queue drained.")

    def stop(self) -> None:
        """Interrupt active playback (both file and stream) immediately."""
        self.stop_stream()
        if self._current_process is not None:
            try:
                if self._current_process.stdin: self._current_process.stdin.close()
            except Exception: pass
            try:
                if self._current_process.stderr: self._current_process.stderr.close()
            except Exception: pass
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=0.3)
            except Exception:
                pass
            self._current_process = None

    def shutdown(self) -> None:
        """Completely shut down the speaker backend and worker threads."""
        self._stream_running = False
        self.stop()
        if hasattr(self, "_stream_thread") and self._stream_thread.is_alive():
            try:
                self._stream_thread.join(timeout=0.5)
            except Exception:
                pass
        logger.info("I2SSpeakerBackend shut down cleanly.")

    def set_volume(self, volume_percent: int) -> None:
        self.volume = max(0, min(100, volume_percent))
        logger.info(f"I2S Speaker volume: {self.volume}%")


class SpeakerInterface:
    """Unified Speaker Manager for LUMI."""

    def __init__(self, backend: Optional[SpeakerBackendBase] = None) -> None:
        self.backend: SpeakerBackendBase = backend or MockSpeakerBackend()
        self._muted = False

    def set_backend(self, backend: SpeakerBackendBase) -> None:
        self.backend = backend

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if hasattr(self.backend, "set_muted"):
            self.backend.set_muted(muted)

    @property
    def is_muted(self) -> bool:
        return getattr(self.backend, "is_muted", self._muted)

    def play_file(self, file_path: str, block: bool = True) -> bool:
        if self.is_muted:
            return False
        return self.backend.play_audio_file(file_path, block=block)

    def play_stream(self, audio_bytes: bytes, sample_rate: int = 24000) -> bool:
        if self.is_muted:
            return False
        return self.backend.play_audio_stream(audio_bytes, sample_rate=sample_rate)

    def stop(self) -> None:
        self.backend.stop()

    def stop_stream(self) -> None:
        if hasattr(self.backend, "stop_stream"):
            self.backend.stop_stream()
        else:
            self.backend.stop()

    def shutdown(self) -> None:
        if hasattr(self.backend, "shutdown"):
            self.backend.shutdown()
        else:
            self.backend.stop()

    def set_volume(self, volume_percent: int) -> None:
        self.backend.set_volume(volume_percent)
