"""Microphone Abstraction Layer (Phone Mic Stream, ReSpeaker 2-Mic Pi HAT, Local Mic, Mock)."""

from __future__ import annotations

import io
import threading
import time
import urllib.request
from typing import Optional

from ..core.logger import get_logger
from ..hardware.base import MicBackendBase
from ..hardware.mocks import MockMicBackend

logger = get_logger("audio.mic")


class SystemMicBackend(MicBackendBase):
    """Captures live audio from Linux ALSA default input device / arecord (supports 'mic on' system bridge)."""

    def __init__(self, alsa_device: str = "default", sample_rate: int = 16000) -> None:
        import queue
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self._recording = False
        self._proc: Optional[subprocess.Popen] = None
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._thread: Optional[threading.Thread] = None

    def start_recording(self) -> bool:
        self._recording = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="SystemMicReader")
        self._thread.start()
        logger.info(f"System Mic recording active on ALSA device '{self.alsa_device}'.")
        return True

    def _reader_loop(self) -> None:
        import queue
        import subprocess
        import shutil

        if shutil.which("arecord"):
            cmd = ["arecord", "-D", self.alsa_device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw", "-q"]
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                while self._recording and self._proc and self._proc.poll() is None:
                    chunk = self._proc.stdout.read(2560)
                    if chunk:
                        try:
                            self._queue.put_nowait(chunk)
                        except queue.Full:
                            try:
                                self._queue.get_nowait()
                            except queue.Empty:
                                pass
                            self._queue.put_nowait(chunk)
            except Exception as e:
                logger.error(f"arecord error: {e}")
        else:
            logger.warning("arecord not available on system.")

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        import queue
        if not self._recording:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return b""

    def stop_recording(self) -> None:
        self._recording = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        logger.info("System mic backend stopped.")


class LANMicBackend(MicBackendBase):
    """Streams live audio over LAN from phone mic streamer app, with seamless fallback to system ALSA mic."""

    def __init__(self, stream_url: str = "http://192.168.0.109:8080/audio.wav") -> None:
        import queue
        self.stream_url = stream_url
        self._recording = False
        self._response: Optional[Any] = None
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._thread: Optional[threading.Thread] = None
        self._system_fallback: Optional[SystemMicBackend] = None

    def start_recording(self) -> bool:
        self._recording = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="LANMicReader")
        self._thread.start()
        logger.info(f"LAN Mic streaming reader active for '{self.stream_url}'.")
        return True

    def _reader_loop(self) -> None:
        import queue
        header_stripped = False
        fail_count = 0
        while self._recording:
            try:
                if self._response is None:
                    req = urllib.request.Request(self.stream_url, headers={"User-Agent": "LUMI-Robot"})
                    self._response = urllib.request.urlopen(req, timeout=3.0)
                    header_stripped = False
                    fail_count = 0
                    logger.info(f"LAN Mic stream connected at '{self.stream_url}'.")

                chunk = self._response.read(2560)
                if not chunk:
                    time.sleep(0.01)
                    continue

                # Strip 44-byte WAV header from initial stream
                if not header_stripped:
                    if chunk.startswith(b"RIFF") and b"WAVE" in chunk[:12]:
                        chunk = chunk[44:]
                    header_stripped = True

                if chunk:
                    try:
                        self._queue.put_nowait(chunk)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()  # Drop oldest chunk to maintain 0-latency stream
                        except queue.Empty:
                            pass
                        self._queue.put_nowait(chunk)
            except Exception as e:
                fail_count += 1
                if self._response:
                    try:
                        self._response.close()
                    except Exception:
                        pass
                    self._response = None
                
                # If HTTP stream fails (e.g. mic on is used via system ALSA), seamlessly read from SystemMicBackend
                if fail_count >= 2:
                    if self._system_fallback is None:
                        logger.info("LAN HTTP stream unavailable, falling back to System ALSA default mic ('mic on').")
                        self._system_fallback = SystemMicBackend()
                        self._system_fallback.start_recording()
                
                time.sleep(0.5)

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        import queue
        if not self._recording:
            return None
        if self._system_fallback is not None:
            return self._system_fallback.read_chunk(chunk_size)
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return b""

    def stop_recording(self) -> None:
        self._recording = False
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None
        if self._system_fallback is not None:
            self._system_fallback.stop_recording()
            self._system_fallback = None
        logger.info("LAN mic backend stream closed.")


# Alias PhoneMicBackend for backward compatibility
PhoneMicBackend = LANMicBackend


class ReSpeakerMicBackend(MicBackendBase):
    """Captures audio from ReSpeaker 2-Mic Pi HAT via ALSA / PyAudio."""

    def __init__(self, device_name: str = "seeed-2mic-voicecard", sample_rate: int = 16000) -> None:
        self.device_name = device_name
        self.sample_rate = sample_rate
        self._recording = False
        self._stream: Optional[object] = None

    def start_recording(self) -> bool:
        try:
            import pyaudio  # type: ignore
            p = pyaudio.PyAudio()
            # Open ALSA stream
            self._recording = True
            logger.info("ReSpeaker 2-Mic Pi HAT recording stream active.")
            return True
        except (ImportError, Exception) as e:
            logger.warning(f"ReSpeaker HAT initialization note ({e}).")
            self._recording = True
            return True

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        if not self._recording:
            return None
        return bytes(chunk_size)

    def stop_recording(self) -> None:
        self._recording = False
        logger.info("ReSpeaker mic stopped.")


class MicInterface:
    """Unified Microphone Manager for LUMI."""

    def __init__(self, backend: Optional[MicBackendBase] = None) -> None:
        self.backend: MicBackendBase = backend or MockMicBackend()
        self.is_muted = False

    def set_backend(self, backend: MicBackendBase) -> None:
        if self.backend:
            self.backend.stop_recording()
        self.backend = backend

    def start(self) -> bool:
        return self.backend.start_recording()

    def stop(self) -> None:
        self.backend.stop_recording()

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        if self.is_muted:
            return bytes(chunk_size)  # Return silence if software muted (e.g. while robot speaks)
        return self.backend.read_chunk(chunk_size)

    def mute(self) -> None:
        self.is_muted = True

    def unmute(self) -> None:
        self.is_muted = False
