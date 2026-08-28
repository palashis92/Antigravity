"""Microphone Abstraction Layer (Local ALSA Mic, Mock)."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from typing import Optional

from ..core.logger import get_logger
from ..hardware.base import MicBackendBase
from ..hardware.mocks import MockMicBackend

logger = get_logger("audio.mic")


class SystemMicBackend(MicBackendBase):
    """Captures live audio from Linux ALSA default input device / arecord."""

    def __init__(self, alsa_device: str = "default", sample_rate: int = 16000) -> None:
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
        import time
        if shutil.which("arecord"):
            # Try Mono first
            channels = "1"
            downmix = False
            
            while self._recording:
                cmd = ["arecord", "-D", self.alsa_device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", channels, "-t", "raw", "-q"]
                try:
                    self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
                    time.sleep(0.5) # Give it time to fail if channels are wrong
                    
                    if self._proc.poll() is not None:
                        err = self._proc.stderr.read().decode('utf-8', errors='ignore') if self._proc.stderr else ""
                        logger.warning(f"arecord -c {channels} failed: {err.strip()}")
                        if channels == "1":
                            logger.info("Retrying with -c 2 (Stereo) and software downmixing...")
                            channels = "2"
                            downmix = True
                            continue
                        else:
                            logger.error("arecord failed even with 2 channels. Mic disabled.")
                            break

                    logger.info(f"arecord running successfully with {channels} channel(s).")
                    
                    leftover = b""
                    
                    while self._recording and self._proc and self._proc.poll() is None:
                        if self._proc.stdout:
                            # Read appropriate chunk size based on channels
                            chunk = self._proc.stdout.read(4096)
                            if chunk:
                                if downmix:
                                    chunk = leftover + chunk
                                    # Ensure chunk is multiple of 4 bytes (2 channels * 2 bytes/sample)
                                    remainder = len(chunk) % 4
                                    if remainder != 0:
                                        leftover = chunk[-remainder:]
                                        chunk = chunk[:-remainder]
                                    else:
                                        leftover = b""
                                        
                                    try:
                                        if len(chunk) > 0:
                                            # Try audioop for proper stereo-to-mono mixing (handles both channels)
                                            try:
                                                import audioop
                                                chunk = audioop.tomono(chunk, 2, 1, 1)
                                            except ImportError:
                                                # Fallback to slicing if audioop is missing (Python 3.13+)
                                                # We take Left channel. If user needs Right, this might fail, but it's fast.
                                                stereo = memoryview(chunk).cast('h')
                                                chunk = stereo[0::2].tobytes()
                                    except Exception as e:
                                        logger.warning(f"Downmix error: {e}")
                                        pass
                                        
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
                    break
        else:
            logger.warning("arecord not available on system.")

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
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
