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
    """Captures live audio from Linux ALSA device using stereo (2-channel) capture.
    
    Uses both microphones on ReSpeaker 2-Mic Pi HAT for spatial audio:
    - Stereo capture enables DOA estimation and beamforming
    - Enhanced mono output has better SNR than naive downmix
    """

    def __init__(self, alsa_device: str = "default", sample_rate: int = 16000) -> None:
        self.alsa_device = alsa_device
        self.sample_rate = sample_rate
        self._recording = False
        self._proc: Optional[subprocess.Popen] = None
        self._queue: queue.Queue = queue.Queue(maxsize=50)       # Stereo chunks
        self._mono_queue: queue.Queue = queue.Queue(maxsize=50)  # Processed mono
        self._thread: Optional[threading.Thread] = None
        self._spatial = None
        self._init_spatial()

    def _init_spatial(self) -> None:
        """Initialize spatial audio processor (graceful fallback)."""
        try:
            from .spatial import SpatialAudioProcessor
            self._spatial = SpatialAudioProcessor()
            logger.info("Spatial audio processor ready (stereo beamforming enabled)")
        except Exception as e:
            logger.warning(f"Spatial processor unavailable ({e}), using mono downmix")
            self._spatial = None

    def start_recording(self) -> bool:
        self._recording = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="SystemMicReader")
        self._thread.start()
        logger.info(f"System Mic recording active on ALSA device '{self.alsa_device}'.")
        return True

    def _reader_loop(self) -> None:
        import time
        if shutil.which("arecord"):
            # Capture stereo (2 channels) to use both mics on ReSpeaker 2-Mic HAT
            channels = "2"
            device = self.alsa_device if self.alsa_device != "default" else "default"
            
            while self._recording:
                cmd = [
                    "arecord", "-D", device,
                    "-f", "S16_LE", "-r", str(self.sample_rate),
                    "-c", channels, "-t", "raw", "-q"
                ]
                try:
                    self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
                    time.sleep(0.5)
                    
                    if self._proc is not None and self._proc.poll() is not None:
                        err = self._proc.stderr.read().decode('utf-8', errors='ignore') if self._proc.stderr else ""
                        logger.error(f"arecord failed with stereo: {err.strip()}")
                        # Fallback to mono capture
                        logger.info("Falling back to mono capture...")
                        self._start_mono_fallback()
                        return

                    logger.info(f"arecord running in STEREO mode (2 channels, {self.sample_rate}Hz)")
                    
                    proc = self._proc
                    while self._recording and proc and proc.poll() is None:
                        if proc.stdout:
                            # Read 8192 bytes = 2048 stereo samples (4 bytes per sample)
                            chunk = proc.stdout.read(8192)
                            if chunk:
                                # Store raw stereo
                                self._enqueue(self._queue, chunk)
                                
                                # Process stereo → enhanced mono
                                mono_chunk = self._stereo_to_mono(chunk)
                                self._enqueue(self._mono_queue, mono_chunk)
                except Exception as e:
                    logger.error(f"arecord error: {e}")
                    break
        else:
            logger.warning("arecord not available on system.")

    def _start_mono_fallback(self) -> None:
        """Fallback to mono capture if stereo fails."""
        import time
        device = "plug:default" if self.alsa_device == "default" else self.alsa_device
        cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw", "-q"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            time.sleep(0.5)
            logger.info("arecord running in MONO fallback mode")
            proc = self._proc
            while self._recording and proc and proc.poll() is None:
                if proc.stdout:
                    chunk = proc.stdout.read(4096)
                    if chunk:
                        self._enqueue(self._mono_queue, chunk)
                        self._enqueue(self._queue, chunk)  # Also put in stereo queue (it's just mono)
        except Exception as e:
            logger.error(f"Mono fallback arecord error: {e}")

    def _stereo_to_mono(self, stereo_chunk: bytes) -> bytes:
        """Convert stereo chunk to enhanced mono via spatial processing or simple mix."""
        if self._spatial:
            try:
                mono_bytes, doa = self._spatial.process_stereo_chunk(stereo_chunk)
                return mono_bytes
            except Exception:
                pass
        
        # Simple fallback: average L and R
        import struct
        n_samples = len(stereo_chunk) // 4  # 4 bytes per stereo sample
        mono = bytearray(n_samples * 2)
        for i in range(n_samples):
            offset = i * 4
            left = struct.unpack_from('<h', stereo_chunk, offset)[0]
            right = struct.unpack_from('<h', stereo_chunk, offset + 2)[0]
            mixed = (left + right) // 2
            struct.pack_into('<h', mono, i * 2, mixed)
        return bytes(mono)

    @staticmethod
    def _enqueue(q: queue.Queue, data: bytes) -> None:
        """Put data in queue, dropping oldest if full."""
        try:
            q.put_nowait(data)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(data)

    def read_chunk(self, chunk_size: int = 1024) -> Optional[bytes]:
        """Read processed mono audio chunk (beamformed if spatial available)."""
        if not self._recording:
            return None
        try:
            return self._mono_queue.get_nowait()
        except queue.Empty:
            return b""

    def read_stereo_chunk(self) -> Optional[bytes]:
        """Read raw stereo audio chunk for spatial analysis."""
        if not self._recording:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return b""

    @property
    def spatial_processor(self):
        """Access the spatial processor for DOA info and beamform steering."""
        return self._spatial

    def stop_recording(self) -> None:
        self._recording = False
        if self._proc is not None:
            try:
                if self._proc.stdout: self._proc.stdout.close()
                if self._proc.stderr: self._proc.stderr.close()
                self._proc.terminate()
                self._proc.wait(timeout=0.5)
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
