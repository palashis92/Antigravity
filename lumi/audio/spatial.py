"""Spatial Audio Processing for LUMI's ReSpeaker 2-Mic Pi HAT.

Exploits the dual-microphone array to estimate Direction of Arrival (DOA),
perform beamforming, and correlate speakers with face positions.

Hardware: ReSpeaker 2-Mic Pi HAT v2.0 (WM8960 codec)
  - 2x MEMS microphones, ~58mm apart
  - Captured as stereo 16kHz 16-bit PCM

The pipeline:
  Stereo PCM → split L/R → DOA estimation → beamforming → enhanced mono

The enhanced mono sent to Gemini has significantly better SNR than
a naive stereo→mono downmix, because beamforming suppresses off-axis noise.
"""

from __future__ import annotations

import struct
import math
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from ..core.logger import get_logger

logger = get_logger("audio.spatial")

# ReSpeaker 2-Mic Pi HAT physical constants
MIC_DISTANCE_M = 0.058       # ~58mm between the two MEMS microphones
SPEED_OF_SOUND = 343.0       # m/s at room temperature
SAMPLE_RATE = 16000           # Hz


class SpatialAudioProcessor:
    """Processes stereo audio from 2-mic array for DOA and beamforming.

    Args:
        mic_distance: Distance between microphones in meters.
        sample_rate: Audio sample rate in Hz.
        beamform_direction: Initial beamforming direction in degrees.
                           0 = center, -90 = full left, +90 = full right.
    """

    def __init__(
        self,
        mic_distance: float = MIC_DISTANCE_M,
        sample_rate: int = SAMPLE_RATE,
        beamform_direction: float = 0.0,
    ) -> None:
        self.mic_distance = mic_distance
        self.sample_rate = sample_rate
        self._beamform_direction = beamform_direction
        self._lock = threading.Lock()

        # DOA state
        self._current_doa: float = 0.0  # degrees, -90 to +90
        self._doa_history: List[float] = []
        self._doa_smoothing_window = 5

        # Max inter-mic delay in samples
        self._max_delay_samples = int(mic_distance / SPEED_OF_SOUND * sample_rate)

        logger.info(
            f"SpatialAudioProcessor initialized (mic_dist={mic_distance*1000:.0f}mm, "
            f"max_delay={self._max_delay_samples} samples)"
        )

    def process_stereo_chunk(self, stereo_pcm: bytes) -> Tuple[bytes, float]:
        """Process a stereo PCM chunk: estimate DOA and produce beamformed mono.

        Args:
            stereo_pcm: Interleaved stereo PCM bytes (L, R, L, R, ...)
                        16-bit signed LE, 2 channels.

        Returns:
            Tuple of (enhanced_mono_pcm_bytes, doa_degrees).
            DOA: -90 = far left, 0 = center, +90 = far right.
        """
        with self._lock:
            # Split interleaved stereo into left and right channels
            left, right = self._split_stereo(stereo_pcm)

            if len(left) < 64:
                # Too short for meaningful processing, just average
                mono = self._simple_mix(left, right)
                return self._to_pcm_bytes(mono), self._current_doa

            # 1. Estimate Direction of Arrival
            doa = self._estimate_doa(left, right)
            self._update_doa(doa)

            # 2. Beamform towards the target direction
            mono = self._beamform(left, right, self._beamform_direction)

            return self._to_pcm_bytes(mono), self._current_doa

    def set_beamform_direction(self, degrees: float) -> None:
        """Set the beamforming steering direction.

        Args:
            degrees: -90 (far left) to +90 (far right). 0 = center/broadside.
        """
        self._beamform_direction = max(-90.0, min(90.0, degrees))

    def steer_towards_face(self, face_center_x: float, frame_width: float) -> None:
        """Steer the beamformer towards a detected face position.

        Maps the face's horizontal position in the camera frame to a
        beamforming direction angle.

        Args:
            face_center_x: X-coordinate of the face center in pixels.
            frame_width: Total width of the camera frame in pixels.
        """
        # Map pixel position to angle: left edge = -60°, right edge = +60°
        # (cameras typically have ~120° horizontal FOV)
        normalized = (face_center_x / frame_width) * 2.0 - 1.0  # -1.0 to +1.0
        angle = normalized * 60.0  # -60° to +60°
        self.set_beamform_direction(angle)

    @property
    def current_doa(self) -> float:
        """Current smoothed Direction of Arrival in degrees."""
        return self._current_doa

    @property
    def speaker_side(self) -> str:
        """Human-readable side of the current speaker."""
        if self._current_doa < -20:
            return "left"
        elif self._current_doa > 20:
            return "right"
        else:
            return "center"

    # ------------------------------------------------------------------
    # Internal Processing
    # ------------------------------------------------------------------

    @staticmethod
    def _split_stereo(stereo_pcm: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Split interleaved stereo PCM into separate L/R float arrays."""
        samples = np.frombuffer(stereo_pcm, dtype=np.int16).astype(np.float32)
        left = samples[0::2]   # Even indices = left channel
        right = samples[1::2]  # Odd indices = right channel
        return left, right

    @staticmethod
    def _simple_mix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Simple average mix of two channels."""
        return (left + right) * 0.5

    @staticmethod
    def _to_pcm_bytes(mono: np.ndarray) -> bytes:
        """Convert float array back to 16-bit PCM bytes."""
        clipped = np.clip(mono, -32768, 32767).astype(np.int16)
        return clipped.tobytes()

    def _estimate_doa(self, left: np.ndarray, right: np.ndarray) -> float:
        """Estimate Direction of Arrival using GCC-PHAT cross-correlation.

        GCC-PHAT (Generalized Cross-Correlation with Phase Transform) is
        the standard algorithm for time-delay estimation between two mics.
        It finds the time lag between L and R channels, which maps to
        the angle of the sound source.
        """
        n = len(left)
        if n < 32:
            return 0.0

        # FFT-based cross-correlation with phase transform (PHAT weighting)
        fft_size = 1 << (2 * n - 1).bit_length()  # Next power of 2
        L = np.fft.rfft(left, n=fft_size)
        R = np.fft.rfft(right, n=fft_size)

        # Cross-power spectrum with PHAT weighting
        cross_spectrum = L * np.conj(R)
        magnitude = np.abs(cross_spectrum)
        # Avoid division by zero
        magnitude[magnitude < 1e-10] = 1e-10
        gcc_phat = cross_spectrum / magnitude

        # Inverse FFT to get cross-correlation
        correlation = np.fft.irfft(gcc_phat, n=fft_size)

        # Search for peak within the physically possible delay range
        max_delay = min(self._max_delay_samples, n // 2)
        if max_delay < 1:
            return 0.0

        # Look at both positive and negative delays
        search_region = np.concatenate([
            correlation[:max_delay + 1],
            correlation[-(max_delay):]
        ])

        peak_idx = np.argmax(np.abs(search_region))

        # Convert index back to delay
        if peak_idx <= max_delay:
            delay = peak_idx
        else:
            delay = peak_idx - len(search_region)

        # Convert delay to angle
        # delay = (d * sin(theta)) / c * sample_rate
        # theta = arcsin(delay * c / (d * sample_rate))
        sin_theta = (delay * SPEED_OF_SOUND) / (self.mic_distance * self.sample_rate)
        sin_theta = max(-1.0, min(1.0, sin_theta))
        angle_rad = math.asin(sin_theta)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _update_doa(self, new_doa: float) -> None:
        """Update smoothed DOA with exponential moving average."""
        self._doa_history.append(new_doa)
        if len(self._doa_history) > self._doa_smoothing_window:
            self._doa_history.pop(0)
        self._current_doa = sum(self._doa_history) / len(self._doa_history)

    def _beamform(
        self, left: np.ndarray, right: np.ndarray, direction_deg: float
    ) -> np.ndarray:
        """Delay-and-sum beamforming towards the specified direction.

        Delays one channel relative to the other to align signals arriving
        from the target direction, then sums them. This enhances audio from
        the target direction and suppresses audio from other directions.

        Args:
            left: Left channel float samples.
            right: Right channel float samples.
            direction_deg: Steering direction in degrees (-90 to +90).

        Returns:
            Enhanced mono audio as float array.
        """
        if abs(direction_deg) < 5.0:
            # Near center: simple sum (no delay needed)
            return (left + right) * 0.5

        # Calculate required delay in samples
        angle_rad = math.radians(direction_deg)
        delay_seconds = self.mic_distance * math.sin(angle_rad) / SPEED_OF_SOUND
        delay_samples = int(round(delay_seconds * self.sample_rate))

        n = len(left)
        if abs(delay_samples) >= n:
            return (left + right) * 0.5

        # Apply delay to align channels
        if delay_samples > 0:
            # Sound from the right: delay left channel
            aligned_left = np.zeros(n)
            aligned_left[delay_samples:] = left[:n - delay_samples]
            output = (aligned_left + right) * 0.5
        else:
            # Sound from the left: delay right channel
            ds = abs(delay_samples)
            aligned_right = np.zeros(n)
            aligned_right[ds:] = right[:n - ds]
            output = (left + aligned_right) * 0.5

        return output
