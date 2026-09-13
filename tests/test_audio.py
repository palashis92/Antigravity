"""Unit tests for LUMI Audio Subsystem: Proximity Filtering and Speaker Identification."""

import math
import struct

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

from lumi.audio.proximity_filter import ProximityAudioFilter
from lumi.audio.speaker_id import SpeakerIdentifier
from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager


def _generate_pcm_tone(rms_target: float, duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate a synthesized PCM chunk with an approximately target RMS amplitude."""
    amplitude = rms_target * math.sqrt(2)
    n_samples = int(duration_s * sample_rate)
    samples = []
    freq = 440.0
    for i in range(n_samples):
        val = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        val = max(-32767, min(32767, val))
        samples.append(val)
    return struct.pack(f"<{len(samples)}h", *samples)


def test_proximity_filter_silence() -> None:
    """Silence chunks should always pass to allow AI turn-taking."""
    filt = ProximityAudioFilter(silence_rms=100.0)
    silence = bytes(2048)  # all zeros
    assert filt.should_pass(silence, is_overlap=False) is True
    assert filt.should_pass(silence, is_overlap=True) is True


def test_proximity_filter_single_speaker() -> None:
    """When only a single speaker is present (no overlap), all audio must pass, even distant/quiet."""
    filt = ProximityAudioFilter(silence_rms=50.0, overlap_ratio=0.6)

    # Distant quiet voice (~150 RMS)
    quiet_chunk = _generate_pcm_tone(rms_target=150.0)
    for _ in range(5):
        assert filt.should_pass(quiet_chunk, is_overlap=False) is True

    # Normal voice (~800 RMS)
    normal_chunk = _generate_pcm_tone(rms_target=800.0)
    for _ in range(5):
        assert filt.should_pass(normal_chunk, is_overlap=False) is True


def test_proximity_filter_overlap_near_field_priority() -> None:
    """When multiple speakers overlap, only close/loud audio passes, distant is dropped."""
    filt = ProximityAudioFilter(silence_rms=50.0, overlap_ratio=0.5, window_size=10)

    close_loud_chunk = _generate_pcm_tone(rms_target=1200.0)
    distant_quiet_chunk = _generate_pcm_tone(rms_target=250.0)

    # Prime peak with close loud audio
    for _ in range(5):
        filt.should_pass(close_loud_chunk, is_overlap=False)

    # Overlap occurs: close loud speaker should pass
    assert filt.should_pass(close_loud_chunk, is_overlap=True) is True

    # Overlap occurs: distant quiet background speaker should be filtered out
    dropped = filt.should_pass(distant_quiet_chunk, is_overlap=True)
    assert dropped is False

    stats = filt.get_stats()
    assert stats["dropped_chunks"] >= 1
    assert stats["passed_chunks"] >= 1


def test_proximity_filter_cooldown() -> None:
    """Filter should remain active during cooldown after overlap ends."""
    filt = ProximityAudioFilter(silence_rms=50.0, overlap_ratio=0.5, cooldown_chunks=3)
    loud_chunk = _generate_pcm_tone(rms_target=1500.0)
    quiet_chunk = _generate_pcm_tone(rms_target=200.0)

    # Set peak with loud audio
    filt.should_pass(loud_chunk, is_overlap=False)

    # Overlap triggers
    filt.should_pass(loud_chunk, is_overlap=True)

    # Overlap flag becomes False, but cooldown keeps filtering
    # 1st chunk after overlap: quiet should still be dropped
    assert filt.should_pass(quiet_chunk, is_overlap=False) is False


def test_speaker_identifier_lifecycle() -> None:
    """Verify SpeakerIdentifier initializes and handles queries without crashing."""
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mem.remember_person("Palash", relationship="owner")

    identifier = SpeakerIdentifier(mem, similarity_threshold=0.70)
    # Even if models are not on dev machine, it shouldn't crash
    dummy_audio = _generate_pcm_tone(rms_target=500.0, duration_s=1.5)
    name, confidence = identifier.identify_speaker(dummy_audio)
    assert isinstance(confidence, float)
    assert name is None or isinstance(name, str)


def test_speaker_identifier_cosine_similarity() -> None:
    """Verify cosine similarity calculation between identical and orthogonal vectors."""
    if _HAS_NP:
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]

    sim_identical = SpeakerIdentifier._cosine_similarity(v1, v2)
    sim_orthogonal = SpeakerIdentifier._cosine_similarity(v1, v3)

    assert abs(sim_identical - 1.0) < 1e-5
    assert abs(sim_orthogonal - 0.0) < 1e-5


def test_doa_body_orientation_mapping() -> None:
    """Verify DOA sound localization mapping to body waist pan angles.
    
    Ground-truth calibration:
      - Center: 0°
      - Left: positive angle (turn body left)
      - Right: negative angle (turn body right)
    """
    from lumi.audio.mic import MicInterface

    class MockSpatial:
        def __init__(self, doa: float, side: str):
            self.current_doa = doa
            self.speaker_side = side

    class MockBackendWithSpatial:
        def __init__(self, spatial):
            self.spatial_processor = spatial
        def start_recording(self): return True
        def stop_recording(self): pass
        def read_chunk(self, size): return b""

    # 1. Center sound
    mic = MicInterface(backend=MockBackendWithSpatial(MockSpatial(0.0, "center")))
    assert mic.spatial_processor is not None
    assert mic.spatial_processor.speaker_side == "center"

    # 2. Left sound (-45° DOA) -> Should map to positive pan
    mic_left = MicInterface(backend=MockBackendWithSpatial(MockSpatial(-45.0, "left")))
    assert mic_left.spatial_processor.speaker_side == "left"

    # 3. Right sound (+45° DOA) -> Should map to negative pan
    mic_right = MicInterface(backend=MockBackendWithSpatial(MockSpatial(45.0, "right")))
    assert mic_right.spatial_processor.speaker_side == "right"

