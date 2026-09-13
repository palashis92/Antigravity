"""Speaker Identification Engine for LUMI.

Uses sherpa-onnx (CAM++ / 3D-Speaker ONNX model) as the primary lightweight engine
for Raspberry Pi 5. Requires only ~45MB disk space and ~40MB RAM (no PyTorch/CUDA needed).
Runs in ~35ms on Pi 5 CPU with ARM NEON optimizations.

Falls back gracefully to resemblyzer if installed, or runs in mock/disabled mode
if no backend is available.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager
    from ..memory.models import Person

logger = get_logger("audio.speaker_id")

# Candidate paths for sherpa-onnx CAM++ speaker verification models
_MODEL_CANDIDATES = [
    "models/speaker_id/campplus.onnx",
    "models/speaker_id/3dspeaker_speech_campplus_sv_zh_en_16k-common.onnx",
    "models/campplus.onnx",
    "data/models/campplus.onnx",
]


class SpeakerIdentifier:
    """Identifies speakers by comparing voice embeddings against known profiles.

    Supports:
        1. sherpa-onnx (Primary, fast, ~45MB footprint, 192-dim CAM++ embeddings)
        2. resemblyzer (Fallback, 256-dim d-vectors)

    Args:
        memory: MemoryManager for accessing Person profiles and voice embeddings.
        similarity_threshold: Minimum cosine similarity for a positive match (0.0-1.0).
                               Higher = stricter matching. 0.70 is recommended for CAM++.
    """

    def __init__(
        self,
        memory: MemoryManager,
        similarity_threshold: float = 0.70,
    ) -> None:
        self.memory = memory
        self.similarity_threshold = similarity_threshold
        self._backend: Optional[str] = None  # "sherpa_onnx" | "resemblyzer"
        self._extractor: Any = None
        self._encoder_available = False
        self._lock = threading.Lock()
        self._last_identified: Optional[str] = None
        self._identification_count: Dict[str, int] = {}

        # Load model in background thread
        load_thread = threading.Thread(
            target=self._load_encoder, daemon=True, name="SpeakerID_Load"
        )
        load_thread.start()

    def _resolve_model_path(self) -> Optional[str]:
        """Locate an existing ONNX model file on disk or auto-download if missing."""
        project_root = Path(__file__).resolve().parent.parent.parent
        for candidate in _MODEL_CANDIDATES:
            p = project_root / candidate
            if p.exists() and p.stat().st_size > 10000:
                return str(p)
            # Also check direct path relative to current working directory
            if os.path.exists(candidate) and os.path.getsize(candidate) > 10000:
                return candidate

        # Auto-download lightweight CAM++ model (~20MB) if missing
        try:
            import urllib.request
            target_dir = project_root / "data" / "models"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "campplus.onnx"
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common.onnx"
            logger.info("Auto-downloading lightweight CAM++ Speaker ID model (~20MB)...")
            urllib.request.urlretrieve(url, str(target_file))
            if target_file.exists() and target_file.stat().st_size > 10000:
                logger.info(f"Speaker ID model downloaded successfully to {target_file}")
                return str(target_file)
        except Exception as e:
            logger.debug(f"Could not auto-download speaker ID model: {e}")

        return None

    def _load_encoder(self) -> None:
        """Attempt to load sherpa-onnx first, then resemblyzer as fallback."""
        start = time.time()

        # 1. Try sherpa-onnx (Recommended for Pi 5)
        try:
            import sherpa_onnx  # type: ignore

            model_path = self._resolve_model_path()
            if model_path:
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=model_path,
                    num_threads=2,
                    provider="cpu",
                )
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
                self._backend = "sherpa_onnx"
                self._encoder_available = True
                elapsed = time.time() - start
                logger.info(
                    f"Speaker ID loaded via sherpa-onnx in {elapsed:.1f}s (model: {model_path})"
                )
                return
            else:
                logger.debug(
                    "sherpa-onnx is installed but no CAM++ model found at models/speaker_id/campplus.onnx"
                )
        except ImportError:
            logger.debug("sherpa-onnx not installed, checking fallback...")
        except Exception as e:
            logger.warning(f"Failed to initialize sherpa-onnx: {e}")

        # 2. Try resemblyzer as fallback
        try:
            from resemblyzer import VoiceEncoder  # type: ignore

            self._extractor = VoiceEncoder("cpu")
            self._backend = "resemblyzer"
            self._encoder_available = True
            elapsed = time.time() - start
            logger.info(f"Speaker ID loaded via resemblyzer in {elapsed:.1f}s")
            return
        except ImportError:
            logger.info(
                "Speaker identification disabled (neither sherpa-onnx nor resemblyzer found). "
                "For Pi 5, install: pip install sherpa-onnx"
            )
        except Exception as e:
            logger.error(f"Failed to load resemblyzer speaker encoder: {e}")

        self._encoder_available = False
        self._backend = None

    def is_available(self) -> bool:
        """Check if any speaker identification engine is ready."""
        return _HAS_NUMPY and self._encoder_available and self._extractor is not None

    @property
    def backend(self) -> Optional[str]:
        """Return the active backend ('sherpa_onnx', 'resemblyzer', or None)."""
        return self._backend

    def _extract_embedding(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """Extract speaker embedding vector using the active backend."""
        if not self.is_available():
            return None

        if self._backend == "sherpa_onnx":
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=audio_array)
            stream.input_finished()
            emb = np.array(self._extractor.compute(stream), dtype=np.float32)
            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

        elif self._backend == "resemblyzer":
            if sample_rate != 16000:
                from resemblyzer import preprocess_wav  # type: ignore
                audio_array = preprocess_wav(audio_array, source_sr=sample_rate)
            emb = self._extractor.embed_utterance(audio_array)
            return np.array(emb, dtype=np.float32)

        return None

    def identify_speaker(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> Tuple[Optional[str], float]:
        """Identify who is speaking from a PCM audio segment.

        Args:
            audio_pcm: Raw PCM 16-bit signed LE mono audio bytes.
                       Should be at least 1.0 second for reliable identification.
            sample_rate: Sample rate of the audio (default 16000 Hz).

        Returns:
            Tuple of (person_name or None, confidence_score).
            Returns (None, 0.0) if no match or engine unavailable.
        """
        if not self.is_available():
            return None, 0.0

        with self._lock:
            try:
                valid_len = (len(audio_pcm) // 2) * 2
                if valid_len == 0:
                    return None, 0.0
                audio_array = np.frombuffer(audio_pcm[:valid_len], dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_array) < int(sample_rate * 0.8):
                    # Utterance too short for reliable identification
                    return None, 0.0

                utterance_embedding = self._extract_embedding(audio_array, sample_rate)
                if utterance_embedding is None:
                    return None, 0.0

                # Compare against known voice profiles
                known_persons = self.memory.list_people()
                best_match: Optional[Person] = None
                best_similarity = 0.0

                for person in known_persons:
                    voice_emb = person.voice_embedding
                    if voice_emb is None:
                        continue

                    stored_embedding = np.array(voice_emb, dtype=np.float32)
                    if len(stored_embedding) != len(utterance_embedding):
                        # Dimension mismatch (e.g. mixed 192-dim vs 256-dim)
                        continue

                    similarity = self._cosine_similarity(utterance_embedding, stored_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person

                if best_match and best_similarity >= self.similarity_threshold:
                    logger.info(
                        f"🎙️ Speaker identified: {best_match.name} "
                        f"(similarity: {best_similarity:.3f}, backend: {self._backend})"
                    )
                    self._last_identified = best_match.name
                    self._identification_count[best_match.name] = (
                        self._identification_count.get(best_match.name, 0) + 1
                    )
                    return best_match.name, best_similarity
                else:
                    if best_match:
                        logger.debug(
                            f"🎙️ Best match was {best_match.name} "
                            f"({best_similarity:.3f}) but below threshold {self.similarity_threshold}"
                        )
                    return None, best_similarity

            except Exception as e:
                logger.error(f"Speaker identification error: {e}")
                return None, 0.0

    def create_voice_profile(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> Optional[List[float]]:
        """Create a voice embedding from an audio segment during enrollment.

        Args:
            audio_pcm: Raw PCM audio bytes (at least 1.5 seconds recommended).
            sample_rate: Sample rate of the audio.

        Returns:
            Normalized voice embedding as a list of floats, or None on failure.
        """
        if not self.is_available():
            return None

        with self._lock:
            try:
                valid_len = (len(audio_pcm) // 2) * 2
                if valid_len == 0:
                    return None
                audio_array = np.frombuffer(audio_pcm[:valid_len], dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_array) < int(sample_rate * 1.2):
                    logger.warning("Audio too short for voice enrollment (need >= 1.2s)")
                    return None

                embedding = self._extract_embedding(audio_array, sample_rate)
                if embedding is not None:
                    logger.info(
                        f"Voice profile created (dim: {len(embedding)}, backend: {self._backend})"
                    )
                    return embedding.tolist()
                return None

            except Exception as e:
                logger.error(f"Voice profile creation failed: {e}")
                return None

    def enroll_voice(self, person_id: str, audio_pcm: bytes, sample_rate: int = 16000) -> bool:
        """Enroll a person's voice by saving their voice embedding.

        Args:
            person_id: The person's ID in the database.
            audio_pcm: Raw PCM audio of them speaking.
            sample_rate: Audio sample rate.

        Returns:
            True if enrollment succeeded.
        """
        embedding = self.create_voice_profile(audio_pcm, sample_rate)
        if embedding is None:
            return False

        people = self.memory.list_people()
        for person in people:
            if person.id == person_id:
                person.voice_embedding = embedding
                self.memory.update_person(person)
                logger.info(f"Voice enrolled for {person.name}")
                return True

        logger.warning(f"Person {person_id} not found for voice enrollment")
        return False

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors (numpy or pure python)."""
        if _HAS_NUMPY and hasattr(a, "dot"):
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))

        # Pure Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @property
    def last_identified_speaker(self) -> Optional[str]:
        """Returns the name of the last identified speaker."""
        return self._last_identified
