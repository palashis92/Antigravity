"""Speaker Identification Engine for LUMI.

Uses resemblyzer to create and match voice embeddings (d-vectors)
for identifying who is currently speaking.

Designed for Raspberry Pi 5 — model loads once at startup (~3-5s),
then each identification takes ~300-500ms per utterance.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager
    from ..memory.models import Person

logger = get_logger("audio.speaker_id")


class SpeakerIdentifier:
    """Identifies speakers by comparing voice embeddings against known profiles.

    Uses resemblyzer's VoiceEncoder to create 256-dimensional speaker
    embeddings (d-vectors) from speech segments.

    Args:
        memory: MemoryManager for accessing Person profiles and voice embeddings.
        similarity_threshold: Minimum cosine similarity for a positive match (0.0-1.0).
                              Higher = stricter matching. 0.75 is a good default.
    """

    def __init__(
        self,
        memory: MemoryManager,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.memory = memory
        self.similarity_threshold = similarity_threshold
        self._encoder = None
        self._encoder_available = False
        self._lock = threading.Lock()
        self._last_identified: Optional[str] = None
        self._identification_count: Dict[str, int] = {}

        # Load encoder in background to not block startup
        load_thread = threading.Thread(
            target=self._load_encoder, daemon=True, name="SpeakerID_Load"
        )
        load_thread.start()

    def _load_encoder(self) -> None:
        """Load the resemblyzer VoiceEncoder model."""
        try:
            start = time.time()
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
            self._encoder_available = True
            elapsed = time.time() - start
            logger.info(f"Speaker ID encoder loaded in {elapsed:.1f}s")
        except ImportError:
            logger.warning(
                "resemblyzer not installed. Speaker identification disabled. "
                "Install with: pip install resemblyzer"
            )
        except Exception as e:
            logger.error(f"Failed to load speaker encoder: {e}")

    def is_available(self) -> bool:
        """Check if the speaker identification engine is ready."""
        return self._encoder_available and self._encoder is not None

    def identify_speaker(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> Tuple[Optional[str], float]:
        """Identify who is speaking from a PCM audio segment.

        Args:
            audio_pcm: Raw PCM 16-bit signed LE mono audio bytes.
                       Should be at least 1.5 seconds for reliable identification.
            sample_rate: Sample rate of the audio.

        Returns:
            Tuple of (person_name or None, confidence_score).
            Returns (None, 0.0) if no match or engine unavailable.
        """
        if not self.is_available():
            return None, 0.0

        with self._lock:
            try:
                # Convert PCM bytes to float32 numpy array
                audio_array = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_array) < int(sample_rate * 1.0):
                    # Less than 1 second, unreliable
                    return None, 0.0

                # Resample if needed (resemblyzer expects 16kHz)
                if sample_rate != 16000:
                    from resemblyzer import preprocess_wav
                    audio_array = preprocess_wav(audio_array, source_sr=sample_rate)

                # Generate embedding for this utterance
                utterance_embedding = self._encoder.embed_utterance(audio_array)

                # Compare against all known voice profiles
                known_persons = self.memory.list_people()
                best_match: Optional[Person] = None
                best_similarity = 0.0

                for person in known_persons:
                    voice_emb = person.voice_embedding
                    if voice_emb is None:
                        continue

                    stored_embedding = np.array(voice_emb, dtype=np.float32)
                    similarity = self._cosine_similarity(utterance_embedding, stored_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person

                if best_match and best_similarity >= self.similarity_threshold:
                    logger.info(
                        f"🎙️ Speaker identified: {best_match.name} "
                        f"(similarity: {best_similarity:.3f})"
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
                            f"({best_similarity:.3f}) but below threshold"
                        )
                    return None, best_similarity

            except Exception as e:
                logger.error(f"Speaker identification error: {e}")
                return None, 0.0

    def create_voice_profile(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> Optional[List[float]]:
        """Create a voice embedding from an audio segment.

        Used during enrollment to save a person's voice profile.

        Args:
            audio_pcm: Raw PCM audio bytes (at least 2 seconds recommended).
            sample_rate: Sample rate of the audio.

        Returns:
            256-dimensional voice embedding as a list of floats, or None on failure.
        """
        if not self.is_available():
            return None

        with self._lock:
            try:
                audio_array = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_array) < int(sample_rate * 1.5):
                    logger.warning("Audio too short for voice enrollment (need >= 1.5s)")
                    return None

                if sample_rate != 16000:
                    from resemblyzer import preprocess_wav
                    audio_array = preprocess_wav(audio_array, source_sr=sample_rate)

                embedding = self._encoder.embed_utterance(audio_array)
                logger.info(f"Voice profile created (embedding dim: {len(embedding)})")
                return embedding.tolist()

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

        # Find person and update their voice embedding
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
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @property
    def last_identified_speaker(self) -> Optional[str]:
        """Returns the name of the last identified speaker."""
        return self._last_identified
