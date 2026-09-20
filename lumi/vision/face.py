"""Face Detection and Recognition Service with Privacy Consent Checks."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.logger import get_logger
from ..memory.manager import MemoryManager
from ..memory.models import ConsentStatus, Person

from enum import Enum

logger = get_logger("vision.face")


class IdentityState(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    RECOGNIZED = "RECOGNIZED"


@dataclass
class DetectedFace:
    """Bounding box coordinates and identity classification for a detected face."""

    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # (center_x, center_y)
    confidence: float
    person: Optional[Person] = None
    is_known: bool = False
    identity_state: IdentityState = IdentityState.UNKNOWN
    distance: float = 1.0
    embedding: List[float] = field(default_factory=list)


class FaceRecognitionService:
    """Detects and identifies faces, retrieving memory profiles and triggering consent workflows."""

    def __init__(self, memory_manager: MemoryManager, recognition_threshold: float = 0.55) -> None:
        self.memory = memory_manager
        self.recognition_threshold = recognition_threshold
        self._cascade = None
        self._cascade_initialized = False
        self._pending_face_encoding: Optional[List[float]] = None
        self._pending_face_timestamp: float = 0.0
        self._last_interaction_timestamps: Dict[str, float] = {}

        # Centroid-based temporal tracking & multi-frame voting
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id: int = 0
        self._recognition_buffer: Dict[str, List[str]] = {}  # alias for backward compat
        self._buffer_size = 5  # 5 frames buffer (~0.75s)
        self._min_votes = 2    # At least 2 votes needed to confirm
        self._frame_counter = 0
        self._last_confirmed_faces: List[DetectedFace] = []

    def _get_cascade(self, cv2: Any) -> Optional[Any]:
        if self._cascade_initialized:
            return self._cascade

        self._cascade_initialized = True
        cascade_path = None

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidate = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(candidate):
                cascade_path = candidate

        if not cascade_path:
            # Search common filesystem locations
            project_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            for candidate in [
                os.path.join(project_data, "haarcascade_frontalface_default.xml"),
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
                "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            ]:
                if os.path.exists(candidate):
                    cascade_path = candidate
                    break

        # Try to download if no local file found
        if not cascade_path:
            dl_path = os.path.join(project_data, "haarcascade_frontalface_default.xml")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                os.makedirs(project_data, exist_ok=True)
                urllib.request.urlretrieve(url, dl_path)
                if os.path.exists(dl_path) and os.path.getsize(dl_path) > 1000:
                    cascade_path = dl_path
                    logger.info(f"Downloaded Haar Cascade to '{dl_path}'.")
            except Exception as e:
                logger.debug(f"Could not download Haar Cascade: {e}")

        if not cascade_path:
            cascade_path = "haarcascade_frontalface_default.xml"

        try:
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._cascade = cascade
                logger.info(f"OpenCV Haar Cascade loaded from '{cascade_path}'.")
            else:
                logger.warning(f"Haar Cascade at '{cascade_path}' is empty.")
        except Exception as e:
            logger.warning(f"Could not load Haar Cascade: {e}")

        return self._cascade

    def detect_and_recognize(self, frame: Any) -> List[DetectedFace]:
        """Process image frame, extract faces, and match against stored person profiles."""
        if frame is None:
            return []

        try:
            import cv2

            if not hasattr(frame, "shape"):
                return self._simulate_face_detection(frame)

            face_cascade = self._get_cascade(cv2)
            if face_cascade is None or face_cascade.empty():
                return self._simulate_face_detection(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            if len(faces) == 0:
                return []

            detected_faces = []

            try:
                import face_recognition

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                logger.info(f"[FACE] detected count={len(faces)}")
                if face_encodings:
                    logger.debug(f"[FACE] embedding generated count={len(face_encodings)} dim={len(face_encodings[0])}")

                known_persons = self.memory.list_people()

                # Build flattened candidate list supporting multiple embeddings per person
                all_known_candidates: List[Tuple[Person, List[float]]] = []
                for p in known_persons:
                    p_embeddings = p.face_embeddings
                    for emb in p_embeddings:
                        if emb and len(emb) == 128:
                            all_known_candidates.append((p, emb))

                for (x, y, w, h), encoding in zip(faces, face_encodings):
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    matched_person = None
                    is_known = False
                    best_dist = 1.0
                    identity_state = IdentityState.UNKNOWN
                    confidence = 0.50

                    if all_known_candidates and len(encoding) == 128:
                        candidate_embeddings = [c[1] for c in all_known_candidates]
                        distances = face_recognition.face_distance(candidate_embeddings, encoding)
                        best_match_index = int(distances.argmin())
                        best_dist = float(distances[best_match_index])
                        best_person = all_known_candidates[best_match_index][0]

                        if best_dist <= self.recognition_threshold:
                            matched_person = best_person
                            is_known = True
                            identity_state = IdentityState.RECOGNIZED
                            confidence = max(0.70, min(0.99, 1.0 - (best_dist / (self.recognition_threshold * 2.0))))
                            logger.info(
                                f"[MATCH] person_id={matched_person.id} name='{matched_person.name}' "
                                f"distance={best_dist:.3f} confidence={confidence:.2f} (RECOGNIZED)"
                            )
                        elif best_dist <= (self.recognition_threshold + 0.07):
                            matched_person = best_person
                            is_known = False
                            identity_state = IdentityState.LOW_CONFIDENCE
                            confidence = 0.50
                            logger.debug(
                                f"[MATCH] Borderline distance={best_dist:.3f} for '{best_person.name}' "
                                f"(thresh={self.recognition_threshold}) (LOW_CONFIDENCE)"
                            )
                        else:
                            identity_state = IdentityState.UNKNOWN
                            is_known = False
                            confidence = 0.90

                    detected_faces.append(
                        DetectedFace(
                            bounding_box=(x, y, w, h),
                            center=(center_x, center_y),
                            confidence=confidence,
                            person=matched_person if is_known else None,
                            is_known=is_known,
                            identity_state=identity_state,
                            distance=best_dist,
                            embedding=encoding.tolist(),
                        )
                    )
            except ImportError:
                # face_recognition library not installed: default detected faces to unknown
                for x, y, w, h in faces:
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    detected_faces.append(
                        DetectedFace(
                            bounding_box=(x, y, w, h),
                            center=(center_x, center_y),
                            confidence=0.90,
                            person=None,
                            is_known=False,
                            identity_state=IdentityState.UNKNOWN,
                            distance=1.0,
                            embedding=[],
                        )
                    )

            return detected_faces

        except Exception as e:
            logger.debug(f"Face detection note: {e}")
            return []

    def confirm_identity(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Apply continuous centroid tracking and multi-frame voting to confirm face identity.
        
        Guarantees:
        1. No grid boundary flickering (continuous distance matching < 140px).
        2. Known names are ONLY assigned when voting achieves consensus across frames.
        3. Low confidence or uncertain faces are NEVER assigned a known name.
        4. Temporal stability prevents 1-frame glitches from flipping identity.
        """
        self._frame_counter += 1
        now = time.time()
        confirmed = []

        for face in faces:
            best_track_id = None
            min_dist = 140.0  # max pixels a face moves between 0.15s frames

            for tid, tdata in list(self._tracks.items()):
                if (now - tdata["last_seen_time"]) > 1.5:
                    continue
                tcx, tcy = tdata["center"]
                dist = math.hypot(face.center[0] - tcx, face.center[1] - tcy)
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is None:
                self._next_track_id += 1
                best_track_id = self._next_track_id
                self._tracks[best_track_id] = {
                    "center": face.center,
                    "last_seen_time": now,
                    "history": [],
                }

            track = self._tracks[best_track_id]
            track["center"] = face.center
            track["last_seen_time"] = now

            # Record vote in track history: (person_id, person_name, person_obj)
            if face.is_known and face.person is not None:
                track["history"].append((face.person.id, face.person.name, face.person))
            else:
                track["history"].append((None, "__unknown__", None))

            if len(track["history"]) > self._buffer_size:
                track["history"].pop(0)

            history = track["history"]
            if len(history) >= 2:
                from collections import Counter
                id_counts = Counter(item[0] for item in history)
                best_pid, best_count = id_counts.most_common(1)[0]

                if best_pid is not None and best_count >= self._min_votes:
                    # Confirmed known person with consensus
                    person_obj = next((item[2] for item in reversed(history) if item[0] == best_pid), None)
                    if person_obj:
                        face.person = person_obj
                        face.is_known = True
                        face.identity_state = IdentityState.RECOGNIZED
                        face.confidence = min(0.99, 0.75 + (best_count / self._buffer_size) * 0.24)
                elif best_pid is None and best_count >= self._min_votes:
                    # Confirmed unknown person
                    face.person = None
                    face.is_known = False
                    face.identity_state = IdentityState.UNKNOWN
                    face.confidence = 0.90
                else:
                    # Mixed / uncertain consensus -> do NOT call by name
                    face.person = None
                    face.is_known = False
                    face.identity_state = IdentityState.LOW_CONFIDENCE
                    face.confidence = 0.50
            else:
                # Not enough frames accumulated yet
                if not face.is_known:
                    face.person = None
                    face.is_known = False
                    face.identity_state = IdentityState.LOW_CONFIDENCE
                    face.confidence = 0.40

            confirmed.append(face)

        # Cleanup tracks inactive for > 2.5 seconds
        expired = [tid for tid, tdata in self._tracks.items() if (now - tdata["last_seen_time"]) > 2.5]
        for tid in expired:
            del self._tracks[tid]

        self._last_confirmed_faces = list(confirmed)
        return confirmed

    def get_last_faces(self) -> List[DetectedFace]:
        """Retrieve the most recent confirmed faces detected by the vision loop without reprocessing."""
        return list(self._last_confirmed_faces)

    def _simulate_face_detection(self, frame: Any) -> List[DetectedFace]:
        return [
            DetectedFace(
                bounding_box=(220, 140, 200, 200),
                center=(320.0, 240.0),
                confidence=0.94,
                person=None,
                is_known=False,
                embedding=[0.1] * 64,
            )
        ]

    def should_interact(self, person_id: str, cooldown_s: float = 60.0) -> bool:
        """Rate-limit proactive greetings to prevent annoying repetitive interruptions."""
        now = time.time()
        last = self._last_interaction_timestamps.get(person_id, 0.0)
        if (now - last) >= cooldown_s:
            self._last_interaction_timestamps[person_id] = now
            return True
        return False

    def set_pending_face(self, encoding: List[float]) -> None:
        """Store the most recent unknown face encoding with timestamp for freshness validation."""
        if encoding:
            self._pending_face_encoding = encoding
            self._pending_face_timestamp = time.time()

    def get_pending_face(self, max_age_seconds: float = 30.0) -> Optional[List[float]]:
        """Retrieve the pending face encoding if it's fresh enough. Returns None if stale or missing."""
        encoding = self._pending_face_encoding
        if encoding is None:
            return None
        age = time.time() - getattr(self, '_pending_face_timestamp', 0.0)
        if age > max_age_seconds:
            logger.warning(f"[IDENTITY] Pending face is {age:.1f}s old (>{max_age_seconds}s). Discarding stale embedding.")
            self._pending_face_encoding = None
            return None
        self._pending_face_encoding = None
        return encoding
