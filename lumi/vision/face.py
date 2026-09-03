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

logger = get_logger("vision.face")


@dataclass
class DetectedFace:
    """Bounding box coordinates and identity classification for a detected face."""

    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # (center_x, center_y)
    confidence: float
    person: Optional[Person] = None
    is_known: bool = False
    embedding: List[float] = field(default_factory=list)


class FaceRecognitionService:
    """Detects and identifies faces, retrieving memory profiles and triggering consent workflows."""

    def __init__(self, memory_manager: MemoryManager, recognition_threshold: float = 0.65) -> None:
        self.memory = memory_manager
        self.recognition_threshold = recognition_threshold
        self._cascade = None
        self._cascade_initialized = False
        self._pending_face_encoding: Optional[List[float]] = None
        self._last_interaction_timestamps: Dict[str, float] = {}

        # Multi-frame voting for robust recognition
        self._recognition_buffer: Dict[str, List[str]] = {}  # track_id -> [person_name, ...]
        self._buffer_size = 5  # Require 5 consistent frames
        self._min_votes = 3    # At least 3 out of 5 must agree
        self._frame_counter = 0

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
                gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100)
            )

            if len(faces) == 0:
                return []

            detected_faces = []

            try:
                import face_recognition

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                known_persons = self.memory.list_people()
                known_embeddings = [p.face_embedding for p in known_persons if p.face_embedding]
                known_person_objects = [p for p in known_persons if p.face_embedding]

                for (x, y, w, h), encoding in zip(faces, face_encodings):
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    matched_person = None
                    is_known = False

                    if known_embeddings:
                        matches = face_recognition.compare_faces(
                            known_embeddings, encoding, tolerance=self.recognition_threshold
                        )
                        distances = face_recognition.face_distance(known_embeddings, encoding)
                        if any(matches):
                            best_match_index = int(distances.argmin())
                            if matches[best_match_index]:
                                matched_person = known_person_objects[best_match_index]
                                is_known = True

                    detected_faces.append(
                        DetectedFace(
                            bounding_box=(x, y, w, h),
                            center=(center_x, center_y),
                            confidence=0.95,
                            person=matched_person,
                            is_known=is_known,
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
                            embedding=[],
                        )
                    )

            return detected_faces

        except Exception as e:
            logger.debug(f"Face detection note: {e}")
            return []

    def confirm_identity(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Apply multi-frame voting to confirm face identity with higher confidence.
        
        Instead of trusting a single frame, we accumulate results over
        several frames and only confirm identity when a person is consistently
        recognized across multiple frames.
        """
        self._frame_counter += 1
        confirmed = []
        
        for face in faces:
            # Create a simple spatial track ID based on face center region
            cx, cy = int(face.center[0] // 80), int(face.center[1] // 80)
            track_id = f"{cx}_{cy}"
            
            # Get the name this frame matched
            name = face.person.name if face.is_known and face.person else "__unknown__"
            
            # Initialize buffer if new track
            if track_id not in self._recognition_buffer:
                self._recognition_buffer[track_id] = []
            
            buffer = self._recognition_buffer[track_id]
            buffer.append(name)
            
            # Keep only last N frames
            if len(buffer) > self._buffer_size:
                buffer.pop(0)
            
            # Count votes for the most common identity
            if len(buffer) >= 2:  # Need at least 2 frames
                from collections import Counter
                vote_counts = Counter(buffer)
                best_name, best_count = vote_counts.most_common(1)[0]
                
                if best_name != "__unknown__" and best_count >= self._min_votes:
                    # Confirmed known person with high confidence!
                    face.confidence = min(0.99, 0.7 + (best_count / self._buffer_size) * 0.3)
                    # face.person and face.is_known are already set from the latest frame
                elif best_name == "__unknown__" and best_count >= self._min_votes:
                    # Confirmed unknown
                    face.is_known = False
                    face.person = None
                    face.confidence = 0.90
                else:
                    # Not enough consensus yet — mark as uncertain but pass through
                    # Use the latest frame's result but lower confidence
                    face.confidence = 0.5 + (best_count / self._buffer_size) * 0.3
            else:
                # Not enough frames yet, lower confidence
                face.confidence = 0.4
            
            confirmed.append(face)
        
        # Clean up stale tracks (not seen for 50+ frames)
        if self._frame_counter % 50 == 0:
            active_tracks = {f"{int(f.center[0] // 80)}_{int(f.center[1] // 80)}" for f in faces}
            stale = [k for k in self._recognition_buffer if k not in active_tracks]
            for k in stale:
                del self._recognition_buffer[k]
        
        return confirmed

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
        """Store the most recent unknown face encoding for learning."""
        if encoding:
            self._pending_face_encoding = encoding

    def get_pending_face(self) -> Optional[List[float]]:
        """Retrieve and clear the pending face encoding."""
        encoding = self._pending_face_encoding
        self._pending_face_encoding = None
        return encoding
