"""Unit tests for Plant disease classification, Chess vision, and Face recognition.

Tests handle gracefully when:
- OpenCV is installed but test input is not a real image frame (numpy array)
- Libraries are completely missing (simulation mode)
"""

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.vision.chess import ChessVision
from lumi.vision.face import FaceRecognitionService
from lumi.vision.plant import PlantDiseaseDetector

# Try to create real numpy frames; fall back to dicts if numpy unavailable
try:
    import numpy as np
    _DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
except ImportError:
    _DUMMY_FRAME = {"data": "mock_frame"}


def test_plant_disease_detection() -> None:
    detector = PlantDiseaseDetector()
    result = detector.analyze_leaf(_DUMMY_FRAME)
    assert result.confidence >= 0.0
    summary = detector.generate_bangla_speech_summary(result)
    assert isinstance(summary, str)
    assert len(summary) > 5


def test_chess_vision_fen() -> None:
    vision = ChessVision()
    res = vision.extract_fen_from_frame(_DUMMY_FRAME)
    assert res.confidence >= 0.0
    assert isinstance(res.fen_string, str)


def test_face_recognition() -> None:
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mem.remember_person("Palash", relationship="owner")

    svc = FaceRecognitionService(mem)
    faces = svc.detect_and_recognize(_DUMMY_FRAME)
    assert isinstance(faces, list)
