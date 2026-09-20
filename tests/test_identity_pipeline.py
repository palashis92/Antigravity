"""Comprehensive unit and scenario tests for LUMI's identity, face recognition, and memory pipeline.

Covers Test Cases A, B, C, D as defined in the end-to-end identity audit.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus, Person
from lumi.vision.face import DetectedFace, FaceRecognitionService, IdentityState


def test_person_model_age_and_multi_embedding():
    """Verify Person dataclass supports age and multi-sample face embeddings."""
    p = Person(name="Mizan", relationship="friend")
    assert p.age is None
    p.age = 28
    assert p.age == 28

    # Single embedding assignment
    emb1 = [0.1] * 128
    p.face_embedding = emb1
    assert p.face_embedding == emb1
    assert len(p.face_embeddings) == 1

    # Multi-sample embedding addition
    emb2 = [0.2] * 128
    p.add_face_embedding(emb2)
    assert len(p.face_embeddings) == 2
    assert p.face_embeddings[0] == emb1
    assert p.face_embeddings[1] == emb2


def test_test_case_a_and_b_recognition_and_isolation():
    """Test Case A & B:
    
    A: Person introduces as Mizan -> profile created -> subsequent frames recognize Mizan.
    B: Mizan in front of camera -> system MUST NOT identify as AnotherPerson.
    """
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)

    # Pre-populate database with another person (e.g. Palash)
    palash = mem.remember_person("Palash", relationship="owner")
    palash.face_embedding = [0.9] * 128
    mem.update_person(palash)

    # 1. New unknown person appears (Mizan's face)
    mizan_embedding = [0.1] * 128
    svc = FaceRecognitionService(mem, recognition_threshold=0.55)
    
    # Store as pending face
    svc.set_pending_face(mizan_embedding)
    pending = svc.get_pending_face()
    assert pending == mizan_embedding

    # 2. User says "আমি মিজান, বয়স ২৫" -> Saved as Mizan
    mizan = mem.remember_person("Mizan", relationship="friend", consent_status=ConsentStatus.GRANTED)
    mizan.face_embedding = mizan_embedding
    mizan.age = 25
    mem.update_person(mizan)

    assert mem.find_person_by_name("Mizan") is not None
    assert mem.find_person_by_name("Mizan").age == 25

    # 3. Mizan appears again (slightly varied embedding simulating angle/light)
    # Euclidean distance between [0.1]*128 and [0.12]*128 is sqrt(128 * 0.02^2) = sqrt(0.0512) ~= 0.226 (< 0.55)
    mizan_angle_sample = [0.12] * 128

    face_detection = DetectedFace(
        bounding_box=(200, 150, 100, 100),
        center=(250.0, 200.0),
        confidence=0.95,
        person=mizan,
        is_known=True,
        identity_state=IdentityState.RECOGNIZED,
        embedding=mizan_angle_sample,
    )

    # Frame 1 through confirm_identity
    res1 = svc.confirm_identity([face_detection])
    # Frame 2 through confirm_identity
    res2 = svc.confirm_identity([face_detection])

    # After 2 votes, identity is confirmed as Mizan
    assert res2[0].is_known is True
    assert res2[0].person is not None
    assert res2[0].person.name == "Mizan"
    assert res2[0].identity_state == IdentityState.RECOGNIZED
    # Test Case B: MUST NOT match Palash
    assert res2[0].person.name != "Palash"


def test_test_case_c_unknown_person_no_hallucinated_name():
    """Test Case C: Unknown person appears -> system MUST NOT guess a previously stored name."""
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)

    # Database has Mizan and Palash
    p1 = mem.remember_person("Palash", relationship="owner")
    p1.face_embedding = [0.1] * 128
    mem.update_person(p1)

    p2 = mem.remember_person("Mizan", relationship="friend")
    p2.face_embedding = [0.5] * 128
    mem.update_person(p2)

    svc = FaceRecognitionService(mem, recognition_threshold=0.55)

    # A stranger appears with an embedding far from both [0.1] and [0.5]
    stranger_face = DetectedFace(
        bounding_box=(200, 150, 100, 100),
        center=(250.0, 200.0),
        confidence=0.90,
        person=None,
        is_known=False,
        identity_state=IdentityState.UNKNOWN,
        embedding=[0.9] * 128,
    )

    # Multi-frame voting
    res1 = svc.confirm_identity([stranger_face])
    res2 = svc.confirm_identity([stranger_face])

    # Stranger must remain UNKNOWN with person=None
    assert res2[0].is_known is False
    assert res2[0].person is None
    assert res2[0].identity_state == IdentityState.UNKNOWN


def test_test_case_d_face_and_name_never_cross_associated():
    """Test Case D: Face A belongs to Mizan. Face B belongs to Karim.
    
    They must never cross-associate or overwrite each other.
    """
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)

    emb_mizan = [0.1] * 128
    emb_karim = [0.8] * 128

    mizan = mem.remember_person("Mizan", relationship="friend")
    mizan.face_embedding = emb_mizan
    mem.update_person(mizan)

    karim = mem.remember_person("Karim", relationship="guest")
    karim.face_embedding = emb_karim
    mem.update_person(karim)

    # Querying by face
    matched_mizan = mem.find_person_by_face(emb_mizan)
    assert matched_mizan is not None
    assert matched_mizan.name == "Mizan"

    matched_karim = mem.find_person_by_face(emb_karim)
    assert matched_karim is not None
    assert matched_karim.name == "Karim"

    # Both profiles remain independent
    assert matched_mizan.id != matched_karim.id


def test_pending_face_timestamp_freshness():
    """Verify that pending face expires after max_age_seconds."""
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    svc = FaceRecognitionService(mem)

    # Store pending face with artificial old timestamp
    svc.set_pending_face([0.1] * 128)
    svc._pending_face_timestamp = time.time() - 35.0  # 35 seconds old

    # Must return None because age > 30.0s
    expired = svc.get_pending_face(max_age_seconds=30.0)
    assert expired is None


if __name__ == "__main__":
    test_person_model_age_and_multi_embedding()
    print("✓ test_person_model_age_and_multi_embedding passed")
    test_test_case_a_and_b_recognition_and_isolation()
    print("✓ test_test_case_a_and_b_recognition_and_isolation passed")
    test_test_case_c_unknown_person_no_hallucinated_name()
    print("✓ test_test_case_c_unknown_person_no_hallucinated_name passed")
    test_test_case_d_face_and_name_never_cross_associated()
    print("✓ test_test_case_d_face_and_name_never_cross_associated passed")
    test_pending_face_timestamp_freshness()
    print("✓ test_pending_face_timestamp_freshness passed")
    print("\nAll Identity Pipeline Scenario Tests Passed Successfully!")
