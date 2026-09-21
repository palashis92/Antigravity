"""Unit tests for Phase 3: Vision & Perception Throughput.

Covers:
- [V3/V4] Decouple Face Detection (15-20 FPS) from 128D Face Recognition (1 FPS)
- [V2] Target-Lost Hysteresis (2.0s delay) & Smooth Search Pan (1.2s)
- [V5] Anjum Mode 3-Frame Confirmation & Adult Auto-Exit
"""

import time
from unittest.mock import MagicMock, patch

from lumi.config.settings import load_settings
from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus, Person
from lumi.vision.face import DetectedFace, FaceRecognitionService, IdentityState
from lumi.core.state_manager import BehaviorState, StateManager
from lumi.core.telemetry import get_telemetry


def test_face_recognition_track_caching_throughput():
    """[V3/V4] Verify spatial track caching bypasses heavy 128D encoding on fast frames."""
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mizan = mem.remember_person("Mizan", relationship="friend", consent_status=ConsentStatus.GRANTED)
    mizan.face_embedding = [0.1] * 128
    mem.update_person(mizan)

    svc = FaceRecognitionService(mem)
    svc._min_recognition_interval = 2.0

    # Simulate a confirmed track in svc._tracks
    now = time.time()
    confirmed_face = DetectedFace(
        bounding_box=(200, 150, 100, 100),
        center=(250.0, 200.0),
        confidence=0.95,
        person=mizan,
        is_known=True,
        identity_state=IdentityState.RECOGNIZED,
        distance=0.1,
        embedding=[0.1] * 128,
    )
    svc._tracks[1] = {
        "center": (250.0, 200.0),
        "last_seen_time": now,
        "history": [(mizan.id, mizan.name, mizan)],
        "cached_face": confirmed_face,
        "last_encoded_time": now,
    }
    svc._last_recognition_time = now

    # Pass mock frame with face at (202, 151, 100, 100) -> matches track 1
    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)

    mock_cascade = MagicMock()
    mock_cascade.empty.return_value = False
    mock_cascade.detectMultiScale.return_value = [(202, 151, 100, 100)]

    mock_cv2 = MagicMock()
    mock_cv2.cvtColor.return_value = mock_frame

    with patch.object(svc, "_get_cascade", return_value=mock_cascade):
        with patch("lumi.vision.face._HAS_CV2", True):
            with patch("lumi.vision.face.cv2", mock_cv2):
                results = svc.detect_and_recognize(mock_frame)
                assert len(results) == 1
                assert results[0].is_known is True
                assert results[0].person.name == "Mizan"


def test_target_lost_search_hysteresis_and_smooth_pan():
    """[V2] Verify target-lost search waits >= 2.0s and pans at 1.2s smooth duration."""
    from lumi.core.lumi_brain import LumiBrain

    settings = load_settings()
    sm = StateManager(BehaviorState.IDLE)
    eb = MagicMock()
    mem = MagicMock()
    servo = MagicMock()
    eyes = MagicMock()
    cam = MagicMock()
    mic = MagicMock()
    spk = MagicMock()

    brain = LumiBrain(
        settings=settings,
        state_manager=sm,
        event_bus=eb,
        memory_manager=mem,
        servo_controller=servo,
        eye_renderer=eyes,
        camera=cam,
        mic=mic,
        speaker=spk,
    )

    now = 1000.0
    brain._had_tracked_face = True
    brain._search_phase = 0
    brain._last_face_exit_side = "right"
    brain.head = MagicMock()
    brain.head.current_pan = 0.0

    # 1. Only 0.8s lost: should NOT trigger search (hysteresis prevents jitter)
    brain._last_face_seen_time = now - 0.8
    with patch("time.time", return_value=now):
        brain.process_person_interaction(None)
    assert brain._search_phase == 0
    brain.head.pan.assert_not_called()

    # 2. 2.2s lost: triggers Phase 1 search with smooth 1.2s duration
    brain._last_face_seen_time = now - 2.2
    with patch("time.time", return_value=now):
        brain.process_person_interaction(None)
    assert brain._search_phase == 1
    brain.head.pan.assert_called_once()
    # Check that duration_s is 1.2s (smooth pan)
    _, kwargs = brain.head.pan.call_args
    assert kwargs.get("duration_s") == 1.2


def test_anjum_mode_three_frame_confirmation_and_adult_exit():
    """[V5] Verify Anjum mode requires 3 consecutive frames and exits when adult appears."""
    from lumi.core.lumi_brain import LumiBrain

    settings = load_settings()
    sm = StateManager(BehaviorState.IDLE)
    eb = MagicMock()
    mem = MagicMock()
    servo = MagicMock()
    eyes = MagicMock()
    cam = MagicMock()
    mic = MagicMock()
    spk = MagicMock()

    brain = LumiBrain(
        settings=settings,
        state_manager=sm,
        event_bus=eb,
        memory_manager=mem,
        servo_controller=servo,
        eye_renderer=eyes,
        camera=cam,
        mic=mic,
        speaker=spk,
    )

    anjum_person = Person(name="Anjum", relationship="daughter")
    anjum_person.age = 6
    anjum_face = DetectedFace(
        bounding_box=(200, 150, 100, 100),
        center=(250.0, 200.0),
        confidence=0.95,
        person=anjum_person,
        is_known=True,
        identity_state=IdentityState.RECOGNIZED,
    )

    adult_person = Person(name="Palash", relationship="owner")
    adult_person.age = 30
    adult_face = DetectedFace(
        bounding_box=(200, 150, 100, 100),
        center=(250.0, 200.0),
        confidence=0.95,
        person=adult_person,
        is_known=True,
        identity_state=IdentityState.RECOGNIZED,
    )

    # Frame 1: Anjum detected -> consecutive_frames = 1, NOT yet in ANJUM_MODE
    with patch.object(brain.face_service, "detect_and_recognize", return_value=[anjum_face]):
        with patch.object(brain.face_service, "confirm_identity", return_value=[anjum_face]):
            brain.process_person_interaction(MagicMock())
    assert brain._anjum_consecutive_frames == 1
    assert brain.state.current_state != BehaviorState.ANJUM_MODE

    # Frame 2: Anjum detected -> consecutive_frames = 2, still NOT yet in ANJUM_MODE
    with patch.object(brain.face_service, "detect_and_recognize", return_value=[anjum_face]):
        with patch.object(brain.face_service, "confirm_identity", return_value=[anjum_face]):
            brain.process_person_interaction(MagicMock())
    assert brain._anjum_consecutive_frames == 2
    assert brain.state.current_state != BehaviorState.ANJUM_MODE

    # Frame 3: Anjum detected -> consecutive_frames = 3 -> ENTER ANJUM_MODE!
    with patch.object(brain.face_service, "detect_and_recognize", return_value=[anjum_face]):
        with patch.object(brain.face_service, "confirm_identity", return_value=[anjum_face]):
            brain.process_person_interaction(MagicMock())
    assert brain._anjum_consecutive_frames == 3
    assert brain.state.current_state == BehaviorState.ANJUM_MODE

    # Now adult Palash appears: Anjum mode must be exited immediately!
    with patch.object(brain.face_service, "detect_and_recognize", return_value=[adult_face]):
        with patch.object(brain.face_service, "confirm_identity", return_value=[adult_face]):
            brain.process_person_interaction(MagicMock())
    assert brain.state.current_state != BehaviorState.ANJUM_MODE
    assert brain._anjum_consecutive_frames == 0
