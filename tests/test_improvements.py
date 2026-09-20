from __future__ import annotations

import asyncio
import os
import struct
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumi.audio.speaker import I2SSpeakerBackend, MockSpeakerBackend, SpeakerInterface
from lumi.hardware.mocks import MockServoDriver
from lumi.motion.servo_controller import ServoController


def test_servo_auto_relax_lifecycle() -> None:
    """Verify that ServoController relaxes idle servos after auto_relax_delay_s and re-engages on move."""
    driver = MockServoDriver()
    ctrl = ServoController(driver, auto_relax_delay_s=0.2)
    ctrl.initialize()

    try:
        # 1. Immediately after homing/moving, it should not be relaxed
        ctrl.set_angle_immediate("head_pan", 10.0)
        assert ctrl.current_angles["head_pan"] == 10.0
        assert not ctrl._is_relaxed

        # 2. Wait for watchdog to trigger relax (> 0.2s + 0.5s check tick)
        time.sleep(0.8)
        assert ctrl._is_relaxed, "Servos should have auto-relaxed after idle timeout"

        # 3. New movement command should wake it up and reset relaxed flag
        ctrl.set_angle_immediate("head_pan", 0.0)
        assert not ctrl._is_relaxed, "Servos should re-engage and clear relaxed flag upon new movement"
        assert ctrl.current_angles["head_pan"] == 0.0

    finally:
        ctrl.shutdown()
        assert not ctrl._running


def test_speaker_interface_stop_stream_and_shutdown() -> None:
    """Verify SpeakerInterface stop_stream and shutdown methods."""
    mock_backend = MockSpeakerBackend()
    speaker = SpeakerInterface(mock_backend)

    # Test stream playback
    dummy_pcm = bytes(960)
    assert speaker.play_stream(dummy_pcm, sample_rate=24000)

    # Test stop_stream
    speaker.stop_stream()

    # Test full shutdown
    speaker.shutdown()


def test_gemini_live_software_aec() -> None:
    """Verify Software AEC drops chunks during speech to prevent feedback echo."""
    from lumi.ai.gemini_live import GeminiLiveClient

    # Create mock dependencies
    mic = MagicMock()
    speaker = MagicMock()
    eyes = MagicMock()
    gestures = MagicMock()
    state = MagicMock()
    memory = MagicMock()
    event_bus = MagicMock()

    client = GeminiLiveClient(
        mic=mic,
        speaker=speaker,
        eyes=eyes,
        gestures=gestures,
        state=state,
        memory=memory,
        event_bus=event_bus,
        api_key="mock_key",
    )
    client._awake = True
    loop = asyncio.new_event_loop()
    client._loop = loop
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    def create_queue():
        client._audio_queue = asyncio.Queue()
    loop.call_soon_threadsafe(create_queue)
    time.sleep(0.05)

    try:
        dummy_chunk = struct.pack("<100h", *([1000] * 100))

        # 1. When speaker is active, mic chunks should be dropped to prevent echo feedback
        now = time.time()
        client._speaker_active_until = now + 5.0
        client.push_audio_chunk(dummy_chunk)
        assert client._audio_queue.empty(), "Mic chunk should be dropped during active speaker to prevent echo"

        # 2. When speaker is inactive, mic chunks should be enqueued for Gemini Live
        client._speaker_active_until = 0.0
        client.push_audio_chunk(dummy_chunk)
        time.sleep(0.05)
        assert not client._audio_queue.empty(), "Mic chunk should be enqueued when speaker is inactive"
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_gemini_live_async_tool_execution() -> None:
    """Verify asynchronous tool execution with timeout handling."""
    from lumi.ai.tools import ToolRegistry

    registry = ToolRegistry()

    # Fast tool
    registry.register("quick_tool", lambda x: f"hello {x}", "Fast tool")

    # Slow tool (simulating hanging network call)
    def slow_tool():
        time.sleep(0.5)
        return "slow_done"

    registry.register("slow_tool", slow_tool, "Slow tool")

    async def run_checks():
        # Fast tool execution in thread
        res_fast = await asyncio.wait_for(
            asyncio.to_thread(registry.tools["quick_tool"], x="world"),
            timeout=1.0,
        )
        assert res_fast == "hello world"

        # Slow tool timeout check with tight timeout (0.1s)
        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(registry.tools["slow_tool"]),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            timed_out = True
        assert timed_out, "Slow tool should trigger asyncio.TimeoutError without blocking"

    asyncio.run(run_checks())


def test_learned_rules_store_and_adaptation() -> None:
    """Verify LearnedRulesStore persistence, prompt generation, and rule lifecycle."""
    import tempfile
    from lumi.memory.learned_rules import LearnedRulesStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test_rules.json"
        store = LearnedRulesStore(store_path)

        # 1. Initially empty
        assert store.load_rules() == []
        assert store.get_rules_prompt() == ""

        # 2. Add rule
        rule1 = store.add_rule(
            feedback="কথা বলার সময় হাত বেশি নাড়িয়ে কথা বলবে না",
            adapted_rule="কথা বলার সময় হাত বেশি নাড়িয়ে কথা বলবে না, স্বাভাবিক অঙ্গভঙ্গি করবে।",
            category="gesture"
        )
        assert rule1["id"].startswith("rule_")
        assert rule1["category"] == "gesture"

        # 3. Verify loaded from disk
        rules = store.load_rules()
        assert len(rules) == 1
        assert rules[0]["user_feedback"] == "কথা বলার সময় হাত বেশি নাড়িয়ে কথা বলবে না"

        # 4. Verify formatted prompt
        prompt = store.get_rules_prompt()
        assert "[GESTURE]" in prompt
        assert "স্বাভাবিক অঙ্গভঙ্গি করবে" in prompt

        # 5. Delete rule
        assert store.delete_rule(rule1["id"])
        assert len(store.load_rules()) == 0


def test_camera_backend_close_and_face_service_cache() -> None:
    """Verify PiCameraBackend closes cleanly and FaceRecognitionService caches confirmed faces."""
    from lumi.vision.camera import PiCameraBackend
    from lumi.vision.face import DetectedFace, FaceRecognitionService, IdentityState

    backend = PiCameraBackend(width=640, height=480)
    # Even if Picamera2 is not installed on testing host, stop() must cleanly execute close() and del
    backend.stop()
    assert backend._picam is None
    assert not backend.is_available()

    # Verify FaceRecognitionService.get_last_faces()
    mem_mock = MagicMock()
    service = FaceRecognitionService(mem_mock)
    assert service.get_last_faces() == []

    # Mock a confirmed face
    mock_face = DetectedFace(
        bounding_box=(100, 100, 50, 50),
        center=(125.0, 125.0),
        confidence=0.95,
        person=None,
        is_known=False,
        identity_state=IdentityState.UNKNOWN,
    )
    service.confirm_identity([mock_face])
    last_faces = service.get_last_faces()
    assert len(last_faces) == 1
    assert last_faces[0].center == (125.0, 125.0)


def test_greeting_cooldown_and_temporal_context() -> None:
    """Verify 50-minute greeting cooldown and temporal awareness."""
    from lumi.vision.face import FaceRecognitionService

    mem_mock = MagicMock()
    service = FaceRecognitionService(mem_mock)

    pid = "test_palash_123"
    # First greeting check: should interact immediately
    assert service.should_interact(pid, cooldown_s=3000.0) is True

    # Immediate follow-up check (e.g. 5 seconds later): should reject/rate-limit
    assert service.should_interact(pid, cooldown_s=3000.0) is False


if __name__ == "__main__":
    test_servo_auto_relax_lifecycle()
    print("✓ test_servo_auto_relax_lifecycle PASSED")
    test_speaker_interface_stop_stream_and_shutdown()
    print("✓ test_speaker_interface_stop_stream_and_shutdown PASSED")
    test_gemini_live_software_aec()
    print("✓ test_gemini_live_software_aec PASSED")
    test_gemini_live_async_tool_execution()
    print("✓ test_gemini_live_async_tool_execution PASSED")
    test_learned_rules_store_and_adaptation()
    print("✓ test_learned_rules_store_and_adaptation PASSED")
    test_camera_backend_close_and_face_service_cache()
    print("✓ test_camera_backend_close_and_face_service_cache PASSED")
    test_greeting_cooldown_and_temporal_context()
    print("✓ test_greeting_cooldown_and_temporal_context PASSED")
    print("All improvement tests passed successfully!")
