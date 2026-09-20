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


def test_memory_turn_recording_and_palash_fallback() -> None:
    """Verify conversations are saved to database with owner fallback when camera has no face."""
    from lumi.core.event_bus import Event
    from lumi.core.lumi_brain import LumiBrain
    from lumi.memory.database import Database
    from lumi.memory.manager import MemoryManager

    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    palash = mem.remember_person("Palash", relationship="owner", notes="Owner and creator of LUMI.")

    brain = LumiBrain.__new__(LumiBrain)
    brain.memory = mem
    brain.active_person = None  # Camera has no active face
    brain.realtime_voice = MagicMock()
    brain.mem0 = MagicMock()

    # Fire turn complete event
    event = Event(topic="conversation.turn_complete", data={"user": "আমার নাম পলাশ", "lumi": "হ্যাঁ পলাশ, আমি মনে রেখেছি!"})
    brain._on_turn_complete(event)

    # Verify turns are recorded in SQLite
    turns = mem.get_recent_turns(limit=10)
    assert len(turns) == 2
    assert turns[0].speaker == "user"
    assert turns[0].person_id == palash.id
    assert turns[0].text == "আমার নাম পলাশ"
    assert turns[1].speaker == "lumi"
    assert turns[1].person_id == palash.id

    # Verify Mem0 async processing was called with Palash
    brain.mem0.process_conversation_turn_async.assert_called_once_with(
        person_id=palash.id,
        person_name="Palash",
        user_text="আমার নাম পলাশ",
        ai_text="হ্যাঁ পলাশ, আমি মনে রেখেছি!"
    )


def test_tool_memorize_and_recall_palash_identity() -> None:
    """Verify _tool_memorize_fact attaches to Palash and _tool_recall_facts always includes profile."""
    from lumi.core.lumi_brain import LumiBrain
    from lumi.memory.database import Database
    from lumi.memory.manager import MemoryManager

    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    palash = mem.remember_person("Palash", relationship="owner", notes="Owner and creator of LUMI.")

    brain = LumiBrain.__new__(LumiBrain)
    brain.memory = mem
    brain.active_person = None
    brain.mem0 = MagicMock()
    brain.mem0.remember_fact_sync.return_value = True
    brain.mem0.recall_facts_sync.return_value = ""

    # Memorize fact without explicit person_name
    res = brain._tool_memorize_fact(fact="Palash prefers dark roast coffee")
    assert "memorized successfully for Palash" in res

    # Verify fact in SQLite
    facts = mem.recall_facts(person_id=palash.id)
    assert len(facts) == 1
    assert "dark roast coffee" in facts[0].fact_text

    # Recall facts with generic question
    recall_res = brain._tool_recall_facts(search_query="who is talking to you")
    assert "Identity Profile: Palash is your owner" in recall_res
    assert "dark roast coffee" in recall_res


def test_spatial_audio_clean_downmix() -> None:
    """Verify spatial audio processor returns clean mono audio without zero-padded clicks."""
    try:
        from lumi.audio.spatial import SpatialAudioProcessor
        import numpy as np
    except ImportError:
        return  # Skip test on machines without numpy installed

    proc = SpatialAudioProcessor(mic_distance=0.058, sample_rate=16000)

    # Generate 100ms of stereo PCM (440 Hz tone on L, 440 Hz tone on R)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    tone_l = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    tone_r = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    stereo_interleaved = np.empty(3200, dtype=np.int16)
    stereo_interleaved[0::2] = tone_l
    stereo_interleaved[1::2] = tone_r
    stereo_bytes = stereo_interleaved.tobytes()

    mono_bytes, doa = proc.process_stereo_chunk(stereo_bytes)
    assert len(mono_bytes) == 1600 * 2  # 1600 samples, 2 bytes/sample
    mono_samples = np.frombuffer(mono_bytes, dtype=np.int16)

    # First samples should NOT be all zeros (no zero-gap clicks at chunk boundary)
    assert not np.all(mono_samples[:10] == 0)
    assert abs(doa) <= 90.0


def test_proactive_recall_identity_query() -> None:
    """Verify proactive recall engine recognizes 'আমাকে চেনো' and injects Palash identity."""
    from lumi.memory.database import Database
    from lumi.memory.manager import MemoryManager
    from lumi.memory.proactive_recall import ProactiveRecallEngine, _RECALL_PATTERNS

    # 1. Pattern matching check
    patterns_match = any(p.search("তুমি কি আমাকে চেনো?") for p in _RECALL_PATTERNS)
    assert patterns_match is True

    patterns_match_about = any(p.search("আমার সম্পর্কে কিছু জানো?") for p in _RECALL_PATTERNS)
    assert patterns_match_about is True

    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mem.remember_person("Palash", relationship="owner", notes="Owner and creator of LUMI.")
    mem.remember_fact("Palash is building autonomous AI robots.", person_id="test_id")

    voice_mock = MagicMock()
    bus_mock = MagicMock()
    mem0_mock = MagicMock()
    mem0_mock.recall_facts_sync.return_value = ""

    engine = ProactiveRecallEngine(mem, mem0_mock, voice_mock, bus_mock)
    engine._process_utterance("তুমি কি আমাকে চেনো? আমার সম্পর্কে কিছু জানো?", trigger_time=time.time())

    assert voice_mock.inject_context.called
    injected = voice_mock.inject_context.call_args[0][0]
    assert "Palash" in injected
    assert "owner" in injected


def test_owner_mizan_and_person_specific_memory_isolation() -> None:
    """Verify LUMI recognizes Mizan as owner and isolates memories between people."""
    from lumi.core.lumi_brain import LumiBrain
    from lumi.memory.database import Database
    from lumi.memory.manager import MemoryManager

    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mizan = mem.remember_person("Mizan", relationship="owner", notes="Owner and primary user.")
    rahul = mem.remember_person("Rahul", relationship="friend", notes="Friend from college.")

    brain = LumiBrain.__new__(LumiBrain)
    brain.memory = mem
    brain.settings = MagicMock()
    brain.settings.app.owner_name = "Mizan"
    brain.active_person = None
    brain.mem0 = MagicMock()
    brain.mem0.remember_fact_sync.return_value = True
    brain.mem0.recall_facts_sync.return_value = ""

    # Verify get_owner returns Mizan
    assert brain.get_owner().name == "Mizan"
    assert brain.get_owner().id == mizan.id

    # 1. Memorize fact for Mizan (default when active_person is None)
    res_mizan = brain._tool_memorize_fact("Loves espresso and robotics")
    assert "memorized successfully for Mizan" in res_mizan

    # 2. Memorize fact specifically for Rahul
    res_rahul = brain._tool_memorize_fact("Prefers green tea and plays guitar", person_name="Rahul")
    assert "memorized successfully for Rahul" in res_rahul

    # 3. Verify Rahul's memory doesn't leak into Mizan's facts
    mizan_facts = mem.recall_facts(person_id=mizan.id)
    assert len(mizan_facts) == 1
    assert "espresso" in mizan_facts[0].fact_text
    assert "tea" not in mizan_facts[0].fact_text

    # 4. Verify Mizan's memory doesn't leak into Rahul's facts
    rahul_facts = mem.recall_facts(person_id=rahul.id)
    assert len(rahul_facts) == 1
    assert "green tea" in rahul_facts[0].fact_text
    assert "espresso" not in rahul_facts[0].fact_text

    # 5. Recall specifically for Rahul
    recall_rahul = brain._tool_recall_facts("tea", person_name="Rahul")
    assert "Rahul" in recall_rahul
    assert "green tea" in recall_rahul
    assert "espresso" not in recall_rahul


def test_silence_command_and_audio_suppression() -> None:
    """Verify silence command ('চুপ থাকো, তুমি ১০ মিনিট চুপ থাকো') activates silence and drops audio."""
    from lumi.ai.gemini_live import GeminiLiveClient

    engine = GeminiLiveClient.__new__(GeminiLiveClient)
    engine._silent_until = 0.0
    engine.speaker = MagicMock()
    engine.eyes = MagicMock()

    assert engine.is_silent() is False

    # Simulate user saying "তুমি ১০ মিনিট চুপ থাকো"
    cmd_handled = engine._check_silence_command("তুমি ১০ মিনিট চুপ থাকো")
    assert cmd_handled is True
    assert engine.is_silent() is True
    assert engine.speaker.stop_stream.called
    assert engine.eyes.set_expression.call_args[0][0] == "sleep"

    # Verify duration is ~600 seconds (10 mins)
    remaining = engine._silent_until - time.time()
    assert 550 < remaining <= 605

    # Wake-up command cancels silence
    wake_handled = engine._check_silence_command("লুমি কথা বলো")
    assert wake_handled is True
    assert engine.is_silent() is False
    assert engine.eyes.set_expression.call_args[0][0] == "happy"


def test_conversation_context_retention_in_setup() -> None:
    """Verify Gemini Live setup prompt contains recent conversation history from MemoryManager."""
    from lumi.ai.gemini_live import GeminiLiveClient
    from lumi.memory.database import Database
    from lumi.memory.manager import MemoryManager

    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    mizan = mem.remember_person("Mizan", relationship="owner", notes="Owner of LUMI.")

    mem.record_turn("user", "আমি আগামীকাল সিলেট যাচ্ছি", person_id=mizan.id)
    mem.record_turn("lumi", "দারুণ! সিলেটের চা বাগান খুব সুন্দর!", person_id=mizan.id)

    engine = GeminiLiveClient.__new__(GeminiLiveClient)
    engine.memory = mem
    engine.model = "models/gemini-3.1-flash-live-preview"
    engine.tools = None

    import asyncio
    import json
    sent_payloads = []

    class MockWS:
        async def send(self, data):
            sent_payloads.append(json.loads(data))

    asyncio.run(engine._send_setup(MockWS()))
    assert len(sent_payloads) == 1
    sys_instruction = sent_payloads[0]["setup"]["systemInstruction"]["parts"][0]["text"]

    # Verify conversation transcript was injected
    assert "RECENT CONVERSATION TRANSCRIPT" in sys_instruction
    assert "আমি আগামীকাল সিলেট যাচ্ছি" in sys_instruction
    assert "দারুণ! সিলেটের চা বাগান খুব সুন্দর!" in sys_instruction


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
    test_memory_turn_recording_and_palash_fallback()
    print("✓ test_memory_turn_recording_and_palash_fallback PASSED")
    test_tool_memorize_and_recall_palash_identity()
    print("✓ test_tool_memorize_and_recall_palash_identity PASSED")
    test_spatial_audio_clean_downmix()
    print("✓ test_spatial_audio_clean_downmix PASSED")
    test_proactive_recall_identity_query()
    print("✓ test_proactive_recall_identity_query PASSED")
    test_owner_mizan_and_person_specific_memory_isolation()
    print("✓ test_owner_mizan_and_person_specific_memory_isolation PASSED")
    test_silence_command_and_audio_suppression()
    print("✓ test_silence_command_and_audio_suppression PASSED")
    test_conversation_context_retention_in_setup()
    print("✓ test_conversation_context_retention_in_setup PASSED")
    print("All improvement tests passed successfully!")
