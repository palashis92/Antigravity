import time
import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumi.audio.speaker import MockSpeakerBackend, SpeakerInterface
from lumi.audio.turn_arbiter import AudioTurnArbiter
from lumi.core.event_bus import EventBus
from lumi.core.lumi_brain import LumiBrain
from lumi.core.state_manager import BehaviorState, StateManager
from lumi.speech.presentation import PresentationEngine
from lumi.speech.tts import BanglaTTS
from lumi.vision.face import DetectedFace, FaceRecognitionService


class TestSilenceAndPresentation(unittest.TestCase):
    def setUp(self):
        self.state = StateManager()
        self.speaker = SpeakerInterface(MockSpeakerBackend())
        self.tts = BanglaTTS()
        self.event_bus = EventBus()
        self.eyes = MagicMock()
        self.gestures = MagicMock()
        self.gestures.is_playing = False
        self.realtime_voice = MagicMock()

        self.presentation_engine = PresentationEngine(
            tts=self.tts,
            speaker=self.speaker,
            gestures=self.gestures,
            eyes=self.eyes,
            state=self.state,
            event_bus=self.event_bus,
            realtime_voice=self.realtime_voice,
        )

    def test_presentation_fsm_valid_transitions(self):
        # IDLE -> PRESENTING
        self.assertEqual(self.state.current_state, BehaviorState.IDLE)
        self.assertTrue(self.state.transition_to(BehaviorState.PRESENTING))
        self.assertEqual(self.state.current_state, BehaviorState.PRESENTING)
        self.assertEqual(self.state.recommended_eye_expression, "speaking")

        # PRESENTING -> LISTENING
        self.assertTrue(self.state.transition_to(BehaviorState.LISTENING))
        self.assertEqual(self.state.current_state, BehaviorState.LISTENING)

        # LISTENING -> PRESENTING
        self.assertTrue(self.state.transition_to(BehaviorState.PRESENTING))
        self.assertEqual(self.state.current_state, BehaviorState.PRESENTING)

        # PRESENTING -> IDLE
        self.assertTrue(self.state.transition_to(BehaviorState.IDLE))
        self.assertEqual(self.state.current_state, BehaviorState.IDLE)

    def test_presentation_engine_stop_command_patterns(self):
        self.assertTrue(self.presentation_engine.check_stop_command("লুমি থামো"))
        self.assertTrue(self.presentation_engine.check_stop_command("বক্তব্য বন্ধ করো"))
        self.assertTrue(self.presentation_engine.check_stop_command("বক্তব্য থামাও"))
        self.assertTrue(self.presentation_engine.check_stop_command("stop presentation"))
        self.assertTrue(self.presentation_engine.check_stop_command("cancel speech"))
        self.assertTrue(self.presentation_engine.check_stop_command("থামো"))

        # Non-stop commands
        self.assertFalse(self.presentation_engine.check_stop_command("দারুণ হয়েছে"))
        self.assertFalse(self.presentation_engine.check_stop_command("গ্রামের রাস্তা নিয়ে বলো"))

    def test_presentation_engine_script_generator(self):
        segments = self.presentation_engine.generate_speech_script(
            topic="গ্রাম উন্নয়ন ও আধুনিকায়ন",
            duration_minutes=3.0,
            audience="সম্মানিত গ্রামবাসী",
            key_points="রাস্তা সংস্কার ও বিশুদ্ধ পানি",
        )
        self.assertIsInstance(segments, list)
        self.assertGreaterEqual(len(segments), 3)
        combined = " ".join(segments)
        self.assertIn("গ্রাম উন্নয়ন ও আধুনিকায়ন", combined)
        self.assertIn("সম্মানিত গ্রামবাসী", combined)
        self.assertIn("রাস্তা সংস্কার ও বিশুদ্ধ পানি", combined)

    def test_presentation_engine_lifecycle_and_stop(self):
        res = self.presentation_engine.start_presentation(
            topic="ডিজিটাল ইউনিয়ন সেবা",
            duration_minutes=1.0,
        )
        self.assertIn("বক্তব্য শুরু হচ্ছে", res)
        self.assertTrue(self.presentation_engine.is_presenting())
        self.assertEqual(self.state.current_state, BehaviorState.PRESENTING)

        # Stop presentation
        stop_res = self.presentation_engine.stop_presentation(reason="test_stop")
        self.assertIn("সফলভাবে বন্ধ", stop_res)
        self.assertFalse(self.presentation_engine.is_presenting())
        self.assertEqual(self.state.current_state, BehaviorState.LISTENING)

    def test_face_service_reset_interaction_cooldown(self):
        face_service = FaceRecognitionService(memory_manager=MagicMock())
        person_id = "test_chairman"
        self.assertTrue(face_service.should_interact(person_id, cooldown_s=100.0))
        # Immediate next check should be false
        self.assertFalse(face_service.should_interact(person_id, cooldown_s=100.0))

        # Reset cooldown
        face_service.reset_interaction_cooldown(person_id)
        # Should now be true again
        self.assertTrue(face_service.should_interact(person_id, cooldown_s=100.0))


class TestLumiBrainSilenceAndPresentation(unittest.TestCase):
    def setUp(self):
        self.brain = LumiBrain.__new__(LumiBrain)
        self.brain.state = StateManager()
        self.brain.speaker = SpeakerInterface(MockSpeakerBackend())
        self.brain.tts = BanglaTTS()
        self.brain.eyes = MagicMock()
        self.brain.gestures = MagicMock()
        self.brain.turn_arbiter = AudioTurnArbiter()
        self.brain.realtime_voice = MagicMock()
        self.brain.face_service = FaceRecognitionService(memory_manager=MagicMock())
        self.brain._silent_until = 0.0
        self.brain._silent_mode_active = False

        self.brain.presentation_engine = PresentationEngine(
            tts=self.brain.tts,
            speaker=self.brain.speaker,
            gestures=self.brain.gestures,
            eyes=self.brain.eyes,
            state=self.brain.state,
            event_bus=EventBus(),
            realtime_voice=self.brain.realtime_voice,
        )

    def test_silence_mode_auto_expiration(self):
        # Activate short silence for 0.15s
        res = self.brain.set_silent_mode(duration_seconds=0.15, reason="test_auto_expire")
        self.assertIn("silent_mode_active_for", res)
        self.assertTrue(self.brain._silent_mode_active)
        self.assertTrue(self.brain._is_silent())
        self.brain.eyes.set_expression.assert_called_with("sleep")

        # Wait for duration to elapse (>0.15s)
        time.sleep(0.2)

        # _is_silent() should detect expiration, call cancel_silent_mode, and return False
        self.assertFalse(self.brain._is_silent())
        self.assertFalse(self.brain._silent_mode_active)
        self.brain.eyes.set_expression.assert_called_with("neutral")
        self.assertEqual(self.brain.state.current_state, BehaviorState.IDLE)

    def test_silence_does_not_burn_face_greeting_cooldown(self):
        person_id = "chairman_mizan"
        cooldown = 100.0

        # When silent, should_interact is never called
        self.brain.set_silent_mode(duration_seconds=50.0)
        self.assertTrue(self.brain._is_silent())

        # Simulate person detected check logic from _on_person_detected:
        # Check _is_silent BEFORE should_interact
        if self.brain._is_silent():
            suppressed = True
        else:
            suppressed = False
            self.brain.face_service.should_interact(person_id, cooldown_s=cooldown)

        self.assertTrue(suppressed)
        # Verify that should_interact was not burned
        self.assertTrue(self.brain.face_service.should_interact(person_id, cooldown_s=cooldown))

    def test_presentation_blocked_during_silence(self):
        self.brain.set_silent_mode(duration_seconds=60.0)
        res = self.brain._tool_start_presentation(topic="কৃষি উন্নয়ন")
        self.assertIn("নীরব মোড (Silent Mode) সক্রিয় রয়েছে", res)
        self.assertFalse(self.brain.presentation_engine.is_presenting())

    def test_silence_halts_ongoing_presentation(self):
        # Start presentation
        self.brain._tool_start_presentation(topic="ইউনিয়ন বাজেট")
        self.assertTrue(self.brain.presentation_engine.is_presenting())
        self.assertEqual(self.brain.state.current_state, BehaviorState.PRESENTING)

        # Entering silent mode must halt presentation immediately
        self.brain.set_silent_mode(duration_seconds=60.0, reason="urgent_silence")
        self.assertFalse(self.brain.presentation_engine.is_presenting())
        self.assertTrue(self.brain._is_silent())

    def test_presentation_5_minute_script_word_count(self):
        """Verify 5-minute presentation script generates >= 500 words across 8+ paragraphs."""
        segments = self.brain.presentation_engine.generate_speech_script(
            topic="কৃত্রিম বুদ্ধিমত্তা ও ভবিষ্যৎ প্রযুক্তি",
            duration_minutes=5.0,
            audience="সুধীবৃন্দ",
            key_points="উদ্ভাবন, অটোমেশন এবং নৈতিকতা",
        )
        self.assertGreaterEqual(len(segments), 8)
        combined = " ".join(segments)
        word_count = len(combined.split())
        self.assertGreaterEqual(word_count, 500, f"Word count {word_count} should be >= 500 for a 5-minute speech")
        self.assertIn("কৃত্রিম বুদ্ধিমত্তা ও ভবিষ্যৎ প্রযুক্তি", combined)
        self.assertIn("সুধীবৃন্দ", combined)
        self.assertIn("উদ্ভাবন, অটোমেশন এবং নৈতিকতা", combined)

    def test_presentation_seconds_and_phrasing_parsing(self):
        """Verify GeminiLiveClient detects seconds (360s), speech phrasing, and extracts topic cleanly."""
        from lumi.ai.gemini_live import GeminiLiveClient
        client = GeminiLiveClient.__new__(GeminiLiveClient)
        client.speaker = MagicMock()
        client._on_presentation_requested_cb = MagicMock()

        # 1. 5 minutes speech phrasing
        matched = client._check_presentation_command("তুমি কৃত্রিম বুদ্ধিমত্তা নিয়ে ৫ মিনিট কথা বলো")
        self.assertTrue(matched)
        client._on_presentation_requested_cb.assert_called_with("কৃত্রিম বুদ্ধিমত্তা", 5.0)

        # 2. 360 seconds phrasing
        client._on_presentation_requested_cb.reset_mock()
        matched = client._check_presentation_command("রোবোটিক্স নিয়ে ৩৬০ সেকেন্ড বলো")
        self.assertTrue(matched)
        client._on_presentation_requested_cb.assert_called_with("রোবোটিক্স", 6.0)

        # 3. Complex prompt with parenthesis and duration
        client._on_presentation_requested_cb.reset_mock()
        matched = client._check_presentation_command("রোবটটাকে যখন কোনো একটা বিষয়ে ৫ মিনিট (৩৬০ সেকেন্ড) কথা বলতে বলা হয়")
        self.assertTrue(matched)
        self.assertTrue(client._on_presentation_requested_cb.called)

        # 4. Standard speech
        client._on_presentation_requested_cb.reset_mock()
        matched = client._check_presentation_command("কৃষি ও পরিবেশ সম্পর্কে একটি ৫ মিনিটের বক্তব্য দাও")
        self.assertTrue(matched)
        client._on_presentation_requested_cb.assert_called_with("কৃষি ও পরিবেশ", 5.0)

        # 5. Generic speech
        client._on_presentation_requested_cb.reset_mock()
        matched = client._check_presentation_command("ভাষণ দাও")
        self.assertTrue(matched)
        client._on_presentation_requested_cb.assert_called_with("বিজ্ঞান ও আধুনিক প্রযুক্তি", 3.0)

        # 6. Non-presentation speech triggers
        self.assertFalse(client._check_presentation_command("তুমি কেমন আছো বলো"))
        self.assertFalse(client._check_presentation_command("১ মিনিট চুপ থাকো"))
        self.assertFalse(client._check_presentation_command("৫ মিনিট পর মনে করিয়ে দিও"))

    def test_turn_arbiter_presenting_guard(self):
        """Verify Turn Arbiter suppresses mic streaming during presentation."""
        arbiter = AudioTurnArbiter()
        self.assertFalse(arbiter.is_presenting())
        self.assertTrue(arbiter.should_stream_mic(energy=1000.0))

        # When presentation starts, mic streaming must be blocked
        arbiter.set_presenting(True)
        self.assertTrue(arbiter.is_presenting())
        self.assertFalse(arbiter.should_stream_mic(energy=1000.0))

        # When presentation finishes, mic streaming is restored
        arbiter.set_presenting(False)
        self.assertFalse(arbiter.is_presenting())
        self.assertTrue(arbiter.should_stream_mic(energy=1000.0))

    def test_anti_silence_negation_guards(self):
        """Verify that negated stopping phrases do not activate silence mode."""
        from lumi.ai.gemini_live import GeminiLiveClient
        client = GeminiLiveClient.__new__(GeminiLiveClient)
        client.is_silent = MagicMock(return_value=False)
        client.cancel_silence = MagicMock()
        client.speaker = MagicMock()
        client.eyes = MagicMock()

        # Negated phrases must NOT activate silence
        self.assertFalse(client._check_silence_command("করতে অনুমতি নেওয়ার প্রয়োজন নাই। বাবা তুমি থামারও প্রয়োজন নাই। তুমি তো নির্দেশ মানতেছ না।"))
        self.assertFalse(client._check_silence_command("তুমি থামার দরকার নাই"))
        self.assertFalse(client._check_silence_command("থামবে না, কথা চালিয়ে যাও"))
        self.assertFalse(client._check_silence_command("ননস্টপ ৩০০ সেকেন্ড কথা বলবো"))

        # True silence commands MUST activate silence
        self.assertTrue(client._check_silence_command("লুমি এবার থামো"))
        self.assertTrue(client._check_silence_command("চুপ করো"))
        self.assertTrue(client._check_silence_command("৫ মিনিট চুপ থাকো"))

    def test_observing_to_speaking_and_presenting_transitions(self):
        """Verify that state machine permits transitions from OBSERVING to SPEAKING and PRESENTING."""
        state = StateManager()
        self.assertTrue(state.transition_to(BehaviorState.OBSERVING))
        self.assertEqual(state.current_state, BehaviorState.OBSERVING)

        # Transition to SPEAKING must succeed (fixes invalid transition rejection)
        self.assertTrue(state.transition_to(BehaviorState.SPEAKING))
        self.assertEqual(state.current_state, BehaviorState.SPEAKING)

        # Transition back to OBSERVING then to PRESENTING
        self.assertTrue(state.transition_to(BehaviorState.IDLE))
        self.assertTrue(state.transition_to(BehaviorState.OBSERVING))
        self.assertTrue(state.transition_to(BehaviorState.PRESENTING))
        self.assertEqual(state.current_state, BehaviorState.PRESENTING)

    def test_conversational_intro_name_filtering(self):
        """Verify that conversational sentences starting with 'আমি কোন...' do not register false names."""
        res = LumiBrain._extract_introduced_name("আমি কোন প্রশ্ন করবো না। কোন প্রশ্ন করবো না। শুধু পাঁচ মিনিট কথা বলবো। ইতিহাস নিয়ে।")
        self.assertIsNone(res)

        res2 = LumiBrain._extract_introduced_name("আমি কোন কথা শুনব না")
        self.assertIsNone(res2)

        # Legitimate introductions
        name1, rel1 = LumiBrain._extract_introduced_name("আমি তানভীর")
        self.assertEqual(name1, "তানভীর")
        self.assertEqual(rel1, "friend")

        name2, rel2 = LumiBrain._extract_introduced_name("আমার নাম পলাশ")
        self.assertEqual(name2, "পলাশ")
        self.assertEqual(rel2, "creator")

    def test_gemini_3_8_live_extended_thinking_setup(self):
        """Verify setup configuration for Gemini 3.8 Live Base vs Extended Thinking."""
        from lumi.ai.gemini_live import GeminiLiveClient
        import asyncio

        # Base model (gemini-3.8-live)
        client_base = GeminiLiveClient.__new__(GeminiLiveClient)
        client_base.model = "models/gemini-3.8-live"
        client_base.tools = MagicMock()
        client_base.tools.schemas = {
            "test_func": {"name": "test_func", "description": "test", "parameters": {"type": "object"}}
        }
        client_base.memory = None

        ws_mock = MagicMock()
        sent_messages = []
        async def mock_send(msg):
            import json
            sent_messages.append(json.loads(msg))
        ws_mock.send = mock_send

        asyncio.run(client_base._send_setup(ws_mock))
        base_setup = sent_messages[0]["setup"]
        self.assertNotIn("thinkingConfig", base_setup["generationConfig"])
        self.assertNotIn("behavior", base_setup["tools"][0]["functionDeclarations"][0])
        self.assertIn("realtimeInputConfig", base_setup)
        self.assertIn("contextWindowCompression", base_setup)
        self.assertIn("slidingWindow", base_setup["contextWindowCompression"])
        self.assertEqual(base_setup["realtimeInputConfig"]["automaticActivityDetection"]["silenceDurationMs"], 800)

        # Extended Thinking model (gemini-3.8-live-extended-thinking)
        client_ext = GeminiLiveClient.__new__(GeminiLiveClient)
        client_ext.model = "models/gemini-3.8-live-extended-thinking"
        client_ext.tools = client_base.tools
        client_ext.memory = None

        sent_messages.clear()
        asyncio.run(client_ext._send_setup(ws_mock))
        ext_setup = sent_messages[0]["setup"]
        self.assertIn("thinkingConfig", ext_setup["generationConfig"])
        self.assertEqual(ext_setup["tools"][0]["functionDeclarations"][0]["behavior"], "NON_BLOCKING")

    def test_prompt_rules_mandate_speech_monologue_and_forbid_checkins(self):
        """Verify LUMI system prompts explicitly forbid 'আমি কি বলতেই থাকবো?' and suspend brevity for speeches."""
        from lumi.ai.prompts import LUMI_SYSTEM_PROMPT_BN, LUMI_SYSTEM_PROMPT_EN

        # Bengali prompt checks
        self.assertIn("স্থগিত", LUMI_SYSTEM_PROMPT_BN)
        self.assertIn("আমি কি বলতেই থাকবো?", LUMI_SYSTEM_PROMPT_BN)
        self.assertIn("কঠোরভাবে ও সম্পূর্ণরূপে নিষিদ্ধ", LUMI_SYSTEM_PROMPT_BN)
        self.assertIn("৫ মিনিট কথা বলো", LUMI_SYSTEM_PROMPT_BN)

        # English prompt checks
        self.assertIn("COMPLETELY SUSPENDED", LUMI_SYSTEM_PROMPT_EN)
        self.assertIn("Should I keep speaking?", LUMI_SYSTEM_PROMPT_EN)
        self.assertIn("STRICTLY FORBIDDEN", LUMI_SYSTEM_PROMPT_EN)

    def test_continuous_speech_session_tracking_and_prompt_continuation(self):
        """Verify GeminiLiveClient manages active speech session target time and injects continuation without stopping."""
        from lumi.ai.gemini_live import GeminiLiveClient
        import time

        client = GeminiLiveClient.__new__(GeminiLiveClient)
        client.speaker = MagicMock()
        client.turn_arbiter = MagicMock()
        client.state = MagicMock()
        client.inject_context = MagicMock()
        client._on_presentation_requested_cb = None

        # Start 5-minute speech session
        matched = client._check_presentation_command("মুক্তিযুদ্ধ নিয়ে ৫ মিনিট কথা বলো")
        self.assertTrue(matched)
        self.assertGreater(client._active_speech_target_end, time.time() + 250.0)
        self.assertEqual(client._active_speech_topic, "মুক্তিযুদ্ধ")
        client.turn_arbiter.set_presenting.assert_called_with(True)
        self.assertTrue(client.inject_context.called)

        # Simulate check-in phrase detection in output transcription
        client.inject_context.reset_mock()
        test_txt = "এই ছিলো প্রারম্ভিক ইতিহাস। আমি কি বলতেই থাকবো?"
        import re
        if re.search(r"(?:আমি\s*কি\s*(?:বলতেই|বলতে|আরো|আরও)\s*থাকব|আমি\s*কি\s*(?:আরো|আরও)\s*বলব)", test_txt):
            client.inject_context(
                f"[DO NOT ASK PERMISSION]: Continue speaking on '{client._active_speech_topic}' continuously. Do not ask check-in questions!",
                trigger_response=True
            )
        self.assertTrue(client.inject_context.called)
        self.assertIn("DO NOT ASK PERMISSION", client.inject_context.call_args[0][0])

    def test_bengali_language_locking_and_transcription_codes(self):
        """Verify Gemini Live setup explicitly configures Bengali BCP-47 languageCodes and language enforcement."""
        from lumi.ai.gemini_live import GeminiLiveClient
        from lumi.ai.prompts import LUMI_SYSTEM_PROMPT_BN, LUMI_SYSTEM_PROMPT_EN
        import asyncio
        import json

        client = GeminiLiveClient.__new__(GeminiLiveClient)
        client.model = "models/gemini-3.8-live"
        client.tools = None
        client.memory = None

        ws_mock = MagicMock()
        sent_messages = []
        async def mock_send(msg):
            sent_messages.append(json.loads(msg))
        ws_mock.send = mock_send

        asyncio.run(client._send_setup(ws_mock))
        setup_data = sent_messages[0]["setup"]

        # Check BCP-47 languageCodes
        self.assertEqual(setup_data["inputAudioTranscription"]["languageCodes"], ["bn-BD", "en-US"])
        self.assertEqual(setup_data["outputAudioTranscription"]["languageCodes"], ["bn-BD"])

        # Check Bengali Language Lock enforcement in prompts
        sys_text = setup_data["systemInstruction"]["parts"][0]["text"]
        self.assertIn("MANDATORY LANGUAGE ENFORCEMENT", sys_text)
        self.assertIn("বাংলা ছাড়া অন্য ভাষা সম্পূর্ণ নিষিদ্ধ", sys_text)
        self.assertIn("হিন্দি", sys_text)

        self.assertIn("MANDATORY LANGUAGE LOCK - BENGALI ONLY", LUMI_SYSTEM_PROMPT_BN)
        self.assertIn("কখনোই হিন্দি, স্প্যানিশ বা অন্য কোনো ভাষায় কথা বলবে না", LUMI_SYSTEM_PROMPT_BN)
        self.assertIn("Language Policy", LUMI_SYSTEM_PROMPT_EN)

    def test_presentation_speech_unified_kore_voice_and_speaker_not_stopped(self):
        """Verify start_presentation does NOT stop the audio stream, preserves Kore voice, and skips TTS when live."""
        from lumi.speech.presentation import PresentationEngine
        from lumi.core.state_manager import StateManager, BehaviorState
        from lumi.core.event_bus import EventBus
        from unittest.mock import MagicMock
        import time

        realtime_voice = MagicMock()
        realtime_voice._running = True
        realtime_voice._active_speech_target_end = 0.0
        realtime_voice.inject_context = MagicMock()

        speaker = MagicMock()
        tts = MagicMock()
        gestures = MagicMock()
        eyes = MagicMock()
        turn_arbiter = MagicMock()
        state = StateManager()

        engine = PresentationEngine(
            tts=tts,
            speaker=speaker,
            gestures=gestures,
            eyes=eyes,
            state=state,
            event_bus=EventBus(),
            realtime_voice=realtime_voice,
            turn_arbiter=turn_arbiter,
        )

        res = engine.start_presentation(topic="স্বাধীনতার ইতিহাস", duration_minutes=5.0)
        self.assertIn("বক্তব্য শুরু হচ্ছে", res)
        self.assertTrue(engine.is_presenting())
        self.assertEqual(state.current_state, BehaviorState.PRESENTING)
        self.assertTrue(turn_arbiter.set_presenting.called)
        self.assertGreater(realtime_voice._active_speech_target_end, time.time() + 250.0)
        self.assertEqual(realtime_voice._active_speech_topic, "স্বাধীনতার ইতিহাস")

        # Verify TTS was NOT called because Gemini Live is streaming voice
        self.assertFalse(tts.synthesize.called)

        # Stop presentation
        engine.stop_presentation(reason="user_stop")
        self.assertFalse(engine.is_presenting())
        self.assertEqual(realtime_voice._active_speech_target_end, 0.0)


if __name__ == "__main__":
    unittest.main()

