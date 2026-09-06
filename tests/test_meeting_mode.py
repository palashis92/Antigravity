"""Unit test for Meeting Mode functionality."""

import os
import tempfile
import unittest

from lumi.audio.speaker import MockSpeakerBackend, SpeakerInterface
from lumi.core.state_manager import BehaviorState, StateManager
from lumi.meeting.manager import MeetingManager, MeetingSession, MeetingUtterance
from lumi.memory.database import Database


class TestMeetingMode(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db = Database(self.temp_db.name)
        self.meeting_manager = MeetingManager(self.db)
        self.state_manager = StateManager()
        self.speaker = SpeakerInterface(MockSpeakerBackend())

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except Exception:
                pass

    def test_state_transitions(self):
        # Initial state should be IDLE
        self.assertEqual(self.state_manager.current_state, BehaviorState.IDLE)

        # Transition to MEETING
        success = self.state_manager.transition_to(BehaviorState.MEETING, reason="test_start")
        self.assertTrue(success)
        self.assertEqual(self.state_manager.current_state, BehaviorState.MEETING)
        self.assertEqual(self.state_manager.recommended_eye_expression, "listening")

        # Transition back to IDLE
        success = self.state_manager.transition_to(BehaviorState.IDLE, reason="test_stop")
        self.assertTrue(success)
        self.assertEqual(self.state_manager.current_state, BehaviorState.IDLE)

    def test_speaker_muting(self):
        self.assertFalse(self.speaker.is_muted)
        self.speaker.set_muted(True)
        self.assertTrue(self.speaker.is_muted)

        # Play should return False when muted
        res = self.speaker.play_stream(b"\x00" * 100)
        self.assertFalse(res)

        self.speaker.set_muted(False)
        self.assertFalse(self.speaker.is_muted)
        res = self.speaker.play_stream(b"\x00" * 100)
        self.assertTrue(res)

    def test_meeting_lifecycle_and_db(self):
        self.assertFalse(self.meeting_manager.is_meeting_active())

        # Start meeting
        session = self.meeting_manager.start_meeting(title="প্রজেক্ট প্ল্যানিং মিটিং")
        self.assertTrue(self.meeting_manager.is_meeting_active())
        self.assertEqual(session.title, "প্রজেক্ট প্ল্যানিং মিটিং")

        # Add utterances
        self.meeting_manager.add_utterance(
            speaker="পলাশ",
            text="আমাদের আগামী মাসের রোডম্যাপ নির্ধারণ করতে হবে।",
            doa_deg=0.0,
        )
        self.meeting_manager.add_utterance(
            speaker="বক্তা (ডানদিক)",
            text="মার্কেটিং বাজেট কত রাখা হবে?",
            doa_deg=35.0,
        )
        self.meeting_manager.add_utterance(
            speaker="পলাশ",
            text="মার্কেটিংয়ের জন্য ৫০ হাজার টাকা বরাদ্দ থাকবে।",
            doa_deg=0.0,
        )

        active = self.meeting_manager.get_active_meeting()
        self.assertIsNotNone(active)
        self.assertEqual(len(active.utterances), 3)
        self.assertIn("পলাশ", active.participants)

        transcript = active.to_transcript_text()
        self.assertIn("আমাদের আগামী মাসের রোডম্যাপ", transcript)
        self.assertIn("বক্তা (ডানদিক)", transcript)

        # Stop meeting
        stopped_session = self.meeting_manager.stop_meeting()
        self.assertFalse(self.meeting_manager.is_meeting_active())
        self.assertIsNotNone(stopped_session)
        self.assertIsNotNone(stopped_session.ended_at)

        # Verify persisted in SQLite
        new_manager = MeetingManager(self.db)
        last_session = new_manager.get_last_meeting()
        self.assertIsNotNone(last_session)
        self.assertEqual(last_session.id, session.id)
        self.assertEqual(len(last_session.utterances), 3)

        # Test analysis formatting
        analysis = new_manager.analyze_meeting(last_session)
        self.assertTrue(len(analysis) > 0)
        print("\n--- Generated Test Analysis ---\n", analysis[:300], "...\n")


if __name__ == "__main__":
    unittest.main()
