"""Unit tests for Music Player and WhatsApp Integration."""

import os
import tempfile
import unittest
from pathlib import Path

from lumi.audio.music_player import MusicPlayer
from lumi.audio.speaker import MockSpeakerBackend, SpeakerInterface
from lumi.integrations.message_polisher import refine_whatsapp_message
from lumi.integrations.whatsapp import WhatsAppClient


class TestMusicPlayer(unittest.TestCase):
    def setUp(self):
        self.speaker = SpeakerInterface(MockSpeakerBackend())
        self.temp_dir = tempfile.TemporaryDirectory()
        self.player = MusicPlayer(speaker=self.speaker, music_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_filename(self):
        cleaned = self.player._sanitize_filename("Stereo/Love: Part 1?*")
        self.assertNotIn("/", cleaned)
        self.assertNotIn(":", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("*", cleaned)

    def test_cached_playback(self):
        # Create a dummy audio file in cache
        cached_song = Path(self.temp_dir.name) / "Stereo Love.mp3"
        cached_song.write_bytes(b"RIFFdummydataWAVEfmt ")

        self.assertFalse(self.player.is_playing)

        # Play cached song
        success, msg = self.player.play("stereo love")
        self.assertTrue(success)
        self.assertTrue(self.player.is_playing)
        self.assertIn("Stereo Love", self.player.current_track)

        # Stop playback
        self.player.stop()
        self.assertFalse(self.player.is_playing)
        self.assertIsNone(self.player.current_track)


class TestWhatsAppIntegration(unittest.TestCase):
    def setUp(self):
        self.client = WhatsAppClient()

    def test_phone_formatting(self):
        # BD local 11 digits
        self.assertEqual(self.client.format_phone("01712345678"), "8801712345678")
        # With spaces and dashes
        self.assertEqual(self.client.format_phone("+880 1712-345678"), "8801712345678")
        # Already standard
        self.assertEqual(self.client.format_phone("8801712345678"), "8801712345678")
        # International number
        self.assertEqual(self.client.format_phone("+1 555-123-4567"), "15551234567")

    def test_simulation_mode_send_text(self):
        success, msg = self.client.send_text("01712345678", "কালকে মিটিং আছে।")
        self.assertTrue(success)
        self.assertIn("8801712345678", msg)

    def test_simulation_mode_send_document(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy pdf content")
            pdf_path = f.name

        try:
            success, msg = self.client.send_document("01812345678", pdf_path, caption="Meeting Report")
            self.assertTrue(success)
            self.assertIn("8801812345678", msg)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_message_polisher(self):
        # Test fallback / execution of polisher
        raw = "কালকে আসতে পারব না"
        polished = refine_whatsapp_message(raw, recipient_name="রহিম", relationship="কলিগ")
        self.assertTrue(len(polished) > 0)
        print(f"\n[Test Polish]: '{raw}' -> '{polished}'\n")


if __name__ == "__main__":
    unittest.main()
