"""Unit and Integration Tests for Phase 1 Hardware, Drivers, and Telemetry."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumi.audio.mic import SystemMicBackend
from lumi.audio.speaker import I2SSpeakerBackend
from lumi.core.telemetry import TelemetryLogger, get_telemetry
from lumi.hardware.display_driver import GC9A01DisplayDriver
from lumi.hardware.mocks import MockServoDriver
from lumi.motion.servo_controller import ServoController


def test_display_driver_single_display_ce0_only() -> None:
    """[H2] Verify single_display=True only initializes CE0 and ignores CE1."""
    driver = GC9A01DisplayDriver(single_display=True)
    assert driver.single_display is True
    # On non-Pi or systems without spidev, initialize returns True (graceful fallback)
    res = driver.initialize()
    assert res is True
    assert driver._gpio_right is None
    assert driver._spi_right is None


def test_display_driver_dual_display_flag() -> None:
    """[H2] Verify single_display=False preserves dual-display mode flag."""
    driver = GC9A01DisplayDriver(single_display=False)
    assert driver.single_display is False


def test_speaker_wm8960_detection() -> None:
    """[H3] Verify I2SSpeakerBackend detects ReSpeaker 2-Mics (WM8960 / seeed) soundcard."""
    fake_aplay_output = (
        "**** List of PLAYBACK Hardware Devices ****\n"
        "card 0: bcm2835_hdmi [bcm2835 HDMI 1], device 0: bcm2835 HDMI 1 [bcm2835 HDMI 1]\n"
        "card 1: seeed2micvoicec [seeed-2mic-voicecard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 []\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["aplay", "-l"], returncode=0, stdout=fake_aplay_output, stderr=""
        )
        backend = I2SSpeakerBackend(alsa_device="default")
        assert backend.alsa_device == "plughw:1,0"


def test_mic_wm8960_detection() -> None:
    """[H3] Verify SystemMicBackend detects ReSpeaker 2-Mics capture device."""
    fake_arecord_output = (
        "**** List of CAPTURE Hardware Devices ****\n"
        "card 1: seeed2micvoicec [seeed-2mic-voicecard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 []\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["arecord", "-l"], returncode=0, stdout=fake_arecord_output, stderr=""
        )
        backend = SystemMicBackend(alsa_device="default")
        assert backend.alsa_device == "plughw:1,0"


def test_servo_slew_rate_limiter() -> None:
    """[H1] Verify multi-channel movements respect slew-rate limits and finish accurately."""
    driver = MockServoDriver()
    ctrl = ServoController(driver)
    ctrl.initialize()

    # Move head from 0 to 70 deg with 0.1s request -> Slew rate clamp will enforce >= 70/360s (~0.194s)
    targets = {"head_pan": 70.0}
    t0 = time.time()
    ctrl.move_multiple(targets, duration_s=0.1)
    elapsed = time.time() - t0

    assert ctrl.current_angles["head_pan"] == 70.0
    assert elapsed >= 0.15  # Clamped above original 0.1s duration to limit velocity


def test_telemetry_logger_non_blocking() -> None:
    """[TEL] Verify TelemetryLogger records metrics and events to JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "telemetry.jsonl")
        logger = TelemetryLogger(log_path=log_file)

        logger.record("TEL-01", 34.5, unit="ms", context="barge_in")
        logger.record_latency("cloud_tts", 210.0, context="gemini")
        logger.record_event("watchdog_reset", context="fsm")

        time.sleep(0.3)
        logger.stop()

        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f.readlines() if line.strip()]

        assert len(lines) >= 3
        assert lines[0]["metric"] == "TEL-01"
        assert lines[0]["val"] == 34.5
        assert lines[1]["metric"] == "cloud_tts"
        assert lines[2]["event"] == "watchdog_reset"


if __name__ == "__main__":
    test_display_driver_single_display_ce0_only()
    test_display_driver_dual_display_flag()
    test_speaker_wm8960_detection()
    test_mic_wm8960_detection()
    test_servo_slew_rate_limiter()
    test_telemetry_logger_non_blocking()
    print("All Phase 1 tests passed!")
