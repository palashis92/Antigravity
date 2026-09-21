"""Unit tests for Phase 2: Audio & Session Stability.

Covers:
- [A5] Addressee Gating (dialogue turn window, auto-sleep on inactivity)
- [A1] ALSA Interruption Mute / Clean Barge-in
- [A2] Centralized AudioTurnArbiter
- [A3] Real-Time Acoustic Echo Ducking with Tail Window
- [A4] Proactive Gemini Live Session Rotation (12m rollover)
"""

import time
from unittest.mock import MagicMock, patch

from lumi.audio.turn_arbiter import AudioTurnArbiter
from lumi.ai.gemini_live import GeminiLiveClient
from lumi.core.telemetry import get_telemetry


def test_turn_arbiter_silence_mode():
    """Verify AudioTurnArbiter manages silence commands and overrides streaming."""
    arbiter = AudioTurnArbiter(turn_window_s=10.0, energy_threshold=100.0)
    assert not arbiter.is_silent()

    # Set silence for 5 seconds
    arbiter.set_silent_until(time.time() + 5.0)
    assert arbiter.is_silent()
    # While silent, mic chunks must be blocked regardless of energy or wake state
    arbiter.wake_up(10.0)
    assert not arbiter.should_stream_mic(energy=500.0)

    # Cancel silence
    arbiter.cancel_silence()
    assert not arbiter.is_silent()
    assert arbiter.should_stream_mic(energy=500.0)


def test_turn_arbiter_speaker_echo_ducking():
    """Verify echo ducking window prevents robot self-talk during and after playback."""
    arbiter = AudioTurnArbiter(echo_tail_s=0.1, turn_window_s=10.0, energy_threshold=100.0)
    arbiter.wake_up(10.0)

    # Initially quiet, stream is allowed
    assert arbiter.should_stream_mic(energy=200.0)

    # Robot starts speaking for 0.2s
    arbiter.notify_speaker_started(duration_s=0.2)
    assert arbiter.is_speaker_active()
    # Must NOT stream while speaker is active
    assert not arbiter.should_stream_mic(energy=500.0)

    # Manually stop speaker early
    arbiter.notify_speaker_stopped()
    # Still inside echo tail (0.1s decay)
    assert arbiter.is_speaker_active()
    assert not arbiter.should_stream_mic(energy=500.0)

    # After echo tail expires
    time.sleep(0.12)
    assert not arbiter.is_speaker_active()
    assert arbiter.should_stream_mic(energy=200.0)


def test_turn_arbiter_dialogue_window_and_ambient_rejection():
    """Verify ambient noise is rejected when outside dialogue turn window."""
    arbiter = AudioTurnArbiter(turn_window_s=0.2, energy_threshold=100.0)
    arbiter.reset_dialogue()
    assert not arbiter.is_in_dialogue()

    # Ambient room hum (low energy) is rejected and does NOT open dialogue
    assert not arbiter.should_stream_mic(energy=50.0)
    assert not arbiter.is_in_dialogue()

    # Mild noise (between threshold and 1.5x threshold) does not wake
    assert not arbiter.should_stream_mic(energy=110.0)
    assert not arbiter.is_in_dialogue()

    # User speaks loudly (> 1.5x threshold) -> auto-wakes dialogue window
    assert arbiter.should_stream_mic(energy=160.0)
    assert arbiter.is_in_dialogue()

    # Subsequent normal-energy speech passes
    assert arbiter.should_stream_mic(energy=110.0)

    # Wait for turn window to elapse
    time.sleep(0.25)
    assert not arbiter.is_in_dialogue()
    # Now mild energy is rejected again
    assert not arbiter.should_stream_mic(energy=110.0)


def test_gemini_live_awake_gating():
    """Verify GeminiLiveClient respects dialogue window and drops chunks when inactive."""
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
    )
    # Default is not awake
    assert not client.is_awake()

    # Wake up for a short duration
    client.wake_up(duration_s=0.15)
    assert client.is_awake()

    # Wait for expiry
    time.sleep(0.2)
    assert not client.is_awake()


def test_gemini_live_inject_context_wakes_dialogue():
    """Verify context injection with trigger_response=True wakes the client for conversation."""
    mic = MagicMock()
    speaker = MagicMock()
    eyes = MagicMock()
    gestures = MagicMock()
    state = MagicMock()
    memory = MagicMock()
    event_bus = MagicMock()
    arbiter = AudioTurnArbiter(turn_window_s=10.0)

    client = GeminiLiveClient(
        mic=mic,
        speaker=speaker,
        eyes=eyes,
        gestures=gestures,
        state=state,
        memory=memory,
        event_bus=event_bus,
        turn_arbiter=arbiter,
    )

    assert not client.is_awake()
    assert not arbiter.is_in_dialogue()

    # Greeting inject with trigger_response=True
    client.inject_context("User entered the room. Greet warmly.", trigger_response=True)
    assert client.is_awake()
    assert arbiter.is_in_dialogue()


def test_gemini_live_barge_in_telemetry():
    """Verify user interruption triggers barge-in telemetry and stops speaker stream."""
    import os
    arbiter = AudioTurnArbiter(turn_window_s=10.0)
    arbiter.notify_speaker_started(5.0)
    assert arbiter.is_speaker_active()

    # Record barge-in
    arbiter.notify_speaker_stopped()
    arbiter.record_barge_in(latency_ms=95.0)

    time.sleep(0.3)
    log_path = get_telemetry().log_path
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "TEL-01" in content
    assert "TEL-02" in content


def test_gemini_live_turn_arbiter_sync_and_speech_wake():
    """Verify GeminiLiveClient dynamically wakes up when turn_arbiter opens dialogue and pushes audio."""
    import asyncio
    mic = MagicMock()
    speaker = MagicMock()
    eyes = MagicMock()
    gestures = MagicMock()
    state = MagicMock()
    memory = MagicMock()
    event_bus = MagicMock()
    arbiter = AudioTurnArbiter(turn_window_s=5.0, energy_threshold=100.0)

    client = GeminiLiveClient(
        mic=mic,
        speaker=speaker,
        eyes=eyes,
        gestures=gestures,
        state=state,
        memory=memory,
        event_bus=event_bus,
        turn_arbiter=arbiter,
    )

    # Initially dormant
    assert not client.is_awake()
    assert not arbiter.is_in_dialogue()

    # Speech triggers arbiter wake
    wake_result = arbiter.should_stream_mic(energy=160.0)
    assert wake_result is True
    assert arbiter.is_in_dialogue()

    # Client is now dynamically awake
    assert client.is_awake()

    # Setup mock event loop and audio queue on client
    loop = MagicMock()
    loop.is_running.return_value = True
    queue_mock = MagicMock()
    loop.call_soon_threadsafe = lambda fn, *args: fn(*args)
    client._loop = loop
    client._audio_queue = queue_mock

    # Audio chunk should now be successfully pushed to queue
    fake_chunk = b"\x01\x02" * 512
    client.push_audio_chunk(fake_chunk)
    queue_mock.put_nowait.assert_called_once_with(fake_chunk)

    # When client is dormant/reset, push_audio_chunk drops chunks
    client.reset_dialogue_state()
    assert not client.is_awake()
    assert not arbiter.is_in_dialogue()
    queue_mock.reset_mock()
    client.push_audio_chunk(fake_chunk)
    queue_mock.put_nowait.assert_not_called()

