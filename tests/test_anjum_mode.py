"""Unit tests for Anjum Speech-Therapy Mode and Companion Engine."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumi.companion.anjum_companion import AnjumCompanionEngine
from lumi.core.state_manager import BehaviorState, StateManager


def test_anjum_state_transitions():
    """Verify ANJUM_MODE state transitions in StateManager."""
    sm = StateManager(BehaviorState.IDLE)
    assert sm.transition_to(BehaviorState.ANJUM_MODE) is True
    assert sm.current_state == BehaviorState.ANJUM_MODE
    assert sm.recommended_eye_expression == "excited"

    # From ANJUM_MODE to SPEAKING and back
    assert sm.transition_to(BehaviorState.SPEAKING) is True
    assert sm.transition_to(BehaviorState.ANJUM_MODE) is True

    # From ANJUM_MODE to IDLE (on timeout)
    assert sm.transition_to(BehaviorState.IDLE) is True
    assert sm.current_state == BehaviorState.IDLE


def test_companion_activation_and_deactivation():
    """Verify engine activate and deactivate greetings."""
    engine = AnjumCompanionEngine(stimulus_cooldown=5.0)
    assert not engine.is_active

    greeting = engine.activate()
    assert engine.is_active
    assert "আঞ্জুম" in greeting

    goodbye = engine.deactivate()
    assert not engine.is_active
    assert "আঞ্জুম" in goodbye


def test_absence_timeout():
    """Verify 10-second absence timeout."""
    engine = AnjumCompanionEngine()
    engine.activate()

    t0 = 1000.0
    engine.mark_seen(t0)

    # 5 seconds passed - not timed out
    assert engine.is_timeout(now=t0 + 5.0, timeout_sec=10.0) is False

    # 10.1 seconds passed - timed out
    assert engine.is_timeout(now=t0 + 10.1, timeout_sec=10.0) is True


def test_proactive_stimulus_cooldown():
    """Verify proactive stimulation respects cooldown and generates energetic prompts."""
    engine = AnjumCompanionEngine(stimulus_cooldown=5.0)
    engine.activate()

    t0 = engine.last_stimulus_time

    # Immediate check should return None due to cooldown
    assert engine.check_proactive_stimulus(now=t0 + 2.0) is None

    # After cooldown expires, returns a stimulus
    stimulus = engine.check_proactive_stimulus(now=t0 + 6.0)
    assert stimulus is not None
    assert len(stimulus) > 0


def test_counting_game_bengali_digits():
    """Verify reverse teaching counting game with Bengali digits."""
    engine = AnjumCompanionEngine()
    engine.activate()

    # Child says "১"
    resp = engine.handle_anjum_speech("১")
    assert resp is not None
    assert "২" in resp  # Should prompt for 2

    # Child says "দুই"
    resp = engine.handle_anjum_speech("দুই")
    assert resp is not None
    assert "৩" in resp  # Should prompt for 3

    # Child reaches "১০"
    resp = engine.handle_anjum_speech("১০")
    assert resp is not None
    assert "১০" in resp
    assert "জিনিয়াস" in resp or "সাবাশ" in resp


def test_counting_game_english_digits():
    """Verify counting game with English numbers spoken by child."""
    engine = AnjumCompanionEngine()
    engine.activate()

    # Child says "one" or "1"
    resp = engine.handle_anjum_speech("one")
    assert resp is not None
    assert "২" in resp


def test_identity_and_praise_speech():
    """Verify identity confirmation and enthusiastic praise for vocalizations."""
    engine = AnjumCompanionEngine()
    engine.activate()

    # Child mentions her name
    resp = engine.handle_anjum_speech("আমি আঞ্জুম")
    assert resp is not None
    assert "আঞ্জুম" in resp

    # Child says any random word
    resp = engine.handle_anjum_speech("আম্মু")
    assert resp is not None
    assert any(word in resp for word in ["সুন্দর", "সাবাশ", "লক্ষ্মী", "ভালো"])


if __name__ == "__main__":
    test_anjum_state_transitions()
    test_companion_activation_and_deactivation()
    test_absence_timeout()
    test_proactive_stimulus_cooldown()
    test_counting_game_bengali_digits()
    test_counting_game_english_digits()
    test_identity_and_praise_speech()
    print("ALL_TESTS_PASSED")
