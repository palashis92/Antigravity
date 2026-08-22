"""Unit tests for Behavior StateManager."""

from lumi.core.state_manager import BehaviorState, StateManager


def test_initial_state() -> None:
    sm = StateManager(BehaviorState.IDLE)
    assert sm.current_state == BehaviorState.IDLE
    assert sm.recommended_eye_expression == "neutral"


def test_valid_transitions() -> None:
    sm = StateManager(BehaviorState.IDLE)
    assert sm.transition_to(BehaviorState.OBSERVING) is True
    assert sm.current_state == BehaviorState.OBSERVING
    assert sm.previous_state == BehaviorState.IDLE

    assert sm.transition_to(BehaviorState.GREETING) is True
    assert sm.current_state == BehaviorState.GREETING
    assert sm.recommended_eye_expression == "happy"


def test_invalid_transition_rejected() -> None:
    sm = StateManager(BehaviorState.SPEAKING)
    # SPEAKING to CHESS_ANALYSIS is not an allowed direct transition
    assert sm.transition_to(BehaviorState.CHESS_ANALYSIS) is False
    assert sm.current_state == BehaviorState.SPEAKING


def test_state_change_listener() -> None:
    sm = StateManager(BehaviorState.IDLE)
    events = []

    def on_change(old, new):
        events.append((old, new))

    sm.add_listener(on_change)
    sm.transition_to(BehaviorState.LISTENING)

    assert len(events) == 1
    assert events[0] == (BehaviorState.IDLE, BehaviorState.LISTENING)
