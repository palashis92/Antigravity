"""Unit tests for Phase 4: Architecture & FSM Robustness.

Covers:
- [S1] StateManager Timeout Watchdogs (auto-recovery from stuck transient states)
- [S5] Persona Segregation (clean dialogue state reset on mode transitions)
- [S3] Unified SQLite Persistence for Learned Rules
"""

import time
from unittest.mock import MagicMock, patch

from lumi.core.state_manager import BehaviorState, StateManager
from lumi.core.telemetry import get_telemetry
from lumi.memory.database import Database
from lumi.memory.learned_rules import LearnedRulesStore
from lumi.ai.gemini_live import GeminiLiveClient


def test_state_manager_watchdog_recovers_stuck_transient_state():
    """[S1] Verify StateManager automatically recovers from stuck transient states to IDLE."""
    sm = StateManager(BehaviorState.IDLE)
    sm.transition_to(BehaviorState.LISTENING)
    sm.transition_to(BehaviorState.THINKING)
    assert sm.current_state == BehaviorState.THINKING

    now = sm._state_entered_time

    # Under timeout: no reset
    triggered = sm.check_watchdog(now=now + 10.0, timeout_s=15.0)
    assert not triggered
    assert sm.current_state == BehaviorState.THINKING

    # Over timeout: forces reset to IDLE and emits TEL-04
    triggered = sm.check_watchdog(now=now + 16.0, timeout_s=15.0)
    assert triggered
    assert sm.current_state == BehaviorState.IDLE


def test_state_manager_watchdog_ignores_stable_states():
    """[S1] Verify StateManager watchdog does NOT reset stable states like IDLE or SLEEP."""
    sm = StateManager(BehaviorState.IDLE)
    now = sm._state_entered_time

    triggered = sm.check_watchdog(now=now + 100.0, timeout_s=15.0)
    assert not triggered
    assert sm.current_state == BehaviorState.IDLE

    sm.transition_to(BehaviorState.SLEEP)
    now = sm._state_entered_time
    triggered = sm.check_watchdog(now=now + 100.0, timeout_s=15.0)
    assert not triggered
    assert sm.current_state == BehaviorState.SLEEP


def test_persona_segregation_reset_dialogue_state():
    """[S5] Verify reset_dialogue_state purges conversation state to prevent persona bleed."""
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
    client.wake_up(10.0)
    assert client.is_awake()

    # Reset dialogue on persona / mode change
    client.reset_dialogue_state()
    assert not client.is_awake()


def test_learned_rules_sqlite_persistence():
    """[S3] Verify LearnedRulesStore persists directly into SQLite database."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(db_path=":memory:", enable_wal=False)
        store = LearnedRulesStore(file_path=Path(tmpdir) / "rules.json", db=db)

        # Initial load is empty
        assert len(store.load_rules()) == 0

    # Add a rule
    rule = store.add_rule(
        feedback="Don't talk too loudly",
        adapted_rule="Keep voice volume moderate",
        category="speech",
    )
    assert rule["adapted_rule"] == "Keep voice volume moderate"

    # Verify rule in SQLite database
    rows = db.execute_query("SELECT id, user_feedback, adapted_rule FROM learned_rules WHERE id = ?", (rule["id"],))
    assert len(rows) == 1
    assert rows[0]["adapted_rule"] == "Keep voice volume moderate"

    # Delete rule
    deleted = store.delete_rule(rule["id"])
    assert deleted
    assert len(db.execute_query("SELECT * FROM learned_rules")) == 0
