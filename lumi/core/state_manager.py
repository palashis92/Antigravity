"""Behavior state machine for LUMI."""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from .logger import get_logger

logger = get_logger("state")


class BehaviorState(str, Enum):
    """The 12 primary behavior states of LUMI."""
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    GREETING = "GREETING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    SEARCHING = "SEARCHING"
    VISION_ANALYSIS = "VISION_ANALYSIS"
    CHESS_ANALYSIS = "CHESS_ANALYSIS"
    REMINDER = "REMINDER"
    ERROR = "ERROR"
    SLEEP = "SLEEP"


# Default mapping from behavior state to corresponding eye expression
STATE_EYE_EXPRESSIONS: Dict[BehaviorState, str] = {
    BehaviorState.IDLE: "neutral",
    BehaviorState.OBSERVING: "curious",
    BehaviorState.GREETING: "happy",
    BehaviorState.LISTENING: "listening",
    BehaviorState.THINKING: "thinking",
    BehaviorState.SPEAKING: "speaking",
    BehaviorState.SEARCHING: "curious",
    BehaviorState.VISION_ANALYSIS: "curious",
    BehaviorState.CHESS_ANALYSIS: "thinking",
    BehaviorState.REMINDER: "excited",
    BehaviorState.ERROR: "sad",
    BehaviorState.SLEEP: "sleepy",
}


# Valid state transitions matrix
VALID_TRANSITIONS: Dict[BehaviorState, Set[BehaviorState]] = {
    BehaviorState.IDLE: {
        BehaviorState.OBSERVING,
        BehaviorState.GREETING,
        BehaviorState.LISTENING,
        BehaviorState.THINKING,
        BehaviorState.SEARCHING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.CHESS_ANALYSIS,
        BehaviorState.REMINDER,
        BehaviorState.ERROR,
        BehaviorState.SLEEP,
    },
    BehaviorState.OBSERVING: {
        BehaviorState.IDLE,
        BehaviorState.GREETING,
        BehaviorState.LISTENING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.ERROR,
        BehaviorState.SLEEP,
    },
    BehaviorState.GREETING: {
        BehaviorState.IDLE,
        BehaviorState.LISTENING,
        BehaviorState.SPEAKING,
        BehaviorState.ERROR,
    },
    BehaviorState.LISTENING: {
        BehaviorState.IDLE,
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.CHESS_ANALYSIS,
        BehaviorState.SEARCHING,
        BehaviorState.ERROR,
    },
    BehaviorState.THINKING: {
        BehaviorState.SPEAKING,
        BehaviorState.SEARCHING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.CHESS_ANALYSIS,
        BehaviorState.ERROR,
        BehaviorState.IDLE,
    },
    BehaviorState.SPEAKING: {
        BehaviorState.IDLE,
        BehaviorState.LISTENING,
        BehaviorState.OBSERVING,
        BehaviorState.ERROR,
    },
    BehaviorState.SEARCHING: {
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.ERROR,
        BehaviorState.IDLE,
    },
    BehaviorState.VISION_ANALYSIS: {
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.ERROR,
        BehaviorState.IDLE,
    },
    BehaviorState.CHESS_ANALYSIS: {
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.ERROR,
        BehaviorState.IDLE,
    },
    BehaviorState.REMINDER: {
        BehaviorState.SPEAKING,
        BehaviorState.LISTENING,
        BehaviorState.IDLE,
        BehaviorState.ERROR,
    },
    BehaviorState.ERROR: {
        BehaviorState.IDLE,
        BehaviorState.SLEEP,
    },
    BehaviorState.SLEEP: {
        BehaviorState.IDLE,
        BehaviorState.OBSERVING,
        BehaviorState.GREETING,
    },
}


class StateManager:
    """Manages the robot's lifecycle state, transitions, and state-change hooks."""

    def __init__(self, initial_state: BehaviorState = BehaviorState.IDLE) -> None:
        self._current_state: BehaviorState = initial_state
        self._previous_state: Optional[BehaviorState] = None
        self._state_entered_time: float = time.time()
        self._listeners: List[Callable[[BehaviorState, BehaviorState], None]] = []
        self._history: List[tuple[BehaviorState, float]] = [(initial_state, self._state_entered_time)]
        logger.info(f"StateManager initialized in state: {initial_state.value}")

    @property
    def current_state(self) -> BehaviorState:
        return self._current_state

    @property
    def previous_state(self) -> Optional[BehaviorState]:
        return self._previous_state

    @property
    def time_in_current_state(self) -> float:
        return time.time() - self._state_entered_time

    @property
    def recommended_eye_expression(self) -> str:
        return STATE_EYE_EXPRESSIONS.get(self._current_state, "neutral")

    def add_listener(self, callback: Callable[[BehaviorState, BehaviorState], None]) -> None:
        """Register a callback that fires when state changes: callback(old_state, new_state)."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[BehaviorState, BehaviorState], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def transition_to(self, new_state: BehaviorState, reason: str = "") -> bool:
        """Attempt to transition to a new behavior state."""
        if new_state == self._current_state:
            return True

        allowed = VALID_TRANSITIONS.get(self._current_state, set())
        if new_state not in allowed:
            logger.warning(
                f"Invalid transition rejected: {self._current_state.value} -> {new_state.value} "
                f"(reason: {reason})"
            )
            return False

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = new_state
        self._state_entered_time = time.time()
        self._history.append((new_state, self._state_entered_time))
        if len(self._history) > 100:
            self._history.pop(0)

        log_msg = f"State transition: {old_state.value} -> {new_state.value}"
        if reason:
            log_msg += f" (reason: {reason})"
        logger.info(log_msg)

        # Notify registered listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state listener {listener}: {e}", exc_info=True)

        return True

    def force_state(self, state: BehaviorState, reason: str = "forced") -> None:
        """Force a state change regardless of transition matrix (e.g. emergency / recovery)."""
        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = state
        self._state_entered_time = time.time()
        logger.warning(f"FORCED state change: {old_state.value} -> {state.value} (reason: {reason})")

        for listener in self._listeners:
            try:
                listener(old_state, state)
            except Exception as e:
                logger.error(f"Error in state listener {listener}: {e}", exc_info=True)
