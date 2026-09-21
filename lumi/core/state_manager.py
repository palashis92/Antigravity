"""Behavior state machine for LUMI."""

from __future__ import annotations

import time
import threading
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
    MEETING = "MEETING"
    ANJUM_MODE = "ANJUM_MODE"


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
    BehaviorState.MEETING: "listening",
    BehaviorState.ANJUM_MODE: "excited",
}

# Transient states subject to watchdog timeout recovery
TRANSIENT_STATES: Set[BehaviorState] = {
    BehaviorState.GREETING,
    BehaviorState.THINKING,
    BehaviorState.LISTENING,
    BehaviorState.VISION_ANALYSIS,
    BehaviorState.CHESS_ANALYSIS,
    BehaviorState.SEARCHING,
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
        BehaviorState.MEETING,
        BehaviorState.ANJUM_MODE,
        BehaviorState.SPEAKING,
    },
    BehaviorState.OBSERVING: {
        BehaviorState.IDLE,
        BehaviorState.GREETING,
        BehaviorState.LISTENING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.ERROR,
        BehaviorState.SLEEP,
        BehaviorState.ANJUM_MODE,
    },
    BehaviorState.GREETING: {
        BehaviorState.IDLE,
        BehaviorState.LISTENING,
        BehaviorState.SPEAKING,
        BehaviorState.OBSERVING,
        BehaviorState.ERROR,
        BehaviorState.ANJUM_MODE,
    },
    BehaviorState.LISTENING: {
        BehaviorState.IDLE,
        BehaviorState.GREETING,
        BehaviorState.OBSERVING,
        BehaviorState.THINKING,
        BehaviorState.SPEAKING,
        BehaviorState.VISION_ANALYSIS,
        BehaviorState.CHESS_ANALYSIS,
        BehaviorState.SEARCHING,
        BehaviorState.ERROR,
        BehaviorState.ANJUM_MODE,
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
        BehaviorState.GREETING,
        BehaviorState.ERROR,
        BehaviorState.ANJUM_MODE,
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
    BehaviorState.MEETING: {
        BehaviorState.IDLE,
        BehaviorState.ERROR,
    },
    BehaviorState.ANJUM_MODE: {
        BehaviorState.IDLE,
        BehaviorState.SPEAKING,
        BehaviorState.LISTENING,
        BehaviorState.ERROR,
    },
}


class StateManager:
    """Manages the robot's lifecycle state, transitions, and state-change hooks."""

    def __init__(self, initial_state: BehaviorState = BehaviorState.IDLE) -> None:
        self._lock = threading.RLock()
        self._current_state: BehaviorState = initial_state
        self._previous_state: Optional[BehaviorState] = None
        self._state_entered_time: float = time.time()
        self._listeners: List[Callable[[BehaviorState, BehaviorState], None]] = []
        self._history: List[tuple[BehaviorState, float]] = [(initial_state, self._state_entered_time)]
        self._watchdog_running: bool = False
        self._watchdog_thread: Optional[threading.Thread] = None
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
        with self._lock:
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

            print(f"\n🧠 [LUMI STATE]: {old_state.name} ➡️ {new_state.name} ({reason})\n")

            log_msg = f"State transition: {old_state.value} -> {new_state.value}"
            if reason:
                log_msg += f" (reason: {reason})"
            logger.info(log_msg)

        # Notify registered listeners outside lock to prevent deadlocks
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

    def check_watchdog(self, now: Optional[float] = None, timeout_s: float = 15.0) -> bool:
        """Check if current state is a transient state that has exceeded timeout_s.
        
        If stuck, automatically forces transition to IDLE and emits TEL-04 telemetry.
        Returns True if a watchdog recovery was triggered.
        """
        current_time = now if now is not None else time.time()
        with self._lock:
            if self._current_state in TRANSIENT_STATES:
                elapsed = current_time - self._state_entered_time
                if elapsed >= timeout_s:
                    stuck_state = self._current_state
                    logger.warning(
                        f"⚠️ [WATCHDOG] State '{stuck_state.value}' stuck for {elapsed:.1f}s (>{timeout_s}s). "
                        "Auto-recovering to IDLE."
                    )
                    try:
                        from .telemetry import get_telemetry
                        get_telemetry().record_event(
                            "TEL-04",
                            context="watchdog_reset",
                            metadata={"from_state": stuck_state.value, "stuck_duration_s": elapsed},
                        )
                    except Exception:
                        pass

                    self.force_state(BehaviorState.IDLE, reason=f"watchdog_timeout_from_{stuck_state.value}")
                    return True
        return False

    def start_watchdog(self, timeout_s: float = 15.0, interval_s: float = 1.0) -> None:
        """Start a background daemon thread that periodically checks for stuck states."""
        with self._lock:
            if self._watchdog_running:
                return
            self._watchdog_running = True
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                args=(timeout_s, interval_s),
                daemon=True,
                name="StateWatchdog",
            )
            self._watchdog_thread.start()
        logger.info(f"StateWatchdog active (timeout={timeout_s}s, interval={interval_s}s).")

    def stop_watchdog(self) -> None:
        """Stop the background watchdog thread."""
        self._watchdog_running = False
        thread = getattr(self, "_watchdog_thread", None)
        if thread and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

    def _watchdog_loop(self, timeout_s: float, interval_s: float) -> None:
        while self._watchdog_running:
            try:
                self.check_watchdog(timeout_s=timeout_s)
            except Exception as e:
                logger.debug(f"Watchdog error: {e}")
            time.sleep(interval_s)
