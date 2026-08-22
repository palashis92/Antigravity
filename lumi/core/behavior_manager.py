"""Proactive Behavior Engine for Social Cues, Idle Wandering, and Context-Aware Actions."""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Optional, Dict

from .event_bus import Event, EventBus
from .logger import get_logger
from .state_manager import BehaviorState, StateManager

logger = get_logger("core.behavior")


class BehaviorManager:
    """Decides proactive social behaviors and idle life-like animations."""

    def __init__(
        self,
        state_manager: StateManager,
        event_bus: EventBus,
        idle_wander_interval_s: float = 6.0,
    ) -> None:
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.idle_wander_interval_s = idle_wander_interval_s
        self._last_idle_move_time = time.time()
        self._running = False
        
        # Interaction cooldown tracking
        self._last_interaction_times: Dict[str, float] = {}
        self._last_convo_times: Dict[str, float] = {}
        self._object_start_time: Optional[float] = None
        self._current_object: Optional[str] = None
        
        self.GREET_COOLDOWN = 300.0  # 5 minutes
        self.CONVO_COOLDOWN = 600.0  # 10 minutes

    def _get_time_of_day_mood(self) -> str:
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def on_person_spotted(self, person_name: str, is_known: bool) -> None:
        """Handle visual detection of a person."""
        curr = self.state_manager.current_state
        if curr in (BehaviorState.SPEAKING, BehaviorState.THINKING, BehaviorState.VISION_ANALYSIS):
            return  # Do not interrupt busy states

        now = time.time()
        last_interact = self._last_interaction_times.get(person_name, 0)
        
        # Cooldown check for greeting
        if now - last_interact < self.GREET_COOLDOWN:
            return
            
        logger.info(f"BehaviorEngine reacting to person spotted: {person_name} (known: {is_known})")
        self._last_interaction_times[person_name] = now
        
        # Time-of-day awareness logic
        mood = self._get_time_of_day_mood()
        
        if not is_known:
            logger.info("Curiosity Trigger: New unrecognized face.")
            self.state_manager.transition_to(
                BehaviorState.OBSERVING, reason=f"person_spotted_unknown"
            )
            # Curiosity logic triggers self introduction via lumi_brain's process_person_interaction
        else:
            self.state_manager.transition_to(
                BehaviorState.OBSERVING, reason=f"person_spotted_{person_name}"
            )
            # Check for memory-driven follow-ups
            last_convo = self._last_convo_times.get(person_name, 0)
            if now - last_convo >= self.CONVO_COOLDOWN:
                self._last_convo_times[person_name] = now
                logger.info(f"Checking memory for {person_name} to suggest follow-up questions.")
                
            if mood == "morning":
                logger.info("Time-of-day: Cheerful greetings.")
            elif mood == "night":
                logger.info("Time-of-day: Sleepy expressions, quieter voice.")
                
    def on_object_detected(self, object_name: str) -> None:
        """Handle continuous object detection."""
        now = time.time()
        if self._current_object == object_name:
            if self._object_start_time and (now - self._object_start_time) > 5.0:
                logger.info(f"Curiosity Trigger: {object_name} in view for > 5s.")
                self.state_manager.transition_to(BehaviorState.OBSERVING, reason="curious_object")
                # Reset to prevent spam
                self._object_start_time = None
        else:
            self._current_object = object_name
            self._object_start_time = now

    def tick_idle(self) -> bool:
        """Called periodically during IDLE state to produce lifelike head, arm & eye wander."""
        if self.state_manager.current_state != BehaviorState.IDLE:
            return False

        now = time.time()
        if now - self._last_idle_move_time >= self.idle_wander_interval_s:
            self._last_idle_move_time = now + random.uniform(-1.5, 2.5)
            self.event_bus.emit("motion.idle_wander", source="behavior_manager")
            return True
        return False
