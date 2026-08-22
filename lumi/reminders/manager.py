"""Background Reminder Scheduler and Due Notification Dispatcher."""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..core.event_bus import EventBus
from ..core.logger import get_logger
from ..memory.manager import MemoryManager

logger = get_logger("reminders.scheduler")


class ReminderScheduler:
    """Monitors scheduled reminders in background and notifies the EventBus when due."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        event_bus: EventBus,
        check_interval_s: float = 15.0,
    ) -> None:
        self.memory = memory_manager
        self.event_bus = event_bus
        self.check_interval_s = check_interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="LumiReminderScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("ReminderScheduler background worker active.")

    def stop(self) -> None:
        """Stop background scheduler."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("ReminderScheduler stopped.")

    def _scheduler_loop(self) -> None:
        while self._running:
            try:
                due_reminders = self.memory.get_due_reminders()
                for rem in due_reminders:
                    logger.info(f"Triggering scheduled reminder: '{rem.title}'")
                    self.event_bus.emit(
                        topic="reminder.due",
                        data={
                            "id": rem.id,
                            "title": rem.title,
                            "description": rem.description,
                            "person_id": rem.person_id,
                        },
                        source="reminder_scheduler",
                    )
                    self.memory.complete_reminder(rem.id)
            except Exception as e:
                logger.error(f"Error checking due reminders: {e}", exc_info=True)

            # Sleep in short increments to allow rapid shutdown
            for _ in range(int(self.check_interval_s * 2)):
                if not self._running:
                    break
                time.sleep(0.5)
