"""Thread-safe event bus for decoupled asynchronous subsystem communication."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .logger import get_logger

logger = get_logger("event_bus")


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable event packet dispatched across the system."""
    topic: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe publish/subscribe event bus."""

    def __init__(self, async_workers: int = 1) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._wildcard_subscribers: List[Callable[[Event], None]] = []
        self._lock = threading.RLock()
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=1000)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._async_workers = max(1, async_workers)

    def start(self) -> None:
        """Start the background worker thread(s) for asynchronous event processing."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for i in range(self._async_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"EventBus-Worker-{i}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            logger.info(f"EventBus started with {self._async_workers} worker thread(s).")

    def stop(self) -> None:
        """Stop background worker threads gracefully."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            # Push sentinel event to wake workers
            for _ in range(self._async_workers):
                self._queue.put(Event(topic="__STOP__"))
            for t in self._threads:
                t.join(timeout=1.0)
            self._threads.clear()
            logger.info("EventBus stopped.")

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Subscribe a handler callback to a specific topic or '*' for all topics."""
        with self._lock:
            if topic == "*":
                if handler not in self._wildcard_subscribers:
                    self._wildcard_subscribers.append(handler)
            else:
                if topic not in self._subscribers:
                    self._subscribers[topic] = []
                if handler not in self._subscribers[topic]:
                    self._subscribers[topic].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__ if hasattr(handler, '__name__') else handler} to topic '{topic}'")

    def unsubscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Unsubscribe a handler from a topic."""
        with self._lock:
            if topic == "*":
                if handler in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(handler)
            elif topic in self._subscribers:
                if handler in self._subscribers[topic]:
                    self._subscribers[topic].remove(handler)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]

    def publish_sync(self, event: Event) -> None:
        """Publish an event and invoke all subscribers synchronously in the calling thread."""
        self._dispatch(event)

    def publish_async(self, event: Event) -> bool:
        """Enqueue an event to be dispatched asynchronously by worker threads."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            logger.error(f"EventBus queue is full! Dropping event on topic '{event.topic}'")
            return False

    def emit(self, topic: str, data: Optional[Dict[str, Any]] = None, source: str = "system") -> None:
        """Convenience method to emit an event asynchronously."""
        evt = Event(topic=topic, data=data or {}, source=source)
        self.publish_async(evt)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if event.topic == "__STOP__":
                self._queue.task_done()
                break

            self._dispatch(event)
            self._queue.task_done()

    def _dispatch(self, event: Event) -> None:
        handlers: List[Callable[[Event], None]] = []
        with self._lock:
            # Topic-specific subscribers
            if event.topic in self._subscribers:
                handlers.extend(self._subscribers[event.topic])
            # Wildcard subscribers
            handlers.extend(self._wildcard_subscribers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in EventBus handler {handler} processing topic '{event.topic}': {e}",
                    exc_info=True,
                )
