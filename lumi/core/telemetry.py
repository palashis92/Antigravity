"""Lightweight, non-blocking telemetry logging system for LUMI.

Stores structured diagnostic metrics (audio latency, barge-in, dropouts, watchdogs)
in .tmp/telemetry.jsonl via an asynchronous ring-buffered worker thread.
Zero behavioral overhead during runtime.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger("core.telemetry")


class TelemetryLogger:
    """Thread-safe, non-blocking diagnostic telemetry collector."""

    _instance: Optional[TelemetryLogger] = None
    _singleton_lock = threading.Lock()

    def __init__(self, log_path: str = ".tmp/telemetry.jsonl", max_queue_size: int = 1000) -> None:
        self.log_path = log_path
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._dropped_count = 0

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self.start()

    @classmethod
    def get_instance(cls) -> TelemetryLogger:
        """Access the global TelemetryLogger singleton."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="TelemetryWriter"
        )
        self._thread.start()

    def stop(self) -> None:
        """Flush remaining queue items and stop writer thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

    def record(
        self,
        metric: str,
        value: float,
        unit: str = "",
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a numeric telemetry metric (non-blocking)."""
        entry = {
            "ts": time.time(),
            "type": "metric",
            "metric": metric,
            "val": value,
            "unit": unit,
            "ctx": context or "",
            "meta": metadata or {},
        }
        self._enqueue(entry)

    def record_latency(
        self,
        name: str,
        duration_ms: float,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method to record a latency measurement in milliseconds."""
        self.record(metric=name, value=duration_ms, unit="ms", context=context, metadata=metadata)

    def record_event(
        self,
        event_name: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a discrete system event (e.g. barge-in, watchdog reset)."""
        entry = {
            "ts": time.time(),
            "type": "event",
            "event": event_name,
            "ctx": context or "",
            "meta": metadata or {},
        }
        self._enqueue(entry)

    def _enqueue(self, item: Dict[str, Any]) -> None:
        """Enqueue event; drop oldest if buffer is full to prevent OOM or latency."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped_count += 1
            try:
                self._queue.get_nowait()  # Drop oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _writer_loop(self) -> None:
        """Background file writer appending JSONL records."""
        while self._running or not self._queue.empty():
            items = []
            try:
                # Wait for at least one item
                item = self._queue.get(timeout=0.2)
                items.append(item)
                # Drain available batch (up to 50 items)
                while len(items) < 50:
                    try:
                        items.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if items:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        for entry in items:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.debug(f"Failed to write telemetry batch: {e}")


def get_telemetry() -> TelemetryLogger:
    """Convenience helper to get global TelemetryLogger."""
    return TelemetryLogger.get_instance()
