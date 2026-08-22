"""Unit tests for EventBus subsystem."""

import time
from lumi.core.event_bus import Event, EventBus


def test_sync_event_publishing() -> None:
    bus = EventBus()
    received = []

    def handler(evt: Event) -> None:
        received.append(evt)

    bus.subscribe("person.detected", handler)
    bus.publish_sync(Event(topic="person.detected", data={"name": "Palash"}))

    assert len(received) == 1
    assert received[0].data["name"] == "Palash"


def test_topic_isolation() -> None:
    bus = EventBus()
    received_a = []
    received_b = []

    bus.subscribe("topic.a", lambda e: received_a.append(e))
    bus.subscribe("topic.b", lambda e: received_b.append(e))

    bus.publish_sync(Event(topic="topic.a"))
    assert len(received_a) == 1
    assert len(received_b) == 0


def test_wildcard_subscription() -> None:
    bus = EventBus()
    wildcard_events = []

    bus.subscribe("*", lambda e: wildcard_events.append(e))
    bus.publish_sync(Event(topic="any.event.1"))
    bus.publish_sync(Event(topic="any.event.2"))

    assert len(wildcard_events) == 2


def test_async_event_publishing() -> None:
    bus = EventBus(async_workers=1)
    bus.start()
    received = []

    bus.subscribe("async.topic", lambda e: received.append(e))
    bus.emit("async.topic", {"msg": "hello"})

    time.sleep(0.2)
    bus.stop()

    assert len(received) == 1
    assert received[0].data["msg"] == "hello"
