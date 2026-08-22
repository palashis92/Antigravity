"""Unit tests for ConversationEngine and ToolRegistry.

Tests handle both online (with API keys) and offline (graceful fallback) modes.
"""

import os

from lumi.ai.conversation import ConversationEngine
from lumi.ai.tools import ToolRegistry
from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus


def test_tool_registry() -> None:
    tools = ToolRegistry()
    # Date/time should always work (no API needed)
    dt_str = tools.get_current_datetime()
    assert len(dt_str) > 5  # Should return some date/time string

    # Weather may call real API or fall back — either way should return a non-empty string
    weather_str = tools.get_weather("Dhaka")
    assert isinstance(weather_str, str)
    assert len(weather_str) > 5


def test_conversation_engine_responses() -> None:
    db = Database(db_path=":memory:", enable_wal=False)
    mem = MemoryManager(db)
    owner = mem.remember_person("Palash", relationship="owner", consent_status=ConsentStatus.GRANTED)

    engine = ConversationEngine(mem)

    # Test greeting — should work via Inworld.ai API or offline fallback
    reply_greet = engine.generate_response("কেমন আছেন?", current_person=owner)
    assert isinstance(reply_greet, str)
    assert len(reply_greet) > 5  # Should generate some meaningful response

    # Test time query — should trigger tool usage or fallback
    reply_time = engine.generate_response("কয়টা বাজে?", current_person=owner)
    assert isinstance(reply_time, str)
    assert len(reply_time) > 5
