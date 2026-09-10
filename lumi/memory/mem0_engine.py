"""
Lumi Mem0 Engine: Advanced Semantic Memory Extraction.

Works like mem0ai, but natively optimized for Raspberry Pi and Gemini.
Runs in a background thread after each conversation turn, using Gemini
to extract, deduplicate, and manage semantic facts autonomously.
"""

from typing import List, Optional
import json
import os
import re
import threading

from ..core.logger import get_logger
from .manager import MemoryManager

logger = get_logger("mem0_engine")

# Lazy imports to avoid startup crash if google-genai is not installed
_genai = None
_types = None


def _ensure_genai():
    global _genai, _types
    if _genai is None:
        try:
            from google import genai
            from google.genai import types
            _genai = genai
            _types = types
        except ImportError:
            logger.warning("google-genai SDK not installed. LumiMem0 Engine disabled.")
    return _genai is not None


class LumiMem0Engine:
    """
    Background worker that intercepts conversations and uses Gemini LLM
    to extract, deduplicate, and manage semantic facts autonomously,
    exactly like the mem0ai library.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        if self.api_key and _ensure_genai():
            self.client = _genai.Client(api_key=self.api_key)
        self._lock = threading.Lock()

    def process_conversation_turn_async(
        self, person_id: str, person_name: str, user_text: str, ai_text: str
    ) -> None:
        """Fires an asynchronous background thread to extract memory."""
        if not self.client:
            return

        thread = threading.Thread(
            target=self._extract_memory_sync,
            args=(person_id, person_name, user_text, ai_text),
            daemon=True,
            name=f"Mem0_Worker_{person_name}",
        )
        thread.start()

    def _extract_memory_sync(
        self, person_id: str, person_name: str, user_text: str, ai_text: str
    ) -> None:
        """Synchronous extraction logic running in background thread."""
        with self._lock:
            try:
                # 1. Fetch current semantic memory state
                current_facts = self.memory.recall_facts(person_id=person_id)
                lines = [f"ID: {f.id} | Fact: {f.fact_text}" for f in current_facts]
                current_facts_str = "\n".join(lines) if lines else "No existing memories."

                # 2. Build the Mem0 Extraction Prompt
                prompt = (
                    "You are Lumi's internal Mem0 Memory Engine. Your job is to "
                    "extract long-term semantic facts from the latest conversation turn.\n"
                    "You must update the user's Memory State by ADDING new facts, "
                    "UPDATING changed facts, or DELETING invalid facts.\n\n"
                    f"USER NAME: {person_name}\n"
                    f"EXISTING MEMORIES:\n{current_facts_str}\n\n"
                    f"LATEST CONVERSATION:\nUser: {user_text}\nLumi: {ai_text}\n\n"
                    "Rules:\n"
                    "1. Only extract important facts (preferences, hobbies, relationships, "
                    "job, current events). Ignore small talk.\n"
                    "2. If the user mentions a new fact, return an ADD action.\n"
                    "3. If the user's statement contradicts or updates an existing memory "
                    "ID, return an UPDATE action for that ID.\n"
                    "4. If a fact is explicitly stated as no longer true, return a DELETE action.\n\n"
                    "Respond ONLY with a JSON array of actions. Example:\n"
                    '[{"action": "ADD", "fact": "User likes black coffee"},'
                    '{"action": "UPDATE", "id": "123-abc", "fact": "User now prefers tea"},'
                    '{"action": "DELETE", "id": "456-def"}]\n'
                    "If nothing important was said, return []."
                )

                # 3. Call LLM for extraction
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=_types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                # 4. Apply Database Actions
                raw_json = response.text.strip() if (response and response.text) else "[]"
                if raw_json.startswith("```"):
                    raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
                    raw_json = re.sub(r"\s*```$", "", raw_json)
                try:
                    actions = json.loads(raw_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM memory actions: {e}")
                    return

                added, updated, deleted = 0, 0, 0

                for action in actions:
                    act = action.get("action", "").upper()

                    if act == "ADD":
                        self.memory.remember_fact(
                            fact_text=action["fact"], person_id=person_id
                        )
                        added += 1

                    elif act == "UPDATE":
                        fact_id = action.get("id")
                        if fact_id:
                            self.memory.forget_fact(fact_id)
                            self.memory.remember_fact(
                                fact_text=action["fact"], person_id=person_id
                            )
                            updated += 1

                    elif act == "DELETE":
                        fact_id = action.get("id")
                        if fact_id:
                            self.memory.forget_fact(fact_id)
                            deleted += 1

                if actions:
                    logger.info(
                        f"LumiMem0 [{person_name}]: "
                        f"+{added} added, ~{updated} updated, -{deleted} deleted."
                    )

            except Exception as e:
                logger.error(f"LumiMem0 Engine Error: {e}")