"""Conversation Management, Context Assembly, and Response Generation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core.logger import get_logger
from ..memory.manager import MemoryManager
from ..memory.models import ConsentStatus, Person
from .prompts import LUMI_SYSTEM_PROMPT_BN
from .tools import ToolRegistry

logger = get_logger("ai.conversation")


class ConversationEngine:
    """Manages natural turn-by-turn conversation and context integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        tool_registry: Optional[ToolRegistry] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.memory = memory_manager
        self.tools = tool_registry or ToolRegistry()
        self.api_key = api_key or os.getenv("INWORLD_API_KEY")
        
        self.client = None
        if self.api_key and OpenAI:
            try:
                self.client = OpenAI(
                    base_url="https://api.inworld.ai/v1",
                    api_key=self.api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize Inworld OpenAI client: {e}")

    def build_context_prompt(self, current_person: Optional[Person]) -> str:
        """Construct prompt context with relevant memories, facts, and relationship."""
        context_parts = [LUMI_SYSTEM_PROMPT_BN]

        if current_person:
            context_parts.append(
                f"\n[বর্তমান শ্রোতা / কথোপকথনকারী]:\n"
                f"- নাম: {current_person.name}\n"
                f"- সম্পর্ক: {current_person.relationship}\n"
                f"- পূর্বে দেখা হয়েছে: {current_person.interaction_count} বার\n"
                f"- পছন্দ/নোট: {current_person.notes or 'নেই'}\n"
            )
            # Retrieve relevant facts
            facts = self.memory.recall_facts(person_id=current_person.id)
            if facts:
                fact_lines = "\n".join(f"  * {f.fact_text}" for f in facts[:5])
                context_parts.append(f"[শ্রোতা সম্পর্কিত স্মরণীয় তথ্য]:\n{fact_lines}\n")

        return "\n".join(context_parts)

    def generate_response(
        self,
        user_text: str,
        current_person: Optional[Person] = None,
    ) -> str:
        """Generate a natural conversational response in Bangla."""
        user_text = user_text.strip()
        if not user_text:
            return ""

        # Record incoming turn
        person_id = current_person.id if current_person else None
        self.memory.record_turn(speaker="user", text=user_text, person_id=person_id)
        
        # Build context
        system_prompt = self.build_context_prompt(current_person)
        
        # Get recent conversation history
        recent_turns = self.memory.get_recent_turns(limit=5, person_id=person_id)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for turn in recent_turns:
            role = "user" if turn.speaker == "user" else "assistant"
            messages.append({"role": role, "content": turn.text})
            
        # Ensure the latest message is the user text, if not already in recent turns
        if not messages or messages[-1]["content"] != user_text:
            messages.append({"role": "user", "content": user_text})

        # Try API
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="auto",
                    messages=messages
                )
                reply = response.choices[0].message.content
                if reply:
                    self.memory.record_turn(speaker="lumi", text=reply, person_id=person_id)
                    return reply
            except Exception as e:
                logger.error(f"Inworld API error: {e}. Falling back to heuristic.")

        # Contextual heuristic / offline responses
        lower = user_text.lower()

        # Greetings
        if any(w in lower for w in ["কেমন আছ", "কেমন আছেন", "kemon acho", "hi", "hello", "সালাম", "assalamu alaikum"]):
            name_part = f" {current_person.name}" if current_person else ""
            reply = f"আসসালামু আলাইকুম{name_part}! আমি লুমি, ভালো আছি। বলুন আপনাকে কীভাবে সাহায্য করতে পারি?"
            self.memory.record_turn(speaker="lumi", text=reply, person_id=person_id)
            return reply

        # Time / Date queries
        if any(w in lower for w in ["কয়টা বাজে", "সময় কত", "আজকের তারিখ", "time", "date"]):
            reply = self.tools.get_current_datetime()
            self.memory.record_turn(speaker="lumi", text=reply, person_id=person_id)
            return reply

        # Weather queries
        if any(w in lower for w in ["আবহাওয়া", "বৃষ্টি", "weather"]):
            reply = self.tools.get_weather("Dhaka")
            self.memory.record_turn(speaker="lumi", text=reply, person_id=person_id)
            return reply

        # Fallback intelligent response
        reply = f"আমি আপনার কথা বুঝতে পেরেছি। '{user_text}' বিষয়ে আমি সাহায্য করতে প্রস্তুত।"
        self.memory.record_turn(speaker="lumi", text=reply, person_id=person_id)
        return reply
