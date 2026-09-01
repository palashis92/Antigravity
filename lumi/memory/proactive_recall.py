"""Proactive Memory Recall Engine for LUMI.

Automatically monitors conversation transcriptions and injects
relevant memories into the Gemini Live session without waiting
for the LLM to call recall_facts.

This solves the core problem: voice-mode LLMs are biased toward
fast responses and rarely voluntarily add tool-call round-trips.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..core.event_bus import Event, EventBus
    from ..memory.manager import MemoryManager

logger = get_logger("memory.proactive_recall")

# Common Bangla stopwords that should not trigger recall
_BANGLA_STOPWORDS: Set[str] = {
    "আমি", "তুমি", "আপনি", "সে", "তারা", "আমরা", "এটা", "ওটা", "সেটা",
    "কি", "কে", "কেন", "কেমন", "কোথায়", "কবে", "কিভাবে", "কত",
    "হচ্ছে", "করছি", "করো", "করেছি", "হয়েছে", "ছিল", "আছে", "নেই",
    "এবং", "বা", "কিন্তু", "তবে", "যদি", "তাহলে", "কারণ",
    "একটু", "একটা", "এই", "ওই", "সেই", "যে", "তো", "না", "হ্যাঁ",
    "ভালো", "খুব", "অনেক", "একটু", "আর", "তা", "দেখো", "বলো",
    "please", "okay", "the", "is", "are", "was", "were", "what", "how",
    "yes", "no", "and", "but", "for", "with", "this", "that",
}

# Patterns that strongly suggest user wants memory recall
_RECALL_PATTERNS = [
    re.compile(r"মনে আছে", re.IGNORECASE),       # "Do you remember..."
    re.compile(r"মনে পড়ে", re.IGNORECASE),       # "Do you recall..."
    re.compile(r"আগে বলেছিলাম", re.IGNORECASE),  # "I told you before..."
    re.compile(r"last time", re.IGNORECASE),
    re.compile(r"remember when", re.IGNORECASE),
    re.compile(r"did I tell you", re.IGNORECASE),
    re.compile(r"কবে .+ হয়েছিল", re.IGNORECASE),  # "When did X happen?"
    re.compile(r"কেমন চলছে", re.IGNORECASE),      # "How's X going?"
]


class ProactiveRecallEngine:
    """Monitors conversation flow and auto-injects relevant memories.

    Instead of hoping the LLM will call recall_facts, this engine
    programmatically detects topics in user speech and silently
    injects matching memories as system context.
    """

    # Minimum seconds between proactive injections
    INJECTION_COOLDOWN = 15.0
    # Maximum number of facts to inject per trigger
    MAX_FACTS_PER_INJECTION = 5
    # Minimum user utterance length to process
    MIN_UTTERANCE_LENGTH = 5

    def __init__(
        self,
        memory: MemoryManager,
        mem0: Any,
        realtime_voice: Any,
        event_bus: EventBus,
    ) -> None:
        self.memory = memory
        self.mem0 = mem0
        self.voice = realtime_voice
        self.event_bus = event_bus

        self._last_injection_time: float = 0.0
        self._injected_fact_ids: Set[str] = set()
        self._known_names_cache: List[str] = []
        self._cache_refresh_time: float = 0.0
        self._lock = threading.Lock()

        # Subscribe to conversation events
        self.event_bus.subscribe("conversation.turn_complete", self._on_turn_complete)
        logger.info("ProactiveRecallEngine initialized and listening.")

    # ------------------------------------------------------------------
    # Event Handler
    # ------------------------------------------------------------------

    def _on_turn_complete(self, event: Event) -> None:
        """Called after every conversation turn. Runs in EventBus worker thread."""
        user_text = event.data.get("user", "")
        if not user_text or len(user_text.strip()) < self.MIN_UTTERANCE_LENGTH:
            return

        # Rate limit
        now = time.time()
        if now - self._last_injection_time < self.INJECTION_COOLDOWN:
            return

        # Run in background to avoid blocking the event bus
        thread = threading.Thread(
            target=self._process_utterance,
            args=(user_text.strip(), now),
            daemon=True,
            name="ProactiveRecall_Worker",
        )
        thread.start()

    # ------------------------------------------------------------------
    # Core Pipeline
    # ------------------------------------------------------------------

    def _process_utterance(self, user_text: str, trigger_time: float) -> None:
        """Main pipeline: extract topics → search memory → inject context."""
        with self._lock:
            try:
                # 1. Detect if user mentioned a known person
                mentioned_people = self._detect_person_mentions(user_text)

                # 2. Check for explicit recall patterns
                has_recall_intent = any(p.search(user_text) for p in _RECALL_PATTERNS)

                # 3. Extract keywords/topics
                topics = self._extract_topics(user_text)

                # 4. Decide whether to recall
                if not mentioned_people and not has_recall_intent and len(topics) < 2:
                    return  # Not enough signal to trigger proactive recall

                # 5. Fetch relevant facts
                all_facts = []

                # 5a. Facts about mentioned people
                for person in mentioned_people:
                    person_facts = self.memory.recall_facts(person_id=person.id)
                    for f in person_facts[:3]:
                        f._mentioned_person = person.name
                    all_facts.extend(person_facts[:3])

                # 5b. Topic-based search
                for topic in topics[:3]:  # Limit to top 3 topics
                    topic_facts = self.memory.recall_facts(search_query=topic)
                    all_facts.extend(topic_facts[:3])

                # 5c. Mem0 Cloud search (if available and has recall)
                if hasattr(self.mem0, "recall_facts_sync") and topics:
                    combined_query = " ".join(topics[:3])
                    cloud_str = self.mem0.recall_facts_sync(
                        person_id="default", query=combined_query
                    )
                    if cloud_str:
                        # Create a pseudo-fact for cloud results
                        from ..memory.models import Fact
                        cloud_fact = Fact(
                            fact_text=cloud_str,
                            category="cloud_recall",
                        )
                        all_facts.append(cloud_fact)

                # 6. Deduplicate and filter already-injected
                unique_facts = self._deduplicate(all_facts)
                new_facts = [
                    f for f in unique_facts
                    if f.id not in self._injected_fact_ids
                ]

                if not new_facts:
                    return

                # 7. Apply recency scoring and take top N
                scored = self._score_by_recency(new_facts)
                top_facts = scored[: self.MAX_FACTS_PER_INJECTION]

                # 8. Format and inject
                context = self._format_injection(top_facts, mentioned_people)
                if context:
                    self.voice.inject_context(context)
                    for f in top_facts:
                        self._injected_fact_ids.add(f.id)
                    self._last_injection_time = time.time()
                    logger.info(
                        f"ProactiveRecall: Injected {len(top_facts)} facts "
                        f"(topics: {topics[:3]}, people: {[p.name for p in mentioned_people]})"
                    )

            except Exception as e:
                logger.error(f"ProactiveRecall error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Topic & Person Detection
    # ------------------------------------------------------------------

    def _detect_person_mentions(self, text: str) -> list:
        """Check if user mentioned any known person's name."""
        # Refresh name cache every 60 seconds
        now = time.time()
        if now - self._cache_refresh_time > 60.0:
            try:
                self._known_names_cache = self.memory.list_people()
                self._cache_refresh_time = now
            except Exception:
                pass

        text_lower = text.lower()
        mentioned = []
        for person in self._known_names_cache:
            # Match whole name or first name
            name_lower = person.name.lower()
            first_name = name_lower.split()[0] if name_lower else ""
            if name_lower in text_lower or (len(first_name) > 2 and first_name in text_lower):
                mentioned.append(person)
        return mentioned

    def _extract_topics(self, text: str) -> List[str]:
        """Extract searchable keywords from user utterance.

        Uses simple heuristics instead of an LLM call:
        - Split into words
        - Remove stopwords (Bangla + English)
        - Keep words longer than 3 characters
        - Preserve multi-word phrases in quotes
        """
        topics: List[str] = []

        # Extract quoted phrases first
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        topics.extend(quoted)

        # Split remaining text into words
        words = re.split(r'[\s,।?!.;:]+', text)
        for word in words:
            word_clean = word.strip().strip('"\'')
            if not word_clean:
                continue
            if len(word_clean) <= 3:
                continue
            if word_clean.lower() in _BANGLA_STOPWORDS:
                continue
            # Skip pure numbers
            if word_clean.isdigit():
                continue
            topics.append(word_clean)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for t in topics:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)
        return unique

    # ------------------------------------------------------------------
    # Scoring & Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, facts: list) -> list:
        """Remove near-duplicate facts based on normalized text."""
        seen_texts: Set[str] = set()
        unique: list = []
        for f in facts:
            # Normalize: lowercase, strip extra whitespace
            normalized = " ".join(f.fact_text.lower().split())
            # Consider facts with >80% overlap as duplicates
            is_dup = False
            for seen in seen_texts:
                if self._text_similarity(normalized, seen) > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                seen_texts.add(normalized)
                unique.append(f)
        return unique

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple word-overlap Jaccard similarity."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _score_by_recency(facts: list) -> list:
        """Score facts by recency: newer facts score higher."""
        from datetime import datetime

        now = datetime.now()
        scored = []
        for f in facts:
            try:
                created = datetime.fromisoformat(f.created_at)
                age_days = max((now - created).days, 0)
            except (ValueError, TypeError):
                age_days = 365  # Assume old if unparseable

            # Exponential decay: halve score every 30 days
            recency_score = 0.5 ** (age_days / 30.0)
            confidence = getattr(f, "confidence", 1.0)
            f._score = confidence * recency_score
            scored.append(f)

        scored.sort(key=lambda x: x._score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Context Formatting
    # ------------------------------------------------------------------

    def _format_injection(self, facts: list, mentioned_people: list) -> str:
        """Format facts into a natural system context injection."""
        if not facts:
            return ""

        lines = []
        for f in facts:
            person_tag = ""
            if hasattr(f, "_mentioned_person"):
                person_tag = f" [about {f._mentioned_person}]"
            lines.append(f"- {f.fact_text}{person_tag}")

        people_str = ""
        if mentioned_people:
            names = ", ".join(p.name for p in mentioned_people)
            people_str = f" The user just mentioned: {names}."

        context = (
            "[PROACTIVE MEMORY CONTEXT: The following relevant memories were "
            "automatically retrieved based on the current conversation."
            f"{people_str} "
            "Use these naturally in your response — do NOT say 'I checked my memory' "
            "or 'according to my records'. Just incorporate the knowledge as if you "
            "naturally remember it.]\n"
            + "\n".join(lines)
        )
        return context

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_injection_cache(self) -> None:
        """Reset the set of already-injected fact IDs.

        Call this when a new person starts interacting or after
        a long period of inactivity to allow re-injection.
        """
        with self._lock:
            self._injected_fact_ids.clear()
            logger.debug("ProactiveRecall injection cache cleared.")
