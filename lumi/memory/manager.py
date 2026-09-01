"""High-level MemoryManager for person profiles, facts, consent, and reminders."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger
from .database import Database
from .models import (
    ConsentStatus,
    ConversationTurn,
    Fact,
    Person,
    Reminder,
    utc_now_iso,
)

logger = get_logger("memory_manager")


class MemoryManager:
    """Manages all persistent memory, privacy consent enforcement, and knowledge recall."""

    def __init__(self, db: Database) -> None:
        self.db = db
        logger.info("MemoryManager initialized.")

    # -------------------------------------------------------------------------
    # People & Consent Management
    # -------------------------------------------------------------------------

    def remember_person(
        self,
        name: str,
        relationship: str = "friend",
        consent_status: ConsentStatus = ConsentStatus.UNKNOWN,
        preferred_language: str = "bn",
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Person:
        """Create or update a person's profile."""
        existing = self.find_person_by_name(name)
        if existing:
            existing.last_seen = utc_now_iso()
            existing.interaction_count += 1
            if relationship and relationship != "friend":
                existing.relationship = relationship
            if consent_status != ConsentStatus.UNKNOWN:
                existing.consent_status = consent_status
            if notes:
                existing.notes = notes
            if metadata:
                existing.metadata.update(metadata)
            self.update_person(existing)
            return existing

        person = Person(
            name=name,
            relationship=relationship,
            consent_status=consent_status,
            first_seen=utc_now_iso(),
            last_seen=utc_now_iso(),
            interaction_count=1,
            preferred_language=preferred_language,
            notes=notes,
            metadata=metadata or {},
        )

        query = """
            INSERT INTO people (id, name, relationship, consent_status, first_seen, last_seen,
                                interaction_count, preferred_language, notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_write(
            query,
            (
                person.id,
                person.name,
                person.relationship,
                person.consent_status.value,
                person.first_seen,
                person.last_seen,
                person.interaction_count,
                person.preferred_language,
                person.notes,
                json.dumps(person.metadata),
            ),
        )
        logger.info(f"Registered new person: '{person.name}' (Consent: {person.consent_status.value})")
        return person

    def get_person(self, person_id: str) -> Optional[Person]:
        """Fetch person by primary ID."""
        rows = self.db.execute_query("SELECT * FROM people WHERE id = ?", (person_id,))
        if not rows:
            return None
        return Person.from_row(rows[0])

    def find_person_by_name(self, name: str) -> Optional[Person]:
        """Find person by case-insensitive name match."""
        rows = self.db.execute_query("SELECT * FROM people WHERE LOWER(name) = LOWER(?)", (name.strip(),))
        if not rows:
            return None
        return Person.from_row(rows[0])

    def list_people(self, only_consented: bool = False) -> List[Person]:
        """List all registered persons."""
        if only_consented:
            rows = self.db.execute_query(
                "SELECT * FROM people WHERE consent_status = ? ORDER BY last_seen DESC",
                (ConsentStatus.GRANTED.value,),
            )
        else:
            rows = self.db.execute_query("SELECT * FROM people ORDER BY last_seen DESC")
        return [Person.from_row(r) for r in rows]

    def update_person(self, person: Person) -> None:
        """Update an existing person profile."""
        query = """
            UPDATE people
            SET name = ?, relationship = ?, consent_status = ?, last_seen = ?,
                interaction_count = ?, preferred_language = ?, notes = ?, metadata = ?
            WHERE id = ?
        """
        self.db.execute_write(
            query,
            (
                person.name,
                person.relationship,
                person.consent_status.value,
                person.last_seen,
                person.interaction_count,
                person.preferred_language,
                person.notes,
                json.dumps(person.metadata),
                person.id,
            ),
        )

    def set_consent(self, person_id: str, status: ConsentStatus) -> bool:
        """Explicitly update privacy consent for a person."""
        person = self.get_person(person_id)
        if not person:
            return False
        person.consent_status = status
        self.update_person(person)
        logger.info(f"Updated consent for '{person.name}' to {status.value}")

        # If consent is denied, purge existing stored facts to respect privacy
        if status == ConsentStatus.DENIED:
            self.purge_person_facts(person_id)
        return True

    def forget_person(self, person_id: str) -> bool:
        """Permanently delete a person and all associated facts (Right to be Forgotten)."""
        person = self.get_person(person_id)
        name = person.name if person else person_id
        self.db.execute_write("DELETE FROM facts WHERE person_id = ?", (person_id,))
        self.db.execute_write("DELETE FROM reminders WHERE person_id = ?", (person_id,))
        self.db.execute_write("DELETE FROM conversations WHERE person_id = ?", (person_id,))
        affected = self.db.execute_write("DELETE FROM people WHERE id = ?", (person_id,))
        logger.info(f"Permanently forgot person '{name}' and purged all associated records.")
        return affected > 0

    # -------------------------------------------------------------------------
    # Facts & Knowledge Storage
    # -------------------------------------------------------------------------

    def remember_fact(
        self,
        fact_text: str,
        person_id: Optional[str] = None,
        category: str = "general",
        confidence: float = 1.0,
    ) -> Optional[Fact]:
        """Save a factual memory item, with privacy consent checks."""
        # Enforce consent policy if associated with a person
        if person_id is not None:
            person = self.get_person(person_id)
            if person and person.consent_status == ConsentStatus.DENIED:
                logger.warning(
                    f"Consent DENIED for person '{person.name}'. Fact will NOT be saved."
                )
                return None

        fact = Fact(
            person_id=person_id,
            category=category,
            fact_text=fact_text.strip(),
            confidence=confidence,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

        query = """
            INSERT INTO facts (id, person_id, category, fact_text, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_write(
            query,
            (
                fact.id,
                fact.person_id,
                fact.category,
                fact.fact_text,
                fact.confidence,
                fact.created_at,
                fact.updated_at,
            ),
        )
        logger.info(f"Saved fact [category: {category}]: '{fact_text}'")
        return fact

    def recall_facts(
        self,
        person_id: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Fact]:
        """Recall saved facts matching optional filters."""
        query = "SELECT * FROM facts WHERE 1=1"
        params: List[Any] = []

        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        if category:
            query += " AND category = ?"
            params.append(category)
        if search_query:
            query += " AND LOWER(fact_text) LIKE ?"
            params.append(f"%{search_query.lower()}%")

        query += " ORDER BY created_at DESC"
        rows = self.db.execute_query(query, tuple(params))
        return [Fact.from_row(r) for r in rows]

    def recall_facts_fts(
        self,
        search_query: str,
        person_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Fact]:
        """Full-text search for facts using SQLite FTS5 with BM25 ranking.

        Falls back to LIKE-based search if FTS5 table is not available.
        """
        try:
            # Build FTS5 query: join each word with OR for broad matching
            words = search_query.strip().split()
            if not words:
                return self.recall_facts(person_id=person_id)
            fts_query = " OR ".join(f'"{w}"' for w in words if len(w) > 1)
            if not fts_query:
                return self.recall_facts(person_id=person_id, search_query=search_query)

            query = """
                SELECT f.*, rank FROM facts f
                JOIN facts_fts ON f.rowid = facts_fts.rowid
                WHERE facts_fts MATCH ?
            """
            params: List[Any] = [fts_query]
            if person_id:
                query += " AND f.person_id = ?"
                params.append(person_id)
            query += " ORDER BY rank LIMIT ?"
            params.append(limit)

            rows = self.db.execute_query(query, tuple(params))
            return [Fact.from_row(r) for r in rows]
        except Exception as e:
            logger.warning(f"FTS5 search failed (falling back to LIKE): {e}")
            return self.recall_facts(person_id=person_id, search_query=search_query)

    def forget_fact(self, fact_id: str) -> bool:
        """Delete a specific fact."""
        affected = self.db.execute_write("DELETE FROM facts WHERE id = ?", (fact_id,))
        return affected > 0

    def purge_person_facts(self, person_id: str) -> int:
        """Remove all facts associated with a person."""
        return self.db.execute_write("DELETE FROM facts WHERE person_id = ?", (person_id,))

    # -------------------------------------------------------------------------
    # Reminders
    # -------------------------------------------------------------------------

    def create_reminder(
        self,
        title: str,
        remind_at_iso: str,
        description: str = "",
        recurrence: str = "none",
        person_id: Optional[str] = None,
    ) -> Reminder:
        """Schedule a new reminder."""
        reminder = Reminder(
            person_id=person_id,
            title=title.strip(),
            description=description.strip(),
            remind_at=remind_at_iso,
            recurrence=recurrence,
            is_completed=False,
            created_at=utc_now_iso(),
        )
        query = """
            INSERT INTO reminders (id, person_id, title, description, remind_at, recurrence, is_completed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_write(
            query,
            (
                reminder.id,
                reminder.person_id,
                reminder.title,
                reminder.description,
                reminder.remind_at,
                reminder.recurrence,
                1 if reminder.is_completed else 0,
                reminder.created_at,
            ),
        )
        logger.info(f"Created reminder: '{title}' at {remind_at_iso}")
        return reminder

    def get_due_reminders(self, as_of_iso: Optional[str] = None) -> List[Reminder]:
        """Fetch pending reminders that are due on or before as_of_iso."""
        target_time = as_of_iso or utc_now_iso()
        query = """
            SELECT * FROM reminders
            WHERE is_completed = 0 AND remind_at <= ?
            ORDER BY remind_at ASC
        """
        rows = self.db.execute_query(query, (target_time,))
        return [Reminder.from_row(r) for r in rows]

    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed."""
        affected = self.db.execute_write(
            "UPDATE reminders SET is_completed = 1 WHERE id = ?", (reminder_id,)
        )
        return affected > 0

    # -------------------------------------------------------------------------
    # Conversation History
    # -------------------------------------------------------------------------

    def record_turn(
        self,
        speaker: str,
        text: str,
        person_id: Optional[str] = None,
        language: str = "bn",
        intent: str = "chat",
    ) -> ConversationTurn:
        """Record a conversation turn."""
        turn = ConversationTurn(
            speaker=speaker,
            person_id=person_id,
            language=language,
            text=text.strip(),
            intent=intent,
            created_at=utc_now_iso(),
        )
        query = """
            INSERT INTO conversations (id, speaker, person_id, language, text, intent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_write(
            query,
            (
                turn.id,
                turn.speaker,
                turn.person_id,
                turn.language,
                turn.text,
                turn.intent,
                turn.created_at,
            ),
        )
        return turn

    def get_recent_turns(self, limit: int = 10, person_id: Optional[str] = None) -> List[ConversationTurn]:
        """Retrieve recent conversation history for context building."""
        query = "SELECT * FROM conversations WHERE 1=1"
        params: List[Any] = []
        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.db.execute_query(query, tuple(params))
        turns = [
            ConversationTurn(
                id=r["id"],
                speaker=r["speaker"],
                person_id=r.get("person_id"),
                language=r.get("language", "bn"),
                text=r["text"],
                intent=r.get("intent", "chat"),
                created_at=r["created_at"],
            )
            for r in rows
        ]
        # Return in chronological order
        return list(reversed(turns))

    # -------------------------------------------------------------------------
    # Convenience Wrappers (Spec Required)
    # -------------------------------------------------------------------------

    def remember(self, key: str, value: str, person_id: Optional[str] = None) -> Optional[Fact]:
        """Convenience wrapper to store a key-value fact."""
        return self.remember_fact(fact_text=f"{key}: {value}", person_id=person_id, category=key)

    def recall(self, key: str, person_id: Optional[str] = None) -> List[Fact]:
        """Convenience wrapper to retrieve facts matching a key."""
        return self.recall_facts(person_id=person_id, category=key)

    def search_memory(self, query: str, person_id: Optional[str] = None) -> List[Fact]:
        """Convenience wrapper to search across all facts."""
        return self.recall_facts(person_id=person_id, search_query=query)

    def update_memory(self, fact_id: str, new_value: str) -> bool:
        """Convenience wrapper to update an existing fact."""
        affected = self.db.execute_write(
            "UPDATE facts SET fact_text = ?, updated_at = ? WHERE id = ?",
            (new_value, utc_now_iso(), fact_id)
        )
        return affected > 0

    def forget(self, fact_id: str) -> bool:
        """Convenience wrapper to delete a specific fact."""
        return self.forget_fact(fact_id)
