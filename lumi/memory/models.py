"""Data models for LUMI memory and knowledge storage."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 string format."""
    return datetime.now(timezone.utc).isoformat()


class ConsentStatus(str, Enum):
    """Explicit privacy and consent status for remembered persons."""
    UNKNOWN = "unknown"        # Temporary / unconfirmed person
    REQUESTED = "requested"    # Robot asked for permission, pending reply
    GRANTED = "granted"        # User explicitly consented to long-term storage
    DENIED = "denied"          # User explicitly refused storage (do not record facts)


@dataclass
class Person:
    """Profile of a recognized or introduced person."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    relationship: str = "friend"  # e.g., "owner", "family", "guest"
    consent_status: ConsentStatus = ConsentStatus.UNKNOWN
    first_seen: str = field(default_factory=utc_now_iso)
    last_seen: str = field(default_factory=utc_now_iso)
    interaction_count: int = 1
    preferred_language: str = "bn"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def face_embedding(self) -> Optional[List[float]]:
        return self.metadata.get("face_embedding")

    @face_embedding.setter
    def face_embedding(self, value: Optional[List[float]]) -> None:
        if value is None:
            self.metadata.pop("face_embedding", None)
        else:
            self.metadata["face_embedding"] = value

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["consent_status"] = self.consent_status.value
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Person:
        meta = row.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return cls(
            id=row["id"],
            name=row["name"],
            relationship=row.get("relationship", "friend"),
            consent_status=ConsentStatus(row.get("consent_status", "unknown")),
            first_seen=row.get("first_seen", utc_now_iso()),
            last_seen=row.get("last_seen", utc_now_iso()),
            interaction_count=row.get("interaction_count", 1),
            preferred_language=row.get("preferred_language", "bn"),
            notes=row.get("notes", ""),
            metadata=meta,
        )


@dataclass
class Fact:
    """A single persistent memory fact tagged to a person or general knowledge."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person_id: Optional[str] = None
    category: str = "general"     # "preference", "event", "interest", "date"
    fact_text: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Fact:
        return cls(
            id=row["id"],
            person_id=row.get("person_id"),
            category=row.get("category", "general"),
            fact_text=row.get("fact_text", ""),
            confidence=float(row.get("confidence", 1.0)),
            created_at=row.get("created_at", utc_now_iso()),
            updated_at=row.get("updated_at", utc_now_iso()),
        )


@dataclass
class Reminder:
    """A scheduled reminder event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person_id: Optional[str] = None
    title: str = ""
    description: str = ""
    remind_at: str = ""           # ISO 8601 target time
    recurrence: str = "none"      # "none", "daily", "weekly"
    is_completed: bool = False
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Reminder:
        return cls(
            id=row["id"],
            person_id=row.get("person_id"),
            title=row.get("title", ""),
            description=row.get("description", ""),
            remind_at=row.get("remind_at", ""),
            recurrence=row.get("recurrence", "none"),
            is_completed=bool(row.get("is_completed", 0)),
            created_at=row.get("created_at", utc_now_iso()),
        )


@dataclass
class ConversationTurn:
    """A single turn in recent conversation history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speaker: str = "user"         # "user" | "lumi"
    person_id: Optional[str] = None
    language: str = "bn"
    text: str = ""
    intent: str = "chat"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
