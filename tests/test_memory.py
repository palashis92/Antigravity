"""Unit tests for MemoryManager: privacy consent, facts, reminders, and forget operations."""

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus


def setup_memory() -> MemoryManager:
    db = Database(db_path=":memory:", enable_wal=False)
    return MemoryManager(db)


def test_person_lifecycle() -> None:
    mem = setup_memory()
    person = mem.remember_person("Tasin", relationship="friend", consent_status=ConsentStatus.GRANTED)
    assert person.name == "Tasin"
    assert person.consent_status == ConsentStatus.GRANTED

    fetched = mem.find_person_by_name("tasin")
    assert fetched is not None
    assert fetched.id == person.id


def test_privacy_consent_enforcement() -> None:
    mem = setup_memory()
    # Create person with denied consent
    person = mem.remember_person("Stranger", consent_status=ConsentStatus.DENIED)

    # Attempt to save a fact about this person
    fact = mem.remember_fact("Likes tea", person_id=person.id, category="preference")
    assert fact is None  # Must be blocked by privacy consent check

    # Allow consent and retry
    mem.set_consent(person.id, ConsentStatus.GRANTED)
    fact_allowed = mem.remember_fact("Likes tea", person_id=person.id, category="preference")
    assert fact_allowed is not None
    assert fact_allowed.fact_text == "Likes tea"


def test_right_to_be_forgotten() -> None:
    mem = setup_memory()
    person = mem.remember_person("Bob", consent_status=ConsentStatus.GRANTED)
    mem.remember_fact("Works at bank", person_id=person.id)
    mem.create_reminder("Call Bob", "2026-08-21T10:00:00Z", person_id=person.id)

    assert len(mem.recall_facts(person_id=person.id)) == 1

    # Forget person
    success = mem.forget_person(person.id)
    assert success is True
    assert mem.get_person(person.id) is None
    assert len(mem.recall_facts(person_id=person.id)) == 0


def test_reminders_and_due_check() -> None:
    mem = setup_memory()
    r1 = mem.create_reminder("Meeting", "2026-08-20T12:00:00Z")
    r2 = mem.create_reminder("Future", "2026-08-25T12:00:00Z")

    due = mem.get_due_reminders(as_of_iso="2026-08-20T15:00:00Z")
    assert len(due) == 1
    assert due[0].id == r1.id

    mem.complete_reminder(r1.id)
    due_after = mem.get_due_reminders(as_of_iso="2026-08-20T15:00:00Z")
    assert len(due_after) == 0


def test_conversation_history() -> None:
    mem = setup_memory()
    mem.record_turn(speaker="user", text="কেমন আছো?", language="bn")
    mem.record_turn(speaker="lumi", text="আমি ভালো আছি!", language="bn")

    history = mem.get_recent_turns(limit=5)
    assert len(history) == 2
    assert history[0].speaker == "user"
    assert history[1].speaker == "lumi"
