"""Unit tests for SQLite database manager."""

from lumi.memory.database import Database


def test_in_memory_database() -> None:
    db = Database(db_path=":memory:", enable_wal=False)
    # Check that schema tables exist
    tables = db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [t["name"] for t in tables]
    assert "people" in table_names
    assert "facts" in table_names
    assert "reminders" in table_names
    assert "conversations" in table_names


def test_insert_and_query() -> None:
    db = Database(db_path=":memory:", enable_wal=False)
    db.execute_write(
        "INSERT INTO system_kv (key, value, updated_at) VALUES (?, ?, ?)",
        ("test_key", "test_val", "2026-08-20T00:00:00Z"),
    )
    rows = db.execute_query("SELECT * FROM system_kv WHERE key = ?", ("test_key",))
    assert len(rows) == 1
    assert rows[0]["value"] == "test_val"
