"""Database connection manager and migration runner."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from ..core.logger import get_logger

logger = get_logger("database")


class Database:
    """SQLite database manager with WAL mode, foreign keys, and connection pooling."""

    def __init__(self, db_path: str | Path = "data/lumi.db", enable_wal: bool = True) -> None:
        self.db_path = str(db_path)
        self.enable_wal = enable_wal
        self._lock = threading.RLock()
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # Persistent single connection for in-memory database
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

        self.initialize_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional database connection context."""
        with self._lock:
            if self._memory_conn is not None:
                yield self._memory_conn
                self._memory_conn.commit()
                return

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                if self.enable_wal:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error during transaction: {e}", exc_info=True)
                raise
            finally:
                conn.close()

    def initialize_schema(self) -> None:
        """Read and apply schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.error(f"Schema file not found at {schema_path}")
            return

        sql = schema_path.read_text(encoding="utf-8")
        with self.get_connection() as conn:
            conn.executescript(sql)
        logger.info(f"Database initialized at '{self.db_path}' (WAL: {self.enable_wal})")

    def execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT query and return rows as dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return the last row ID or affected count."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid or cursor.rowcount
