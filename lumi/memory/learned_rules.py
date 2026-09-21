"""Persistent store for rules, advice, and behavioral adaptations taught by the user."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger

logger = get_logger("learned_rules")


class LearnedRulesStore:
    """Manages persistent behavioral rules learned from user feedback."""

    def __init__(self, file_path: Optional[str | Path] = None, db: Optional[Any] = None) -> None:
        self.db = db
        if file_path is None:
            # Default to data/learned_rules.json relative to repository root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.file_path = base_dir / "data" / "learned_rules.json"
        else:
            self.file_path = Path(file_path)

        self._lock = threading.Lock()
        self._ensure_table()
        self._ensure_file()

    def _ensure_table(self) -> None:
        if self.db is not None:
            try:
                self.db.execute_write(
                    "CREATE TABLE IF NOT EXISTS learned_rules ("
                    "id TEXT PRIMARY KEY, "
                    "user_feedback TEXT NOT NULL, "
                    "adapted_rule TEXT NOT NULL, "
                    "category TEXT NOT NULL DEFAULT 'general', "
                    "created_at TEXT NOT NULL, "
                    "timestamp REAL NOT NULL"
                    ");"
                )
            except Exception as e:
                logger.debug(f"Could not initialize learned_rules table: {e}")

    def _ensure_file(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not initialize learned rules file at {self.file_path}: {e}")

    def load_rules(self) -> List[Dict[str, Any]]:
        """Load all stored rules from SQLite or disk."""
        with self._lock:
            if self.db is not None:
                try:
                    rows = self.db.execute_query(
                        "SELECT id, user_feedback, adapted_rule, category, created_at, timestamp "
                        "FROM learned_rules ORDER BY timestamp ASC;"
                    )
                    if rows:
                        return rows
                except Exception as e:
                    logger.debug(f"DB load rules note: {e}")

            if not self.file_path.exists():
                return []
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Failed to load learned rules: {e}")
                return []

    def save_rules(self, rules: List[Dict[str, Any]]) -> bool:
        """Write rules atomically to disk."""
        with self._lock:
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.file_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(rules, f, indent=2, ensure_ascii=False)
                # Atomic replace
                temp_path.replace(self.file_path)
                return True
            except Exception as e:
                logger.error(f"Failed to save learned rules: {e}")
                return False

    def add_rule(self, feedback: str, adapted_rule: str, category: str = "general") -> Dict[str, Any]:
        """Add a new learned behavioral rule and persist to SQLite and disk."""
        rules = self.load_rules()
        rule_entry = {
            "id": f"rule_{uuid.uuid4().hex[:8]}",
            "timestamp": time.time(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_feedback": feedback.strip(),
            "adapted_rule": adapted_rule.strip(),
            "category": category.strip().lower() if category else "general",
        }
        rules.append(rule_entry)
        self.save_rules(rules)

        if self.db is not None:
            try:
                self.db.execute_write(
                    "INSERT OR REPLACE INTO learned_rules (id, user_feedback, adapted_rule, category, created_at, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?);",
                    (
                        rule_entry["id"],
                        rule_entry["user_feedback"],
                        rule_entry["adapted_rule"],
                        rule_entry["category"],
                        rule_entry["created_at"],
                        rule_entry["timestamp"],
                    ),
                )
            except Exception as e:
                logger.debug(f"DB add rule note: {e}")

        logger.info(f"Learned rule added: '{adapted_rule}' (from: '{feedback}')")
        return rule_entry

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by its ID."""
        rules = self.load_rules()
        initial_len = len(rules)
        filtered = [r for r in rules if r.get("id") != rule_id]
        if len(filtered) < initial_len:
            self.save_rules(filtered)
            if self.db is not None:
                try:
                    self.db.execute_write("DELETE FROM learned_rules WHERE id = ?;", (rule_id,))
                except Exception as e:
                    logger.debug(f"DB delete rule note: {e}")
            logger.info(f"Deleted learned rule {rule_id}.")
            return True
        return False

    def clear_all(self) -> None:
        """Clear all stored rules."""
        self.save_rules([])
        if self.db is not None:
            try:
                self.db.execute_write("DELETE FROM learned_rules;")
            except Exception as e:
                logger.debug(f"DB clear rules note: {e}")
        logger.info("Cleared all learned rules.")

    def get_rules_prompt(self) -> str:
        """Format all stored rules into concise instructions for Gemini system prompt."""
        rules = self.load_rules()
        if not rules:
            return ""

        lines = []
        for r in rules:
            category = r.get("category", "general").upper()
            rule_text = r.get("adapted_rule", "")
            feedback = r.get("user_feedback", "")
            if rule_text:
                if feedback:
                    lines.append(f"- [{category}] {rule_text} (ইউজারের দেওয়া দিকনির্দেশনা: \"{feedback}\")")
                else:
                    lines.append(f"- [{category}] {rule_text}")

        return "\n".join(lines)
