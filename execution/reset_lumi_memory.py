#!/usr/bin/env python3
"""
Reset LUMI Memory & Face Data.

Clears all stored facts, conversation turns, reminders, messages,
and face/voice embeddings, giving LUMI a completely fresh start.
Automatically re-initializes the clean default owner ('Palash') with
no corrupted memories or face data.

Usage:
    python execution/reset_lumi_memory.py [--no-backup]
"""

import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus


def reset_memory(keep_backup: bool = True) -> None:
    db_file = PROJECT_ROOT / "data" / "lumi.db"
    
    print("=" * 60)
    print("🧹 LUMI MEMORY & FACE DATA RESET UTILITY")
    print("=" * 60)

    if not db_file.exists():
        print(f"ℹ️ Database file not found at: {db_file}")
        print("Creating a fresh database...")
    else:
        if keep_backup:
            bak_name = f"lumi.db.bak_{int(time.time())}"
            bak_path = db_file.parent / bak_name
            shutil.copy2(db_file, bak_path)
            print(f"📦 Backup created at: data/{bak_name}")

    # Initialize Database & Manager
    db = Database(db_path=db_file)
    manager = MemoryManager(db)

    print("🗑️ Clearing all stored facts, conversations, and people...")

    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Clear all memory-related tables
        tables = [
            "facts",
            "conversations",
            "reminders",
            "messages",
            "people",
            "system_kv",
        ]
        for tbl in tables:
            try:
                cursor.execute(f"DELETE FROM {tbl};")
                print(f"  ✓ Cleared table: {tbl}")
            except Exception as e:
                print(f"  ⚠️ Could not clear {tbl}: {e}")

        # Clear FTS5 search index if it exists
        try:
            cursor.execute("DELETE FROM facts_fts;")
            print("  ✓ Cleared FTS5 search index (facts_fts)")
        except Exception:
            pass

    # Re-register clean Owner
    print("👤 Re-registering clean Owner profile ('Palash')...")
    owner = manager.remember_person(
        name="Palash",
        relationship="owner",
        consent_status=ConsentStatus.GRANTED,
        notes="Owner and creator of LUMI.",
        metadata={},  # completely empty face_embedding and voice_embedding
    )
    print(f"  ✓ Owner profile created: {owner.name} (ID: {owner.id})")
    print("  ✓ Face embedding: None (ready for fresh capture)")
    print("  ✓ Memory facts: 0")

    # Clean documents folder if present
    docs_dir = PROJECT_ROOT / "data" / "documents"
    if docs_dir.exists():
        for doc in docs_dir.glob("lumi_report_*.txt"):
            try:
                doc.unlink()
            except Exception:
                pass
        print("  ✓ Cleaned temporary diagnostic reports")

    print("=" * 60)
    print("✨ SUCCESS: LUMI's memory and face data have been completely reset!")
    print("LUMI is now fresh, clean, and ready to learn anew.")
    print("=" * 60)


if __name__ == "__main__":
    keep_bak = "--no-backup" not in sys.argv
    reset_memory(keep_backup=keep_bak)
