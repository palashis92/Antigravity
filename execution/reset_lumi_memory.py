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

    # Reset Mem0 Cloud memories
    reset_mem0_cloud()

    print("=" * 60)
    print("✨ SUCCESS: LUMI's local and cloud memories have been completely reset!")
    print("LUMI is now fresh, clean, and ready to learn anew.")
    print("=" * 60)


def reset_mem0_cloud() -> None:
    """Wipe all memories and users stored in Mem0 Cloud API (api.mem0.ai)."""
    import json
    import urllib.request
    import urllib.error

    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MEM0_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        print("\n☁️ MEM0_API_KEY not found in environment or .env — skipping Mem0 Cloud wipe.")
        print("  (If you use Mem0 Cloud, ensure MEM0_API_KEY is set in ~/Antigravity/.env on the Pi)")
        return

    print("\n☁️ Connecting to Mem0 Cloud API (api.mem0.ai) to wipe cloud memories...")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    # 1. Delete all memories
    try:
        mem_url = "https://api.mem0.ai/v1/memories/"
        req = urllib.request.Request(mem_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        memories = data if isinstance(data, list) else data.get("results", [])
        if memories:
            print(f"  Found {len(memories)} memories in Mem0 Cloud. Deleting...")
            del_count = 0
            for m in memories:
                mid = m.get("id")
                if not mid:
                    continue
                del_req = urllib.request.Request(f"{mem_url}{mid}/", headers=headers, method="DELETE")
                try:
                    with urllib.request.urlopen(del_req, timeout=10):
                        del_count += 1
                except Exception as e:
                    print(f"  ⚠️ Could not delete memory {mid}: {e}")
            print(f"  ✓ Deleted {del_count}/{len(memories)} memories from Mem0 Cloud.")
        else:
            print("  ✓ No memories found in Mem0 Cloud.")
    except Exception as e:
        print(f"  ⚠️ Error wiping Mem0 Cloud memories: {e}")

    # 2. Delete all users from Mem0 Cloud
    try:
        users_url = "https://api.mem0.ai/v1/users/"
        req = urllib.request.Request(users_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            users_data = json.loads(response.read().decode())

        users = users_data if isinstance(users_data, list) else users_data.get("results", [])
        if users:
            print(f"  Found {len(users)} users in Mem0 Cloud. Deleting...")
            del_user_count = 0
            for u in users:
                uid = u.get("id") or u.get("user_id")
                if not uid:
                    continue
                del_req = urllib.request.Request(f"{users_url}{uid}/", headers=headers, method="DELETE")
                try:
                    with urllib.request.urlopen(del_req, timeout=10):
                        del_user_count += 1
                except Exception as e:
                    print(f"  ⚠️ Could not delete user {uid}: {e}")
            print(f"  ✓ Deleted {del_user_count}/{len(users)} users from Mem0 Cloud.")
    except Exception as e:
        # Some accounts may not support users endpoint or it might be empty
        pass


if __name__ == "__main__":
    keep_bak = "--no-backup" not in sys.argv
    reset_memory(keep_backup=keep_bak)
