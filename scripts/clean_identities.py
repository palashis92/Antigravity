"""Database identity cleanup utility for LUMI.

Removes accidentally enrolled stopword profiles ('চলছে', 'আমি', 'অলরেডি'),
and allows consolidating/purging misidentified test profiles ('ছিন্না', 'মর্তুজা হাসান')
into Mizan (Owner).
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and manage LUMI identity database.")
    parser.add_argument("--purge-corrupted", action="store_true", help="Purge known accidental profiles (চলছে, আমি, অলরেডি, ছিন্না, মর্তুজা হাসান)")
    parser.add_argument("--delete-name", type=str, default="", help="Delete a specific person profile by name")
    args = parser.parse_args()

    db_path = Path("data/lumi.db")
    if not db_path.exists():
        print(f"[!] Database file not found at {db_path.resolve()}")
        return

    db = Database(db_path=str(db_path), enable_wal=False)
    mem = MemoryManager(db)

    # 1. Automatic purge of invalid stopword names
    purged = mem.purge_invalid_persons()
    if purged > 0:
        print(f"[OK] Automatically purged {purged} invalid/stopword profile(s).")

    # 2. Targeted name deletion if requested
    if args.delete_name:
        p = mem.find_person_by_name(args.delete_name)
        if p:
            mem.delete_person(p.id)
            print(f"[OK] Deleted profile '{p.name}' (id={p.id}).")
        else:
            print(f"[-] Profile '{args.delete_name}' not found.")

    # 3. Purge corrupted test profiles if requested
    if args.purge_corrupted:
        targets = ["চলছে", "আমি", "অলরেডি", "ছিন্না", "মর্তুজা হাসান"]
        for target in targets:
            p = mem.find_person_by_name(target)
            if p:
                mem.delete_person(p.id)
                print(f"[OK] Purged corrupted profile '{p.name}' (id={p.id}).")

    # 4. Print current registered persons
    print("\n--- Current Registered Profiles in data/lumi.db ---")
    people = mem.list_people()
    if not people:
        print("  (No people registered)")
    else:
        for idx, p in enumerate(people, 1):
            emb_count = len(p.face_embeddings)
            print(f"  {idx}. Name: '{p.name}' | Role: {p.relationship} | Face Embeddings: {emb_count} | ID: {p.id}")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    main()
