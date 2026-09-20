"""Setup/Migrate LUMI database to ensure Mizan is Owner and Palash is Creator."""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus


def main() -> None:
    db_path = Path("data/lumi.db")
    db = Database(db_path=str(db_path), enable_wal=False)
    mem = MemoryManager(db)

    # 1. Setup Mizan as Owner
    mizan = mem.find_person_by_name("Mizan")
    if mizan:
        db.execute_write(
            "UPDATE people SET relationship = ?, consent_status = ?, notes = ? WHERE id = ?",
            ("owner", ConsentStatus.GRANTED.value, "Owner and primary user of LUMI.", mizan.id),
        )
        print(f"[OK] Updated Mizan (ID: {mizan.id}) as OWNER.")
    else:
        mizan = mem.remember_person(
            name="Mizan",
            relationship="owner",
            consent_status=ConsentStatus.GRANTED,
            notes="Owner and primary user of LUMI.",
        )
        print(f"[OK] Registered Mizan (ID: {mizan.id}) as OWNER.")

    # 2. Setup Palash as Creator/Developer
    palash = mem.find_person_by_name("Palash")
    if palash:
        db.execute_write(
            "UPDATE people SET relationship = ?, consent_status = ?, notes = ? WHERE id = ?",
            ("creator", ConsentStatus.GRANTED.value, "Creator and developer of LUMI.", palash.id),
        )
        print(f"[OK] Updated Palash (ID: {palash.id}) as CREATOR.")
    else:
        palash = mem.remember_person(
            name="Palash",
            relationship="creator",
            consent_status=ConsentStatus.GRANTED,
            notes="Creator and developer of LUMI.",
        )
        print(f"[OK] Registered Palash (ID: {palash.id}) as CREATOR.")

    print("Database owner configuration complete: Mizan = Owner, Palash = Creator.")


if __name__ == "__main__":
    main()
