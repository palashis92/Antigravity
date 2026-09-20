"""Setup/Migrate LUMI database to ensure Mizan is Owner and Palash is Creator."""

import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path("data/lumi.db")
    if not db_path.exists():
        print(f"Database {db_path} does not exist yet. It will be created on first start.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='people'")
    if not cur.fetchone():
        print("Table 'people' does not exist yet.")
        conn.close()
        return

    cur.execute("SELECT id, name, relationship FROM people WHERE LOWER(name) LIKE '%mizan%'")
    mizan = cur.fetchone()

    cur.execute("SELECT id, name, relationship FROM people WHERE LOWER(name) LIKE '%palash%'")
    palash = cur.fetchone()

    if mizan:
        cur.execute(
            "UPDATE people SET relationship='owner', consent_status='granted', notes='Owner and primary user of LUMI.' WHERE id=?",
            (mizan[0],)
        )
        print(f"[OK] Updated Mizan (ID: {mizan[0]}) as OWNER.")
    else:
        cur.execute(
            """INSERT INTO people (id, name, relationship, notes, consent_status, created_at, updated_at)
               VALUES ('person_mizan_owner', 'Mizan', 'owner', 'Owner and primary user of LUMI.', 'granted', datetime('now'), datetime('now'))"""
        )
        print("[OK] Registered Mizan as OWNER.")

    if palash:
        cur.execute(
            "UPDATE people SET relationship='creator', consent_status='granted', notes='Creator and developer of LUMI.' WHERE id=?",
            (palash[0],)
        )
        print(f"[OK] Updated Palash (ID: {palash[0]}) as CREATOR.")
    else:
        cur.execute(
            """INSERT INTO people (id, name, relationship, notes, consent_status, created_at, updated_at)
               VALUES ('person_palash_creator', 'Palash', 'creator', 'Creator and developer of LUMI.', 'granted', datetime('now'), datetime('now'))"""
        )
        print("[OK] Registered Palash as CREATOR.")

    conn.commit()
    conn.close()
    print("Database owner configuration complete: Mizan = Owner, Palash = Creator.")


if __name__ == "__main__":
    main()
