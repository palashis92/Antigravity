import sqlite3
import os

def clean_alvarez():
    db_path = 'data/lumi.db'
    if not os.path.exists(db_path):
        print(f'Database not found at {db_path}')
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        c.execute('SELECT id, name FROM people')
        print('People before cleanup:', c.fetchall())

        c.execute("DELETE FROM people WHERE name='Álvarez'")
        deleted = c.rowcount
        conn.commit()

        print(f'Successfully deleted {deleted} ghost records named "Álvarez" from the local database.')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    clean_alvarez()
