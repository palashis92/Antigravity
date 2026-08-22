-- LUMI SQLite Relational Memory Schema

CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    relationship TEXT NOT NULL DEFAULT 'friend',
    consent_status TEXT NOT NULL DEFAULT 'unknown',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 1,
    preferred_language TEXT NOT NULL DEFAULT 'bn',
    notes TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);
CREATE INDEX IF NOT EXISTS idx_people_consent ON people(consent_status);
CREATE INDEX IF NOT EXISTS idx_people_last_seen ON people(last_seen);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    person_id TEXT,
    category TEXT NOT NULL DEFAULT 'general',
    fact_text TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_facts_person ON facts(person_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    person_id TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    remind_at TEXT NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'none',
    is_completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders(remind_at, is_completed);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    speaker TEXT NOT NULL,
    person_id TEXT,
    language TEXT NOT NULL DEFAULT 'bn',
    text TEXT NOT NULL,
    intent TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);

CREATE TABLE IF NOT EXISTS system_kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
