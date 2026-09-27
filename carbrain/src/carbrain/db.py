"""SQLite schema and connection helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

-- Vehicle identity: brand > family > generation > version > model year > configuration revision
CREATE TABLE IF NOT EXISTS brand (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS family (
    id TEXT PRIMARY KEY,
    brand_id TEXT NOT NULL REFERENCES brand(id),
    name TEXT NOT NULL,
    segment TEXT,
    body TEXT,
    status TEXT NOT NULL DEFAULT 'on_sale',
    powertrains TEXT NOT NULL DEFAULT '[]',
    notes TEXT
);
CREATE TABLE IF NOT EXISTS generation (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES family(id),
    name TEXT NOT NULL,
    start_model_year INTEGER,
    end_model_year INTEGER
);
CREATE TABLE IF NOT EXISTS version (
    id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES family(id),
    generation_id TEXT REFERENCES generation(id),
    name TEXT NOT NULL,
    engine TEXT,
    transmission TEXT,
    fuel TEXT
);
CREATE TABLE IF NOT EXISTS version_year (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES version(id),
    model_year INTEGER NOT NULL,
    config_revision INTEGER NOT NULL DEFAULT 1,
    UNIQUE (version_id, model_year, config_revision)
);

-- How each source names a vehicle, mapped to our IDs
CREATE TABLE IF NOT EXISTS external_mapping (
    source_id TEXT NOT NULL,
    external_key TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('family', 'generation', 'version', 'version_year')),
    target_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('auto', 'confirmed', 'rejected')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, external_key)
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    external_key TEXT NOT NULL,
    candidate_level TEXT,
    candidate_id TEXT,
    confidence REAL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT,
    UNIQUE (source_id, external_key)
);
-- Every distinct vehicle label seen in a source, with volume, for mapping review
CREATE TABLE IF NOT EXISTS label_inventory (
    source_id TEXT NOT NULL,
    label TEXT NOT NULL,
    as_of TEXT NOT NULL,
    quantity REAL NOT NULL,
    PRIMARY KEY (source_id, label, as_of)
);

-- A vehicle as one source describes it (e.g. a PBEV table row), with its raw attributes
CREATE TABLE IF NOT EXISTS source_vehicle (
    source_id TEXT NOT NULL,
    external_key TEXT NOT NULL,
    as_of TEXT NOT NULL,
    attributes TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (source_id, external_key, as_of)
);

-- Publishers of car information (specialist media, creators) and their channels
CREATE TABLE IF NOT EXISTS publisher (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    focus TEXT NOT NULL DEFAULT '[]',
    evidence_kind TEXT NOT NULL DEFAULT '[]',
    notes TEXT
);
CREATE TABLE IF NOT EXISTS channel (
    id TEXT PRIMARY KEY,
    publisher_id TEXT NOT NULL REFERENCES publisher(id),
    platform TEXT NOT NULL,
    handle TEXT,
    url TEXT,
    external_id TEXT,
    status TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    checked_on TEXT,
    stats TEXT,
    stats_fetched_at TEXT
);
-- One article, video or post, as metadata only (rights decide what is kept, and how long)
CREATE TABLE IF NOT EXISTS content_item (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    channel_id TEXT NOT NULL REFERENCES channel(id),
    external_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    url TEXT,
    title TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    expires_at TEXT,
    UNIQUE (channel_id, external_id)
);
CREATE TABLE IF NOT EXISTS content_mention (
    content_id INTEGER NOT NULL REFERENCES content_item(id) ON DELETE CASCADE,
    family_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    matched_text TEXT,
    PRIMARY KEY (content_id, family_id)
);

-- Raw archive: every fetched file, content-hashed
CREATE TABLE IF NOT EXISTS snapshot (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    content_type TEXT,
    path TEXT NOT NULL,
    parsed_at TEXT,
    parser_version TEXT,
    UNIQUE (source_id, url, sha256)
);

-- Facts. A changed value for the same key becomes a new revision; nothing is overwritten.
CREATE TABLE IF NOT EXISTS observation (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    dims TEXT NOT NULL DEFAULT '{}',
    as_of_start TEXT NOT NULL,
    as_of_end TEXT NOT NULL,
    value REAL,
    unit TEXT NOT NULL,
    price_type TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    snapshot_id INTEGER REFERENCES snapshot(id),
    revision INTEGER NOT NULL DEFAULT 1,
    UNIQUE (source_id, metric, subject_type, subject_id, dims, as_of_start, as_of_end, revision)
);
CREATE INDEX IF NOT EXISTS observation_lookup
    ON observation (metric, subject_type, subject_id, as_of_start);
CREATE VIEW IF NOT EXISTS observation_current AS
SELECT o.* FROM observation o
WHERE o.revision = (
    SELECT MAX(o2.revision) FROM observation o2
    WHERE o2.source_id = o.source_id AND o2.metric = o.metric
      AND o2.subject_type = o.subject_type AND o2.subject_id = o.subject_id
      AND o2.dims = o.dims AND o2.as_of_start = o.as_of_start AND o2.as_of_end = o.as_of_end
);

-- Text evidence (owner reports, comments) with a hard expiry from the rights registry
CREATE TABLE IF NOT EXISTS text_item (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    vehicle_level TEXT,
    vehicle_id TEXT,
    match_confidence REAL,
    text TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    expires_at TEXT,
    UNIQUE (source_id, external_id)
);

-- Structural breaks (taxes, protocols, launches) that explain moves in the data
CREATE TABLE IF NOT EXISTS event (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    scope TEXT,
    source_url TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    files_fetched INTEGER NOT NULL DEFAULT 0,
    files_new INTEGER NOT NULL DEFAULT 0,
    observations_inserted INTEGER NOT NULL DEFAULT 0,
    observations_revised INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database, creating the schema on first use."""
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
