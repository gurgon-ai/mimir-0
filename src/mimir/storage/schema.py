"""The SQLite schema, as a versioned migration ladder.

``schema_version`` is the first table created and the first thing checked at boot.
DESIGN §10: schema versioning + a tiny migration runner + a startup check are a v0
doctrine, not a later nicety. A misconfigured store fails loud — it NEVER silently
falls back to another store.

To evolve the schema: append a new ``(version, [statements])`` tuple to ``MIGRATIONS``
and bump nothing else — ``CURRENT_SCHEMA_VERSION`` is derived from the ladder length.
Never edit a past migration; only append. Each migration runs once, in order, inside a
transaction (see ``migrate.py``).
"""

from __future__ import annotations

# Each entry: (target_version, [SQL statements to reach it from the previous version]).
# Migration N takes the DB from version N-1 to version N.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # The version marker itself. Single-row table; the runner maintains it.
            "CREATE TABLE schema_version (version INTEGER NOT NULL)",
            # The one knowledge table for the v0 spine. Typed layers beyond `memory`
            # (understanding, procedural, …) arrive as new `kind` values and/or new
            # tables in later migrations — never by mutating this one in place.
            """
            CREATE TABLE memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT    NOT NULL,
                kind          TEXT    NOT NULL DEFAULT 'memory',
                evidence_tier TEXT    NOT NULL DEFAULT 'conversation',
                confidence    REAL    NOT NULL DEFAULT 0.7,
                salience      REAL    NOT NULL DEFAULT 1.0,
                embedding     BLOB,
                provenance    TEXT    NOT NULL DEFAULT 'conversation',
                user          TEXT,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                meta          TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX idx_memories_user ON memories(user)",
            "CREATE INDEX idx_memories_kind ON memories(kind)",
        ],
    ),
    (
        2,
        [
            # v0.1 document ingestion: a chunk is just a memory with evidence_tier='document'
            # and a `source` pointing at the file it came from. The column lets re-ingest
            # replace a document's chunks cleanly (delete-by-source). NULL for non-document rows.
            "ALTER TABLE memories ADD COLUMN source TEXT",
            "CREATE INDEX idx_memories_source ON memories(source)",
        ],
    ),
    (
        3,
        [
            # Identity anchors: foundational, operator-established identity facts (name, operator,
            # location, purpose) that ground the always-on self-model from the very first boot,
            # before any history exists. Key-value with upsert semantics — re-answering updates.
            "CREATE TABLE identity ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL,"
            " established_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        4,
        [
            # Entity graph: subject-relation-object triples — the 'what is connected' layer
            # (DESIGN §3a). Indexed on subject and object for 1-2 hop traversal; a unique
            # expression index dedupes case-insensitively so the same triple isn't stored twice.
            "CREATE TABLE triples ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " subject TEXT NOT NULL,"
            " relation TEXT NOT NULL,"
            " object TEXT NOT NULL,"
            " user TEXT,"
            " provenance TEXT NOT NULL DEFAULT 'conversation',"
            " confidence REAL NOT NULL DEFAULT 0.8,"
            " created_at REAL NOT NULL"
            ")",
            "CREATE INDEX idx_triples_subject ON triples(subject)",
            "CREATE INDEX idx_triples_object ON triples(object)",
            "CREATE UNIQUE INDEX idx_triples_unique ON triples("
            " lower(subject), lower(relation), lower(object), coalesce(user, ''))",
        ],
    ),
    (
        5,
        [
            # Sleep/consolidation: archived memories are excluded from active recall but kept in
            # the store (archiving ≠ disbelieving, DESIGN §3c) — a resurfaced one is still trusted.
            "ALTER TABLE memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0",
            "CREATE INDEX idx_memories_archived ON memories(archived)",
        ],
    ),
    (
        6,
        [
            # Procedural memory: learned reasoning habits as trigger→procedure pairs (DESIGN §3a).
            # The trigger is embedded for cosine matching; `uses` tracks how proven a habit is.
            "CREATE TABLE procedures ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " trigger TEXT NOT NULL,"
            " procedure TEXT NOT NULL,"
            " trigger_embedding BLOB,"
            " user TEXT,"
            " confidence REAL NOT NULL DEFAULT 0.7,"
            " uses INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        7,
        [
            # The fleet catalogue (DESIGN §5): a persisted snapshot of every (node, model) found,
            # with its weight/family/quant. return_time + quality are filled by Phase 2 benchmarking
            # (NULL until then). Rebuilt on each scan.
            "CREATE TABLE model_catalogue ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " node TEXT NOT NULL,"
            " model TEXT NOT NULL,"
            " family TEXT NOT NULL DEFAULT '',"
            " params_b REAL NOT NULL DEFAULT 0,"
            " quantization TEXT NOT NULL DEFAULT '',"
            " context_length INTEGER NOT NULL DEFAULT 0,"
            " capabilities TEXT NOT NULL DEFAULT '[]',"
            " return_time REAL,"
            " quality REAL,"
            " scanned_at REAL NOT NULL"
            ")",
        ],
    ),
]

# Derived, never hand-edited: the version this code expects an opened DB to be at.
CURRENT_SCHEMA_VERSION: int = MIGRATIONS[-1][0] if MIGRATIONS else 0

# What the startup check asserts actually exists, so a half-migrated or foreign DB is
# caught loudly rather than limping along. Maps table -> required columns.
EXPECTED_SHAPE: dict[str, set[str]] = {
    "schema_version": {"version"},
    "memories": {
        "id",
        "text",
        "kind",
        "evidence_tier",
        "confidence",
        "salience",
        "embedding",
        "provenance",
        "user",
        "created_at",
        "last_accessed",
        "access_count",
        "meta",
        "source",
        "archived",
    },
    "identity": {"key", "value", "established_at"},
    "triples": {
        "id",
        "subject",
        "relation",
        "object",
        "user",
        "provenance",
        "confidence",
        "created_at",
    },
    "procedures": {
        "id",
        "trigger",
        "procedure",
        "trigger_embedding",
        "user",
        "confidence",
        "uses",
        "created_at",
    },
    "model_catalogue": {
        "id",
        "node",
        "model",
        "family",
        "params_b",
        "quantization",
        "context_length",
        "capabilities",
        "return_time",
        "quality",
        "scanned_at",
    },
}
