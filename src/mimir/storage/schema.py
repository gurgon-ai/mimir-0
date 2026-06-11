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
    },
}
