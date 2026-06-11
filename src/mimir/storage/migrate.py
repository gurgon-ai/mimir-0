"""The migration runner and the startup schema check.

Two jobs, both load-bearing for the DESIGN §10 doctrine:

1. ``run_migrations`` walks the ladder in ``schema.py``, applying each pending step once,
   in order, each inside a transaction. It is idempotent: a fully-migrated DB is a no-op.
2. ``check_schema`` is the startup guard. It asserts the opened DB is at the version this
   code expects and has the columns this code reads. A mismatch raises ``SchemaError``
   with an instruction — it never silently limps on or falls back to another store.
"""

from __future__ import annotations

import sqlite3

from ..errors import MigrationError, SchemaError
from .schema import CURRENT_SCHEMA_VERSION, EXPECTED_SHAPE, MIGRATIONS


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def current_version(conn: sqlite3.Connection) -> int:
    """The schema version recorded in the DB, or 0 if it has never been migrated."""
    if not _table_exists(conn, "schema_version"):
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        # The table exists but holds no marker — a half-built or corrupt store.
        raise SchemaError(
            "schema_version table exists but is empty — the database is in an "
            "inconsistent state. Move the .db file aside and let Mimir recreate it, "
            "or restore a known-good backup."
        )
    return int(row[0])


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply every pending migration in order. Returns the version after running.

    Guards against the two silent-death traps: a foreign database that happens to sit at
    our path, and a code/DB version skew where the DB is *newer* than this code knows how
    to handle. Both fail loud (DESIGN §10).
    """
    version = current_version(conn)

    # A foreign / pre-existing store that was never stamped by Mimir but already holds
    # our tables: do NOT blindly run migration 1 over it. Fail loud instead.
    if version == 0 and _table_exists(conn, "memories"):
        raise SchemaError(
            "found a 'memories' table but no schema_version marker — this database was "
            "not created by Mimir, or is corrupt. Refusing to migrate over it. Point "
            "Mimir at a fresh path, or move this file aside."
        )

    if version > CURRENT_SCHEMA_VERSION:
        raise SchemaError(
            f"database is at schema version {version}, but this build only understands "
            f"up to {CURRENT_SCHEMA_VERSION}. You are running older code against a newer "
            f"store. Upgrade Mimir, or restore a matching backup. Refusing to proceed."
        )

    for target, statements in MIGRATIONS:
        if target <= version:
            continue
        try:
            with conn:  # one transaction per migration step
                for stmt in statements:
                    conn.execute(stmt)
                if target == 1:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (target,))
                else:
                    conn.execute("UPDATE schema_version SET version = ?", (target,))
        except sqlite3.Error as exc:
            # Re-raise as a typed, loud failure — never swallow (DESIGN §10).
            raise MigrationError(
                f"migration to schema version {target} failed: {exc}"
            ) from exc
        version = target

    return version


def check_schema(conn: sqlite3.Connection) -> None:
    """Assert the DB matches what this code expects. Raise ``SchemaError`` if not.

    Run at boot, after ``run_migrations``. This is the guard that turns 'the store
    quietly drifted' from an invisible bug into a loud, actionable failure.
    """
    version = current_version(conn)
    if version != CURRENT_SCHEMA_VERSION:
        raise SchemaError(
            f"schema version mismatch: database is at {version}, code expects "
            f"{CURRENT_SCHEMA_VERSION}. Run migrations, or check you opened the right file."
        )

    for table, required in EXPECTED_SHAPE.items():
        if not _table_exists(conn, table):
            raise SchemaError(
                f"required table {table!r} is missing from the database. The store is "
                f"incomplete or foreign — refusing to use it."
            )
        present = _columns(conn, table)
        missing = required - present
        if missing:
            raise SchemaError(
                f"table {table!r} is missing required column(s): {sorted(missing)}. "
                f"The store does not match this code — refusing to use it."
            )
