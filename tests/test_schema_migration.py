"""Executable spec for the schema doctrine (DESIGN §10): versioned, checked, fail-loud."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mimir.errors import SchemaError
from mimir.storage.gateway import StorageGateway
from mimir.storage.migrate import check_schema, current_version, run_migrations
from mimir.storage.schema import CURRENT_SCHEMA_VERSION, EXPECTED_SHAPE, MIGRATIONS


def test_fresh_db_migrates_and_checks(db_path: str) -> None:
    with StorageGateway(db_path):
        pass
    conn = sqlite3.connect(db_path)
    try:
        assert current_version(conn) == CURRENT_SCHEMA_VERSION
        for table in EXPECTED_SHAPE:
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
        check_schema(conn)  # must not raise
    finally:
        conn.close()


def test_migration_is_idempotent(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        assert run_migrations(conn) == CURRENT_SCHEMA_VERSION
        # Running again changes nothing and does not error.
        assert run_migrations(conn) == CURRENT_SCHEMA_VERSION
    finally:
        conn.close()


def test_foreign_db_fails_loud(tmp_path: Path) -> None:
    """A 'memories' table with no version marker must be refused, not migrated over."""
    db = str(tmp_path / "foreign.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, junk TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaError, match="not created by Mimir"):
        StorageGateway(db)


def test_newer_db_than_code_fails_loud(tmp_path: Path) -> None:
    db = str(tmp_path / "newer.db")
    conn = sqlite3.connect(db)
    try:
        run_migrations(conn)
        conn.execute("UPDATE schema_version SET version = ?", (CURRENT_SCHEMA_VERSION + 5,))
        conn.commit()
        with pytest.raises(SchemaError, match="only understands"):
            run_migrations(conn)
    finally:
        conn.close()


def test_v1_db_upgrades_to_current(tmp_path: Path) -> None:
    """A store stamped at v1 is migrated forward (e.g. the v2 `source` column is added)."""
    db = str(tmp_path / "v1.db")
    conn = sqlite3.connect(db)
    try:
        # Build a v1-shaped store and stamp it at version 1.
        for stmt in MIGRATIONS[0][1]:
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.commit()
        v1_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "source" not in v1_cols  # v1 predates document ingestion

        # Upgrade forward and confirm the v2 column landed and the schema check passes.
        assert run_migrations(conn) == CURRENT_SCHEMA_VERSION
        v_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "source" in v_cols
        check_schema(conn)
    finally:
        conn.close()


def test_missing_column_fails_loud(tmp_path: Path) -> None:
    """If the store is stamped current but is missing a column, the check is loud."""
    db = str(tmp_path / "shape.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_SCHEMA_VERSION,))
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT)")
        conn.commit()
        with pytest.raises(SchemaError, match="missing required column"):
            check_schema(conn)
    finally:
        conn.close()
