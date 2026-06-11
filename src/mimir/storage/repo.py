"""Persistence operations for memories, expressed over the storage gateway.

Every write here goes through ``gateway.submit`` (the single-writer law); every read
through ``gateway.read``. This module is the only place that knows the SQL for the
``memories`` table — callers work in ``Memory`` objects, never rows.
"""

from __future__ import annotations

import json
import sqlite3
import time

from .gateway import Priority, StorageGateway
from .models import (
    EvidenceTier,
    Memory,
    MemoryKind,
    blob_to_embedding,
    embedding_to_blob,
)

_COLUMNS = (
    "id, text, kind, evidence_tier, confidence, salience, embedding, "
    "provenance, user, created_at, last_accessed, access_count, meta"
)
# The same columns minus the auto-assigned id, for INSERT.
_INSERT_COLUMNS = (
    "text, kind, evidence_tier, confidence, salience, embedding, "
    "provenance, user, created_at, last_accessed, access_count, meta"
)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        text=row["text"],
        kind=MemoryKind(row["kind"]),
        evidence_tier=EvidenceTier.from_key(row["evidence_tier"]),
        confidence=row["confidence"],
        salience=row["salience"],
        embedding=blob_to_embedding(row["embedding"]),
        provenance=row["provenance"],
        user=row["user"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        meta=json.loads(row["meta"]),
    )


def save_memory(gateway: StorageGateway, mem: Memory) -> int:
    """Persist a memory and return its new row id. Mutates ``mem.id`` in place.

    Timestamps default to 'now' at write time if the caller left them at 0.0, so callers
    don't have to thread a clock through everything.
    """
    now = time.time()
    if mem.created_at == 0.0:
        mem.created_at = now
    if mem.last_accessed == 0.0:
        mem.last_accessed = mem.created_at

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            f"""
            INSERT INTO memories ({_INSERT_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem.text,
                mem.kind.value,
                mem.evidence_tier.key,
                mem.confidence,
                mem.salience,
                embedding_to_blob(mem.embedding),
                mem.provenance,
                mem.user,
                mem.created_at,
                mem.last_accessed,
                mem.access_count,
                json.dumps(mem.meta),
            ),
        )
        return int(cur.lastrowid or 0)

    mem.id = gateway.submit(_write)
    return mem.id


def get_memory(gateway: StorageGateway, memory_id: int) -> Memory | None:
    def _read(conn: sqlite3.Connection) -> Memory | None:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    return gateway.read(_read)


def list_memories(
    gateway: StorageGateway,
    *,
    user: str | None = None,
    kind: MemoryKind = MemoryKind.MEMORY,
) -> list[Memory]:
    """All memories of a kind, optionally scoped to a user (plus user-agnostic rows).

    User scoping is inclusive of rows with no user (``user IS NULL``) so shared/global
    facts surface alongside a specific user's.
    """

    def _read(conn: sqlite3.Connection) -> list[Memory]:
        if user is None:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ?", (kind.value,)
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? AND (user = ? OR user IS NULL)",
                (kind.value, user),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    return gateway.read(_read)


def latest_sentinel_note(gateway: StorageGateway, user: str | None) -> Memory | None:
    """The most recent sentinel note for a user — the high-attention end slot's content."""

    def _read(conn: sqlite3.Connection) -> Memory | None:
        if user is None:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? "
                f"ORDER BY created_at DESC, id DESC LIMIT 1",
                (MemoryKind.SENTINEL_NOTE.value,),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? "
                f"AND (user = ? OR user IS NULL) ORDER BY created_at DESC, id DESC LIMIT 1",
                (MemoryKind.SENTINEL_NOTE.value, user),
            ).fetchone()
        return _row_to_memory(row) if row else None

    return gateway.read(_read)


def record_access(gateway: StorageGateway, memory_ids: list[int]) -> None:
    """Mark memories as accessed: bump access_count, refresh last_accessed, bump salience.

    Access measures *relevance, not truth* (DESIGN §3c): it raises salience, never
    confidence. Salience is capped at 1.0 so a hot memory saturates rather than runs away.

    Routed fire-and-forget at TOUCH priority — these are the highest-frequency, least-critical
    writes, so they must never make a user turn wait. They are NOT coalesced: the count
    increment is non-idempotent, so collapsing touches would lose increments. Tests/shutdown
    that need them observed call ``gateway.flush()``.
    """
    if not memory_ids:
        return
    now = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ?, "
            "salience = MIN(1.0, salience + 0.1) WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )

    gateway.submit_async(_write, priority=Priority.TOUCH)


def count_memories(gateway: StorageGateway, *, kind: MemoryKind | None = None) -> int:
    def _read(conn: sqlite3.Connection) -> int:
        if kind is None:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE kind = ?", (kind.value,)
            ).fetchone()
        return int(row[0])

    return gateway.read(_read)
