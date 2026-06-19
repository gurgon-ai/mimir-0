"""Persistence operations for memories, expressed over the storage gateway.

Every write here goes through ``gateway.submit`` (the single-writer law); every read
through ``gateway.read``. This module is the only place that knows the SQL for the
``memories`` table — callers work in ``Memory`` objects, never rows.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from .gateway import Priority, StorageGateway
from .models import (
    CatalogueEntry,
    EvidenceTier,
    LibraryClaim,
    LibraryDocument,
    LibraryPage,
    Memory,
    MemoryKind,
    Procedure,
    Triple,
    blob_to_embedding,
    embedding_to_blob,
)

_COLUMNS = (
    "id, text, kind, evidence_tier, confidence, salience, embedding, "
    "provenance, user, source, created_at, last_accessed, access_count, meta, archived"
)
# The same columns minus the auto-assigned id, for INSERT.
_INSERT_COLUMNS = (
    "text, kind, evidence_tier, confidence, salience, embedding, "
    "provenance, user, source, created_at, last_accessed, access_count, meta, archived"
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
        source=row["source"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        archived=bool(row["archived"]),
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                mem.source,
                mem.created_at,
                mem.last_accessed,
                mem.access_count,
                json.dumps(mem.meta),
                int(mem.archived),
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


def delete_memory(gateway: StorageGateway, memory_id: int) -> None:
    """Permanently remove a memory by id (a hard delete, distinct from sleep's archive).

    Used for user-governed edits — re-answering an onboarding question replaces its row, and the
    Profile panel can drop a fact outright. Goes through the single writer like every mutation.
    """
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    gateway.submit(_write)


def update_memory(
    gateway: StorageGateway, memory_id: int, *,
    text: str | None = None, salience: float | None = None, confidence: float | None = None,
) -> None:
    """Edit a memory's text, salience, and/or confidence in place (user-governed review e.g. the
    graph editor; or the deep-idle convergence boost — a re-derived insight earns belief).

    Only the provided fields change. ``embedding`` is left as-is; a re-embed on text change is a
    later refinement (recall still works on the old vector + keyword overlap)."""
    sets, params = [], []
    if text is not None:
        sets.append("text = ?")
        params.append(text)
    if salience is not None:
        sets.append("salience = ?")
        params.append(salience)
    if confidence is not None:
        sets.append("confidence = ?")
        params.append(confidence)
    if not sets:
        return
    params.append(memory_id)

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", tuple(params))

    gateway.submit(_write)


def record_interaction(
    gateway: StorageGateway, ts: float, user: str | None = None, *, keep: int = 5000
) -> None:
    """Append one interaction timestamp (one per turn) to the durable log, pruning to the most
    recent ``keep`` rows so it never grows unbounded. Powers temporal-awareness baselines (§3e)."""
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO interactions (ts, user) VALUES (?, ?)", (ts, user))
        conn.execute(
            "DELETE FROM interactions WHERE id NOT IN "
            "(SELECT id FROM interactions ORDER BY ts DESC LIMIT ?)",
            (keep,),
        )

    gateway.submit(_write)


def kv_get(gateway: StorageGateway, key: str) -> str | None:
    """Read a value from the generic key→value store, or ``None`` if unset."""
    def _read(conn: sqlite3.Connection) -> str | None:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    return gateway.read(_read)


def kv_set(gateway: StorageGateway, key: str, value: str) -> None:
    """Upsert a value into the generic key→value store (opaque text; callers JSON-encode)."""
    ts = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, ts),
        )

    gateway.submit(_write)


# -- the council forum (DESIGN §5a) ---------------------------------------------------

def create_forum_thread(
    gateway: StorageGateway, *, question: str, source: str = "council", verdict: str = "",
) -> int:
    """Open a forum thread for a deliberation; returns its id."""
    ts = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "INSERT INTO forum_threads (question, status, source, verdict, created_at) "
            "VALUES (?, 'open', ?, ?, ?)",
            (question, source, verdict, ts),
        )
        return int(cur.lastrowid or 0)

    return gateway.submit(_write)


def add_forum_post(
    gateway: StorageGateway, *, thread_id: int, author: str, kind: str, content: str,
    model: str = "", node: str = "",
) -> None:
    """Append a post (persona position, verdict, or user comment) to a thread."""
    ts = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO forum_posts (thread_id, author, kind, model, node, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, author, kind, model, node, content, ts),
        )

    gateway.submit(_write)


def set_forum_thread_verdict(gateway: StorageGateway, thread_id: int, verdict: str) -> None:
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE forum_threads SET verdict = ? WHERE id = ?", (verdict, thread_id))

    gateway.submit(_write)


def set_forum_thread_status(gateway: StorageGateway, thread_id: int, status: str) -> None:
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE forum_threads SET status = ? WHERE id = ?", (status, thread_id))

    gateway.submit(_write)


def delete_forum_thread(gateway: StorageGateway, thread_id: int) -> None:
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM forum_posts WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM forum_threads WHERE id = ?", (thread_id,))

    gateway.submit(_write)


def delete_forum_post(gateway: StorageGateway, post_id: int) -> None:
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM forum_posts WHERE id = ?", (post_id,))

    gateway.submit(_write)


def prune_forum_threads(gateway: StorageGateway, keep: int) -> int:
    """Keep only the most recent ``keep`` forum threads; delete older ones and their posts.
    Returns threads removed. Each deliberation's verdict is *also* stored as a memory, so trimming
    old threads bounds the browsable forum — it does not lose the recalled understanding."""

    def _write(conn: sqlite3.Connection) -> int:
        stale = [
            r[0] for r in conn.execute(
                "SELECT id FROM forum_threads ORDER BY created_at DESC, id DESC LIMIT -1 OFFSET ?",
                (max(0, keep),),
            ).fetchall()
        ]
        if not stale:
            return 0
        marks = ",".join("?" * len(stale))
        conn.execute(f"DELETE FROM forum_posts WHERE thread_id IN ({marks})", stale)
        conn.execute(f"DELETE FROM forum_threads WHERE id IN ({marks})", stale)
        return len(stale)

    return gateway.submit(_write)


def list_forum_threads(gateway: StorageGateway) -> list[dict[str, Any]]:
    """All threads, newest first, each with its post count (for the forum list)."""
    def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT t.id, t.question, t.status, t.source, t.verdict, t.created_at, "
            "       (SELECT COUNT(*) FROM forum_posts p WHERE p.thread_id = t.id) "
            "FROM forum_threads t ORDER BY t.created_at DESC"
        ).fetchall()
        return [
            {"id": r[0], "question": r[1], "status": r[2], "source": r[3], "verdict": r[4],
             "created_at": r[5], "posts": r[6]}
            for r in rows
        ]

    return gateway.read(_read)


def get_forum_thread(gateway: StorageGateway, thread_id: int) -> dict[str, Any] | None:
    """One thread with all its posts in time order, or ``None`` if it's gone."""
    def _read(conn: sqlite3.Connection) -> dict[str, Any] | None:
        t = conn.execute(
            "SELECT id, question, status, source, verdict, created_at FROM forum_threads "
            "WHERE id = ?", (thread_id,)
        ).fetchone()
        if t is None:
            return None
        posts = conn.execute(
            "SELECT id, author, kind, model, node, content, created_at FROM forum_posts "
            "WHERE thread_id = ? ORDER BY created_at ASC, id ASC", (thread_id,)
        ).fetchall()
        return {
            "id": t[0], "question": t[1], "status": t[2], "source": t[3], "verdict": t[4],
            "created_at": t[5],
            "posts": [
                {"id": p[0], "author": p[1], "kind": p[2], "model": p[3], "node": p[4],
                 "content": p[5], "created_at": p[6]}
                for p in posts
            ],
        }

    return gateway.read(_read)


def save_narrative(
    gateway: StorageGateway, *, scope: str, period: str, narrative: str,
    source_count: int = 0, created_at: float | None = None,
) -> None:
    """Upsert one temporal narrative — regenerating a (scope, period) replaces it (idempotent)."""
    ts = time.time() if created_at is None else created_at

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO narratives "
            "(scope, period, narrative, source_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (scope, period, narrative, source_count, ts),
        )

    gateway.submit(_write)


def list_narratives(gateway: StorageGateway, scope: str) -> list[dict[str, Any]]:
    """All narratives in a scope, newest period first."""
    def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT period, narrative, source_count, created_at FROM narratives "
            "WHERE scope = ? ORDER BY period DESC",
            (scope,),
        ).fetchall()
        return [
            {"period": r[0], "narrative": r[1], "source_count": r[2], "created_at": r[3]}
            for r in rows
        ]

    return gateway.read(_read)


def get_narrative(gateway: StorageGateway, scope: str, period: str) -> str | None:
    """The narrative text for one (scope, period), or ``None`` if not yet generated."""
    def _read(conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT narrative FROM narratives WHERE scope = ? AND period = ?", (scope, period)
        ).fetchone()
        return row[0] if row else None

    return gateway.read(_read)


def prune_narratives(gateway: StorageGateway, scope: str, keep: int) -> None:
    """Keep only the ``keep`` newest periods in a scope (retention cap; older ones compress up)."""
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "DELETE FROM narratives WHERE scope = ? AND period NOT IN "
            "(SELECT period FROM narratives WHERE scope = ? ORDER BY period DESC LIMIT ?)",
            (scope, scope, keep),
        )

    gateway.submit(_write)


def record_conversation_turn(
    gateway: StorageGateway, *, user: str | None, user_text: str, reply: str,
    session_id: str | None = None, keep: int = 500, created_at: float | None = None,
) -> None:
    """Append one exchange to the durable conversation log, pruning to the most recent ``keep``.

    This is the lasting turn history (for UI restore + model continuity), distinct from the capped
    EXCHANGE recency buffer that working-memory compression clears. ``session_id`` groups it into a
    distinct conversation."""
    ts = time.time() if created_at is None else created_at

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO conversation (user, user_text, reply, created_at, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (user, user_text, reply, ts, session_id),
        )
        conn.execute(
            "DELETE FROM conversation WHERE id NOT IN "
            "(SELECT id FROM conversation ORDER BY id DESC LIMIT ?)",
            (keep,),
        )

    gateway.submit(_write)


def recent_conversation(
    gateway: StorageGateway, *, user: str | None = None, limit: int = 20,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """The most recent exchanges, oldest→newest (for restore + replaying to the model). With a
    ``user``, includes user-agnostic rows too. With ``session_id``, only that conversation."""
    where, params = [], []
    if user is not None:
        where.append("(user = ? OR user IS NULL)")
        params.append(user)
    if session_id is not None:
        where.append("session_id = ?")
        params.append(session_id)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"SELECT user_text, reply, created_at FROM conversation{clause} "
            "ORDER BY id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [{"user_text": r[0], "reply": r[1], "created_at": r[2]} for r in reversed(rows)]

    return gateway.read(_read)


def last_conversation_meta(gateway: StorageGateway) -> dict[str, Any] | None:
    """The newest turn's ``session_id`` + ``created_at`` (or ``None``) — for deciding whether to
    continue the last session or start a new one after a gap/restart."""
    def _read(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT session_id, created_at FROM conversation ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {"session_id": row[0], "created_at": row[1]} if row else None

    return gateway.read(_read)


def list_sessions(
    gateway: StorageGateway, *, user: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Distinct conversations, most recent first: ``session_id``, a one-line ``summary`` (the
    session's first user message), ``started``/``last`` timestamps, and turn ``count``."""
    scope = "WHERE (user = ? OR user IS NULL)" if user is not None else ""
    args: tuple[Any, ...] = (user,) if user is not None else ()

    def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"""
            SELECT session_id, COUNT(*) AS n, MIN(created_at) AS started, MAX(created_at) AS last,
                   (SELECT user_text FROM conversation c2
                    WHERE c2.session_id IS s.session_id ORDER BY c2.id ASC LIMIT 1) AS summary
            FROM conversation s {scope}
            GROUP BY session_id ORDER BY last DESC LIMIT ?
            """,
            (*args, limit),
        ).fetchall()
        return [
            {"session_id": r[0], "count": r[1], "started": r[2], "last": r[3], "summary": r[4]}
            for r in rows
        ]

    return gateway.read(_read)


def interaction_history(
    gateway: StorageGateway, *, user: str | None = None, since_ts: float = 0.0
) -> list[float]:
    """Interaction timestamps at/after ``since_ts``, oldest→newest. With ``user`` set, includes
    user-agnostic rows too (matching the rest of the store's user scoping)."""
    def _read(conn: sqlite3.Connection) -> list[float]:
        if user is None:
            rows = conn.execute(
                "SELECT ts FROM interactions WHERE ts >= ? ORDER BY ts", (since_ts,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts FROM interactions WHERE ts >= ? AND (user = ? OR user IS NULL) "
                "ORDER BY ts",
                (since_ts, user),
            ).fetchall()
        return [float(r[0]) for r in rows]

    return gateway.read(_read)


def list_memories(
    gateway: StorageGateway,
    *,
    user: str | None = None,
    kind: MemoryKind = MemoryKind.MEMORY,
    exclude_sources: set[str] | None = None,
) -> list[Memory]:
    """All memories of a kind, optionally scoped to a user (plus user-agnostic rows).

    User scoping is inclusive of rows with no user (``user IS NULL``) so shared/global
    facts surface alongside a specific user's. ``exclude_sources`` drops rows from those source
    files at the SQL layer (so toggled-off documents aren't even loaded — recall speed back).
    """
    excl = sorted(exclude_sources) if exclude_sources else []

    def _read(conn: sqlite3.Connection) -> list[Memory]:
        clauses = ["kind = ?", "archived = 0"]
        params: list[object] = [kind.value]
        if user is not None:
            clauses.append("(user = ? OR user IS NULL)")
            params.append(user)
        if excl:
            ph = ",".join("?" * len(excl))
            clauses.append(f"(source IS NULL OR source NOT IN ({ph}))")
            params.extend(excl)
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    return gateway.read(_read)


def recent_by_kind(
    gateway: StorageGateway,
    kind: MemoryKind,
    *,
    user: str | None = None,
    limit: int = 5,
) -> list[Memory]:
    """The most recent rows of a kind, newest first. Used for notes and the self-model.

    With ``user`` set, includes user-agnostic rows (``user IS NULL``) too, so shared content
    surfaces alongside a specific user's.
    """

    def _read(conn: sqlite3.Connection) -> list[Memory]:
        if user is None:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (kind.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? AND (user = ? OR user IS NULL) "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (kind.value, user, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    return gateway.read(_read)


def latest_sentinel_note(gateway: StorageGateway, user: str | None) -> Memory | None:
    """The most recent sentinel note for a user — the high-attention end slot's content."""
    rows = recent_by_kind(gateway, MemoryKind.SENTINEL_NOTE, user=user, limit=1)
    return rows[0] if rows else None


def latest_self_model(gateway: StorageGateway) -> Memory | None:
    """The most recent synthesized self-model (always-on identity; shared, not user-scoped)."""
    rows = recent_by_kind(gateway, MemoryKind.SELF_MODEL, user=None, limit=1)
    return rows[0] if rows else None


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


def browse_memories(
    gateway: StorageGateway,
    *,
    kind: MemoryKind = MemoryKind.MEMORY,
    query: str | None = None,
    limit: int = 100,
) -> list[Memory]:
    """List memories of a kind for inspection, newest first, optionally keyword-filtered.

    A simple ``LIKE`` filter on the text — this is for the human-facing memory browser, not the
    ranked retrieval path (that is ``retrieval.hybrid``).
    """

    def _read(conn: sqlite3.Connection) -> list[Memory]:
        if query:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? AND text LIKE ? "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (kind.value, f"%{query}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (kind.value, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    return gateway.read(_read)


def delete_by_source(gateway: StorageGateway, source: str) -> int:
    """Delete all chunks that came from a given document source. Returns rows removed.

    Used by re-ingest to make ``ingest(path)`` idempotent — the old chunks are cleared
    before the new ones are written, so a document never accumulates stale duplicates.
    """

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute("DELETE FROM memories WHERE source = ?", (source,))
        return cur.rowcount

    return gateway.submit(_write)


def set_identity_anchor(gateway: StorageGateway, key: str, value: str) -> None:
    """Upsert a foundational identity anchor (name/operator/location/purpose, …).

    Re-establishing an anchor updates it in place rather than duplicating — identity is a
    single coherent record, not an accreting log.
    """
    now = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO identity (key, value, established_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "established_at = excluded.established_at",
            (key, value, now),
        )

    gateway.submit(_write)


def get_identity_anchors(gateway: StorageGateway) -> dict[str, str]:
    """All established identity anchors as a ``{key: value}`` map."""

    def _read(conn: sqlite3.Connection) -> dict[str, str]:
        rows = conn.execute("SELECT key, value FROM identity").fetchall()
        return {r["key"]: r["value"] for r in rows}

    return gateway.read(_read)


def prune_kind(gateway: StorageGateway, kind: MemoryKind, keep: int) -> int:
    """Keep only the most recent ``keep`` rows of a kind; delete the rest. Returns rows removed.

    Bounds the recency logs (e.g. EXCHANGE) so they never grow without limit.
    """

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "DELETE FROM memories WHERE kind = ? AND id NOT IN "
            "(SELECT id FROM memories WHERE kind = ? ORDER BY created_at DESC, id DESC LIMIT ?)",
            (kind.value, kind.value, keep),
        )
        return cur.rowcount

    return gateway.submit(_write)


def delete_kind(gateway: StorageGateway, kind: MemoryKind) -> int:
    """Delete all rows of a kind (e.g. clear the EXCHANGE log after folding it into a summary)."""

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute("DELETE FROM memories WHERE kind = ?", (kind.value,))
        return cur.rowcount

    return gateway.submit(_write)


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


# -- consolidation (sleep) ------------------------------------------------------------


def apply_decay(gateway: StorageGateway, updates: list[tuple[float, float, int]]) -> None:
    """Batch-apply (salience, confidence, id) updates from a decay pass."""
    if not updates:
        return

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "UPDATE memories SET salience = ?, confidence = ? WHERE id = ?", updates
        )

    gateway.submit(_write)


# Re-embed update batches: (vector | None, row id). None clears a vector (keyword fallback).
_EmbeddingUpdates = list[tuple[list[float] | None, int]]


def reembed_memories(gateway: StorageGateway, updates: _EmbeddingUpdates) -> None:
    """Batch-replace memory embeddings — ``[(vector, id)]`` — for a re-embed pass (model change)."""
    if not updates:
        return

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            [(embedding_to_blob(v), i) for v, i in updates],
        )

    gateway.submit(_write)


def reembed_claims(gateway: StorageGateway, updates: _EmbeddingUpdates) -> None:
    """Batch-replace library-claim embeddings — ``[(vector, id)]`` — for a re-embed pass."""
    if not updates:
        return

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "UPDATE library_claims SET embedding = ? WHERE id = ?",
            [(embedding_to_blob(v), i) for v, i in updates],
        )

    gateway.submit(_write)


def reembed_procedures(gateway: StorageGateway, updates: _EmbeddingUpdates) -> None:
    """Batch-replace procedure trigger embeddings — ``[(vector, id)]`` — for a re-embed pass."""
    if not updates:
        return

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "UPDATE procedures SET trigger_embedding = ? WHERE id = ?",
            [(embedding_to_blob(v), i) for v, i in updates],
        )

    gateway.submit(_write)


def archive_memories(gateway: StorageGateway, ids: list[int]) -> int:
    """Mark memories archived (excluded from recall, kept in store). Returns rows affected."""
    if not ids:
        return 0

    def _write(conn: sqlite3.Connection) -> int:
        ph = ",".join("?" * len(ids))
        cur = conn.execute(f"UPDATE memories SET archived = 1 WHERE id IN ({ph})", ids)
        return cur.rowcount

    return gateway.submit(_write)


def delete_memories(gateway: StorageGateway, ids: list[int]) -> int:
    """Hard-delete memories by id (used to drop exact duplicates). Returns rows removed."""
    if not ids:
        return 0

    def _write(conn: sqlite3.Connection) -> int:
        ph = ",".join("?" * len(ids))
        cur = conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
        return cur.rowcount

    return gateway.submit(_write)


def retier_by_provenance(gateway: StorageGateway, provenance: str, tier: EvidenceTier) -> int:
    """Re-tier every memory with this exact ``provenance`` to ``tier``. Maintenance for when a
    speaker was ingested at the wrong trust level (e.g. a peer baked before ``primary_user`` was
    set). Returns the number of rows changed."""
    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "UPDATE memories SET evidence_tier = ? WHERE provenance = ?", (tier.key, provenance)
        )
        return cur.rowcount

    return gateway.submit(_write)


def bump_memory(
    gateway: StorageGateway, memory_id: int, *, access_count: int, salience: float
) -> None:
    """Set a surviving memory's access_count and salience after merging duplicates into it."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE memories SET access_count = ?, salience = ? WHERE id = ?",
            (access_count, salience, memory_id),
        )

    gateway.submit(_write)


def delete_triples(gateway: StorageGateway, ids: list[int]) -> int:
    """Delete triples by id (used to resolve contradictions). Returns rows removed."""
    if not ids:
        return 0

    def _write(conn: sqlite3.Connection) -> int:
        ph = ",".join("?" * len(ids))
        cur = conn.execute(f"DELETE FROM triples WHERE id IN ({ph})", ids)
        return cur.rowcount

    return gateway.submit(_write)


# -- fleet catalogue ------------------------------------------------------------------

_C_SELECT = (
    "node, model, family, params_b, quantization, context_length, capabilities, "
    "return_time, quality, scanned_at, talk, tools, code, coherence, discipline, epistemics, "
    "reasoning, vision"
)
# Scan only sets discovery fields; benchmark scores (talk/tools/code/coherence/return_time/quality)
# are filled later by update_catalogue_scores.
_C_INSERT = (
    "node, model, family, params_b, quantization, context_length, capabilities, scanned_at"
)


def replace_catalogue(gateway: StorageGateway, entries: list[CatalogueEntry]) -> None:
    """Rebuild the fleet catalogue from a fresh scan (clear then insert)."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM model_catalogue")
        conn.executemany(
            f"INSERT INTO model_catalogue ({_C_INSERT}) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    e.node,
                    e.model,
                    e.family,
                    e.params_b,
                    e.quantization,
                    e.context_length,
                    json.dumps(e.capabilities),
                    e.scanned_at,
                )
                for e in entries
            ],
        )

    gateway.submit(_write)


def merge_catalogue(gateway: StorageGateway, entries: list[CatalogueEntry]) -> None:
    """Reconcile the catalogue with a fresh inventory WITHOUT wiping benchmark scores: insert
    newly-discovered ``(node, model)`` rows, refresh the discovery fields on existing ones (size /
    quant / capabilities can change), and drop rows for pairs no longer installed. Unlike
    ``replace_catalogue`` (clear-then-insert, which zeroes every score for a from-scratch run), this
    lets "qualify new models" add models without re-scoring the whole fleet (DESIGN §4)."""

    def _write(conn: sqlite3.Connection) -> None:
        present = {(e.node, e.model) for e in entries}
        existing = {
            (str(r[0]), str(r[1]))
            for r in conn.execute("SELECT node, model FROM model_catalogue").fetchall()
        }
        for node, mdl in existing - present:   # uninstalled since the last scan → drop
            conn.execute("DELETE FROM model_catalogue WHERE node=? AND model=?", (node, mdl))
        for e in entries:
            row = (e.family, e.params_b, e.quantization, e.context_length,
                   json.dumps(e.capabilities), e.scanned_at, e.node, e.model)
            if (e.node, e.model) in existing:   # refresh discovery fields, keep the scores
                conn.execute(
                    "UPDATE model_catalogue SET family=?, params_b=?, quantization=?, "
                    "context_length=?, capabilities=?, scanned_at=? WHERE node=? AND model=?", row
                )
            else:
                conn.execute(
                    f"INSERT INTO model_catalogue ({_C_INSERT}) VALUES (?,?,?,?,?,?,?,?)",
                    (e.node, e.model, e.family, e.params_b, e.quantization, e.context_length,
                     json.dumps(e.capabilities), e.scanned_at),
                )

    gateway.submit(_write)


def update_catalogue_scores(
    gateway: StorageGateway,
    model: str,
    *,
    return_time: float | None = None,
    quality: float | None,
    talk: float | None,
    tools: float | None,
    code: float | None,
    coherence: float | None = None,   # legacy column, no longer scored (kept so the DB shape holds)
    discipline: float | None = None,
    epistemics: float | None = None,
    reasoning: float | None = None,
    vision: float | None = None,
) -> None:
    """Write the **node-independent** scores (quality + capability dims) to every catalogue row of a
    model. ``return_time`` is **per-node** — it belongs to ``update_catalogue_speed`` and is written
    here only when explicitly given (omitted from the SQL when ``None``), so the benchmark, which
    records speed per node, never clobbers a model's per-node times with one model-wide value.
    """

    def _write(conn: sqlite3.Connection) -> None:
        cols = ["quality=?", "talk=?", "tools=?", "code=?",
                "coherence=?", "discipline=?", "epistemics=?", "reasoning=?", "vision=?"]
        params: list[object] = [quality, talk, tools, code, coherence, discipline,
                                epistemics, reasoning, vision]
        if return_time is not None:   # legacy/single-node callers may still set it model-wide
            cols.insert(0, "return_time=?")
            params.insert(0, return_time)
        conn.execute(
            f"UPDATE model_catalogue SET {', '.join(cols)} WHERE model=?", (*params, model),
        )

    gateway.submit(_write)


def update_catalogue_speed(
    gateway: StorageGateway, node: str, model: str, return_time: float
) -> None:
    """Set the measured response time for one specific (node, model) — speed is per-node."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE model_catalogue SET return_time=? WHERE node=? AND model=?",
            (return_time, node, model),
        )

    gateway.submit(_write)


# -- model preferences (user enable/disable for `auto` routing) -----------------------


def set_model_enabled(gateway: StorageGateway, model: str, enabled: bool) -> None:
    """Record a user's enable/disable choice for a model (upsert). Disabled → `auto` skips it."""
    now = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO model_prefs (model, enabled, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(model) DO UPDATE SET "
            "enabled=excluded.enabled, updated_at=excluded.updated_at",
            (model, 1 if enabled else 0, now),
        )

    gateway.submit(_write)


def disabled_models(gateway: StorageGateway) -> set[str]:
    """The set of models the user has explicitly disabled (everything else is enabled)."""

    def _read(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("SELECT model FROM model_prefs WHERE enabled = 0").fetchall()
        return {r["model"] for r in rows}

    return gateway.read(_read)


def set_node_enabled(gateway: StorageGateway, node: str, enabled: bool) -> None:
    """Record a user's enable/disable choice for a fleet node (upsert). Disabled → discovery,
    qualification, and routing all skip it, even if it's reachable (DESIGN §5)."""
    now = time.time()

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO node_prefs (node, enabled, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(node) DO UPDATE SET "
            "enabled=excluded.enabled, updated_at=excluded.updated_at",
            (node, 1 if enabled else 0, now),
        )

    gateway.submit(_write)


def disabled_nodes(gateway: StorageGateway) -> set[str]:
    """The set of fleet nodes the user has explicitly disabled (everything else is enabled)."""

    def _read(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("SELECT node FROM node_prefs WHERE enabled = 0").fetchall()
        return {r["node"] for r in rows}

    return gateway.read(_read)


def list_catalogue(gateway: StorageGateway) -> list[CatalogueEntry]:
    def _read(conn: sqlite3.Connection) -> list[CatalogueEntry]:
        rows = conn.execute(
            f"SELECT {_C_SELECT} FROM model_catalogue ORDER BY node, params_b DESC"
        ).fetchall()
        return [
            CatalogueEntry(
                node=r["node"],
                model=r["model"],
                family=r["family"],
                params_b=r["params_b"],
                quantization=r["quantization"],
                context_length=r["context_length"],
                capabilities=json.loads(r["capabilities"]),
                return_time=r["return_time"],
                quality=r["quality"],
                talk=r["talk"],
                tools=r["tools"],
                code=r["code"],
                coherence=r["coherence"],
                discipline=r["discipline"],
                epistemics=r["epistemics"],
                reasoning=r["reasoning"],
                vision=r["vision"],
                scanned_at=r["scanned_at"],
            )
            for r in rows
        ]

    return gateway.read(_read)


# -- procedural memory ----------------------------------------------------------------

_P_COLUMNS = "id, trigger, procedure, trigger_embedding, user, confidence, uses, created_at"


def _row_to_procedure(row: sqlite3.Row) -> Procedure:
    return Procedure(
        id=row["id"],
        trigger=row["trigger"],
        procedure=row["procedure"],
        trigger_embedding=blob_to_embedding(row["trigger_embedding"]),
        user=row["user"],
        confidence=row["confidence"],
        uses=row["uses"],
        created_at=row["created_at"],
    )


def save_procedure(gateway: StorageGateway, proc: Procedure) -> int:
    if proc.created_at == 0.0:
        proc.created_at = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "INSERT INTO procedures "
            "(trigger, procedure, trigger_embedding, user, confidence, uses, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                proc.trigger,
                proc.procedure,
                embedding_to_blob(proc.trigger_embedding),
                proc.user,
                proc.confidence,
                proc.uses,
                proc.created_at,
            ),
        )
        return int(cur.lastrowid or 0)

    proc.id = gateway.submit(_write)
    return proc.id


def list_procedures(
    gateway: StorageGateway, *, user: str | None = None, limit: int = 500
) -> list[Procedure]:
    """All procedures (optionally scoped to a user plus shared ones), newest first."""

    def _read(conn: sqlite3.Connection) -> list[Procedure]:
        if user is None:
            rows = conn.execute(
                f"SELECT {_P_COLUMNS} FROM procedures ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_P_COLUMNS} FROM procedures WHERE user = ? OR user IS NULL "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (user, limit),
            ).fetchall()
        return [_row_to_procedure(r) for r in rows]

    return gateway.read(_read)


def bump_procedure_uses(gateway: StorageGateway, ids: list[int]) -> None:
    """Increment the use counter for procedures that fired this turn (fire-and-forget)."""
    if not ids:
        return

    def _write(conn: sqlite3.Connection) -> None:
        conn.executemany("UPDATE procedures SET uses = uses + 1 WHERE id = ?", [(i,) for i in ids])

    gateway.submit_async(_write, priority=Priority.TOUCH)


def count_procedures(gateway: StorageGateway) -> int:
    def _read(conn: sqlite3.Connection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0])

    return gateway.read(_read)


# -- entity graph (triples) -----------------------------------------------------------

_T_COLUMNS = "id, subject, relation, object, user, provenance, confidence, created_at"


def _row_to_triple(row: sqlite3.Row) -> Triple:
    return Triple(
        id=row["id"],
        subject=row["subject"],
        relation=row["relation"],
        object=row["object"],
        user=row["user"],
        provenance=row["provenance"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


def save_triple(gateway: StorageGateway, triple: Triple) -> int:
    """Persist a triple, deduped case-insensitively. Returns its row id (0 if a duplicate)."""
    if triple.created_at == 0.0:
        triple.created_at = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "INSERT OR IGNORE INTO triples "
            "(subject, relation, object, user, provenance, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                triple.subject,
                triple.relation,
                triple.object,
                triple.user,
                triple.provenance,
                triple.confidence,
                triple.created_at,
            ),
        )
        # rowcount, not lastrowid: an IGNORE'd duplicate leaves lastrowid at the previous row.
        return int(cur.lastrowid or 0) if cur.rowcount > 0 else 0

    triple.id = gateway.submit(_write)
    return triple.id


def all_entities(gateway: StorageGateway, *, limit: int = 1000) -> list[str]:
    """The distinct entity nodes (subjects ∪ objects), for matching against a query."""

    def _read(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT subject FROM triples UNION SELECT object FROM triples LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in rows]

    return gateway.read(_read)


def traverse_from_entities(
    gateway: StorageGateway,
    entities: list[str],
    *,
    user: str | None = None,
    limit: int = 20,
) -> list[Triple]:
    """Triples touching any of ``entities`` (as subject or object) — one hop, best-confidence."""
    keys = [e.lower() for e in entities if e.strip()]
    if not keys:
        return []

    def _read(conn: sqlite3.Connection) -> list[Triple]:
        ph = ",".join("?" * len(keys))
        sql = (
            f"SELECT {_T_COLUMNS} FROM triples "
            f"WHERE (lower(subject) IN ({ph}) OR lower(object) IN ({ph}))"
        )
        params: list[object] = [*keys, *keys]
        if user is not None:
            sql += " AND (user = ? OR user IS NULL)"
            params.append(user)
        sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_triple(r) for r in rows]

    return gateway.read(_read)


def browse_triples(
    gateway: StorageGateway, *, query: str | None = None, limit: int = 100
) -> list[Triple]:
    """List triples for the graph browser, newest first, optionally filtered by entity/relation."""

    def _read(conn: sqlite3.Connection) -> list[Triple]:
        if query:
            like = f"%{query}%"
            rows = conn.execute(
                f"SELECT {_T_COLUMNS} FROM triples "
                f"WHERE subject LIKE ? OR relation LIKE ? OR object LIKE ? "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_T_COLUMNS} FROM triples ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_triple(r) for r in rows]

    return gateway.read(_read)


def count_triples(gateway: StorageGateway) -> int:
    def _read(conn: sqlite3.Connection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0])

    return gateway.read(_read)


# -- the Library layer (docs/LIBRARY.md): documents -> claims -> composite pages ---------

def upsert_library_document(gateway: StorageGateway, doc: LibraryDocument) -> int:
    """Record/refresh a source document by ``path`` (the ground-truth pointer), returning its id.
    Preserves id + ingested_at on update; refreshes filename/size/hash/title."""
    now = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        conn.execute(
            "INSERT INTO library_documents "
            "(path, filename, size_bytes, content_hash, title, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET filename=excluded.filename, "
            "size_bytes=excluded.size_bytes, content_hash=excluded.content_hash, "
            "title=excluded.title",
            (doc.path, doc.filename, doc.size_bytes, doc.content_hash, doc.title, now),
        )
        return int(conn.execute(
            "SELECT id FROM library_documents WHERE path = ?", (doc.path,)
        ).fetchone()[0])

    doc.id = gateway.submit(_write)
    return doc.id


def list_library_documents(gateway: StorageGateway) -> list[LibraryDocument]:
    def _read(conn: sqlite3.Connection) -> list[LibraryDocument]:
        rows = conn.execute(
            "SELECT * FROM library_documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [
            LibraryDocument(
                id=r["id"], path=r["path"], filename=r["filename"], size_bytes=r["size_bytes"],
                content_hash=r["content_hash"], title=r["title"], ingested_at=r["ingested_at"],
            )
            for r in rows
        ]

    return gateway.read(_read)


def delete_library_document(gateway: StorageGateway, path: str) -> int:
    """Drop a source document and all derived from it: its claims and their page links."""
    def _write(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT id FROM library_documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            return 0
        doc_id = int(row[0])
        claim_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM library_claims WHERE document_id = ?", (doc_id,)
            ).fetchall()
        ]
        if claim_ids:
            ph = ",".join("?" * len(claim_ids))
            conn.execute(f"DELETE FROM library_page_claims WHERE claim_id IN ({ph})", claim_ids)
        conn.execute("DELETE FROM library_claims WHERE document_id = ?", (doc_id,))
        cur = conn.execute("DELETE FROM library_documents WHERE id = ?", (doc_id,))
        return cur.rowcount

    return gateway.submit(_write)


def delete_library_page(gateway: StorageGateway, page_id: int) -> int:
    """Drop a composite page and its claim links (the source claims/documents are untouched)."""
    def _write(conn: sqlite3.Connection) -> int:
        conn.execute("DELETE FROM library_page_claims WHERE page_id = ?", (page_id,))
        cur = conn.execute("DELETE FROM library_pages WHERE id = ?", (page_id,))
        return cur.rowcount

    return gateway.submit(_write)


def _row_to_claim(row: sqlite3.Row) -> LibraryClaim:
    return LibraryClaim(
        id=row["id"], document_id=row["document_id"], text=row["text"], locator=row["locator"],
        embedding=blob_to_embedding(row["embedding"]), confidence=row["confidence"],
        created_at=row["created_at"],
    )


def replace_document_claims(
    gateway: StorageGateway, document_id: int, claims: list[LibraryClaim]
) -> int:
    """Replace all claims for a document (re-extraction is idempotent); returns the count written.
    Also clears stale page links to the old claims."""
    now = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        old = [
            r[0] for r in conn.execute(
                "SELECT id FROM library_claims WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]
        if old:
            ph = ",".join("?" * len(old))
            conn.execute(f"DELETE FROM library_page_claims WHERE claim_id IN ({ph})", old)
        conn.execute("DELETE FROM library_claims WHERE document_id = ?", (document_id,))
        conn.executemany(
            "INSERT INTO library_claims "
            "(document_id, text, locator, embedding, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(document_id, c.text, c.locator, embedding_to_blob(c.embedding), c.confidence, now)
             for c in claims],
        )
        return len(claims)

    return gateway.submit(_write)


def list_library_claims(
    gateway: StorageGateway, *, exclude_doc_ids: set[int] | None = None
) -> list[LibraryClaim]:
    """All claims (the retrievable spine) with embeddings. ``exclude_doc_ids`` drops claims from
    toggled-off documents at the SQL layer, so an unselected book costs nothing to recall over."""
    excl = sorted(exclude_doc_ids) if exclude_doc_ids else []

    def _read(conn: sqlite3.Connection) -> list[LibraryClaim]:
        if excl:
            ph = ",".join("?" * len(excl))
            rows = conn.execute(
                f"SELECT * FROM library_claims WHERE document_id NOT IN ({ph})", excl
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM library_claims").fetchall()
        return [_row_to_claim(r) for r in rows]

    return gateway.read(_read)


def claims_for_document(gateway: StorageGateway, document_id: int) -> list[LibraryClaim]:
    def _read(conn: sqlite3.Connection) -> list[LibraryClaim]:
        rows = conn.execute(
            "SELECT * FROM library_claims WHERE document_id = ? ORDER BY id", (document_id,)
        ).fetchall()
        return [_row_to_claim(r) for r in rows]

    return gateway.read(_read)


def upsert_library_page(gateway: StorageGateway, page: LibraryPage) -> int:
    """Record/refresh a composite page by ``path``, returning its id (preserves id/created_at)."""
    now = time.time()

    def _write(conn: sqlite3.Connection) -> int:
        conn.execute(
            "INSERT INTO library_pages "
            "(path, title, summary, content_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET title=excluded.title, summary=excluded.summary, "
            "content_hash=excluded.content_hash, updated_at=excluded.updated_at",
            (page.path, page.title, page.summary, page.content_hash, now, now),
        )
        return int(conn.execute(
            "SELECT id FROM library_pages WHERE path = ?", (page.path,)
        ).fetchone()[0])

    page.id = gateway.submit(_write)
    return page.id


def _row_to_page(row: sqlite3.Row) -> LibraryPage:
    return LibraryPage(
        id=row["id"], path=row["path"], title=row["title"], summary=row["summary"],
        content_hash=row["content_hash"], created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def get_library_page(gateway: StorageGateway, page_id: int) -> LibraryPage | None:
    def _read(conn: sqlite3.Connection) -> LibraryPage | None:
        row = conn.execute("SELECT * FROM library_pages WHERE id = ?", (page_id,)).fetchone()
        return _row_to_page(row) if row is not None else None

    return gateway.read(_read)


def list_library_pages(gateway: StorageGateway) -> list[LibraryPage]:
    def _read(conn: sqlite3.Connection) -> list[LibraryPage]:
        rows = conn.execute("SELECT * FROM library_pages ORDER BY updated_at DESC").fetchall()
        return [
            LibraryPage(
                id=r["id"], path=r["path"], title=r["title"], summary=r["summary"],
                content_hash=r["content_hash"], created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    return gateway.read(_read)


def set_page_claims(gateway: StorageGateway, page_id: int, claim_ids: list[int]) -> None:
    """Set which claims composed a page (replaces prior links)."""
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM library_page_claims WHERE page_id = ?", (page_id,))
        conn.executemany(
            "INSERT INTO library_page_claims (page_id, claim_id) VALUES (?, ?)",
            [(page_id, cid) for cid in claim_ids],
        )

    gateway.submit(_write)


def claims_for_page(gateway: StorageGateway, page_id: int) -> list[LibraryClaim]:
    """The claims that composed a page — provenance from the composite down to source citations."""
    def _read(conn: sqlite3.Connection) -> list[LibraryClaim]:
        rows = conn.execute(
            "SELECT c.* FROM library_claims c "
            "JOIN library_page_claims pc ON pc.claim_id = c.id "
            "WHERE pc.page_id = ? ORDER BY c.id",
            (page_id,),
        ).fetchall()
        return [_row_to_claim(r) for r in rows]

    return gateway.read(_read)


def pages_for_claims(gateway: StorageGateway, claim_ids: list[int]) -> dict[int, list[int]]:
    """Map each claim id → the composite page id(s) it helped compose (claim → its composites).
    The reverse of ``claims_for_page`` — used to offer 'load the page this fact came from'."""
    if not claim_ids:
        return {}

    def _read(conn: sqlite3.Connection) -> dict[int, list[int]]:
        ph = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"SELECT claim_id, page_id FROM library_page_claims WHERE claim_id IN ({ph})",
            claim_ids,
        ).fetchall()
        out: dict[int, list[int]] = {}
        for cid, pid in rows:
            out.setdefault(int(cid), []).append(int(pid))
        return out

    return gateway.read(_read)
