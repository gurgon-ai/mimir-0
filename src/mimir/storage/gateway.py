"""The storage gateway — one of the two chokepoints (DESIGN §5).

**The law:** every write to the store goes through here, and through the single dedicated
writer thread it owns. There is no other path to a write. Reads stay direct (SQLite WAL allows
many readers alongside the one writer), so reads do not queue behind writes.

Hardened internals (the patterns proven in the parent system's single-writer, reimplemented
clean):

- **Priority queue** — interactive writes jump ahead of background ones.
- **Batching** — ops that arrive close together are applied in one transaction, so the write
  lock is taken once per batch instead of once per op.
- **Coalescing** — idempotent fire-and-forget ops sharing a ``coalesce_key`` collapse to the
  latest (last-writer-wins); sync ops are never coalesced (each owes a result).
- **Retry-on-locked** — a batch that hits ``database is locked`` retries once with a short
  backoff, then falls back to applying ops individually so one bad op can't fail its neighbors.
- **Flush barrier** — ``flush()`` blocks until everything queued so far has landed (shutdown).

Two submit paths: ``submit`` (synchronous, blocks on the result — for writes the caller needs
confirmed, e.g. a bake whose id is returned and read back) and ``submit_async`` (fire-and-forget,
for high-frequency non-critical writes like access touches). The seam is unchanged; callers that
used ``submit`` still work.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from ..errors import StorageError
from .migrate import check_schema, run_migrations

log = logging.getLogger("mimir.storage")

T = TypeVar("T")

WriteFn = Callable[[sqlite3.Connection], Any]


class Priority:
    """Write priority — lower runs first (mirrors the parent system's tiers)."""

    CRITICAL = 0  # interactive writes a user is waiting on
    USER_ADJACENT = 10  # bakes, sentinel notes — post-turn, someone may read them back
    BACKGROUND = 50  # background cognition writes
    TOUCH = 70  # access bookkeeping — coalescible, lowest urgency


@dataclass
class _WriteOp:
    fn: WriteFn | None
    priority: int
    coalesce_key: str | None = None
    future: Future[Any] | None = None  # None => fire-and-forget
    barrier: bool = False  # a flush() marker — carries no work, just resolves when reached
    stop: bool = False  # tells the writer thread to drain and exit


@dataclass(order=True)
class _Queued:
    priority: int
    seq: int
    op: _WriteOp = field(compare=False)


# Batch tuning. Small intervals keep latency low while still collapsing bursts.
_FLUSH_INTERVAL_S = 0.02  # max time to gather more ops after the first
_BATCH_MAX = 64  # max ops per transaction


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")


def _is_locked_error(exc: Exception) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "database is locked" in str(exc).lower()


class StorageGateway:
    """Serializes all writes through one thread; serves reads directly.

    Construct it and it is ready: migrations have run and the schema check has passed, or
    construction has raised loudly. There is no half-open state.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        retry_sleep_s: float = 0.05,
        batch_max: int = _BATCH_MAX,
        flush_interval_s: float = _FLUSH_INTERVAL_S,
    ) -> None:
        self._path = str(path)
        self._retry_sleep_s = retry_sleep_s
        self._batch_max = batch_max
        self._flush_interval_s = flush_interval_s
        self._queue: queue.PriorityQueue[_Queued] = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._ready: Future[None] = Future()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "submitted": 0,
            "written": 0,
            "coalesced": 0,
            "batches": 0,
            "errors": 0,
            "retries": 0,
        }
        self._thread = threading.Thread(
            target=self._writer_loop, name="mimir-storage-writer", daemon=True
        )
        self._thread.start()
        # Block until the writer has opened the DB, migrated, and checked the schema. Any
        # failure there is re-raised here, loud, at construction time.
        self._ready.result()

    # -- enqueue ----------------------------------------------------------------------

    def _next_seq(self) -> int:
        with self._seq_lock:
            n = self._seq
            self._seq += 1
            return n

    def _enqueue(self, op: _WriteOp) -> None:
        with self._stats_lock:
            self._stats["submitted"] += 1
        self._queue.put(_Queued(op.priority, self._next_seq(), op))

    # -- public API -------------------------------------------------------------------

    def submit(
        self, fn: Callable[[sqlite3.Connection], T], *, priority: int = Priority.USER_ADJACENT
    ) -> T:
        """Run ``fn(write_conn)`` on the writer thread, in a transaction. Blocks for the result.

        The only way to write something you need confirmed (e.g. a new row id, read back later).
        """
        if self._closed:
            raise StorageError("storage gateway is closed; cannot accept writes")
        fut: Future[T] = Future()
        self._enqueue(_WriteOp(fn=fn, priority=priority, future=fut))
        return fut.result()

    def submit_async(
        self,
        fn: WriteFn,
        *,
        priority: int = Priority.TOUCH,
        coalesce_key: str | None = None,
    ) -> None:
        """Fire-and-forget write. Returns immediately; loss on crash is acceptable for the
        non-critical, high-frequency writes this path is for (e.g. access touches).

        Pass a ``coalesce_key`` ONLY for idempotent ops (last-writer-wins) — non-idempotent
        ops (counters) must not be coalesced or increments are lost.
        """
        if self._closed:
            raise StorageError("storage gateway is closed; cannot accept writes")
        self._enqueue(_WriteOp(fn=fn, priority=priority, coalesce_key=coalesce_key))

    def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(read_conn)`` on a fresh connection. Direct, no queue (WAL)."""
        if self._closed:
            raise StorageError("storage gateway is closed; cannot serve reads")
        conn = sqlite3.connect(self._path)
        try:
            _apply_pragmas(conn)
            conn.row_factory = sqlite3.Row
            return fn(conn)
        finally:
            conn.close()

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything queued so far has been written. False on timeout."""
        if self._closed:
            return True
        fut: Future[None] = Future()
        # Lowest urgency so all real writes ahead of it drain first.
        self._enqueue(_WriteOp(fn=None, priority=Priority.TOUCH + 1, future=fut, barrier=True))
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, int]:
        with self._stats_lock:
            s = dict(self._stats)
        s["queue_depth"] = self._queue.qsize()
        return s

    def close(self) -> None:
        """Drain pending writes, stop the writer thread, and wait for it. Idempotent."""
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._enqueue(_WriteOp(fn=None, priority=Priority.TOUCH + 2, stop=True))
        self._thread.join(timeout=10)

    def __enter__(self) -> StorageGateway:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the writer thread ------------------------------------------------------------

    def _writer_loop(self) -> None:
        try:
            conn = sqlite3.connect(self._path)
            _apply_pragmas(conn)
            run_migrations(conn)
            check_schema(conn)
        except BaseException as exc:  # surface boot failure to the constructor, loud
            self._ready.set_exception(exc)
            return
        self._ready.set_result(None)

        try:
            while True:
                batch = self._drain_batch()
                if not batch:
                    continue
                stop = any(op.stop for op in batch)
                work_and_barriers = [op for op in batch if not op.stop]
                if work_and_barriers:
                    self._apply_batch(conn, work_and_barriers)
                if stop:
                    return
        finally:
            conn.close()

    def _drain_batch(self) -> list[_WriteOp]:
        """Block for the first op, then gather more (up to batch_max / flush_interval)."""
        try:
            first = self._queue.get(timeout=1.0)
        except queue.Empty:
            return []
        batch = [first.op]
        deadline = time.monotonic() + self._flush_interval_s
        while len(batch) < self._batch_max:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining).op)
            except queue.Empty:
                break
        return batch

    def _coalesce(self, ops: list[_WriteOp]) -> list[_WriteOp]:
        """Collapse fire-and-forget ops sharing a coalesce_key to the latest one."""
        seen: dict[str, int] = {}
        out: list[_WriteOp] = []
        dropped = 0
        for op in ops:
            if op.coalesce_key is not None and op.future is None:
                if op.coalesce_key in seen:
                    out[seen[op.coalesce_key]] = op  # keep latest
                    dropped += 1
                    continue
                seen[op.coalesce_key] = len(out)
            out.append(op)
        if dropped:
            with self._stats_lock:
                self._stats["coalesced"] += dropped
        return out

    def _apply_batch(self, conn: sqlite3.Connection, raw_ops: list[_WriteOp]) -> None:
        ops = self._coalesce(raw_ops)
        barriers = [o for o in ops if o.barrier]
        work = [o for o in ops if not o.barrier]

        if work:
            if not self._apply_atomic(conn, work):
                # The batch failed as a unit (non-lock error or still-locked after retry).
                # Re-apply each op on its own so one bad op doesn't fail its neighbors.
                self._apply_individually(conn, work)
            else:
                with self._stats_lock:
                    self._stats["batches"] += 1
                    self._stats["written"] += len(work)

        for b in barriers:
            if b.future is not None and not b.future.done():
                b.future.set_result(None)

    def _apply_atomic(self, conn: sqlite3.Connection, work: list[_WriteOp]) -> bool:
        """Apply the whole batch in one transaction. Retry once on lock. Return success."""
        attempt = 0
        while True:
            results: dict[int, Any] = {}
            try:
                for idx, op in enumerate(work):
                    assert op.fn is not None
                    results[idx] = op.fn(conn)
                conn.commit()
            except Exception as exc:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                if _is_locked_error(exc) and attempt == 0:
                    attempt += 1
                    with self._stats_lock:
                        self._stats["retries"] += 1
                    time.sleep(self._retry_sleep_s)
                    continue
                return False
            else:
                for idx, op in enumerate(work):
                    if op.future is not None and not op.future.done():
                        op.future.set_result(results.get(idx))
                return True

    def _apply_individually(self, conn: sqlite3.Connection, work: list[_WriteOp]) -> None:
        """Fallback: each op in its own transaction, so failures are isolated."""
        for op in work:
            assert op.fn is not None
            try:
                with conn:
                    result = op.fn(conn)
            except Exception as exc:
                with self._stats_lock:
                    self._stats["errors"] += 1
                wrapped = StorageError(f"write failed: {exc}")
                if op.future is not None and not op.future.done():
                    op.future.set_exception(wrapped)
                else:
                    # Fire-and-forget loss is acceptable here, but never silent (DESIGN §10).
                    log.error("storage: dropped a fire-and-forget write: %s", exc)
            else:
                with self._stats_lock:
                    self._stats["written"] += 1
                if op.future is not None and not op.future.done():
                    op.future.set_result(result)
