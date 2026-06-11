"""Executable spec for the hardened storage-gateway internals (behind the unchanged seam)."""

from __future__ import annotations

import sqlite3

from mimir.storage.gateway import StorageGateway
from mimir.storage.models import Memory
from mimir.storage.repo import get_memory, save_memory


def test_async_write_lands_after_flush(db_path: str) -> None:
    with StorageGateway(db_path) as gw:
        mid = save_memory(gw, Memory(text="orig"))

        def _update(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE memories SET text='changed' WHERE id=?", (mid,))

        gw.submit_async(_update)
        gw.flush()
        got = get_memory(gw, mid)
        assert got is not None and got.text == "changed"


def test_coalescing_collapses_idempotent_writes(db_path: str) -> None:
    # Large flush interval so the rapid-fire ops land in one batch and coalesce.
    with StorageGateway(db_path, flush_interval_s=0.3) as gw:
        mid = save_memory(gw, Memory(text="orig"))
        for value in ("a", "b", "c", "d"):
            gw.submit_async(
                lambda conn, v=value: conn.execute(
                    "UPDATE memories SET text=? WHERE id=?", (v, mid)
                ),
                coalesce_key=f"text:{mid}",
            )
        gw.flush()
        got = get_memory(gw, mid)
        assert got is not None and got.text == "d"  # last writer wins
        assert gw.get_stats()["coalesced"] >= 1  # earlier ones collapsed


def test_retry_on_locked_then_succeeds(db_path: str) -> None:
    with StorageGateway(db_path, retry_sleep_s=0.0) as gw:
        state = {"n": 0}

        def _flaky(conn: sqlite3.Connection) -> int:
            state["n"] += 1
            if state["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return 42

        assert gw.submit(_flaky) == 42
        assert state["n"] == 2
        assert gw.get_stats()["retries"] >= 1


def test_batch_failure_falls_back_to_isolated_ops(db_path: str) -> None:
    """One bad op in a batch must not fail its neighbors — they apply individually."""
    with StorageGateway(db_path, flush_interval_s=0.3, retry_sleep_s=0.0) as gw:
        mid = save_memory(gw, Memory(text="x"))

        def _bad(conn: sqlite3.Connection) -> None:
            raise ValueError("boom")  # non-lock error → whole-batch apply fails

        def _good(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE memories SET text='good' WHERE id=?", (mid,))

        gw.submit_async(_bad)
        gw.submit_async(_good)
        gw.flush()
        got = get_memory(gw, mid)
        assert got is not None and got.text == "good"  # neighbor survived
        assert gw.get_stats()["errors"] >= 1
