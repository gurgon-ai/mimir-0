"""Executable spec for the storage gateway: single-writer correctness, reads see writes."""

from __future__ import annotations

import threading

import pytest

from mimir.storage.gateway import StorageGateway
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import (
    count_memories,
    get_memory,
    list_memories,
    record_access,
    save_memory,
)


def test_save_and_get_roundtrip(db_path: str) -> None:
    with StorageGateway(db_path) as gw:
        mem = Memory(
            text="the sky is blue",
            evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
            embedding=[0.1, 0.2, 0.3],
            provenance="stated by alex",
            user="alex",
        )
        mid = save_memory(gw, mem)
        assert mid > 0 and mem.id == mid

        got = get_memory(gw, mid)
        assert got is not None
        assert got.text == "the sky is blue"
        assert got.evidence_tier is EvidenceTier.STATED_BY_PRIMARY_USER
        assert got.embedding == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)  # float32 round-trip
        assert got.user == "alex"


def test_list_filters_by_kind_and_user(db_path: str) -> None:
    with StorageGateway(db_path) as gw:
        save_memory(gw, Memory(text="alex fact", user="alex"))
        save_memory(gw, Memory(text="global fact", user=None))
        save_memory(gw, Memory(text="bob fact", user="bob"))
        save_memory(
            gw, Memory(text="a note", kind=MemoryKind.SENTINEL_NOTE, user="alex")
        )

        alex = {m.text for m in list_memories(gw, user="alex", kind=MemoryKind.MEMORY)}
        assert alex == {"alex fact", "global fact"}  # includes user-agnostic rows

        assert count_memories(gw, kind=MemoryKind.MEMORY) == 3
        assert count_memories(gw, kind=MemoryKind.SENTINEL_NOTE) == 1


def test_record_access_bumps_salience_and_count(db_path: str) -> None:
    with StorageGateway(db_path) as gw:
        mem = Memory(text="x", salience=0.5)
        mid = save_memory(gw, mem)
        record_access(gw, [mid])
        gw.flush()  # touches are fire-and-forget now; wait for them to land
        got = get_memory(gw, mid)
        assert got is not None
        assert got.access_count == 1
        assert got.salience == 0.6  # +0.1, capped at 1.0


def test_concurrent_writes_are_serialized(db_path: str) -> None:
    """Many threads writing at once: the single writer must not drop or corrupt any."""
    with StorageGateway(db_path) as gw:
        n = 50

        def worker(i: int) -> None:
            save_memory(gw, Memory(text=f"mem-{i}"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert count_memories(gw, kind=MemoryKind.MEMORY) == n
        texts = {m.text for m in list_memories(gw, kind=MemoryKind.MEMORY)}
        assert texts == {f"mem-{i}" for i in range(n)}
