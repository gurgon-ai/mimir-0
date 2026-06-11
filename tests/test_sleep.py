"""Executable spec for sleep / consolidation: dedup, decay, archive, contradictions (DESIGN §5)."""

from __future__ import annotations

import time

from mimir.brain import Mimir
from mimir.cognition.graph import store_triples
from mimir.cognition.sleep import SleepReport, consolidate
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import (
    browse_memories,
    browse_triples,
    count_memories,
    get_memory,
    list_memories,
    save_memory,
)

_DAY = 86_400.0


def test_exact_duplicates_merge(brain: Mimir) -> None:
    save_memory(brain._storage, Memory(text="the sky is blue", user="g", access_count=2))
    save_memory(brain._storage, Memory(text="The sky  is blue", user="g", access_count=3))
    report = consolidate(brain._storage)
    assert report.deduped == 1
    survivors = list_memories(brain._storage, kind=MemoryKind.MEMORY)
    assert len(survivors) == 1
    assert survivors[0].access_count == 5  # access counts summed into the survivor


def test_near_duplicates_merge(brain: Mimir) -> None:
    save_memory(brain._storage, Memory(text="alpha one", embedding=[1.0, 0.0, 0.0], user="g"))
    save_memory(brain._storage, Memory(text="beta two", embedding=[1.0, 0.0, 0.0], user="g"))
    consolidate(brain._storage)
    assert count_memories(brain._storage, kind=MemoryKind.MEMORY) == 1  # cosine 1.0 → merged


def test_salience_decays_with_disuse(brain: Mimir) -> None:
    now = time.time()
    old = now - 60 * _DAY  # two salience half-lives
    mid = save_memory(
        brain._storage,
        Memory(text="x", salience=1.0, created_at=old, last_accessed=old),
    )
    consolidate(brain._storage, now=now)
    got = get_memory(brain._storage, mid)
    assert got is not None and got.salience < 0.3  # ~0.25


def test_confidence_decays_only_for_low_tier(brain: Mimir) -> None:
    now = time.time()
    old = now - 60 * _DAY
    conv = save_memory(
        brain._storage,
        Memory(
            text="a", confidence=0.8, evidence_tier=EvidenceTier.CONVERSATION,
            created_at=old, last_accessed=old,
        ),
    )
    prim = save_memory(
        brain._storage,
        Memory(
            text="b", confidence=0.9, evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
            created_at=old, last_accessed=old,
        ),
    )
    consolidate(brain._storage, now=now)
    assert get_memory(brain._storage, conv).confidence < 0.8  # provisional decays
    assert get_memory(brain._storage, prim).confidence == 0.9  # authority never does


def test_archives_low_salience_provisionals_only(brain: Mimir) -> None:
    now = time.time()
    ancient = now - 365 * _DAY
    save_memory(
        brain._storage,
        Memory(
            text="provisional", salience=0.1, confidence=0.5,
            evidence_tier=EvidenceTier.INFERRED, created_at=ancient, last_accessed=ancient,
        ),
    )
    save_memory(
        brain._storage,
        Memory(
            text="solid fact", salience=0.1, confidence=0.9,
            evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
            created_at=ancient, last_accessed=ancient,
        ),
    )
    consolidate(brain._storage, now=now)
    active = {m.text for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)}
    assert "provisional" not in active  # archived → out of recall
    assert "solid fact" in active  # confident facts are never archived (no death spiral)
    # archived memory is kept in the store, not deleted (archiving ≠ disbelieving)
    all_rows = browse_memories(brain._storage, kind=MemoryKind.MEMORY)
    assert any(m.text == "provisional" and m.archived for m in all_rows)


def test_resolves_functional_contradiction(brain: Mimir) -> None:
    store_triples(brain._storage, [["Greg", "lives in", "Colorado"]], user="g", confidence=0.8)
    store_triples(brain._storage, [["Greg", "lives in", "Texas"]], user="g", confidence=0.9)
    report = consolidate(brain._storage)
    assert report.contradictions_resolved == 1
    objects = [t.object for t in browse_triples(brain._storage) if t.relation == "lives in"]
    assert objects == ["Texas"]  # higher-confidence value wins; stale one dropped


def test_leaves_nonfunctional_relations_alone(brain: Mimir) -> None:
    store_triples(
        brain._storage, [["Greg", "likes", "tea"], ["Greg", "likes", "coffee"]], user="g"
    )
    consolidate(brain._storage)
    likes = sorted(t.object for t in browse_triples(brain._storage) if t.relation == "likes")
    assert likes == ["coffee", "tea"]  # 'likes' is many-valued — not a contradiction


def test_brain_sleep_returns_report(brain: Mimir) -> None:
    report = brain.sleep()
    assert isinstance(report, SleepReport)
    assert report.total_changes >= 0
