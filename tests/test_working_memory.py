"""Executable spec for working memory: recency + compression (DESIGN §3a, §3e)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.working_memory import (
    MAX_EXCHANGES,
    current_working_memory,
    latest_working_memory,
    recent_exchanges,
    record_exchange,
    synthesize_working_memory,
)
from mimir.config import Config
from mimir.context.build import build_context
from mimir.embed.base import EmbeddingMode
from mimir.retrieval.hybrid import ScoredMemory
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import count_memories


def test_recency_log_is_chronological_and_capped(brain: Mimir) -> None:
    for i in range(MAX_EXCHANGES + 3):
        record_exchange(brain._storage, user="g", user_text=f"msg{i}", reply="ok")
    assert count_memories(brain._storage, kind=MemoryKind.EXCHANGE) == MAX_EXCHANGES  # pruned
    chron = recent_exchanges(brain._storage, 100)
    assert len(chron) == MAX_EXCHANGES
    assert f"msg{MAX_EXCHANGES + 2}" in chron[-1].text  # newest last


def test_synthesis_folds_oldest_keeps_recent(brain: Mimir) -> None:
    for i in range(10):
        record_exchange(brain._storage, user="g", user_text=f"point {i}", reply="noted")
    mem = synthesize_working_memory(brain._model, brain._storage, fold_threshold=10, keep_recent=4)
    assert mem is not None and mem.kind is MemoryKind.WORKING_MEMORY
    # the oldest 6 folded into the rolling summary; the most recent 4 stay raw
    assert count_memories(brain._storage, kind=MemoryKind.EXCHANGE) == 4
    kept = recent_exchanges(brain._storage, 100)
    assert "point 9" in kept[-1].text                              # newest retained
    assert all("point 0" not in e.text for e in kept)             # oldest folded away
    assert latest_working_memory(brain._storage) is not None
    # below threshold now (only 4 left) → no-op
    assert synthesize_working_memory(brain._model, brain._storage, fold_threshold=10) is None


def test_current_working_memory_composition(brain: Mimir) -> None:
    assert current_working_memory(brain._storage) is None  # empty
    record_exchange(brain._storage, user="g", user_text="my favorite color is teal", reply="ok")
    composed = current_working_memory(brain._storage)
    assert composed is not None
    assert "Most recent exchanges" in composed
    assert "teal" in composed.lower()


def test_build_context_places_working_memory_before_sentinel() -> None:
    scored = ScoredMemory(
        memory=Memory(text="a fact", evidence_tier=EvidenceTier.CONVERSATION, id=1),
        score=0.5,
        keyword=0.5,
        vector=0.0,
    )
    bundle = build_context(
        query="what's up?",
        user=None,
        identity="id",
        retrieved=[scored],
        sentinel_note=Memory(text="a note"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
        working_memory="carrying recent context",
    )
    names = [s.name for s in bundle.sections]
    assert names.index("knowledge") < names.index("working_memory") < names.index("sentinel_note")


def test_brain_carries_recent_context_forward(brain: Mimir) -> None:
    brain.turn("My favorite color is teal.", user="greg")
    brain.wait_for_sentinel()
    r2 = brain.turn("Tell me more.", user="greg")
    wm = next((s for s in r2.context.sections if s.name == "working_memory"), None)
    assert wm is not None
    assert "teal" in wm.body.lower()  # the prior exchange is carried into this turn


def test_compression_fires_on_threshold(mock_config: Config) -> None:
    mock_config.working_memory_fold_threshold = 3
    mock_config.working_memory_keep_recent = 1
    with Mimir(mock_config) as m:
        for i in range(3):
            m.turn(f"thing {i}", user="g")  # the 3rd turn crosses the threshold → folds
            m.wait_for_sentinel()
        assert latest_working_memory(m._storage) is not None
        # folded the oldest, kept the most recent 1 raw
        assert count_memories(m._storage, kind=MemoryKind.EXCHANGE) == 1


def test_compression_can_be_disabled(mock_config: Config) -> None:
    mock_config.working_memory_fold_threshold = 0
    with Mimir(mock_config) as m:
        for i in range(4):
            m.turn(f"hello {i}", user="g")
            m.wait_for_sentinel()
        assert latest_working_memory(m._storage) is None  # no compression
        assert count_memories(m._storage, kind=MemoryKind.EXCHANGE) >= 1  # recency still works
