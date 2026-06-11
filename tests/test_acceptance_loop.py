"""The §6 acceptance loop — the definition of done for v0.

Boot empty → converse → bake → a later turn recalls with correct provenance & tier via
build_context() → the sentinel fired and left a usable note. Runs on the mock provider:
no Ollama, no GPU, no network.
"""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.selftest import run_self_test
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import count_memories, latest_sentinel_note


def test_boot_empty_bake_recall_sentinel(brain: Mimir) -> None:
    # Boot empty.
    assert count_memories(brain._storage, kind=MemoryKind.MEMORY) == 0

    # Converse: state a fact. It should bake via the storage gateway.
    r1 = brain.turn("My favorite color is teal.", user="alex")
    brain.wait_for_sentinel()
    assert len(r1.baked) == 1
    assert count_memories(brain._storage, kind=MemoryKind.MEMORY) == 1

    # A later turn whose answer depends on that memory: it must be recalled and cited.
    r2 = brain.turn("What is my favorite color?", user="alex")
    assert "teal" in r2.reply.lower()
    assert r2.context.source_count >= 1

    # Recalled with correct evidence tier + provenance (single-user → primary user tier).
    knowledge = next(s for s in r2.context.sections if s.name == "knowledge")
    assert "tier=stated_by_primary_user" in knowledge.body
    assert "source=stated by alex" in knowledge.body

    # The sentinel fired and left a usable note for the next turn.
    brain.wait_for_sentinel()
    assert brain.last_sentinel_error is None
    note = latest_sentinel_note(brain._storage, "alex")
    assert note is not None and note.text.strip()
    assert note.evidence_tier is EvidenceTier.INFERRED


def test_questions_do_not_bake(brain: Mimir) -> None:
    """A question is not a durable fact — baking it would pollute the store."""
    r = brain.turn("What is my favorite color?", user="alex")
    brain.wait_for_sentinel()
    assert r.baked == []
    assert count_memories(brain._storage, kind=MemoryKind.MEMORY) == 0


def test_runtime_self_test_passes() -> None:
    """The loop, run as the shipped runtime self-test, passes including the canary."""
    report = run_self_test()
    assert report.ok
    assert report.baked and report.recalled and report.correct_tier
    assert report.sentinel_fired and report.canary_held
