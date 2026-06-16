"""Bidirectional (output-triggered) RAG (DESIGN §5a): after a reply, retrieve memory relevant to the
MODEL'S OWN words and surface it into the next turn — grounding a thread the model itself opened."""

from __future__ import annotations

import time

from mimir.brain import Mimir
from mimir.cognition.burst import ResponseContext
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import save_memory


def _seed(brain: Mimir, text: str, *, age_s: float) -> None:
    save_memory(brain._storage, Memory(
        text=text, kind=MemoryKind.MEMORY, evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
        embedding=brain._embedder.embed(text), created_at=time.time() - age_s))


def test_surfaces_memory_relevant_to_the_reply(brain: Mimir) -> None:
    _seed(brain, "The north gate latch was replaced in March.", age_s=3600)
    note = brain._output_rag("I'll go check whether the gate latch is still holding.", None)
    assert note and "gate latch" in note.lower()


def test_skips_trivial_replies(brain: Mimir) -> None:
    _seed(brain, "The north gate latch was replaced in March.", age_s=3600)
    assert brain._output_rag("ok", None) is None      # too short to bother retrieving on


def test_excludes_facts_just_baked_this_turn(brain: Mimir) -> None:
    # A memory created in this turn (the just-baked echo of the reply) must NOT be surfaced back.
    _seed(brain, "The cellar floods every spring thaw.", age_s=0)
    note = brain._output_rag("Noting that the cellar floods every spring thaw.", None)
    assert note is None or "cellar floods" not in note.lower()


def test_task_emits_a_surface_for_the_next_turn(brain: Mimir) -> None:
    _seed(brain, "Greg keeps bees on the south slope.", age_s=3600)
    ctx = ResponseContext(user_text="...", reply="Tell me more about the bees on the south slope.")
    result = brain._output_rag_task(ctx)()
    assert result.surface and "bees" in result.surface.lower()
