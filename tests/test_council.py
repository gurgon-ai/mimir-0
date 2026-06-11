"""Executable spec for the inner council: discovery, parallel positions, verdict (DESIGN §4, §5)."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition.council import deliberate
from mimir.prompts import COUNCIL_PERSONAS
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import get_memory, list_memories


def test_deliberation_gathers_all_personas(brain: Mimir) -> None:
    result = brain.deliberate("Should the project prioritize breadth or depth?")
    assert len(result.positions) == len(COUNCIL_PERSONAS)
    assert {p.persona for p in result.positions} == {name for name, _ in COUNCIL_PERSONAS}
    # every persona produced a non-empty, persona-specific position
    assert all(p.text and p.persona in p.text for p in result.positions)


def test_personas_spread_across_discovered_models(brain: Mimir) -> None:
    # the mock advertises several models → assignment should use more than one
    result = brain.deliberate("An open question.")
    assert len({p.model for p in result.positions}) > 1


def test_verdict_is_stored_as_recallable_understanding(brain: Mimir) -> None:
    result = brain.deliberate("What matters most here?")
    assert result.verdict
    assert result.memory_id is not None
    mem = get_memory(brain._storage, result.memory_id)
    assert mem is not None
    assert mem.kind is MemoryKind.MEMORY
    assert mem.evidence_tier is EvidenceTier.INFERRED
    assert mem.provenance == "inner council"
    # it joins the recallable knowledge layer
    recalled = list_memories(brain._storage, kind=MemoryKind.MEMORY)
    assert any(m.id == result.memory_id for m in recalled)


def test_falls_back_to_configured_model_without_discovery(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    # if discovery returns nothing, the council still runs on a configured role model
    monkeypatch.setattr(brain._model, "available_models", lambda: [])
    result = deliberate(
        brain._model, brain._storage, brain._embedder, question="Anything?"
    )
    assert len(result.positions) == len(COUNCIL_PERSONAS)
    assert all(p.model == "mock" for p in result.positions)  # default_council_model → role model
