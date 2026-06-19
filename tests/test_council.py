"""Executable spec for the inner council: discovery, parallel positions, verdict (DESIGN §4, §5)."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition.council import (
    _confidence_from_consensus,
    _parse_verdict,
    deliberate,
)
from mimir.prompts import COUNCIL_PERSONAS
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import get_forum_thread, get_memory, list_memories


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


def test_rebuttal_round_runs_for_every_persona(brain: Mimir) -> None:
    result = brain.deliberate("Should we ship now or harden first?")
    # round two: every voice answers the floor, distinct from its opening
    assert len(result.rebuttals) == len(COUNCIL_PERSONAS)
    assert {p.persona for p in result.rebuttals} == {name for name, _ in COUNCIL_PERSONAS}
    assert all("rebutting" in p.text for p in result.rebuttals)
    assert all(p.persona in p.text for p in result.rebuttals)


def test_forum_thread_carries_both_rounds_in_order(brain: Mimir) -> None:
    result = brain.deliberate("A genuinely open question.")
    assert result.thread_id is not None
    thread = get_forum_thread(brain._storage, result.thread_id)
    assert thread is not None
    kinds = [p["kind"] for p in thread["posts"]]
    assert kinds.count("position") == len(COUNCIL_PERSONAS)
    assert kinds.count("rebuttal") == len(COUNCIL_PERSONAS)
    # openings precede rebuttals, which precede the verdict — the debate reads in order
    last_position = max(i for i, k in enumerate(kinds) if k == "position")
    first_rebuttal = min(i for i, k in enumerate(kinds) if k == "rebuttal")
    last_rebuttal = max(i for i, k in enumerate(kinds) if k == "rebuttal")
    assert last_position < first_rebuttal
    assert last_rebuttal < kinds.index("verdict")


def test_parse_verdict_reads_labelled_fields() -> None:
    raw = (
        "VERDICT: The plan holds, with caveats.\n"
        "DISSENT: The cost estimate is unproven.\n"
        "DISSENT_BY: skeptic\n"
        "CONSENSUS: 0.8"
    )
    v = _parse_verdict(raw)
    assert v.summary == "The plan holds, with caveats."
    assert v.dissent == "The cost estimate is unproven."
    assert v.dissent_by == "skeptic"
    assert v.consensus == 0.8


def test_parse_verdict_falls_back_to_plain_text() -> None:
    # a model that ignores the format must not lose its verdict — the whole reply is the conclusion
    v = _parse_verdict("Just a paragraph with no labels at all.")
    assert v.summary == "Just a paragraph with no labels at all."
    assert v.dissent == ""
    assert v.dissent_by == ""


def test_parse_verdict_treats_none_dissent_as_agreement() -> None:
    raw = "VERDICT: All voices align.\nDISSENT: none\nDISSENT_BY: none\nCONSENSUS: 1.0"
    v = _parse_verdict(raw)
    assert v.dissent == ""
    assert v.dissent_by == ""  # no objection ⇒ no holder, even if the model named one
    assert v.consensus == 1.0


def test_confidence_rises_with_consensus() -> None:
    # a split deliberation is worth less as understanding than a unanimous one — but stays modest
    assert _confidence_from_consensus(0.0) < _confidence_from_consensus(0.5)
    assert _confidence_from_consensus(0.5) < _confidence_from_consensus(1.0)
    assert _confidence_from_consensus(0.5) == 0.6  # a 50/50 split == the old flat default
    assert _confidence_from_consensus(1.0) <= 0.75  # never escapes the INFERRED band


def test_surviving_objection_rides_into_recallable_memory(brain: Mimir) -> None:
    # the mock synthesizer emits a structured verdict with a dissent from the skeptic
    result = brain.deliberate("Is breadth or depth the better bet?")
    assert result.dissent
    assert result.dissent_by == "skeptic"
    mem = get_memory(brain._storage, result.memory_id)
    assert mem is not None
    # the conclusion AND the surviving objection are both in the stored memory text
    assert result.verdict in mem.text
    assert "Surviving objection (skeptic):" in mem.text
    assert result.dissent in mem.text
    # confidence reflects the council's consensus, not a hardcoded constant
    assert mem.confidence == _confidence_from_consensus(result.consensus)


def test_dissent_persists_as_its_own_forum_post(brain: Mimir) -> None:
    result = brain.deliberate("A contested question.")
    assert result.thread_id is not None
    thread = get_forum_thread(brain._storage, result.thread_id)
    assert thread is not None
    dissent_posts = [p for p in thread["posts"] if p["kind"] == "dissent"]
    assert len(dissent_posts) == 1
    assert dissent_posts[0]["author"] == "skeptic"
    assert dissent_posts[0]["content"] == result.dissent


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
