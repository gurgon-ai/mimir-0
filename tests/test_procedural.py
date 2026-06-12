"""Executable spec for procedural memory: trigger → procedure, matched and injected (DESIGN §3a)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.procedural import retrieve_procedures
from mimir.storage.repo import count_procedures, list_procedures


def test_learn_and_retrieve_by_trigger(brain: Mimir) -> None:
    brain.learn_procedure(
        "the user asks for a summary",
        "give three bullet points, then a one-line takeaway",
    )
    assert count_procedures(brain._storage) == 1
    hits = retrieve_procedures(brain._storage, brain._embedder, "can you give me a summary please?")
    assert len(hits) == 1
    assert "bullet points" in hits[0].procedure


def test_unrelated_query_matches_nothing(brain: Mimir) -> None:
    brain.learn_procedure("the user asks for a summary", "use bullet points")
    hits = retrieve_procedures(brain._storage, brain._embedder, "what's the weather in Paris?")
    assert hits == []


def test_procedure_injected_and_use_counted(brain: Mimir) -> None:
    brain.learn_procedure("the user wants directions", "list numbered steps, shortest path first")
    r = brain.turn("I want directions to the office", user="g")
    section = next((s for s in r.context.sections if s.name == "procedures"), None)
    assert section is not None
    assert "numbered steps" in section.body
    # firing the procedure bumps its use count (a structural relevance signal)
    procs = list_procedures(brain._storage)
    assert procs[0].uses >= 1


def test_procedures_are_guidance_not_a_source(brain: Mimir) -> None:
    """A matched procedure is method, not evidence — it must not satisfy the uncertainty gate."""
    brain.learn_procedure("the user asks about meaning", "answer plainly")
    r = brain.turn("what is the meaning of it all?", user="g")
    # the procedures section is present but source_count stays 0 (no factual grounding)
    assert any(s.name == "procedures" for s in r.context.sections)
    assert r.context.source_count == 0
