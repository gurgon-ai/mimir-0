"""Bidirectional (output-triggered) RAG (DESIGN §5a): after a reply, retrieve memory relevant to the
MODEL'S OWN words and surface it into the next turn — grounding a thread the model itself opened."""

from __future__ import annotations

import time

import pytest

from mimir.brain import Mimir
from mimir.cognition.burst import ResponseContext
from mimir.cognition.output_rag import (
    authority_beliefs,
    correction_surface,
    parse_output_check,
)
from mimir.retrieval.hybrid import ScoredMemory
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import save_memory


def _seed(
    brain: Mimir, text: str, *, age_s: float,
    tier: EvidenceTier = EvidenceTier.STATED_BY_PRIMARY_USER,
) -> None:
    save_memory(brain._storage, Memory(
        text=text, kind=MemoryKind.MEMORY, evidence_tier=tier,
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
    _seed(brain, "Alex keeps bees on the south slope.", age_s=3600)
    ctx = ResponseContext(user_text="...", reply="Tell me more about the bees on the south slope.")
    result = brain._output_rag_task(ctx)()
    assert result.surface and "bees" in result.surface.lower()


# -- self-correction (#1) ---------------------------------------------------------------------

def _mem(text: str, tier: EvidenceTier) -> ScoredMemory:
    return ScoredMemory(memory=Memory(text=text, evidence_tier=tier), score=1.0, keyword=1.0,
                        vector=1.0)


def test_parse_output_check_reads_a_contradiction() -> None:
    check = parse_output_check("CONTRADICTS: 2\nNOTE: the barn is red, not blue", 3)
    assert check is not None and check.index == 2
    assert check.note == "the barn is red, not blue"


def test_parse_output_check_none_and_out_of_range_are_no_conflict() -> None:
    assert parse_output_check("CONTRADICTS: none\nNOTE: none", 3) is None
    assert parse_output_check("CONTRADICTS: 9\nNOTE: x", 3) is None   # out of range → ignored
    assert parse_output_check("a model that ignored the format entirely", 3) is None


def test_authority_beliefs_keeps_only_outranking_tiers() -> None:
    scored = [
        _mem("primary fact", EvidenceTier.STATED_BY_PRIMARY_USER),
        _mem("a doc fact", EvidenceTier.DOCUMENT),
        _mem("just conversation", EvidenceTier.CONVERSATION),
        _mem("a peer said", EvidenceTier.STATED_BY_PEER),
        _mem("my own musing", EvidenceTier.INFERRED),
    ]
    kept = {m.text for m in authority_beliefs(scored)}
    # conversation/peer/inferred don't outrank a generated reply
    assert kept == {"primary fact", "a doc fact"}


def test_self_check_surfaces_a_correction_on_contradiction(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(brain, "The barn is red.", age_s=3600)  # a primary-user fact that outranks the reply
    monkeypatch.setattr(
        brain, "_background_chat",
        lambda messages: "CONTRADICTS: 1\nNOTE: the barn is red, not blue",
    )
    note = brain._output_rag("Actually the barn is blue, I repainted it last week.", None)
    assert note is not None and note.startswith("Self-check:")
    assert "barn is red" in note  # the surviving authoritative fact is named


def test_self_check_silent_when_the_model_finds_no_conflict(brain: Mimir) -> None:
    # the mock self-check returns 'none' → no correction, falls through to the grounding note
    _seed(brain, "The barn is red.", age_s=3600)
    note = brain._output_rag("I should check whether the barn needs a fresh coat.", None)
    assert note is not None and not note.startswith("Self-check:")


def test_self_check_can_be_disabled(brain: Mimir, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(brain, "The barn is red.", age_s=3600)
    brain.config.output_rag_self_check = False
    # even if the model would flag a contradiction, the gate is off → never runs the check
    monkeypatch.setattr(brain, "_background_chat",
                        lambda messages: "CONTRADICTS: 1\nNOTE: conflict")
    note = brain._output_rag("Actually the barn is blue now.", None)
    assert note is not None and not note.startswith("Self-check:")


def test_correction_surface_omits_empty_note() -> None:
    belief = Memory(text="The well is on the north field.")
    assert correction_surface(belief, "") == (
        'Self-check: what you just said may conflict with something you hold — '
        '"The well is on the north field."'
    )
