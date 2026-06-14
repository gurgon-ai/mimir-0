"""Executable spec for the seeding interview (DESIGN §9; docs/mimir_foundational_interview.md).

The operator's answers are the orienting bedrock: stored as top-tier (stated_by_primary_user),
provenance="onboarding" memories — one editable row per question, living in one place — with the
identity-anchored ones (name/operator/location) mirrored into the always-on self-model.
"""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.onboarding import ONBOARDING_PROVENANCE, ONBOARDING_QUESTIONS
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import list_memories


def _answers(profile: list[dict]) -> dict[str, str | None]:
    return {q["key"]: q["answer"] for q in profile}


def test_answer_is_stored_as_a_top_tier_onboarding_fact(brain: Mimir) -> None:
    brain.record_onboarding_answer("work", "I run a small farm in Mission, BC.")
    rows = [m for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)
            if m.provenance == ONBOARDING_PROVENANCE]
    assert len(rows) == 1
    mem = rows[0]
    assert mem.evidence_tier is EvidenceTier.STATED_BY_PRIMARY_USER  # the orienting bedrock (1.30x)
    assert "small farm" in mem.text
    assert mem.meta["onboarding_key"] == "work"
    assert mem.meta["answer"] == "I run a small farm in Mission, BC."
    assert mem.salience > 1.0  # load-bearing, surfaces ahead of incidental chatter


def test_profile_is_the_one_editable_place(brain: Mimir) -> None:
    # Every question appears, answered or not — the single panel the user reviews/edits.
    profile = brain.onboarding_profile()
    assert [q["key"] for q in profile] == [q.key for q in ONBOARDING_QUESTIONS]
    assert all(q["answer"] is None for q in profile)  # nothing captured yet
    brain.record_onboarding_answer("interests", "woodworking, local AI, gardening")
    assert _answers(brain.onboarding_profile())["interests"] == "woodworking, local AI, gardening"


def test_anchored_questions_mirror_into_the_self_model(brain: Mimir) -> None:
    # What to call the AI / who you are / where this is also become identity anchors (injected
    # verbatim into the always-on self-model), not just memories.
    brain.record_onboarding_answer("assistant_name", "Mimir")
    brain.record_onboarding_answer("location", "a farm in Mission, BC")
    anchors = brain.identity_anchors()
    assert anchors["name"] == "Mimir"
    assert anchors["location"] == "a farm in Mission, BC"


def test_reanswering_updates_in_place(brain: Mimir) -> None:
    brain.record_onboarding_answer("pets", "a dog named Rex")
    brain.record_onboarding_answer("pets", "a dog named Rex and two cats")  # edit, any time
    rows = [m for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)
            if m.provenance == ONBOARDING_PROVENANCE and m.meta.get("onboarding_key") == "pets"]
    assert len(rows) == 1  # upsert — one row per question, not an append
    assert "two cats" in rows[0].text


def test_blank_answer_clears_the_fact(brain: Mimir) -> None:
    brain.record_onboarding_answer("household", "my partner")
    brain.record_onboarding_answer("household", "   ")  # cleared
    assert _answers(brain.onboarding_profile())["household"] is None


def test_pending_drives_the_interview_and_setup_prompt(brain: Mimir) -> None:
    assert len(brain.pending_onboarding()) == len(ONBOARDING_QUESTIONS)  # first run: all pending
    for q in ONBOARDING_QUESTIONS:
        brain.record_onboarding_answer(q.key, f"answer for {q.key}")
    assert brain.pending_onboarding() == []  # complete


def test_unknown_key_is_ignored_not_miswritten(brain: Mimir) -> None:
    assert brain.record_onboarding_answer("not_a_question", "x") is None
    assert [m for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)
            if m.provenance == ONBOARDING_PROVENANCE] == []
