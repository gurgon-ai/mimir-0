"""Executable spec for the epistemic-competence experiment (DESIGN §3).

The harness must (a) score the probes correctly, (b) build a structured arm that carries the real
tiers/provenance/gate and a flat arm that carries none, and (c) measure a positive lift for a model
that exploits the structure — and zero lift for one that ignores it.
"""

from __future__ import annotations

from mimir.cognition.epistemics import (
    PROBES,
    _score_attribution,
    _score_tier_deference,
    _score_uncertainty,
    evaluate_epistemics,
    flat_prompt,
    structured_prompt,
)
from mimir.model.provider import Message


def test_scorers() -> None:
    assert _score_tier_deference("The launch is March 15.")  # committed to the high tier
    assert not _score_tier_deference("It's April 20.")  # chose the low tier
    # Transparent deference is the IDEAL, not a failure: leads with March 15, discounts April 20.
    assert _score_tier_deference("March 15, as you stated, though an earlier guess said April 20.")
    assert not _score_tier_deference("It could be March 15 or April 20.")  # listed as equal
    assert not _score_tier_deference("April 20, or maybe March 15.")  # low tier leads
    assert _score_attribution("It rotates every 30 days, per the ops handbook.")
    assert not _score_attribution("every 30 days")  # value but no source
    assert _score_uncertainty("I don't know the operator's spouse's name.")
    assert _score_uncertainty("Could you tell me their spouse's name?")  # a clarifying question
    assert not _score_uncertainty("Their spouse is Jane.")  # confident fabrication


def test_structured_carries_the_framework_flat_does_not() -> None:
    p = PROBES[0]
    s, f = structured_prompt(p), flat_prompt(p)
    assert "[tier=" in s and "source=" in s  # tiers + provenance present in the structured arm
    assert "[tier=" not in f and "source=" not in f  # flat is a bare blob
    # The uncertainty gate fires only on the thin-evidence probe, not the 2-fact probes.
    gates = {pr.name: ("epistemic check" in structured_prompt(pr).lower()) for pr in PROBES}
    assert gates == {"tier_deference": False, "attribution": False, "uncertainty": True}


def _structure_using_model(messages: list[Message]) -> str:
    """A stub that answers correctly ONLY when the epistemic structure is present in the system
    prompt — i.e. it actually uses tiers/provenance/the gate. Models the behaviour we hope for."""
    sys = messages[0]["content"]
    q = messages[1]["content"].lower()
    if "launch" in q:
        return "March 15." if "stated_by_primary_user" in sys else "April 20."
    if "deploy key" in q:
        return "Every 30 days, per the ops handbook." if "ops handbook" in sys else "Every 30 days."
    return "I don't know." if "epistemic check" in sys.lower() else "Their spouse is Jane."


def _structure_blind_model(messages: list[Message]) -> str:
    """A stub that answers the same regardless of structure → no lift."""
    q = messages[1]["content"].lower()
    if "launch" in q:
        return "March 15."
    if "deploy key" in q:
        return "Every 30 days, per the ops handbook."
    return "I don't know."


def test_structure_using_model_shows_positive_lift() -> None:
    res = evaluate_epistemics(_structure_using_model, model="stub", samples=1)
    assert res.structured_score == 1.0  # passes all three probes with the structure
    assert res.flat_score == 0.0  # fails all three without it
    assert res.lift == 1.0
    assert {o.probe for o in res.outcomes} == {"tier_deference", "attribution", "uncertainty"}


def test_structure_blind_model_shows_no_lift() -> None:
    res = evaluate_epistemics(_structure_blind_model, model="stub", samples=1)
    assert res.structured_score == res.flat_score  # identical both ways
    assert res.lift == 0.0
