"""Executable spec for the epistemic-competence experiment (DESIGN §3).

The harness must (a) score the probes correctly, (b) build a structured arm that carries the real
tiers/provenance/gate and a flat arm that carries none, and (c) measure a positive lift for a model
that exploits the structure — and zero lift for one that ignores it.
"""

from __future__ import annotations

from mimir.cognition.epistemics import (
    PROBES,
    _score_attribution,
    _score_bird_color,
    _score_secret_word,
    _score_tier_deference,
    _score_uncertainty,
    _score_vault_passphrase,
    evaluate_epistemics,
    flat_prompt,
    score_epistemic_competence,
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
    # Layered tier-deference: high-tier 'blue' beats low-tier 'red' under noise; the prompt asks for
    # one word, so a clean 'blue' passes and anything still mentioning 'red' fails.
    assert _score_bird_color("Blue.")  # committed to the high tier, one clean word
    assert _score_bird_color("The birds are blue.")
    assert not _score_bird_color("Red.")  # chose the low tier
    assert not _score_bird_color("It's either blue or red.")  # didn't commit to the high tier
    assert not _score_bird_color("Blue, but an older note said red.")  # 'red' still present
    # Grounding: must surface the nonce that lives only in context.
    assert _score_secret_word("The secret command word is zephyr-quartz.")
    assert _score_secret_word("ZEPHYR QUARTZ")
    assert not _score_secret_word("I don't see a secret word.")


def test_structured_carries_the_framework_flat_does_not() -> None:
    p = PROBES[0]
    s, f = structured_prompt(p), flat_prompt(p)
    assert "[tier=" in s and "source=" in s  # tiers + provenance present in the structured arm
    assert "[tier=" not in f and "source=" not in f  # flat is a bare blob
    # The uncertainty gate fires only on the thin-evidence probe, not the 2-fact probes.
    gates = {pr.name: ("epistemic check" in structured_prompt(pr).lower()) for pr in PROBES}
    assert gates == {
        "tier_deference": False, "attribution": False, "uncertainty": True,
        "layered_tier_deference": False,  # 7 facts → plenty of evidence → gate stays quiet
    }


def _structure_using_model(messages: list[Message]) -> str:
    """A stub that answers correctly ONLY when the epistemic structure is present in the system
    prompt — i.e. it actually uses tiers/provenance/the gate. Models the behaviour we hope for."""
    sys = messages[0]["content"]
    q = messages[1]["content"].lower()
    if "launch" in q:
        return "March 15." if "stated_by_primary_user" in sys else "April 20."
    if "deploy key" in q:
        return "Every 30 days, per the ops handbook." if "ops handbook" in sys else "Every 30 days."
    if "color" in q:  # the layered gauntlet: defer to high-tier 'blue' only when tiers are shown
        return "Blue." if "stated_by_primary_user" in sys else "Red."
    if "secret" in q:  # grounding: the nonce is in both arms, so a reader answers it either way
        return "zephyr-quartz"
    if "passphrase" in q:  # long-context needle: a reader of the long document finds it
        return "quokka-lantern"
    return "I don't know." if "epistemic check" in sys.lower() else "Their spouse is Jane."


def _structure_blind_model(messages: list[Message]) -> str:
    """A stub that answers the same regardless of structure → no lift."""
    q = messages[1]["content"].lower()
    if "launch" in q:
        return "March 15."
    if "deploy key" in q:
        return "Every 30 days, per the ops handbook."
    if "color" in q:
        return "Blue."
    if "secret" in q:
        return "zephyr-quartz"
    if "passphrase" in q:
        return "quokka-lantern"
    return "I don't know."


def test_structure_using_model_shows_positive_lift() -> None:
    res = evaluate_epistemics(_structure_using_model, model="stub", samples=1)
    assert res.structured_score == 1.0  # passes every lift probe with the structure
    assert res.flat_score == 0.0  # fails them all without it
    assert res.lift == 1.0
    assert {o.probe for o in res.outcomes} == {
        "tier_deference", "attribution", "uncertainty", "layered_tier_deference",
    }


def test_structure_blind_model_shows_no_lift() -> None:
    res = evaluate_epistemics(_structure_blind_model, model="stub", samples=1)
    assert res.structured_score == res.flat_score  # identical both ways
    assert res.lift == 0.0


def test_chat_qualifier_scores_grounding_and_layered_deference() -> None:
    # The chat qualification signal (structured arm) spans the lift probes — including the big
    # layered conflicting-tier gauntlet (defer to 'blue') — AND the grounding floor (recall the
    # nonce that lives only in context). A model that reads the context and honours tiers passes;
    # one that ignores both fails, so it can't qualify for the identity-bearing chat role.
    grounded = score_epistemic_competence(_structure_using_model, samples=1)
    assert grounded == 1.0  # date + attribution + uncertainty + bird + secret + needle all pass

    def context_blind(messages: list[Message]) -> str:
        q = messages[1]["content"].lower()
        if "color" in q:
            return "Red."  # ignores the high tier
        if "secret" in q:
            return "I have no secret word."  # ignores the context
        if "passphrase" in q:
            return "There is no passphrase in here."  # didn't read the long document
        return "Their spouse is Jane."  # confabulates instead of hedging

    assert score_epistemic_competence(context_blind, samples=1) == 0.0  # fails every probe


def test_long_context_needle_scales_with_window_and_survives_assembly() -> None:
    # The haystack SIZES to num_ctx: a bigger window → a proportionally bigger haystack (so the test
    # actually stresses the context we qualify at, not a fixed 2k gesture). The needle survives the
    # assembly when build_context gets a window-sized budget.
    from mimir.cognition.epistemics import _long_context_probe, _long_haystack

    hay = _long_haystack("the vault passphrase is quokka-lantern", target_tokens=2000)
    assert "quokka-lantern" in hay
    assert len(hay) > 5000  # ~2000 tokens of real filler
    assert _score_vault_passphrase("The passphrase is quokka-lantern.")
    assert not _score_vault_passphrase("I couldn't find a passphrase.")
    # Scales: a 32k window gives a much bigger haystack than an 8k one.
    small, big = _long_context_probe(8192), _long_context_probe(32768)
    assert len(big.facts[0][0]) > 3 * len(small.facts[0][0])
    # The needle survives the real pipeline when the budget matches the window.
    assert "quokka-lantern" in structured_prompt(small, budget_tokens=8192)
