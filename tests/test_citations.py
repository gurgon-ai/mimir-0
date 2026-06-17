"""The citation guard (DESIGN §10): a reply may cite only sources the system actually holds; an
invented citation (training-data knowledge wearing a source) is flagged, never silently passed."""

from __future__ import annotations

from mimir.cognition.citations import (
    annotate_unverified_citations,
    unverified_citations,
)

KNOWN = {"Servus Group OHS Manual Nov 2024.docx", "Servus Group Safe Work Practices Nov 2024.docx"}


def test_real_citation_to_a_held_document_passes() -> None:
    # Both the heading-locator and the comma-locator citation forms the model actually emits.
    reply = (
        "Report it to your supervisor [Servus Group OHS Manual Nov 2024, Worker Rights]. "
        "Bag soaked laundry [Servus Group Safe Work Practices Nov 2024.docx:Bloodborne Pathogens]."
    )
    assert unverified_citations(reply, KNOWN) == []
    assert annotate_unverified_citations(reply, KNOWN) == reply  # untouched


def test_invented_source_is_flagged() -> None:
    reply = "Follow the standard [National Fire Code 2020] and [OSHA 1910.120] for this."
    bad = unverified_citations(reply, KNOWN)
    assert "National Fire Code 2020" in bad
    out = annotate_unverified_citations(reply, KNOWN)
    assert "⚠ Unverified citation" in out and out.startswith(reply)


def test_prose_brackets_are_not_mistaken_for_citations() -> None:
    reply = "First do step [1], then [see above]. It's [important] to wear PPE."
    assert unverified_citations(reply, KNOWN) == []


def test_guard_is_silent_with_no_known_sources() -> None:
    # A fresh install that holds no documents shouldn't accuse anything (nothing to verify against).
    reply = "As per [Some Manual 2024], do the thing."
    assert unverified_citations(reply, set()) == []
