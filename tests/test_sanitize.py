"""The epistemic-tag stripper: internal provenance tags never reach a human (DESIGN §10)."""

from __future__ import annotations

from mimir.sanitize import StreamTagStripper, strip_epistemic_tags


def test_strip_removes_tier_and_source_tags() -> None:
    raw = "Greetings, Greg. [tier=deduction; source=Mimir's internal assessment] I am ready."
    assert strip_epistemic_tags(raw) == "Greetings, Greg. I am ready."


def test_strip_removes_invented_single_field_tags() -> None:
    raw = "What validation do you use? [tier=question] And biases? [tier=focus]"
    assert strip_epistemic_tags(raw) == "What validation do you use? And biases?"


def test_strip_removes_epistemic_check_flag() -> None:
    assert strip_epistemic_tags("I'm unsure. [epistemic check] Tell me more.") == (
        "I'm unsure. Tell me more."
    )


def test_strip_leaves_ordinary_brackets_alone() -> None:
    text = "See item [1] and the list [a, b, c] here."
    assert strip_epistemic_tags(text) == text


def test_strip_is_idempotent() -> None:
    raw = "Fact. [tier=stated_by_primary_user; source=Greg] Done."
    once = strip_epistemic_tags(raw)
    assert strip_epistemic_tags(once) == once


def _stream(stripper: StreamTagStripper, deltas: list[str]) -> str:
    out = [stripper.feed(d) for d in deltas]
    out.append(stripper.flush())
    return "".join(out)


def test_streaming_strips_tag_split_across_deltas() -> None:
    # The tag is fragmented exactly the way token streaming would fragment it.
    deltas = ["Hello Greg. ", "[tier", "=ded", "uction; source=x", "] ", "Ready."]
    assert _stream(StreamTagStripper(), deltas) == "Hello Greg. Ready."


def test_streaming_emits_plain_text_promptly() -> None:
    stripper = StreamTagStripper()
    # Bracketless text flows through; only a trailing whitespace run is briefly held (it could
    # precede a tag in the next delta), arriving with the following chunk.
    assert stripper.feed("Hello ") == "Hello"
    assert stripper.feed("Greg") == " Greg"


def test_streaming_releases_non_tag_bracket() -> None:
    deltas = ["See ", "[1", "] ", "there."]
    assert _stream(StreamTagStripper(), deltas) == "See [1] there."


def test_streaming_flush_releases_unclosed_bracket() -> None:
    stripper = StreamTagStripper()
    # An unclosed '[' at end of stream is ordinary content, not a tag — nothing dropped overall.
    assert _stream(stripper, ["Note ", "[draft"]) == "Note [draft"
