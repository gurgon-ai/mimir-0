"""The vision benchmark dimension (DESIGN §4 "Round 4"): vision capability is determined EMPIRICALLY
by the probe image, not advertised metadata. A model that reads the word + counts the shapes scores
1.0; a text-only model that can't see the image scores ~0 — that failure is the determination."""

from __future__ import annotations

from mimir.cognition import benchmark


def _vision_capable(messages):
    # Pretends to see the probe: answers correctly only when an image is attached.
    has_image = any(m.get("images") for m in messages)
    q = messages[-1]["content"].lower()
    if not has_image:
        return "I can't see any image."
    return "GLYPHON" if "word" in q else "There are 3 red circles."


def _text_only(messages):
    # Ignores the image (no vision): guesses, can't know the made-up word.
    q = messages[-1]["content"].lower()
    return "maybe two?" if "circle" in q else "I'm not sure what it says."


def test_vision_probe_scores_a_seeing_model_full() -> None:
    assert benchmark._VISION_PROBE.is_file()              # the committed probe ships with the repo
    assert benchmark.score_vision(_vision_capable) == 1.0


def test_vision_probe_scores_a_text_only_model_low() -> None:
    # Can't read GLYPHON (needs vision) → at most the lucky count; never full marks.
    assert benchmark.score_vision(_text_only) < 1.0


def test_count_only_fluke_stays_below_the_role_floor() -> None:
    # A text model that ignores the image but guesses '3' gets only the count weight (0.4) — below
    # the 0.5 capability floor, so it can't masquerade as vision-capable.
    def count_guesser(messages):
        return "3" if "circle" in messages[-1]["content"].lower() else "no idea"
    assert benchmark.score_vision(count_guesser) < 0.5


def test_vision_is_none_when_probe_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(benchmark, "_VISION_PROBE", tmp_path / "nope.png")
    assert benchmark.score_vision(_vision_capable) is None   # not tested, not a false zero
