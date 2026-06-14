"""Executable spec for temporal grounding — time context + the deterministic intercept (DESIGN §3e).

Pure functions take the moment as an argument, so they're tested against a fixed instant. The brain
injects the time line every turn and short-circuits explicit time questions with no model call.
"""

from __future__ import annotations

import datetime as dt

from mimir.brain import Mimir
from mimir.cognition.temporal import (
    answer_time_query,
    humanize_duration,
    next_season,
    season_of,
    time_prefix,
)

_WINTER = dt.datetime(2026, 1, 15, 14, 30)   # mid-January
_SUMMER = dt.datetime(2026, 7, 4, 9, 0)      # early July


def test_season_respects_hemisphere() -> None:
    assert season_of(_WINTER, "north") == "winter"
    assert season_of(_SUMMER, "north") == "summer"
    # Southern hemisphere runs the opposite season on the same date.
    assert season_of(_WINTER, "south") == "summer"
    assert season_of(_SUMMER, "south") == "winter"


def test_next_season_counts_forward() -> None:
    name, days = next_season(dt.datetime(2026, 3, 1), "north")  # before spring (3/20)
    assert name == "spring" and days == 19


def test_time_prefix_states_date_and_season() -> None:
    line = time_prefix(_WINTER, "north")
    assert "Thursday" in line and "January" in line and "2026" in line
    assert "2:30 PM" in line
    assert "winter" in line


def test_intercept_answers_plain_time_questions() -> None:
    ans = answer_time_query("what day is it?", _WINTER, "north")
    assert ans and "Thursday" in ans and "January" in ans


def test_intercept_handles_season_countdown() -> None:
    ans = answer_time_query("how long until summer?", dt.datetime(2026, 6, 1), "north")
    assert ans and "Summer begins in 20 days" in ans


def test_intercept_passes_non_time_queries_to_the_model() -> None:
    # No time trigger → None (let the model answer).
    assert answer_time_query("tell me about the garden", _WINTER) is None
    # A long query that merely contains a trigger phrase is NOT intercepted.
    assert answer_time_query(
        "what day should I prune the apple trees this spring for best yield", _WINTER
    ) is None


def test_humanize_duration_is_coarse_and_natural() -> None:
    assert humanize_duration(10) == "just now"
    assert humanize_duration(120) == "2 minutes"
    assert humanize_duration(7200) == "2 hours"
    assert humanize_duration(2 * 86400) == "2 days"


def test_turn_injects_the_time_section(brain: Mimir) -> None:
    result = brain.turn("hello there")
    assert any(s.name == "time" for s in result.context.sections)


def test_turn_intercepts_a_time_query_without_the_model(brain: Mimir) -> None:
    result = brain.turn("what time is it?")
    assert "Today is" in result.reply and "currently" in result.reply
    assert result.baked == []  # a deterministic time answer learns nothing


def test_recalled_facts_carry_their_age() -> None:
    # Component 2: a recalled fact renders with a relative-age tag so the model can reason about
    # recency. build_context is pure — it gets `now_ts`, so this is deterministic.
    import time as _t

    from mimir.context.build import _memory_line
    from mimir.storage.models import EvidenceTier, Memory

    now = _t.time()
    mem = Memory(text="the gate was fixed", evidence_tier=EvidenceTier.CONVERSATION,
                 provenance="conversation", created_at=now - 3 * 86400)
    line = _memory_line(mem, now)
    assert "3 days ago" in line
    assert "tier=conversation" in line and "source=conversation" in line
    # Without now_ts, the age tag is omitted (backwards-compatible).
    assert "ago" not in _memory_line(mem)
