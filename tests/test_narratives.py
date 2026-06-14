"""Executable spec for temporal narratives — hierarchical daily→weekly→monthly, lossy by design.

Generic sources only (the conversation + what was learned); generated off the hot path; injected as
a [Recent history:] section.
"""

from __future__ import annotations

import datetime as dt

from mimir.brain import Mimir
from mimir.cognition.narratives import (
    compress_monthly,
    compress_weekly,
    generate_daily,
    recent_narratives,
    render_recent_history,
)
from mimir.storage.repo import get_narrative, list_narratives, save_narrative

_NOW = dt.datetime(2026, 6, 14, 22, 0)


def test_daily_narrative_is_generated_and_idempotent(brain: Mimir) -> None:
    brain.turn("I fixed the north gate today and ordered new hinges.")  # gives material to journal
    first = generate_daily(brain._model, brain._storage, now=_NOW)
    assert first and get_narrative(brain._storage, "daily", "2026-06-14") == first
    # Re-running the same day returns the existing entry — no duplicate.
    again = generate_daily(brain._model, brain._storage, now=_NOW)
    assert again == first
    assert len(list_narratives(brain._storage, "daily")) == 1


def test_empty_day_writes_no_journal(brain: Mimir) -> None:
    # Nothing happened (no exchanges, no facts) → no empty entry.
    assert generate_daily(brain._model, brain._storage, now=_NOW) is None


def test_weekly_compresses_older_dailies_and_prunes(brain: Mimir) -> None:
    for d in range(1, 8):  # seven daily entries, 2026-06-01 .. 2026-06-07
        save_narrative(brain._storage, scope="daily", period=f"2026-06-0{d}",
                       narrative=f"Day {d}: did things.", source_count=1)
    weekly = compress_weekly(brain._model, brain._storage, now=_NOW)
    assert weekly is not None
    rows = list_narratives(brain._storage, "weekly")
    assert len(rows) == 1
    # It compressed the 4 oldest (skipping the 3 most recent), keyed newest_to_oldest.
    assert rows[0]["period"] == "2026-06-04_to_2026-06-01"


def test_monthly_compresses_older_weeklies(brain: Mimir) -> None:
    for w in range(1, 8):  # seven weekly entries
        save_narrative(brain._storage, scope="weekly", period=f"2026-W0{w}",
                       narrative=f"Week {w}: themes.", source_count=4)
    monthly = compress_monthly(brain._model, brain._storage, now=_NOW)
    assert monthly is not None
    assert len(list_narratives(brain._storage, "monthly")) == 1
    assert len(list_narratives(brain._storage, "weekly")) == 5  # pruned to retention


def test_recent_history_renders_coarsest_first(brain: Mimir) -> None:
    save_narrative(brain._storage, scope="monthly", period="2026-M1", narrative="A big arc.")
    save_narrative(brain._storage, scope="daily", period="2026-06-14", narrative="Today's detail.")
    recent = recent_narratives(brain._storage)
    assert recent["monthly"] and recent["daily"]
    rendered = render_recent_history(brain._storage)
    assert rendered is not None
    # Monthly (the long arc) is read before the daily detail.
    assert rendered.index("big arc") < rendered.index("Today's detail")


def test_turn_injects_recent_history_when_present(brain: Mimir) -> None:
    save_narrative(brain._storage, scope="daily", period="2026-06-13",
                   narrative="Yesterday I set up the new node.")
    result = brain.turn("morning")
    assert any(s.name == "recent_history" for s in result.context.sections)


def test_brain_generate_narratives_runs_the_cycle(brain: Mimir) -> None:
    brain.turn("Planted the south bed with garlic.")
    stats = brain.generate_narratives()
    assert "daily" in stats and "weekly" in stats and "monthly" in stats
