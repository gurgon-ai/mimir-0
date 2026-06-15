"""Temporal grounding — the system's clock + calendar sense (DESIGN §3e).

A stripped, universal extraction of the home AI's temporal awareness. Two pieces here (component 1):

- ``time_prefix`` — a compact "[Time: …; season; N days to …]" line injected each turn so the model
  can answer relative-time questions ("how long ago was that?", "what season is it?") instead of
  hallucinating a date.
- ``answer_time_query`` — a deterministic intercept that answers plain time/date/season questions
  with **zero model cost**.

Timezone and hemisphere are config (``[locale]``), so nothing about a place is baked into core — the
default is the host's local zone, northern seasons. The functions are **pure** (they take the moment
as an argument); only ``local_now`` reads the wall clock, so everything else is unit-testable
with a fixed instant.
"""

from __future__ import annotations

import datetime as _dt
import re
import statistics

# Northern-hemisphere meteorological/astronomical season starts (month, day). Close enough for
# conversation. The southern hemisphere runs the opposite season on the same dates.
_SEASON_STARTS_NORTH = [(3, 20, "spring"), (6, 21, "summer"), (9, 22, "fall"), (12, 21, "winter")]
_SEASON_STARTS_SOUTH = [(3, 20, "fall"), (6, 21, "winter"), (9, 22, "spring"), (12, 21, "summer")]


# A fixed UTC offset: "UTC", "UTC-7", "UTC+05:30", "GMT-8", "-07:00". Resolves with pure stdlib
# arithmetic (no tz database / no `tzdata` package) — the zero-dep way to pin a zone, sans DST.
_OFFSET_RE = re.compile(r"^\s*(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?\s*$", re.IGNORECASE)


def resolve_timezone(timezone: str | None) -> _dt.tzinfo | None:
    """A ``tzinfo`` for ``timezone``, or ``None`` to mean 'use the host's local clock'.

    Order: a literal UTC offset (always works, no package) → an IANA name via ``zoneinfo`` (needs
    the OS tz database or the optional ``tzdata`` extra) → ``None`` (host local) if neither. So
    ``UTC``/``UTC-08:00`` work everywhere; ``America/Vancouver`` works where a tz db is present.
    """
    if not timezone:
        return None
    name = timezone.strip()
    if name.upper() in ("UTC", "GMT", "Z"):
        return _dt.UTC
    match = _OFFSET_RE.match(name)
    if match:
        hours, minutes = int(match.group(2)), int(match.group(3) or 0)
        if hours <= 14 and minutes < 60:
            sign = 1 if match.group(1) == "+" else -1
            return _dt.timezone(sign * _dt.timedelta(hours=hours, minutes=minutes))
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # no tz db for this name — caller falls back to host local
        return None


def local_now(timezone: str | None = None) -> _dt.datetime:
    """The current moment as an aware datetime. The ONE wall-clock read — everything else is pure.

    ``timezone`` is a UTC offset (``UTC-08:00``) or IANA name (``America/Vancouver``); ``None`` uses
    the host's local zone. Anything that doesn't resolve falls back to host-local with no crash — a
    clock is never load-bearing enough to fail a boot, and host-local is correct when the machine
    runs in your timezone (the common home case).
    """
    tz = resolve_timezone(timezone)
    if tz is not None:
        return _dt.datetime.now(tz)
    return _dt.datetime.now().astimezone()


def _season_starts(hemisphere: str) -> list[tuple[int, int, str]]:
    return _SEASON_STARTS_SOUTH if hemisphere.lower().startswith("s") else _SEASON_STARTS_NORTH


def season_of(now: _dt.datetime, hemisphere: str = "north") -> str:
    """The current season name for the hemisphere."""
    starts = _season_starts(hemisphere)
    for month, day, name in reversed(starts):
        if (now.month, now.day) >= (month, day):
            return name
    return starts[-1][2]  # before the first start of the year → the prior (wrapped) season


def _days_until(now: _dt.datetime, month: int, day: int) -> int:
    target = now.date().replace(month=month, day=day)
    if target <= now.date():
        target = target.replace(year=now.year + 1)
    return (target - now.date()).days


def next_season(now: _dt.datetime, hemisphere: str = "north") -> tuple[str, int]:
    """``(next_season_name, days_until_it_starts)``."""
    for month, day, name in _season_starts(hemisphere):
        if (now.month, now.day) < (month, day):
            return name, _days_until(now, month, day)
    first = _season_starts(hemisphere)[0]
    return first[2], _days_until(now, first[0], first[1])


def time_prefix(now: _dt.datetime, hemisphere: str = "north") -> str:
    """The compact time/season line injected each turn (DESIGN §3e)."""
    h = now.hour % 12 or 12
    clock = f"{h}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    season = season_of(now, hemisphere)
    nxt, days = next_season(now, hemisphere)
    return (
        f"It is {now:%A}, {now:%B} {now.day} {now.year}, {clock}. "
        f"Season: {season} ({nxt} in {days} day{'s' if days != 1 else ''})."
    )


# -- the deterministic time-query intercept (no model cost) ----------------------------

_TIME_TRIGGERS = (
    "what time", "what day", "what date", "what month", "what year", "what season",
    "is it spring", "is it summer", "is it fall", "is it autumn", "is it winter",
    "days until", "days till", "days to", "how long until", "how long till",
    "when is spring", "when is summer", "when is fall", "when is autumn", "when is winter",
)
_SEASON_COUNTDOWN_WORDS = ("until", "till", "to", "when", "long", "days", "start", "begin")
_SEASONS = ("spring", "summer", "fall", "autumn", "winter")


def answer_time_query(
    text: str, now: _dt.datetime, hemisphere: str = "north"
) -> str | None:
    """A direct answer to an explicit time/date/season question, else ``None``.

    ``None`` lets the model handle it. Zero model cost. Guarded to short, clearly time-focused
    queries — a long question that merely contains "what day" (e.g. "what day should I prune the
    apples") is left to the model.
    """
    if not text:
        return None
    norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
    if not any(t in norm for t in _TIME_TRIGGERS) or len(norm.split()) > 8:
        return None

    starts = {name: (m, d) for m, d, name in _season_starts(hemisphere)}
    starts.setdefault("autumn", starts.get("fall", (9, 22)))
    for season in _SEASONS:
        if season in norm and any(w in norm for w in _SEASON_COUNTDOWN_WORDS):
            m, d = starts[season]
            days = _days_until(now, m, d)
            label = "fall" if season == "autumn" else season
            if days == 0:
                return f"{label.capitalize()} begins today."
            target = now.date() + _dt.timedelta(days=days)
            return (f"{label.capitalize()} begins in {days} day{'s' if days != 1 else ''} "
                    f"({target:%B} {target.day}).")

    h = now.hour % 12 or 12
    clock = f"{h}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    season = season_of(now, hemisphere)
    nxt, days = next_season(now, hemisphere)
    return (
        f"Today is {now:%A}, {now:%B} {now.day}, {now.year}. It's {clock}. "
        f"It's currently {season}; {nxt} begins in {days} day{'s' if days != 1 else ''}."
    )


# -- shared duration formatting (used by timestamps + baselines, components 2-3) -------

def humanize_duration(seconds: float) -> str:
    """A coarse, natural duration: ``"3 days"``, ``"2 hours"``, ``"45 minutes"``, ``"just now"``."""
    seconds = max(0.0, seconds)
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        m = round(seconds / 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if seconds < 86400:
        h = round(seconds / 3600)
        return f"{h} hour{'s' if h != 1 else ''}"
    d = round(seconds / 86400)
    return f"{d} day{'s' if d != 1 else ''}"


def relative_age(then_ts: float, now_ts: float) -> str:
    """How long ago ``then_ts`` was, as a recency tag for a recalled fact: ``"3 days ago"``,
    ``"just now"``. Future/zero timestamps collapse to ``"just now"`` (never a negative age)."""
    seconds = now_ts - then_ts
    if seconds < 45:
        return "just now"
    return f"{humanize_duration(seconds)} ago"


# -- temporal-awareness baseline: is this gap normal for this user? (component 3) ------

_GAP_FLOOR_S = 4 * 3600  # don't remark on gaps shorter than this (no nagging about a lunch break)


def gap_insight(
    timestamps: list[float], now_ts: float, *, min_events: int = 5, window_days: int = 30
) -> str | None:
    """A note when the time since the last interaction is **notable for this user's own history**,
    else ``None`` (normal, or not enough history). Pure statistics — zero model cost (DESIGN §3e).

    ``timestamps`` are PRIOR interaction epochs (not including the current turn). We measure the
    distribution of past gaps over a rolling window and compare the current gap to it: beyond the
    longest ever → "longest gap recorded"; beyond the 90th percentile (or 2× median) → "longer than
    usual". Anything within normal rhythm, or below a floor, stays silent — awareness, not chatter.
    """
    cutoff = now_ts - window_days * 86400
    recent = sorted(t for t in timestamps if t >= cutoff)
    if len(recent) < min_events:
        return None
    gaps = sorted(recent[i + 1] - recent[i] for i in range(len(recent) - 1))
    if not gaps:
        return None
    current = now_ts - recent[-1]
    if current < _GAP_FLOOR_S:
        return None
    median = statistics.median(gaps)
    p90 = gaps[min(int(len(gaps) * 0.9), len(gaps) - 1)]
    longest = gaps[-1]
    if current > longest * 1.05:
        return (f"It's been {humanize_duration(current)} since you were last here — "
                f"the longest gap I've recorded.")
    if current > p90 or current > median * 2:
        return (f"You haven't been around in {humanize_duration(current)} — longer than usual "
                f"(typically about every {humanize_duration(median)}).")
    return None
