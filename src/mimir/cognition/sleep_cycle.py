"""The sleep cycle — a wall-clock maintenance window, resumable phase-by-phase (DESIGN §5a).

The post-response *burst* worker assumes the model sits idle while the user reads the reply. That
assumption breaks twice over: with **streaming** chat the model is busy until the last token, and on
a **slow machine** a single turn can eat the whole window. So the heavy, model-touching maintenance
(consolidation + narratives) gets its own **wall-clock window** when nobody's around — sleep —
instead of fighting for scraps between turns.

This module is the generic mechanism; *what* runs in each phase is bound by the brain. It is pure
and clock-injected so tests drive it deterministically:

- **Window math** — ``in_window`` / ``minutes_remaining_in_window`` understand a window that crosses
  midnight (e.g. ``23:00``→``06:00``).
- **Phase budgeting** — before starting a phase, compare the minutes left in the window to the
  phase's declared minimum; if it won't fit, **skip it** rather than start work that can't finish.
- **Resume + once-a-day** — each phase's status is checkpointed (via injected ``load_state`` /
  ``save_state``) under the cycle's date, so a same-night restart skips completed phases and the
  cycle never runs twice in a day. A new date starts fresh.
- **Yield to foreground** — between phases an ``is_busy`` predicate defers the rest to the next
  check if a turn arrives mid-cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger("mimir.sleep_cycle")


def _hhmm_to_minutes(value: str) -> int:
    h, m = (int(p) for p in value.strip().split(":"))
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad HH:MM time: {value!r}")
    return h * 60 + m


def in_window(now: datetime, start: str, end: str) -> bool:
    """Is ``now`` (a local-time datetime) inside the [start, end) window? Cross-midnight aware."""
    s, e = _hhmm_to_minutes(start), _hhmm_to_minutes(end)
    cur = now.hour * 60 + now.minute
    if s == e:
        return False  # zero-length window — disabled
    if s < e:
        return s <= cur < e
    return cur >= s or cur < e  # window crosses midnight


def minutes_remaining_in_window(now: datetime, start: str, end: str) -> float:
    """Minutes until the current window closes; ``0.0`` if ``now`` is not inside it."""
    if not in_window(now, start, end):
        return 0.0
    eh, em = divmod(_hhmm_to_minutes(end), 60)
    end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    if end_dt <= now:  # the closing time is tomorrow (we're past midnight inside the window)
        end_dt += timedelta(days=1)
    return (end_dt - now).total_seconds() / 60.0


@dataclass(slots=True)
class Phase:
    """One unit of maintenance: a name, the minutes it needs to be worth starting, and the work."""

    name: str
    min_minutes: float
    run: Callable[[], object]  # returns anything (stats); the brain binds consolidate/narratives


@dataclass(slots=True)
class CycleReport:
    """The outcome of one ``run_cycle`` call — what ran, what was skipped/failed, did it finish."""

    date: str
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)  # yielded to foreground; resumes next check
    completed: bool = False
    already_done: bool = False


_TERMINAL = ("completed", "skipped", "failed")


def _budget_label(remaining: float) -> str:
    """Human-readable budget for the phase-start log. A forced run has no window limit (``inf``);
    say so plainly instead of printing a bare ``-1 min left`` that reads like a negative budget."""
    return "forced, no window limit" if remaining == float("inf") else f"{remaining:.0f} min left"


def run_cycle(
    phases: list[Phase],
    *,
    clock: Callable[[], datetime],
    window_start: str,
    window_end: str,
    load_state: Callable[[], dict[str, Any]],
    save_state: Callable[[dict[str, Any]], None],
    is_busy: Callable[[], bool] = lambda: False,
    force: bool = False,
) -> CycleReport:
    """Run the maintenance phases in order, honouring the window budget and the daily checkpoint.

    ``force`` bypasses the window check and the once-a-day guard (the manual "run sleep now" path) —
    it still records progress so a later scheduled run sees the work as done for the day.
    """
    now = clock()
    today = now.strftime("%Y-%m-%d")
    state = load_state() or {}

    if not force and state.get("date") == today and state.get("completed"):
        return CycleReport(date=today, completed=True, already_done=True)

    if state.get("date") != today:
        if state.get("date") and not state.get("completed"):
            log.info("sleep: prior cycle (%s) incomplete; starting fresh for %s",
                     state.get("date"), today)
        state = {"date": today, "phases": {}, "completed": False}
        save_state(state)

    report = CycleReport(date=today)
    phase_status: dict[str, Any] = state.setdefault("phases", {})

    for phase in phases:
        if phase_status.get(phase.name) == "completed":
            continue  # already done this date (same-night restart) — don't repeat

        if is_busy():  # a turn arrived — defer the rest to the next check, keep what's done
            log.info("sleep: foreground active; deferring %s and the rest", phase.name)
            report.deferred = [p.name for p in phases if phase_status.get(p.name) != "completed"]
            break

        remaining = float("inf") if force else minutes_remaining_in_window(
            clock(), window_start, window_end
        )
        if remaining < phase.min_minutes:
            phase_status[phase.name] = "skipped"
            report.skipped.append(phase.name)
            save_state(state)
            log.info("sleep: skip %s — %.0f min left, needs %.0f", phase.name, remaining,
                     phase.min_minutes)
            continue

        log.info("sleep: phase %s starting (%s)", phase.name, _budget_label(remaining))
        try:
            phase.run()
            phase_status[phase.name] = "completed"
            report.ran.append(phase.name)
        except Exception as exc:  # one phase failing never aborts the cycle (§10)
            phase_status[phase.name] = "failed"
            report.failed.append(phase.name)
            log.error("sleep: phase %s failed: %s", phase.name, exc, exc_info=True)
        save_state(state)

    if all(phase_status.get(p.name) in _TERMINAL for p in phases):
        state["completed"] = True
        report.completed = True
        save_state(state)
    return report
