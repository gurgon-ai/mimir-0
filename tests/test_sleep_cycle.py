"""The wall-clock sleep cycle (DESIGN §5a): window math + the resumable phase orchestrator.

Pure and clock-injected, so these run without threads or real time. The last test exercises the
brain's manual ``run_sleep_cycle(force=True)`` path end to end against the mock-backed fixture.
"""

from __future__ import annotations

from datetime import datetime

from mimir.brain import Mimir
from mimir.cognition.sleep_cycle import (
    Phase,
    in_window,
    minutes_remaining_in_window,
    run_cycle,
)


def _state_io() -> tuple[dict, callable, callable]:
    """An in-memory load/save pair standing in for the kv checkpoint."""
    box: dict = {}
    return box, (lambda: box.get("v", {})), (lambda s: box.__setitem__("v", dict(s)))


# -- window math ---------------------------------------------------------------------

def test_in_window_same_day() -> None:
    assert in_window(datetime(2026, 6, 14, 3, 0), "02:00", "06:00")
    assert not in_window(datetime(2026, 6, 14, 1, 59), "02:00", "06:00")
    assert not in_window(datetime(2026, 6, 14, 6, 0), "02:00", "06:00")  # half-open: [start, end)


def test_in_window_crosses_midnight() -> None:
    assert in_window(datetime(2026, 6, 14, 23, 30), "22:00", "06:00")
    assert in_window(datetime(2026, 6, 14, 5, 0), "22:00", "06:00")
    assert not in_window(datetime(2026, 6, 14, 12, 0), "22:00", "06:00")


def test_zero_length_window_is_disabled() -> None:
    assert not in_window(datetime(2026, 6, 14, 3, 0), "02:00", "02:00")


def test_minutes_remaining() -> None:
    assert minutes_remaining_in_window(datetime(2026, 6, 14, 3, 0), "02:00", "06:00") == 180.0
    # past midnight inside a cross-midnight window: closing time is later today
    assert minutes_remaining_in_window(datetime(2026, 6, 14, 5, 0), "22:00", "06:00") == 60.0
    # not in the window → zero
    assert minutes_remaining_in_window(datetime(2026, 6, 14, 12, 0), "02:00", "06:00") == 0.0


# -- the orchestrator ----------------------------------------------------------------

def test_runs_all_phases_then_marks_done() -> None:
    ran: list[str] = []
    phases = [Phase("a", 2.0, lambda: ran.append("a")), Phase("b", 2.0, lambda: ran.append("b"))]
    _, load, save = _state_io()
    now = datetime(2026, 6, 14, 3, 0)  # 3h left
    report = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save)
    assert ran == ["a", "b"]
    assert report.completed and report.ran == ["a", "b"] and not report.skipped


def test_skips_phase_that_wont_fit() -> None:
    ran: list[str] = []
    phases = [Phase("fast", 2.0, lambda: ran.append("fast")),
              Phase("slow", 30.0, lambda: ran.append("slow"))]
    _, load, save = _state_io()
    now = datetime(2026, 6, 14, 5, 55)  # only 5 min left → slow can't fit
    report = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save)
    assert ran == ["fast"]
    assert report.ran == ["fast"] and report.skipped == ["slow"]
    assert report.completed  # skipped-for-time is terminal for the day


def test_once_a_day_guard() -> None:
    calls: list[str] = []
    phases = [Phase("a", 2.0, lambda: calls.append("a"))]
    _, load, save = _state_io()
    now = datetime(2026, 6, 14, 3, 0)
    first = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                      load_state=load, save_state=save)
    second = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save)
    assert first.completed and not first.already_done
    assert second.already_done and calls == ["a"]  # not re-run


def test_resumes_completed_phases_after_restart() -> None:
    """A same-night re-run (e.g. process restart) skips work already done, runs the rest."""
    box, load, save = _state_io()
    ran: list[str] = []
    now = datetime(2026, 6, 14, 3, 0)
    # First pass: only 'a' fits (b needs more than is left in this contrived window check).
    phases1 = [Phase("a", 2.0, lambda: ran.append("a")), Phase("b", 600.0, lambda: ran.append("b"))]
    run_cycle(phases1, clock=lambda: now, window_start="02:00", window_end="06:00",
              load_state=load, save_state=save)
    assert ran == ["a"]
    # Second pass with room for b — 'a' must not repeat.
    box["v"]["completed"] = False  # pretend the day isn't finished yet
    phases2 = [Phase("a", 2.0, lambda: ran.append("a")), Phase("b", 2.0, lambda: ran.append("b"))]
    run_cycle(phases2, clock=lambda: now, window_start="02:00", window_end="06:00",
              load_state=load, save_state=save)
    assert ran == ["a", "b"]  # 'a' skipped as already completed, 'b' now run


def test_force_bypasses_window_and_guard() -> None:
    ran: list[str] = []
    phases = [Phase("a", 999.0, lambda: ran.append("a"))]  # huge min — never fits a real window
    _, load, save = _state_io()
    noon = datetime(2026, 6, 14, 12, 0)  # outside any night window
    report = run_cycle(phases, clock=lambda: noon, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save, force=True)
    assert ran == ["a"] and report.completed


def test_defers_to_foreground() -> None:
    ran: list[str] = []
    phases = [Phase("a", 2.0, lambda: ran.append("a")), Phase("b", 2.0, lambda: ran.append("b"))]
    _, load, save = _state_io()
    now = datetime(2026, 6, 14, 3, 0)
    report = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save, is_busy=lambda: True)
    assert ran == [] and not report.completed
    assert "a" in report.deferred and "b" in report.deferred


def test_failed_phase_does_not_abort_cycle() -> None:
    ran: list[str] = []

    def _boom() -> None:
        raise RuntimeError("kaboom")

    phases = [Phase("a", 2.0, _boom), Phase("b", 2.0, lambda: ran.append("b"))]
    _, load, save = _state_io()
    now = datetime(2026, 6, 14, 3, 0)
    report = run_cycle(phases, clock=lambda: now, window_start="02:00", window_end="06:00",
                       load_state=load, save_state=save)
    assert ran == ["b"]  # b still ran after a failed
    assert report.failed == ["a"] and report.ran == ["b"] and report.completed


# -- brain integration ---------------------------------------------------------------

def test_settings_override_config(brain: Mimir) -> None:
    base = brain.settings()
    assert base["sleep_window_start"] == "02:00"  # config default
    assert base["overridden"] == []
    updated = brain.update_settings({"sleep_window_start": "23:30", "sleep_enabled": False})
    assert updated["sleep_window_start"] == "23:30"
    assert updated["sleep_enabled"] is False
    assert "sleep_window_start" in updated["overridden"]
    # the effective window the scheduler/status read reflects the override
    status = brain.sleep_cycle_status()
    assert status["window_start"] == "23:30" and status["enabled"] is False


def test_settings_reject_bad_values(brain: Mimir) -> None:
    import pytest

    from mimir.errors import ConfigError
    with pytest.raises((ConfigError, ValueError)):
        brain.update_settings({"sleep_window_start": "9999"})
    with pytest.raises((ConfigError, ValueError)):
        brain.update_settings({"timezone": "Not/AZone"})
    with pytest.raises(ConfigError):
        brain.update_settings({"bogus_key": "x"})


def test_timezone_setting_persists(brain: Mimir) -> None:
    # "UTC" is always offered (real zoneinfo list or the curated fallback when tzdata is absent),
    # so it is accepted and stored regardless of whether the host has a tz database.
    brain.update_settings({"timezone": "UTC"})
    status = brain.sleep_cycle_status()
    assert status["timezone"] == "UTC"
    assert "timezone_active" in status  # reports whether tzdata actually resolved it
    assert "UTC" in brain.available_timezones()


def test_brain_manual_sleep_cycle(brain: Mimir) -> None:
    status_before = brain.sleep_cycle_status()
    assert status_before["last_cycle_date"] is None
    report = brain.run_sleep_cycle(force=True)
    assert "consolidate" in report.ran  # consolidation always runs on the force path
    status_after = brain.sleep_cycle_status()
    assert status_after["last_cycle_date"] is not None
    assert status_after["completed"]
    assert status_after["phases"].get("consolidate") == "completed"
