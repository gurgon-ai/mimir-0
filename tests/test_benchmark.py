"""Executable spec for fleet benchmarking: capability checks, scoring, canary (DESIGN §4)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.benchmark import (
    _check_add,
    _check_json_ok,
    _check_pong,
    _check_three_numbered,
    _check_tool_call,
    is_approved,
    score_capability,
)
from mimir.storage.repo import list_catalogue


def test_capability_checkers_are_strict_but_lenient() -> None:
    assert _check_pong("PONG") and _check_pong("PONG.")
    assert not _check_pong("I think the answer is pong")
    assert _check_json_ok('{"ok": true}') and _check_json_ok('Sure: {"ok": true}')
    assert not _check_json_ok('{"ok": false}')
    assert _check_tool_call('{"tool": "get_weather", "args": {"city": "Paris"}}')
    assert not _check_tool_call('{"foo": 1}')
    assert _check_add("def add(a, b):\n    return a + b")
    assert _check_add("```python\ndef add(a, b): return a + b\n```")  # fenced
    assert not _check_add("def sub(a, b): return a - b")
    assert _check_three_numbered("1. apple\n2. pear\n3. plum")
    assert not _check_three_numbered("- apple\n- pear")


def test_score_capability_perfect_and_zero() -> None:
    def perfect(messages: list[dict]) -> str:
        prompt = messages[0]["content"]
        if "PONG" in prompt:
            return "PONG"
        if '{"ok": true}' in prompt:
            return '{"ok": true}'
        return "1. apple\n2. pear\n3. plum"

    assert score_capability(perfect, "talk") == 1.0
    assert score_capability(lambda m: "nope", "talk") == 0.0


def test_is_approved_matches_families() -> None:
    assert is_approved("gemma4") and is_approved("qwen2") and is_approved("command-r")
    assert not is_approved("nomic-bert-moe")


def test_benchmark_fleet_writes_scores(brain: Mimir) -> None:
    # mock fleet families (alpha/beta/gamma) aren't on the allowlist → benchmark all; mock judges
    # can't return a number, so the canary fails and coherence is skipped (judges_ok False).
    result = brain.benchmark_fleet(only_approved=False, judge=True, limit=8)
    assert result.benchmarked == 3
    assert result.judges_ok is False  # canary correctly distrusts the mock judges
    entries = list_catalogue(brain._storage)
    assert entries and all(e.quality is not None for e in entries)
    assert all(e.talk is not None for e in entries)  # capability scores written
