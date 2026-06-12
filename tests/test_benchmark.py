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


def test_smallest_first_and_size_cap(brain: Mimir) -> None:
    # mock fleet sizes: mock-a 3B, mock-b 8B, mock-c 27B. A 10B cap keeps only a + b.
    result = brain.benchmark_fleet(only_approved=False, judge=False, max_params_b=10.0)
    assert {b.model for b in result.results} == {"mock-a", "mock-b"}  # mock-c (27B) skipped
    # benchmarked smallest-first
    assert result.results[0].model == "mock-a"


def _craft_scores(brain: Mimir) -> None:
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()  # catalogue has mock-a/b/c
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.7,
        talk=0.9, tools=0.6, code=0.6, coherence=None,
    )
    update_catalogue_scores(
        brain._storage, "mock-b", return_time=5.0, quality=0.95,
        talk=1.0, tools=0.9, code=0.9, coherence=None,
    )


def test_recommendations_pick_per_role(brain: Mimir) -> None:
    _craft_scores(brain)
    recs = brain.fleet_recommendations()
    assert recs["bake"]["model"] == "mock-b"  # bake prefers quality → the high-quality model
    assert recs["chat"] is not None  # chat is balanced; either could win, just must resolve
    assert recs["bake"]["node"]  # the fastest node holding the model is named


def test_apply_recommendations_repoints_roles(brain: Mimir) -> None:
    _craft_scores(brain)
    applied = brain.apply_recommendations()
    assert applied["bake"] == "mock-b"  # bake re-pointed to the recommended model
    assert brain.config.roles["bake"].model == "mock-b"  # live routing updated


def test_per_node_speed_skips_non_url_nodes(brain: Mimir) -> None:
    # mock catalogue nodes are not URLs, so per-node speed probing is skipped (no crash).
    from mimir.cognition.benchmark import _measure_node_speed

    assert _measure_node_speed("endpoint-0", "mock-a") is None


def test_benchmark_fleet_writes_scores(brain: Mimir) -> None:
    # mock fleet families (alpha/beta/gamma) aren't on the allowlist → benchmark all; mock judges
    # can't return a number, so the canary fails and coherence is skipped (judges_ok False).
    result = brain.benchmark_fleet(only_approved=False, judge=True, limit=8)
    assert result.benchmarked == 3
    assert result.judges_ok is False  # canary correctly distrusts the mock judges
    entries = list_catalogue(brain._storage)
    assert entries and all(e.quality is not None for e in entries)
    assert all(e.talk is not None for e in entries)  # capability scores written
