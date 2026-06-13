"""Executable spec for fleet benchmarking: capability checks, scoring, canary (DESIGN §4)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.benchmark import (
    _check_add,
    _check_json_ok,
    _check_no_brackets,
    _check_no_dog_or_cat,
    _check_pong,
    _check_reverse_python,
    _check_three_numbered,
    _check_tool_call,
    _expect_int,
    _last_int,
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


def test_discipline_checkers_catch_the_tag_leak() -> None:
    # The core failure mode: any square bracket when told not to is a fail.
    assert _check_no_brackets("Mona's favorite tea is genmaicha.")
    assert not _check_no_brackets("Genmaicha [tier=stated_by_user; source=Mona].")
    assert not _check_no_brackets("She likes genmaicha [1].")  # any bracket, even a citation
    assert not _check_no_brackets("")  # silence is not discipline
    # Negative lexical constraint.
    assert _check_no_dog_or_cat("Fish") and _check_no_dog_or_cat("a hamster")
    assert not _check_no_dog_or_cat("A cat")
    assert not _check_no_dog_or_cat("I would suggest perhaps a goldfish or a parakeet")  # too long


def test_reasoning_checkers_require_the_right_answer() -> None:
    # The reasoning dimension grades problem-solving, not formatting: the final integer must match.
    assert _last_int("after draining and refilling, the answer is 242.") == 242
    assert _last_int("that's 1,234 widgets total") == 1234
    assert _last_int("no number here") is None
    check242 = _expect_int(242)
    assert check242("242") and check242("So the tank holds 242 liters.")
    assert not check242("It holds 250 liters.")  # wrong answer fails, however fluent
    assert _expect_int(3)("3") and not _expect_int(3)("the letter appears 2 times")
    # An instruction transform: 'PYTHON' reversed + lowercased.
    assert _check_reverse_python("nohtyp") and _check_reverse_python("The result is NOHTYP.")
    assert not _check_reverse_python("python")


def test_outside_in_ordering() -> None:
    from mimir.cognition.benchmark import _outside_in

    # smallest→largest in → big, small, big, small, … (so an ETA samples both extremes early)
    assert _outside_in(["a", "b", "c", "d", "e"]) == ["e", "a", "d", "b", "c"]
    assert _outside_in(["small", "large"]) == ["large", "small"]
    assert _outside_in([]) == [] and _outside_in(["x"]) == ["x"]


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


def test_size_cap_and_outside_in_order(brain: Mimir) -> None:
    # mock fleet sizes: mock-a 3B, mock-b 8B, mock-c 27B. A 10B cap keeps only a + b.
    result = brain.benchmark_fleet(only_approved=False, judge=False, max_params_b=10.0)
    assert {b.model for b in result.results} == {"mock-a", "mock-b"}  # mock-c (27B) skipped
    # outside-in order = biggest first, so mock-b (8B) is scored before mock-a (3B).
    assert result.results[0].model == "mock-b"


def test_size_floor_excludes_tiny_models(brain: Mimir) -> None:
    # A 5B floor drops mock-a (3B) so it can't out-compete the bigger models on capable hardware.
    result = brain.benchmark_fleet(only_approved=False, judge=False, min_params_b=5.0)
    assert {b.model for b in result.results} == {"mock-b", "mock-c"}  # mock-a (3B) under the floor
    assert result.skipped_too_small == 1


def test_tournament_only_models_restricts_the_round(brain: Mimir) -> None:
    # A later tournament round re-tests only the survivors the user kept.
    result = brain.benchmark_fleet(only_approved=False, judge=False, only_models={"mock-b"})
    assert {b.model for b in result.results} == {"mock-b"}
    assert result.eligible == 1  # the round's pool is just the survivor


def test_tournament_triage_skips_the_framework_and_can_be_ephemeral(brain: Mimir) -> None:
    from mimir.storage.repo import list_catalogue

    # Triage (framework=False) runs the cheap dimensions only and, ephemeral, writes nothing.
    result = brain.benchmark_fleet(
        only_approved=False, judge=True, framework=False, persist=False,
    )
    assert result.benchmarked == 3  # all three mock models triaged
    assert all(b.coherence is None for b in result.results)  # judge skipped in triage
    # Ephemeral: the catalogue still has no scores (the scouting round didn't pollute it).
    assert all(e.quality is None for e in list_catalogue(brain._storage))


def _craft_scores(brain: Mimir) -> None:
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()  # catalogue has mock-a/b/c
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.7,
        talk=0.9, tools=0.6, code=0.6, coherence=None, discipline=0.9, epistemics=0.9,
        reasoning=0.8,
    )
    update_catalogue_scores(
        brain._storage, "mock-b", return_time=5.0, quality=0.95,
        talk=1.0, tools=0.9, code=0.9, coherence=None, discipline=0.9, epistemics=0.9,
        reasoning=0.9,
    )


def test_recommendations_pick_per_role(brain: Mimir) -> None:
    _craft_scores(brain)
    recs = brain.fleet_recommendations()
    assert recs["bake"]["model"] == "mock-b"  # bake prefers quality → the high-quality model
    assert recs["chat"] is not None  # chat ranks on quality under the cap; must resolve to a model
    assert recs["bake"]["node"]  # the fastest node holding the model is named


def test_leaky_model_is_barred_from_identity_roles(brain: Mimir) -> None:
    # A model that's fluent (high talk) but leaks tags (low discipline) must NOT be recommended for
    # chat/reasoning, while a disciplined one is. bake (gated on talk) can still take the leaker.
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.6, talk=1.0, tools=0.6, code=0.6,
        coherence=None, discipline=0.0, epistemics=1.0, reasoning=1.0,  # capable but LEAKS tags
    )
    update_catalogue_scores(
        brain._storage, "mock-b", return_time=1.0, quality=0.9, talk=1.0, tools=0.9, code=0.9,
        coherence=None, discipline=1.0, epistemics=1.0, reasoning=1.0,  # disciplined + epistemic
    )
    recs = brain.fleet_recommendations()
    assert recs["chat"]["model"] == "mock-b"  # only the disciplined model qualifies for chat
    assert recs["reasoning"]["model"] == "mock-b"  # ...and for reasoning (self-model synthesis)
    assert recs["bake"] is not None  # bake gates on talk, not discipline — still resolvable


def test_tier_blind_model_is_barred_from_identity_roles(brain: Mimir) -> None:
    # A model that's disciplined but epistemically incompetent (ignores evidence tiers) must NOT be
    # recommended for chat/reasoning — the framework is never handed to a model that won't use it.
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.7, talk=1.0, tools=0.7, code=0.7,
        coherence=None, discipline=1.0, epistemics=0.1, reasoning=1.0,  # ignores tiers/provenance
    )
    update_catalogue_scores(
        brain._storage, "mock-b", return_time=1.0, quality=0.9, talk=1.0, tools=0.9, code=0.9,
        coherence=None, discipline=1.0, epistemics=1.0, reasoning=1.0,  # uses the framework
    )
    recs = brain.fleet_recommendations()
    assert recs["chat"]["model"] == "mock-b"
    assert recs["reasoning"]["model"] == "mock-b"
    assert recs["bake"] is not None  # bake doesn't require epistemics


def test_reasoning_incompetent_model_is_barred_from_thinking_roles(brain: Mimir) -> None:
    # A model that's fluent + disciplined + epistemic but CAN'T SOLVE PROBLEMS (low reasoning) must
    # NOT win chat/reasoning/code — those roles need actual problem-solving, not just good manners.
    # This is the gate that stops a model 'that can't do the job' from sweeping on format alone.
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.9, talk=1.0, tools=1.0, code=1.0,
        coherence=None, discipline=1.0, epistemics=1.0, reasoning=0.1,  # polished but can't solve
    )
    update_catalogue_scores(
        brain._storage, "mock-b", return_time=1.0, quality=0.8, talk=1.0, tools=0.9, code=0.9,
        coherence=None, discipline=1.0, epistemics=1.0, reasoning=0.9,  # actually solves problems
    )
    recs = brain.fleet_recommendations()
    assert recs["chat"]["model"] == "mock-b"  # the solver wins the thinking roles, despite lower q
    assert recs["reasoning"]["model"] == "mock-b"
    assert recs["code"]["model"] == "mock-b"
    # bake gates on talk only (not reasoning) → the higher-quality polished model still wins it.
    assert recs["bake"]["model"] == "mock-a"


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
    assert all(e.discipline is not None for e in entries)  # incl. the discipline dimension
    assert all(e.epistemics is not None for e in entries)  # ...and the epistemics dimension
    assert all(e.reasoning is not None for e in entries)  # ...and the reasoning dimension
