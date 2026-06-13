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


def test_choose_test_node_picks_fast_enough_else_fastest_never_fails() -> None:
    from mimir.cognition.benchmark import _choose_test_node

    def probe(speeds: dict[str, float | None]):
        seen: list[str] = []
        def p(n: str) -> float | None:
            seen.append(n)
            return speeds.get(n)
        p.seen = seen  # type: ignore[attr-defined]
        return p

    # First node under budget → chosen, and probing STOPS there (B, C never probed).
    p = probe({"A": 5.0, "B": 1.0, "C": 1.0})
    node, sp = _choose_test_node(["A", "B", "C"], p, test_budget=10.0)
    assert node == "A" and sp == {"A": 5.0} and p.seen == ["A"]  # type: ignore[attr-defined]

    # None under budget → the FASTEST that ran wins (capability never failed on speed); all probed.
    node, sp = _choose_test_node(["A", "B", "C"], probe({"A": 40.0, "B": 20.0, "C": 30.0}), 10.0)
    assert node == "B" and sp == {"A": 40.0, "B": 20.0, "C": 30.0}

    # A node that can't run it (probe → None) is skipped; the next viable one is chosen.
    node, sp = _choose_test_node(["A", "B"], probe({"A": None, "B": 3.0}), 10.0)
    assert node == "B" and sp == {"B": 3.0}

    # Installed only on nodes that all fail → no viable node (caller records it, not a quality cut).
    node, sp = _choose_test_node(["A", "B"], probe({"A": None, "B": None}), 10.0)
    assert node is None and sp == {}


def test_complete_speed_matrix_times_only_untimed_acceptable_http_pairings(db_path: str) -> None:
    from mimir.cognition.benchmark import complete_speed_matrix
    from mimir.storage.gateway import StorageGateway
    from mimir.storage.models import CatalogueEntry
    from mimir.storage.repo import (
        replace_catalogue,
        update_catalogue_scores,
        update_catalogue_speed,
    )

    sg = StorageGateway(db_path)
    n = "http://127.0.0.1:9"   # a refused port → the probe fails fast (no real Ollama needed)
    try:
        replace_catalogue(sg, [
            CatalogueEntry(node=n, model="good:8b", family="x", params_b=8.0, scanned_at=1.0),
            CatalogueEntry(node=n, model="timed:8b", family="x", params_b=8.0, scanned_at=1.0),
            CatalogueEntry(node=n, model="weak:8b", family="x", params_b=8.0, scanned_at=1.0),
            CatalogueEntry(node="endpoint-0", model="local:8b", family="x", params_b=8.0,
                           scanned_at=1.0),
            CatalogueEntry(node=n, model="nomic-embed", family="x", params_b=0.1, scanned_at=1.0),
        ])
        sc = dict(talk=1.0, tools=1.0, code=1.0, coherence=None, discipline=1.0, epistemics=1.0,
                  reasoning=1.0)
        update_catalogue_scores(sg, "good:8b", quality=0.9, **sc)    # acceptable, untimed → probe
        update_catalogue_scores(sg, "timed:8b", quality=0.9, **sc)
        update_catalogue_speed(sg, n, "timed:8b", 5.0)               # already timed → skip
        update_catalogue_scores(sg, "weak:8b", quality=0.3, **sc)    # below the floor → skip
        update_catalogue_scores(sg, "local:8b", quality=0.9, **sc)   # non-http node → skip
        # nomic-embed: never scored (embed) → skip
        timed = complete_speed_matrix(sg, min_quality=0.5, num_ctx=2048)
        assert timed == 1   # only good:8b qualifies for a time trial
    finally:
        sg.close()


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


def test_inverted_size_band_is_swapped_not_left_empty(brain: Mimir) -> None:
    # min > max is always a transposed pair — it must SWAP (to [3, 10]) and qualify the models in
    # that band, never dead-end with an empty run (the "0 eligible / empty round" bug).
    result = brain.benchmark_fleet(only_approved=False, judge=False,
                                   max_params_b=3.0, min_params_b=10.0)
    assert {b.model for b in result.results} == {"mock-a", "mock-b"}  # 3B + 8B in [3,10]; 27B out


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


def test_failed_latency_probe_is_unmeasured_not_instant() -> None:
    # A latency probe that raises (timeout/transport) must record None, NEVER 0.0 — else a
    # timing-out model sorts as the FASTEST and sails under any latency cap. (Regression: a model
    # aced the short capability probes but timed out on the longer latency generation and showed
    # 0.0s/turn, winning 'fastest node'.)
    from mimir.cognition.benchmark import _measure_turn_latency

    def boom(_messages: list) -> str:
        raise TimeoutError("generation timed out")

    assert _measure_turn_latency(boom) is None
    # A successful probe yields a measured number, not None (the instant mock rounds to ~0.0, but
    # the point is it's measured — distinct from the None 'unmeasured/failed' signal).
    assert _measure_turn_latency(lambda _m: "x" * 400) is not None


def test_placement_matrix_shows_each_model_on_every_node_with_a_winner(db_path: str) -> None:
    # The placement matrix must list a model under EVERY node it runs on (with that node's speed),
    # unlike the results board (one row per model on its test node), and crown each node's winner.
    from mimir.cognition.fleet import placement_matrix
    from mimir.storage.gateway import StorageGateway
    from mimir.storage.models import CatalogueEntry
    from mimir.storage.repo import (
        replace_catalogue,
        update_catalogue_scores,
        update_catalogue_speed,
    )

    beast_node, edge_node = "http://127.0.0.1:11434", "http://192.168.2.60:11434"
    sg = StorageGateway(db_path)
    try:
        # mid:12b on the beast + a slow edge; small:3b on the edge only.
        def entry(node: str, model: str, b: float) -> CatalogueEntry:
            return CatalogueEntry(node=node, model=model, family="x", params_b=b, scanned_at=1.0)
        replace_catalogue(sg, [
            entry(beast_node, "mid:12b", 12.0),
            entry(edge_node, "mid:12b", 12.0),
            entry(edge_node, "small:3b", 3.0),
        ])
        sc = dict(talk=1.0, tools=1.0, code=1.0, coherence=None,
                  discipline=1.0, epistemics=1.0, reasoning=0.9)
        update_catalogue_scores(sg, "mid:12b", quality=0.86, **sc)
        update_catalogue_scores(sg, "small:3b", quality=0.6, **sc)
        update_catalogue_speed(sg, beast_node, "mid:12b", 1.5)    # fast on the beast
        update_catalogue_speed(sg, edge_node, "mid:12b", 62.0)    # slow on the edge
        update_catalogue_speed(sg, edge_node, "small:3b", 4.0)

        pm = placement_matrix(sg)
        beast, edge = pm["by_node"][beast_node], pm["by_node"][edge_node]
        # mid:12b appears on BOTH nodes, each with its OWN per-node speed.
        assert next(m for m in beast if m["model"] == "mid:12b")["return_time"] == 1.5
        assert next(m for m in edge if m["model"] == "mid:12b")["return_time"] == 62.0
        # The beast's winner is mid:12b; on the edge it's also mid:12b (quality beats speed)...
        assert next(m for m in beast if m.get("champion"))["model"] == "mid:12b"
        assert next(m for m in edge if m.get("champion"))["model"] == "mid:12b"
        # ...but small:3b is the FASTEST on the edge (4s < 62s), so it earns the ⚡ tag.
        assert next(m for m in edge if m.get("fastest"))["model"] == "small:3b"
    finally:
        sg.close()


def test_bar_reason_names_the_failing_floor() -> None:
    # The shared role gate: None when a model clears every floor, else a reason naming the first
    # failing capability + the floor it missed. This is what the leaderboard renders instead of a
    # silent drop, and it's the SAME predicate recommend_roles uses to pick winners.
    from mimir.cognition.fleet import _bar_reason

    needs = ("discipline", "epistemics", "reasoning")
    assert _bar_reason({"quality": 0.8, "discipline": 0.9, "epistemics": 0.7, "reasoning": 0.8},
                       needs) is None
    assert _bar_reason({"quality": 0.6, "discipline": 0.25, "epistemics": 0.9, "reasoning": 0.9},
                       needs) == "discipline 0.25 < 0.50"
    assert _bar_reason({"quality": None}, ("talk",)) == "not benchmarked yet"


def test_model_pool_explains_why_a_model_is_barred(brain: Mimir) -> None:
    # The model pool (leaderboard) must EXPLAIN the verdict, not drop barred models silently: a
    # fluent-but-leaky model (discipline 0) is barred from chat with the reason, yet still eligible
    # for bake (gated on talk). DESIGN §10 — no silent state.
    from mimir.storage.repo import update_catalogue_scores

    brain.scan_fleet()
    update_catalogue_scores(
        brain._storage, "mock-a", return_time=0.5, quality=0.6, talk=1.0, tools=0.6, code=0.6,
        coherence=None, discipline=0.0, epistemics=1.0, reasoning=1.0,  # leaks tags
    )
    row = next(m for m in brain.model_pool()["models"] if m["model"] == "mock-a")
    assert row["barred"]["chat"] == "discipline 0.00 < 0.50"   # explained, not hidden
    assert "reasoning" in row["barred"]                         # identity roles both gate on it
    assert "bake" in row["eligible_roles"]                      # gated on talk, which it has
    assert "chat" not in row["eligible_roles"]


def test_tournament_finals_restricts_to_kept_finalists(brain: Mimir) -> None:
    # The finals round recommends only among the survivors the user carried in. mock-b is the better
    # model, but if the user keeps ONLY mock-a, the finals must champion mock-a (the veto wins).
    _craft_scores(brain)  # mock-a + mock-b benchmarked; mock-b higher quality
    full = brain.fleet_recommendations()
    assert full["bake"]["model"] == "mock-b"  # unrestricted, the better model wins
    finals = brain.tournament_finals({"mock-a"})
    assert finals["bake"]["model"] == "mock-a"  # restricted to the kept finalist


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
