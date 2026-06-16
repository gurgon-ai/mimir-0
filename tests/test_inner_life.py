"""The live inner life — the low-frequency idle loop that thinks between turns (DESIGN §5a).

One spec per load-bearing claim: the chat-priority/edge gates (``should_think``), stimulus building
from universal signals, deterministic picking, and the brain integration — a forced tick stores one
low-confidence, decaying memory, while the loop is off by default and yields to a live turn.
"""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.inner_life import (
    Stimulus,
    compose_thought,
    gather_stimuli,
    pick_stimulus,
    should_think,
)
from mimir.storage.models import MemoryKind
from mimir.storage.repo import list_memories

OK = dict(  # a baseline "everything clear" set of gate inputs
    enabled=True, turn_active=False, degraded=False, now=1000.0,
    last_turn_at=0.0, last_thought_at=0.0, cadence_s=300.0, idle_floor_s=30.0,
)


def test_should_think_passes_when_clear() -> None:
    ok, reason = should_think(**OK)
    assert ok and reason == "ok"


def test_should_think_respects_chat_priority_and_cost_gates() -> None:
    # Each gate, in the order the loop checks them.
    assert should_think(**{**OK, "enabled": False}) == (False, "disabled")
    assert should_think(**{**OK, "turn_active": True}) == (False, "turn in flight")
    assert should_think(**{**OK, "degraded": True}) == (False, "fleet degraded")
    # A turn ended 5s ago — under the 30s idle floor.
    assert should_think(**{**OK, "last_turn_at": 995.0}) == (False, "too soon after a turn")
    # Last thought 100s ago — under the 300s cadence.
    assert should_think(**{**OK, "last_thought_at": 900.0}) == (False, "within cadence")
    # Floors of 0 mean "never" — they don't block.
    assert should_think(**{**OK, "last_turn_at": 0.0, "last_thought_at": 0.0})[0]


def test_pick_stimulus_prefers_priority_then_avoids_repeat() -> None:
    stims = [
        Stimulus("memory", "m", "k1"),
        Stimulus("error", "e", "k2"),
        Stimulus("conflict", "c", "k3"),
    ]
    # Highest priority is "error".
    assert pick_stimulus(stims).kind == "error"
    # Avoiding "error" falls to the next-highest, "conflict".
    assert pick_stimulus(stims, avoid_kind="error").kind == "conflict"
    assert pick_stimulus([]) is None


def test_compose_thought_uses_injected_chat() -> None:
    seen: dict[str, object] = {}

    def fake_chat(messages: list[dict[str, str]]) -> str:
        seen["messages"] = messages
        return "  a quiet thought.  "

    out = compose_thought(fake_chat, Stimulus("memory", "dwell on this", "k"))
    assert out == "a quiet thought."
    msgs = seen["messages"]
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "dwell on this"


def test_gather_stimuli_draws_on_universal_sources(brain: Mimir) -> None:
    brain.turn("My favorite color is blue.")  # bake a real memory
    stimuli = gather_stimuli(
        brain._storage, embedder=brain._embedder,
        recent_errors=["socket timeout on node X"], working_memory_text="we were discussing colors",
    )
    kinds = {s.kind for s in stimuli}
    assert "error" in kinds          # the supplied error
    assert "memory" in kinds         # the baked fact
    assert "working_memory" in kinds  # the supplied rolling summary


def test_inner_life_off_by_default(brain: Mimir) -> None:
    # No opt-in → the loop is a no-op even though the daemon is running.
    assert brain.run_inner_life_tick() == {"ran": False, "reason": "disabled"}


def test_forced_tick_stores_one_low_confidence_memory(brain: Mimir) -> None:
    brain.turn("My favorite color is blue.")  # give it something to muse on
    before = len(list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY))

    result = brain.run_inner_life_tick(force=True)
    assert result["ran"] is True
    assert result["thought"]

    mems = list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)
    assert len(mems) == before + 1
    musings = [m for m in mems if (m.provenance or "") == "inner life"]
    assert len(musings) == 1
    assert musings[0].confidence <= 0.5  # a musing, not a fact


def test_forced_tick_skips_a_duplicate_musing(brain: Mimir) -> None:
    # The mock returns the same reflection text each time, so a second forced tick is a verbatim
    # repeat — it must be skipped, not piled up (the over-retention distillation guards against).
    def musings() -> list:
        return [m for m in list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)
                if (m.provenance or "") == "inner life"]

    brain.turn("My favorite color is blue.")
    first = brain.run_inner_life_tick(force=True)
    assert first["ran"] is True
    assert len(musings()) == 1

    second = brain.run_inner_life_tick(force=True)
    assert second == {"ran": False, "reason": "duplicate musing"}
    assert len(musings()) == 1  # no new row


def test_inner_life_memory_starts_faint(brain: Mimir) -> None:
    brain.turn("My favorite color is blue.")
    brain.run_inner_life_tick(force=True)
    m = next(m for m in list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)
             if (m.provenance or "") == "inner life")
    assert m.salience <= 0.3 and m.confidence <= 0.3  # faint + low-belief, so it decays out fast


def test_forced_tick_still_yields_to_a_live_turn(brain: Mimir) -> None:
    brain._turn_active = True
    try:
        assert brain.run_inner_life_tick(force=True) == {"ran": False, "reason": "turn in flight"}
    finally:
        brain._turn_active = False
