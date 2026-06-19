"""Deep-idle dialogue — inner-life Slice 3 (DESIGN §5a): when the quiet runs long, two asymmetric
voices argue one matter, then an insight is distilled and stored (or reinforced on reconverge)."""

from __future__ import annotations

import time

import pytest

from mimir.brain import Mimir
from mimir.cognition.deep_idle import (
    REFLECT,
    SKEPTIC,
    parse_insight,
    run_dialogue,
)
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import get_memory, save_memory


def _seed_belief(brain: Mimir, text: str) -> None:
    save_memory(brain._storage, Memory(
        text=text, kind=MemoryKind.MEMORY, evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
        embedding=brain._embedder.embed(text)))


# -- the pure dialogue ------------------------------------------------------------------------

def _voice_of(messages: list[dict[str, str]]) -> str:
    return SKEPTIC if "skeptical inner voice" in messages[0]["content"] else REFLECT


def test_dialogue_alternates_voices_and_caps_turns() -> None:
    def chat(messages: list[dict[str, str]]) -> str:
        return "challenge" if _voice_of(messages) == SKEPTIC else "reflect"
    turns = run_dialogue(chat, "the matter", "recent stuff", max_turns=4)
    assert [t.voice for t in turns] == [REFLECT, SKEPTIC, REFLECT, SKEPTIC]


def test_skeptic_never_sees_recent_context_but_reflective_does() -> None:
    # the load-bearing asymmetry: the skeptic can't see recent context, so it can't take a "what I
    # just said" claim on trust — it must demand grounding.
    seen: list[tuple[str, str]] = []

    def chat(messages: list[dict[str, str]]) -> str:
        seen.append((_voice_of(messages), messages[-1]["content"]))
        return "challenge" if _voice_of(messages) == SKEPTIC else "reflect"

    run_dialogue(chat, "the matter", "RECENT_MARKER context", max_turns=4)
    for voice, user in seen:
        if voice == SKEPTIC:
            assert "RECENT_MARKER" not in user
        else:
            assert "RECENT_MARKER" in user


def test_dialogue_stops_early_on_an_empty_voice() -> None:
    def chat(messages: list[dict[str, str]]) -> str:
        return "" if _voice_of(messages) == SKEPTIC else "reflect"
    turns = run_dialogue(chat, "m", "", max_turns=4)
    assert [t.voice for t in turns] == [REFLECT]  # opening only; the skeptic returned nothing


def test_parse_insight_reads_structured_fields() -> None:
    insight = parse_insight("INSIGHT: the well may be low\nTYPE: gap\nCONFIDENCE: 0.7")
    assert insight is not None
    assert insight.text == "the well may be low"
    assert insight.kind == "gap"
    assert insight.confidence == 0.7


def test_parse_insight_defaults_unknown_type_and_bad_confidence() -> None:
    insight = parse_insight("INSIGHT: foo\nTYPE: banana\nCONFIDENCE: very high")
    assert insight is not None and insight.kind == "self_knowledge" and insight.confidence == 0.4


def test_parse_insight_fallback_and_empty() -> None:
    assert parse_insight("a sentence with no labels").text == "a sentence with no labels"
    assert parse_insight("") is None
    assert parse_insight("INSIGHT:\nTYPE: gap") is None  # empty insight body → nothing usable


# -- the brain integration --------------------------------------------------------------------

def test_force_deep_idle_stores_a_flagged_insight(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain, "_pool_degraded", lambda: False)
    _seed_belief(brain, "The orchard has twelve apple trees on the south slope.")
    res = brain.run_deep_idle_tick(force=True)
    assert res["ran"] and res["deep"] and not res["converged"]
    assert res["transcript"] and res["transcript"][0]["voice"] == REFLECT  # the dialogue happened
    mem = get_memory(brain._storage, res["memory_id"])
    assert mem is not None
    assert mem.provenance == "deep idle"
    assert mem.evidence_tier is EvidenceTier.INFERRED and mem.confidence <= 0.4
    # surfaces in the Mind tab, flagged as a deep insight (not a one-shot musing)
    deep = [t for t in brain.recent_thoughts() if t.get("deep")]
    assert any(t["text"] == res["insight"] for t in deep)


def test_reconverged_insight_is_reinforced_not_duplicated(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain, "_pool_degraded", lambda: False)
    _seed_belief(brain, "The cellar floods every spring thaw.")
    first = brain.run_deep_idle_tick(force=True)
    assert not first["converged"]
    before = get_memory(brain._storage, first["memory_id"]).confidence
    # the mock distils the same insight again → it re-derives, so it reinforces, not duplicates
    second = brain.run_deep_idle_tick(force=True)
    assert second["converged"] and second["memory_id"] == first["memory_id"]
    after = get_memory(brain._storage, first["memory_id"]).confidence
    assert after > before  # convergence-as-validation: belief earned, not duplicated


def test_should_deep_idle_gates_on_enable_idle_and_cooldown(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain, "_pool_degraded", lambda: False)
    now = time.time()
    assert not brain._should_deep_idle(now)  # off by default
    brain.update_settings({"deep_idle_enabled": True})
    brain._last_turn_at = now  # a fresh turn → not been quiet long enough
    assert not brain._should_deep_idle(now)
    brain._last_turn_at = now - brain.config.deep_idle_after_s - 1  # quiet has run long
    brain._last_deep_idle_at = 0.0
    assert brain._should_deep_idle(now)
    brain._last_deep_idle_at = now  # just held one → cooldown blocks the next
    assert not brain._should_deep_idle(now)
