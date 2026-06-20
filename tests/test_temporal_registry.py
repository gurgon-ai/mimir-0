"""The Temporal Registry — STATE vs NARRATIVE (docs/EXTENSIBILITY.md).

A small, authoritative, dated, status-tagged ledger of milestones (what is true *now*) that sits
beside the mixed-tense narrative memory store: it answers status questions from current state and,
in the sleep pass, reconciles stale-state memories it supersedes. Each load-bearing claim below is
one executable spec (the six acceptance tests from the extraction spec)."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition import temporal_registry as tr
from mimir.cognition.sleep import consolidate
from mimir.storage.models import EvidenceTier, Memory
from mimir.storage.repo import list_memories, save_memory


def _save(brain: Mimir, text: str, *, tier: EvidenceTier = EvidenceTier.CONVERSATION,
          salience: float = 1.0) -> int:
    return save_memory(brain._storage, Memory(
        text=text, evidence_tier=tier, salience=salience,
        embedding=brain._embedder.embed(text)))


def _find(brain: Mimir, needle: str) -> Memory:
    hits = [m for m in list_memories(brain._storage) if needle in m.text]
    assert len(hits) == 1, f"expected exactly one memory containing {needle!r}, got {len(hits)}"
    return hits[0]


def test_state_beats_narrative(brain: Mimir) -> None:
    # An older planning memory and a later "it's done" milestone coexist; the milestone is the
    # authority. It surfaces high-attention in the turn, and reconcile flags the planning note.
    _save(brain, "We are planning to do the Zephyr migration next week.")
    tr.record_milestone(brain._storage, "Zephyr migration",
                        "The Zephyr migration is finished.", "done")

    r = brain.turn("what's the status of the Zephyr migration?", user="alex")
    names = [s["name"] for s in r.context.introspect()["sections"]]
    assert "timeline" in names  # the authoritative STATE block made it into the prompt
    assert "Zephyr migration is finished" in r.context.prompt

    report = tr.reconcile(brain._storage)
    assert report.demoted == 1
    stale = _find(brain, "planning to do the Zephyr migration")
    assert stale.meta.get("superseded_by_milestone")  # flagged as superseded by the milestone
    assert stale.salience < 0.1  # and demoted toward deprioritization


def test_reconcile_is_safe_on_generic_overlap(brain: Mimir) -> None:
    # A milestone with a distinctive token (Foo9000) must NOT touch an unrelated memory that only
    # shares a generic word ("system"). Distinctive-token guard → zero false demotions.
    before = _save(brain, "We will upgrade the system soon.", salience=0.8)
    tr.record_milestone(brain._storage, "Foo9000 rollout", "Foo9000 rollout is done.", "done")

    report = tr.reconcile(brain._storage)
    assert report.demoted == 0 and report.examined == 0
    after = _find(brain, "upgrade the system")
    assert after.id == before and after.salience == pytest.approx(0.8)
    assert not after.meta.get("superseded_by_milestone")


def test_milestone_is_decay_exempt(brain: Mimir) -> None:
    # Milestones live in their own table — the salience/decay/archival pass never touches them.
    mid = tr.record_milestone(brain._storage, "Barn raised", "The new barn is built.", "done")
    consolidate(brain._storage)  # full decay + archive pass
    survivor = tr.get_milestone(brain._storage, mid)
    assert survivor is not None and survivor.status == "done"
    # And it is genuinely separate state — not a row in the narrative memory store.
    assert not any("barn is built" in m.text for m in list_memories(brain._storage))


def test_current_config_protection(brain: Mimir) -> None:
    # A faded memory that AGREES with a current milestone (shared distinctive token, present-tense)
    # survives an archival pass it would otherwise fail — the registry lifts it above the floor.
    _save(brain, "The Helios server is the primary database host.",
          tier=EvidenceTier.CONVERSATION, salience=0.01)
    tr.record_milestone(brain._storage, "Helios server",
                        "Helios server is the primary database host.", "in_progress",
                        is_current_config=True)

    consolidate(brain._storage, reconcile_milestones=True)
    survivor = _find(brain, "Helios server is the primary")  # lifted, not archived
    assert survivor.meta.get("confirmed_by_milestone")
    assert survivor.salience >= 0.1


def test_current_config_archived_without_protection(brain: Mimir) -> None:
    # Control for the test above: with reconcile off, the same faded memory IS archived.
    _save(brain, "The Helios server is the primary database host.",
          tier=EvidenceTier.CONVERSATION, salience=0.01)
    consolidate(brain._storage, reconcile_milestones=False)
    assert not any("Helios server" in m.text for m in list_memories(brain._storage))  # archived out


def test_self_model_pins_current_config(brain: Mimir) -> None:
    # current_config() statements are pinned into the authored self-model text ("how am I set up").
    tr.record_milestone(brain._storage, "Edge fleet",
                        "Inference runs on three edge nodes off the RTX host.", "in_progress",
                        is_current_config=True)
    self_text = brain._compose_self_knowledge()
    assert self_text and "three edge nodes off the RTX host" in self_text


def test_gateway_compliance(brain: Mimir, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every milestone write routes through the storage gateway's single writer (no direct path).
    calls = {"n": 0}
    real_submit = brain._storage.submit

    def spy(fn: object, **k: object) -> object:
        calls["n"] += 1
        return real_submit(fn, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(brain._storage, "submit", spy)
    tr.record_milestone(brain._storage, "Gateway", "Gateway check is done.", "done")
    assert calls["n"] >= 1  # the write went through the gateway, not a private connection
    assert tr.get_milestone_by_title(brain._storage, "Gateway") is not None
