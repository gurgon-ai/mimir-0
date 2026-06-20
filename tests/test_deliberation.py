"""Self-directed deliberation (DESIGN §5a): conflict surfacing, the hybrid curator, and the
brain's sleep-phase / manual path that submits the system's own conflicts to the council."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition.deliberation import Conflict, curate, surface_conflicts
from mimir.storage.models import Triple
from mimir.storage.repo import save_triple


def _seed_tension(brain: Mimir) -> None:
    # "wants" is neither functional (consolidation's job) nor additive (a pure list) — it's an
    # ambiguous preference relation whose values *can* genuinely compete, so it's surfaced for the
    # council/curator to judge (quiet farm vs busy startup is a real lifestyle tension).
    save_triple(brain._storage, Triple(subject="Greg", relation="wants", object="a quiet farm"))
    save_triple(brain._storage, Triple(subject="Greg", relation="wants", object="a busy startup"))


def test_surface_graph_tension(brain: Mimir) -> None:
    _seed_tension(brain)
    conflicts = surface_conflicts(brain._storage, embedder=brain._embedder)
    assert any(c.key.startswith("graph:greg|wants") for c in conflicts)
    q = next(c.question for c in conflicts if c.key.startswith("graph:greg|wants"))
    assert "quiet farm" in q and "busy startup" in q


def test_functional_relations_are_not_surfaced(brain: Mimir) -> None:
    # "lives in" IS functional → consolidation's job, not the council's.
    save_triple(brain._storage, Triple(subject="Greg", relation="lives in", object="Mission"))
    save_triple(brain._storage, Triple(subject="Greg", relation="lives in", object="Vancouver"))
    conflicts = surface_conflicts(brain._storage, embedder=brain._embedder)
    assert not any("lives in" in c.key for c in conflicts)


def test_additive_relations_are_not_surfaced(brain: Mimir) -> None:
    # Regression for the worst real failure: "has 16-core; has 64GB; has RTX 5090" is a spec LIST,
    # not a disagreement — the council was reasoning itself out of true hardware facts.
    for obj in ("a 16-core CPU", "64 GB of RAM", "an RTX 5090"):
        save_triple(brain._storage, Triple(subject="Parent system", relation="has", object=obj))
    conflicts = surface_conflicts(brain._storage, embedder=brain._embedder)
    assert not any(c.key.startswith("graph:parent system|has") for c in conflicts)


def test_curate_caps_to_limit(brain: Mimir) -> None:
    many = [Conflict(key=f"k{i}", question=f"Q{i}?", weight=float(i)) for i in range(8)]
    chosen = curate(brain._model, many, limit=3)
    assert len(chosen) == 3


def test_curate_drops_all_when_curator_says_none(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The curator is now a FILTER: if it judges none a real conflict, no council runs.
    monkeypatch.setattr(brain._model, "chat", lambda *a, **k: "none — these all simply coexist")
    many = [Conflict(key=f"k{i}", question=f"Q{i}?", weight=float(i)) for i in range(4)]
    assert curate(brain._model, many, limit=3) == []


def test_curate_honors_explicit_selection(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain._model, "chat", lambda *a, **k: "0, 2")
    many = [Conflict(key=f"k{i}", question=f"Q{i}?", weight=float(8 - i)) for i in range(4)]
    assert [c.key for c in curate(brain._model, many, limit=3)] == ["k0", "k2"]


def test_deliberate_open_questions_runs_council(brain: Mimir) -> None:
    _seed_tension(brain)
    report = brain.deliberate_open_questions(force=True)
    assert report["enabled"] and report["surfaced"] >= 1
    assert len(report["ran"]) >= 1
    assert report["ran"][0]["verdict"]  # the council produced a verdict (mock model)
    # the verdict was stored as a recallable "sleep deliberation" memory
    from mimir.storage.models import MemoryKind
    from mimir.storage.repo import list_memories
    mems = list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)
    assert any(m.provenance == "sleep deliberation" for m in mems)


def test_deliberation_skips_already_seen(brain: Mimir) -> None:
    _seed_tension(brain)
    first = brain.deliberate_open_questions(force=True)
    assert len(first["ran"]) >= 1
    # Same conflicts, nothing new → no fresh items to argue the second time.
    second = brain.deliberate_open_questions(force=True)
    assert second["fresh"] == 0 and second["ran"] == []


def test_deliberation_disabled_is_noop(brain: Mimir) -> None:
    _seed_tension(brain)
    brain.update_settings({"deliberation_enabled": False})
    report = brain.deliberate_open_questions()  # non-force respects the toggle
    assert report["enabled"] is False and report["ran"] == []


def test_memory_conflicts_skip_documents_and_self_output(brain: Mimir) -> None:
    # Regression: the council was "reconciling" overlapping README/DESIGN chunks (DOCUMENT) and its
    # own musings/verdicts (INFERRED). Only what someone *stated* should surface as a tension.
    from types import SimpleNamespace

    from mimir.embed.base import EmbeddingMode
    from mimir.storage.models import EvidenceTier, Memory
    from mimir.storage.repo import save_memory

    semantic = SimpleNamespace(mode=EmbeddingMode.ENDPOINT)  # _memory_conflicts needs semantic mode
    va, vb = [1.0, 0.0, 0.0], [0.92, 0.39, 0.0]  # cosine ~0.92 — inside the tension band

    def pair(tier: EvidenceTier, prov: str, a: str, b: str) -> None:
        save_memory(brain._storage, Memory(text=a, embedding=va, evidence_tier=tier,
                                           user="g", provenance=prov))
        save_memory(brain._storage, Memory(text=b, embedding=vb, evidence_tier=tier,
                                           user="g", provenance=prov))

    pair(EvidenceTier.DOCUMENT, "README.md", "the fleet qualifies models", "the fleet qualifies it")
    pair(EvidenceTier.INFERRED, "inner life", "i wonder about the fleet", "i keep wondering on it")
    assert surface_conflicts(brain._storage, embedder=semantic) == []  # reference + self → nothing

    pair(EvidenceTier.CONVERSATION, "stated by g", "gate is north", "gate sits at the north fence")
    keys = [c.key for c in surface_conflicts(brain._storage, embedder=semantic)]
    assert any(k.startswith("mem:") for k in keys)  # the stated pair IS a real tension
