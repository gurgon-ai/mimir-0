"""Self-directed deliberation (DESIGN §5a): conflict surfacing, the hybrid curator, and the
brain's sleep-phase / manual path that submits the system's own conflicts to the council."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.deliberation import Conflict, curate, surface_conflicts
from mimir.storage.models import Triple
from mimir.storage.repo import save_triple


def _seed_tension(brain: Mimir) -> None:
    # A non-functional relation with two values for the same subject = a genuine tension.
    # ("likes" is not in FUNCTIONAL_RELATIONS, so consolidation leaves it for the council.)
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


def test_curate_caps_to_limit(brain: Mimir) -> None:
    many = [Conflict(key=f"k{i}", question=f"Q{i}?", weight=float(i)) for i in range(8)]
    chosen = curate(brain._model, many, limit=3)
    assert len(chosen) == 3


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
