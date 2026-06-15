"""Self-knowledge: the system bakes its own README into memory in the nightly cycle, so it can
answer about what it is and how it works. Content-hashed → re-embeds only when the doc changes."""

from __future__ import annotations

from pathlib import Path

from mimir.brain import Mimir
from mimir.storage.models import MemoryKind
from mimir.storage.repo import list_memories


def test_bake_is_idempotent_and_refreshes_on_change(brain: Mimir, tmp_path: Path) -> None:
    doc = tmp_path / "README.md"
    doc.write_text("# Mimir\n\nA local cognition core. It bakes and recalls memory.\n", "utf-8")
    brain.config.self_knowledge_doc = str(doc)

    first = brain.bake_self_knowledge()
    assert first["baked"] and first["chunks"] >= 1

    again = brain.bake_self_knowledge()
    assert not again["baked"] and again["reason"] == "unchanged"   # hash guard skips re-embed

    doc.write_text("# Mimir\n\nA local cognition core. Now with a new sentence.\n", "utf-8")
    changed = brain.bake_self_knowledge()
    assert changed["baked"]                                        # content changed → re-baked

    # the doc is now recallable, tagged with its source
    sourced = [m for m in list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)
               if (m.provenance or "").startswith("README.md")]
    assert sourced and "cognition core" in " ".join(m.text for m in sourced).lower()


def test_disabled_and_missing_are_soft(brain: Mimir) -> None:
    brain.config.self_knowledge_doc = None
    assert brain.bake_self_knowledge()["reason"] == "disabled"
    brain.config.self_knowledge_doc = "definitely/not/a/real/path.md"
    assert brain.bake_self_knowledge()["reason"] == "not found"


def test_self_knowledge_is_a_sleep_phase(brain: Mimir) -> None:
    names = [p.name for p in brain._sleep_phases()]
    assert "self_knowledge" in names
