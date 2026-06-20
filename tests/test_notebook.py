"""Notebook — lossless, name-addressable working memory (docs/EXTENSIBILITY.md). Distinct from the
lossy memory store: never decayed/deduped; deliberate, self-curated, section-addressable; and a
re-read re-triggers recall so the note reconnects to live memory."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition import notebook as nb
from mimir.errors import NotebookError
from mimir.storage.models import EvidenceTier, Memory
from mimir.storage.repo import save_memory


def test_write_read_round_trip_is_lossless(brain: Mimir) -> None:
    body = "## Plan\n- step one\n- step two\n\n## Notes\nverbatim *markdown* is kept."
    nb.write(brain._storage, "build", body)
    assert nb.read(brain._storage, "build") == body  # byte-identical, no dedup/decay touches it


def test_section_addressing_edits_in_place(brain: Mimir) -> None:
    nb.write(brain._storage, "doc", "## A\nalpha\n\n## B\nbeta\n\n## C\ngamma")
    nb.edit(brain._storage, "doc", "B", "BETA REWRITTEN")
    assert nb.read(brain._storage, "doc", section="B") == "## B\nBETA REWRITTEN"
    body = nb.read(brain._storage, "doc")  # other sections + ordering preserved
    assert "alpha" in body and "gamma" in body and body.index("## A") < body.index("## C")


def test_name_addressing_and_owner_isolation(brain: Mimir) -> None:
    nb.write(brain._storage, "shared", "self body", owner="__self__")
    nb.write(brain._storage, "shared", "alex body", owner="alex")  # same title, diff owner
    assert nb.read(brain._storage, "shared", owner="__self__") == "self body"
    assert nb.read(brain._storage, "shared", owner="alex") == "alex body"


def test_read_with_memory_re_triggers_recall(brain: Mimir) -> None:
    fact = "The orchard has twelve apple trees."
    save_memory(brain._storage, Memory(text=fact, evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
                                       embedding=brain._embedder.embed(fact)))
    nb.write(brain._storage, "orchard", "Counting the apple trees in the orchard this spring.")
    passage, mems = nb.read_with_memory(brain._storage, brain._embedder, "orchard")
    assert "apple trees" in passage
    assert any("apple trees" in m.text for m in mems)  # the note reconnected to live memory


def test_soft_cap_surfaces_and_never_drops(brain: Mimir) -> None:
    for i in range(3):
        nb.write(brain._storage, f"n{i}", "x", soft_cap=3)
    with pytest.raises(NotebookError):
        nb.write(brain._storage, "n3", "x", soft_cap=3)  # creating past the cap → loud, not silent
    assert nb.read(brain._storage, "n0") == "x"  # existing intact
    nb.write(brain._storage, "n0", "updated", soft_cap=3)  # replacing an existing one is fine
    assert nb.read(brain._storage, "n0") == "updated"


def test_append_index_and_delete(brain: Mimir) -> None:
    nb.write(brain._storage, "log", "## Day 1\nstarted")
    nb.append(brain._storage, "log", "## Day 2\ncontinued")
    body = nb.read(brain._storage, "log")
    assert "Day 1" in body and "Day 2" in body
    idx = nb.index(brain._storage)
    assert "log" in idx and "started" not in idx  # the index is the catalog, never the bodies
    assert nb.delete(brain._storage, "log") is True
    assert nb.read(brain._storage, "log") == ""


def test_rename(brain: Mimir) -> None:
    nb.write(brain._storage, "old", "body")
    assert nb.rename(brain._storage, "old", "new") is True
    assert nb.read(brain._storage, "new") == "body" and nb.read(brain._storage, "old") == ""


# -- the connector wiring: the index section (sensory port) + the tool (motor port) --------------

def test_notebook_index_section_appears_in_a_turn(brain: Mimir) -> None:
    nb.write(brain._storage, "plans", "## Q3\nship it")
    r = brain.turn("hello there", user="alex")  # no notebook keyword needed for the ambient index
    assert "plans" in r.context.prompt
    assert any(s["name"] == "notebooks" for s in r.context.introspect()["sections"])
    assert "ship it" not in r.context.prompt  # the index is the catalog, not the bodies


def test_notebooks_facade_exposes_bodies_for_the_ui(brain: Mimir) -> None:
    # Mimir.notebooks() is the read-only window the /api/notebooks route serves — title, sections,
    # size, and the full body, newest first (so the UI shows what the model is working on).
    assert brain.notebooks() == []
    nb.write(brain._storage, "Greenhouse", "## Settings\nThermostat set to 12 overnight.")
    out = brain.notebooks()
    assert len(out) == 1
    entry = out[0]
    assert entry["title"] == "Greenhouse"
    assert entry["sections"] == ["Settings"]
    assert "Thermostat set to 12 overnight." in entry["body"]
    assert entry["size"] > 0 and entry["owner"] == "__self__"


def test_notebook_tool_writes_through_a_turn(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chat(role: str, messages: list, **k: object) -> str:
        sys = messages[0]["content"]
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if role == "chat" and "Tools you may call" in sys and not has_assistant:
            return ('<TOOL name="notebook" '
                    'args={"op": "write", "title": "ideas", "body": "first idea"}>')
        return "Noted."

    monkeypatch.setattr(brain._model, "chat", fake_chat)
    r = brain.turn("jot this in a notebook for me", user="alex")  # 'jot'/'notebook' select the tool
    assert any(a.tool == "notebook" and a.status == "ok" for a in r.actions)
    assert nb.read(brain._storage, "ideas") == "first idea"  # the tool actually wrote it
