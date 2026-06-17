"""Library Phase 1b — the cited claims spine: extraction, retrieval, idle indexing, and the
cited Library section in a turn. (Data foundation is covered by test_library_storage.py.)"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mimir.brain import Mimir
from mimir.cognition.library import (
    ScoredClaim,
    extract_claims,
    render_claims,
    retrieve_claims,
)
from mimir.config import Config
from mimir.storage.models import LibraryClaim
from mimir.storage.repo import claims_for_document, list_library_claims, list_library_documents


def test_extract_claims_parses_and_degrades() -> None:
    out = extract_claims(lambda m: '{"claims": ["Bees make honey", "A hive has one queen"]}', "...")
    assert out == ["Bees make honey", "A hive has one queen"]
    assert extract_claims(lambda m: "not json", "...") == []   # lenient: nothing parseable → []


def test_retrieve_and_render_claims_cite_sources() -> None:
    claims = [
        LibraryClaim(document_id=1, text="Garlic is planted in October", locator="p.2"),
        LibraryClaim(document_id=2, text="Torque is rotational force", locator="p.9"),
    ]
    hits = retrieve_claims("when do I plant garlic", None, claims, top_k=2)
    assert hits and hits[0].claim.text.startswith("Garlic")     # on-topic claim ranks first
    assert all(isinstance(h, ScoredClaim) for h in hits)
    rendered = render_claims(hits, {1: "Gardening", 2: "Cars"})
    assert "[Gardening, p.2]" in rendered                       # every fact carries its citation


def _libbrain(mock_config: Config, tmp_path) -> Mimir:
    # Source of truth = the documents folder; composites written to a separate library folder.
    cfg = dataclasses.replace(
        mock_config,
        documents_folder=str(tmp_path / "documents"),
        library_folder=str(tmp_path / "library"),
    )
    return Mimir(cfg)


def test_idle_extraction_records_document_and_cited_claims(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "garden.md").write_text(
            "# Garlic\n\nGarlic is planted in October. Harvest garlic in July.")
        report = brain.ingest_pending_library()
        assert "garden.md" in report["documents"] and report["claims"] >= 1

        doc = list_library_documents(brain._storage)[0]
        assert doc.filename == "garden.md" and doc.size_bytes > 0   # exact filename + size tracked
        claims = claims_for_document(brain._storage, doc.id)
        assert claims and all(c.locator for c in claims)            # each claim cites a locator
        assert any("Garlic" in c.text for c in claims)

        # Unchanged re-scan is a no-op; removing the file drops the doc + cascades its claims.
        assert brain.ingest_pending_library()["claims"] == 0
        (folder / "garden.md").unlink()
        assert brain.ingest_pending_library()["dropped"] == 1
        assert list_library_documents(brain._storage) == []
        assert list_library_claims(brain._storage) == []
    finally:
        brain.close()


def test_library_claims_surface_cited_in_a_turn(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("Beekeepers inspect hives. Each hive has a single queen.")
        brain.ingest_pending_library()
        result = brain.turn("tell me about hives")
        prompt = result.context.prompt
        assert "hive" in prompt.lower() and "[bees" in prompt    # cited library claim in the prompt
    finally:
        brain.close()


def test_idle_compiles_a_linked_composite_with_citations(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "garden.md").write_text(
            "# Garlic\n\nGarlic is planted in October. Harvest garlic in July.")
        report = brain.ingest_pending_library()
        assert report["composed"] >= 1

        overview = brain.library_overview()
        page = overview["pages"][0]
        assert Path(page["path"]).is_file()              # the composite MD is on disk
        full = brain.library_page(page["id"])
        assert full["markdown"]                          # full composite loaded on demand
        assert full["citations"] and all(c["title"] for c in full["citations"])  # traces to source

        # A verbatim source is fetchable for quoting/checking.
        doc = overview["documents"][0]
        assert "Garlic" in brain.library_source(doc["id"])["text"]
    finally:
        brain.close()


def test_hand_edited_composite_is_not_clobbered(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "note.md").write_text("# Note\n\nA fact about the farm.")
        brain.ingest_pending_library()
        page_path = Path(brain.library_overview()["pages"][0]["path"])
        page_path.write_text("# Note\n\nMY HAND-EDITED VERSION.")   # user edits the composite
        brain.ingest_pending_library(force=True)                    # re-derive attempt
        assert "HAND-EDITED" in page_path.read_text()               # respected, not clobbered
    finally:
        brain.close()


def test_no_source_folder_is_a_quiet_noop(brain: Mimir) -> None:
    assert brain.ingest_pending_library() == {
        "folder": None, "documents": [], "claims": 0, "composed": 0, "dropped": 0}
    assert brain._library_gist("anything", None) is None


def test_loaded_page_is_pinned_into_the_next_turn(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fences.md").write_text("# Fences\n\nThe north fence is cedar.")
        brain.ingest_pending_library()
        page_id = brain.library_overview()["pages"][0]["id"]
        # A query that wouldn't surface the gist on its own; the pinned page is loaded regardless.
        result = brain.turn("what's the weather", loaded_pages=[page_id])
        prompt = result.context.prompt
        assert "Full pages you've loaded" in prompt and "Fences" in prompt
    finally:
        brain.close()
