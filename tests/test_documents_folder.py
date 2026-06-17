"""Drop-folder document pipeline: upload → ingest, idle scan → ingest + wiki summary (DESIGN §8)."""

from __future__ import annotations

import dataclasses

import pytest

from mimir.brain import Mimir
from mimir.config import Config
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import list_memories


@pytest.fixture
def docbrain(mock_config: Config, tmp_path) -> Mimir:
    cfg = dataclasses.replace(mock_config, documents_folder=str(tmp_path / "documents"))
    m = Mimir(cfg)
    try:
        yield m
    finally:
        m.close()


def _doc_chunks(brain: Mimir) -> list:
    return [m for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)
            if m.evidence_tier is EvidenceTier.DOCUMENT]


def test_upload_saves_to_folder_and_ingests(docbrain: Mimir) -> None:
    out = docbrain.upload_document("notes.md", b"# Title\n\nThe north field grows garlic.")
    assert out["chunks"] >= 1
    folder = docbrain._docs_folder()
    assert (folder / "notes.md").is_file()              # saved to the drop folder
    assert _doc_chunks(docbrain)                         # ingested as document-tier knowledge
    assert any(d["name"] == "notes.md" for d in docbrain.documents())  # listed in the wiki


def test_upload_rejects_unsupported_type(docbrain: Mimir) -> None:
    from mimir.errors import IngestError
    with pytest.raises(IngestError):
        docbrain.upload_document("evil.exe", b"...")


def test_idle_scan_ingests_dropped_files_and_summarizes(docbrain: Mimir) -> None:
    folder = docbrain._docs_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dropped.txt").write_text("Beekeeping notes: inspect the hives every two weeks.")
    report = docbrain.ingest_pending_documents()
    assert "dropped.txt" in report["ingested"]
    assert report["summarized"] >= 1
    doc = next(d for d in docbrain.documents() if d["name"] == "dropped.txt")
    assert doc.get("summary")                            # the idle pass generated a wiki summary

    # Re-scan is a no-op (content unchanged, already summarized).
    again = docbrain.ingest_pending_documents()
    assert again["ingested"] == [] and again["summarized"] == 0


def test_no_folder_configured_is_a_quiet_noop(brain: Mimir) -> None:
    assert brain.ingest_pending_documents() == {
        "folder": None, "ingested": [], "summarized": 0, "failed": []}
