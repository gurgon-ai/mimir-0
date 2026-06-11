"""Executable spec for document ingestion end-to-end: ingest → recall with provenance (v0.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimir.brain import Mimir
from mimir.errors import IngestError
from mimir.storage.models import EvidenceTier, MemoryKind
from mimir.storage.repo import count_memories, list_memories


def test_ingest_then_recall_with_document_provenance(brain: Mimir, tmp_path: Path) -> None:
    doc = tmp_path / "facts.md"
    doc.write_text(
        "# Landmarks\nThe Eiffel Tower in Paris is 330 meters tall.\n",
        encoding="utf-8",
    )
    result = brain.ingest(doc)
    assert result.chunks_written >= 1
    assert result.chunks_replaced == 0

    # The chunk is stored as a document-tier memory (a memory, per DESIGN §8).
    docs = [
        m
        for m in list_memories(brain._storage, kind=MemoryKind.MEMORY)
        if m.evidence_tier is EvidenceTier.DOCUMENT
    ]
    assert docs and docs[0].source == str(doc.resolve())

    # A later turn recalls it, attributed to the file + section.
    r = brain.turn("How tall is the Eiffel Tower?")
    assert "330" in r.reply
    assert r.context.source_count >= 1
    knowledge = next(s for s in r.context.sections if s.name == "knowledge")
    assert "tier=document" in knowledge.body
    assert "facts.md:Landmarks" in knowledge.body


def test_reingest_replaces_not_duplicates(brain: Mimir, tmp_path: Path) -> None:
    doc = tmp_path / "v.txt"
    doc.write_text("original content about widgets", encoding="utf-8")
    brain.ingest(doc)
    after_first = count_memories(brain._storage, kind=MemoryKind.MEMORY)

    doc.write_text("revised content about widgets and gadgets", encoding="utf-8")
    result = brain.ingest(doc)
    assert result.chunks_replaced >= 1

    after_second = count_memories(brain._storage, kind=MemoryKind.MEMORY)
    # Re-ingest replaced the prior chunk(s) rather than piling new ones on top.
    assert after_second == after_first


def test_ingest_missing_file_fails_loud(brain: Mimir, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no such file"):
        brain.ingest(tmp_path / "nope.txt")


def test_ingest_empty_document_fails_loud(brain: Mimir, tmp_path: Path) -> None:
    doc = tmp_path / "blank.md"
    doc.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(IngestError, match="no extractable text"):
        brain.ingest(doc)
