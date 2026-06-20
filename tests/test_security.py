"""Security: the path-based document ingest/forget surface must stay confined to the configured
documents/library folder, so an exposed API can't read/delete arbitrary files (path traversal)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimir.brain import Mimir
from mimir.errors import IngestError


def test_ingest_path_confined_to_the_doc_folder(brain: Mimir, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.txt").write_text("hello", encoding="utf-8")
    brain.config.documents_folder = str(docs)
    brain.config.library_folder = None
    # inside the folder → resolves to the real path
    assert brain.resolve_ingest_path(str(docs / "ok.txt")).endswith("ok.txt")
    # a sibling outside the folder → refused
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(IngestError):
        brain.resolve_ingest_path(str(outside))
    # traversal out of the folder → refused
    with pytest.raises(IngestError):
        brain.resolve_ingest_path(str(docs / ".." / "secret.txt"))


def test_ingest_path_disabled_without_a_configured_folder(brain: Mimir, tmp_path: Path) -> None:
    brain.config.documents_folder = None
    brain.config.library_folder = None
    with pytest.raises(IngestError):
        brain.resolve_ingest_path(str(tmp_path / "anything.txt"))


def test_forget_refuses_to_delete_outside_the_folder(brain: Mimir, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    brain.config.documents_folder = str(docs)
    brain.config.library_folder = None
    outside = tmp_path / "keep.txt"
    outside.write_text("important", encoding="utf-8")
    # forget an arbitrary path with delete_file=True must NOT unlink a file outside the doc folder
    brain.forget_document(str(outside), delete_file=True)
    assert outside.exists()  # spared
