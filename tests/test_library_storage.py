"""The Library data foundation (docs/LIBRARY.md): documents -> claims -> composite pages, and the
provenance links between them. The DB is the spine; this proves the round-trips and the cascade."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.storage.models import LibraryClaim, LibraryDocument, LibraryPage
from mimir.storage.repo import (
    claims_for_document,
    claims_for_page,
    delete_library_document,
    list_library_claims,
    list_library_documents,
    list_library_pages,
    replace_document_claims,
    set_page_claims,
    upsert_library_document,
    upsert_library_page,
)


def test_document_claims_page_provenance_roundtrip(brain: Mimir) -> None:
    s = brain._storage
    doc_id = upsert_library_document(s, LibraryDocument(
        path="/books/bees.md", filename="bees.md", size_bytes=1234,
        content_hash="abc", title="Beekeeping"))
    assert [d.filename for d in list_library_documents(s)] == ["bees.md"]
    assert list_library_documents(s)[0].size_bytes == 1234   # exact size tracked

    # Claims cite their document + exact locator, and carry an embedding.
    n = replace_document_claims(s, doc_id, [
        LibraryClaim(document_id=doc_id, text="Inspect hives fortnightly", locator="p.3",
                     embedding=[0.1, 0.2, 0.3]),
        LibraryClaim(document_id=doc_id, text="A hive has one queen", locator="p.5"),
    ])
    assert n == 2
    claims = claims_for_document(s, doc_id)
    assert {c.locator for c in claims} == {"p.3", "p.5"}
    assert claims[0].embedding == pytest.approx([0.1, 0.2, 0.3])  # vector round-trips (float32)

    # A composite page links to the claims it was composed from (provenance both ways).
    page_id = upsert_library_page(s, LibraryPage(path="/lib/bees.md", title="Bees", summary="gist"))
    set_page_claims(s, page_id, [c.id for c in claims])
    page_claims = claims_for_page(s, page_id)
    assert {c.text for c in page_claims} == {"Inspect hives fortnightly", "A hive has one queen"}
    # …and each still carries its source citation.
    assert all(c.document_id == doc_id and c.locator for c in page_claims)


def test_replace_claims_is_idempotent(brain: Mimir) -> None:
    s = brain._storage
    doc_id = upsert_library_document(s, LibraryDocument(
        path="/d.md", filename="d.md", content_hash="h"))
    replace_document_claims(s, doc_id, [LibraryClaim(document_id=doc_id, text="old")])
    replace_document_claims(s, doc_id, [LibraryClaim(document_id=doc_id, text="new")])
    assert [c.text for c in list_library_claims(s)] == ["new"]   # replaced, not appended


def test_delete_document_cascades_claims_and_links(brain: Mimir) -> None:
    s = brain._storage
    doc_id = upsert_library_document(s, LibraryDocument(
        path="/x.md", filename="x.md", content_hash="h"))
    replace_document_claims(s, doc_id, [LibraryClaim(document_id=doc_id, text="f", locator="p.1")])
    page_id = upsert_library_page(s, LibraryPage(path="/lib/x.md", title="X"))
    set_page_claims(s, page_id, [c.id for c in claims_for_document(s, doc_id)])

    delete_library_document(s, "/x.md")
    assert list_library_documents(s) == []
    assert list_library_claims(s) == []
    assert claims_for_page(s, page_id) == []         # the page's links were cleared too
    assert [p.title for p in list_library_pages(s)] == ["X"]   # the page row itself remains
