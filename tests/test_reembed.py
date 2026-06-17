"""Re-embedding the store with the current embed model. Switching embed models silently corrupts
recall (same-dimension vectors from different models are not comparable), so ``Mimir.reembed()`` is
the rebuild that makes the change safe. This proves it touches every vector-bearing store and is
non-destructive on a degraded embedder."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.embed.base import EmbeddingMode
from mimir.embed.endpoint import NullEmbedder
from mimir.storage.models import LibraryClaim, LibraryDocument, Memory, Procedure
from mimir.storage.repo import (
    browse_memories,
    claims_for_document,
    list_procedures,
    replace_document_claims,
    save_memory,
    save_procedure,
    upsert_library_document,
)


def test_reembed_covers_memories_claims_and_procedures(brain: Mimir) -> None:
    s = brain._storage
    # Seed each vector-bearing store with NO embedding (as if embedded under a dead model).
    save_memory(s, Memory(text="The orchard floods in spring", embedding=None))
    doc_id = upsert_library_document(s, LibraryDocument(
        path="/b.md", filename="b.md", content_hash="h"))
    replace_document_claims(s, doc_id, [
        LibraryClaim(document_id=doc_id, text="Bees swarm when crowded", embedding=None)])
    save_procedure(s, Procedure(trigger="user asks for a plan", procedure="enumerate steps",
                                trigger_embedding=None))

    counts = brain.reembed()

    assert counts["memories"] == 1
    assert counts["claims"] == 1
    assert counts["procedures"] == 1
    assert counts["failed"] == 0
    # Every store now carries a real vector (bootstrap embedder produces them).
    assert browse_memories(s)[0].embedding is not None
    assert claims_for_document(s, doc_id)[0].embedding is not None
    assert list_procedures(s)[0].trigger_embedding is not None


def test_reembed_aborts_when_embedder_degraded(brain: Mimir) -> None:
    s = brain._storage
    save_memory(s, Memory(text="kept as-is", embedding=[0.1, 0.2, 0.3]))
    brain._embedder = NullEmbedder()
    assert brain._embedder.mode is EmbeddingMode.DEGRADED

    counts = brain.reembed()

    assert counts.get("aborted") == 1
    assert counts["memories"] == 0
    # The existing vector is untouched — a degraded run is never destructive.
    assert browse_memories(s)[0].embedding is not None
