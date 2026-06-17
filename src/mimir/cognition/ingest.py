"""Document ingestion: extract → chunk → embed → store as document-tier memories (DESIGN §8).

A document chunk is *just a memory whose evidence tier is ``document``* — it is written through
the same storage gateway, embedded by the same embedder, and later recalled through the same
``build_context()`` path as any other knowledge. What distinguishes it is the ``DOCUMENT``
evidence tier (a gentle retrieval boost + an honest provenance tag) and a ``source`` pointing at
the originating file.

Re-ingest is idempotent: a document's existing chunks are deleted by ``source`` before the new
ones are written, so re-ingesting an edited file replaces rather than duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..documents.chunk import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, chunk_units
from ..documents.extract import extract
from ..embed.base import Embedder
from ..errors import IngestError
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import delete_by_source, save_memory

log = logging.getLogger("mimir.ingest")

# Document chunks are well-sourced but not user-asserted truth — a confident-but-not-authority tier.
_DOCUMENT_CONFIDENCE = 0.8


# Document types the drop-folder scan will pick up. Extensionless files are deliberately excluded
# (a drop folder shouldn't sweep in stray non-documents); `.pdf` needs the [documents] extra.
SUPPORTED_SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown", ".mdown", ".pdf"})


def list_documents(folder: str | Path) -> list[Path]:
    """Supported document files directly in ``folder`` (non-recursive), sorted; ``[]`` if absent."""
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(
        f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )


@dataclass(slots=True)
class IngestResult:
    """What an ingest produced."""

    source: str
    units: int
    chunks_written: int
    chunks_replaced: int  # prior chunks removed for this source (re-ingest)


def ingest_document(
    storage: StorageGateway,
    embedder: Embedder,
    *,
    path: str | Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> IngestResult:
    """Ingest one document into the store. Raises ``IngestError`` if it can't be read/chunked."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"no such file to ingest: {p}")
    source = str(p.resolve())

    units = extract(p)
    chunks = chunk_units(units, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        raise IngestError(f"no extractable text found in {p.name}")

    replaced = delete_by_source(storage, source)

    for idx, chunk in enumerate(chunks):
        locator = chunk.locator or f"#{idx + 1}"
        provenance = f"{p.name}:{locator}"
        mem = Memory(
            text=chunk.text,
            kind=MemoryKind.MEMORY,  # a document chunk is just a memory (DESIGN §8)
            evidence_tier=EvidenceTier.DOCUMENT,
            confidence=_DOCUMENT_CONFIDENCE,
            salience=1.0,
            embedding=embedder.embed(chunk.text),
            provenance=provenance,
            user=None,  # documents are shared knowledge, not scoped to one speaker
            source=source,
        )
        save_memory(storage, mem)

    log.info(
        "ingest: %s → %d chunk(s) from %d unit(s)%s",
        p.name,
        len(chunks),
        len(units),
        f" (replaced {replaced})" if replaced else "",
    )
    return IngestResult(
        source=source,
        units=len(units),
        chunks_written=len(chunks),
        chunks_replaced=replaced,
    )
