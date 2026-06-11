"""Document ingestion (v0.1): extraction + chunking into the document-tier knowledge layer."""

from __future__ import annotations

from .chunk import Chunk, chunk_units
from .extract import ExtractedUnit, extract

__all__ = ["Chunk", "chunk_units", "ExtractedUnit", "extract"]
