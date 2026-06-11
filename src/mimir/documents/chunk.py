"""Chunking for document ingestion (DESIGN §8).

Packs extracted units into ~token-sized chunks with overlap, preserving each unit's locator so
the provenance (page/section) survives onto every chunk. The splitter degrades gracefully:
paragraphs first, then sentences for an oversized paragraph, then a word window for an oversized
sentence — so no single chunk blows the budget regardless of how the source is formatted.

Token counts use the same cheap ruler as ``build_context`` (``estimate_tokens``), so chunk sizes
and the prompt budget speak the same units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..context.sections import estimate_tokens
from .extract import ExtractedUnit

_PARA_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_TARGET_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32


@dataclass(slots=True)
class Chunk:
    text: str
    locator: str


def _split_words(text: str, target: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    window: list[str] = []
    for word in words:
        window.append(word)
        if estimate_tokens(" ".join(window)) >= target:
            out.append(" ".join(window))
            window = []
    if window:
        out.append(" ".join(window))
    return out


def _segments(text: str, target: int) -> list[str]:
    """Break text into packable segments, none larger than ``target`` where avoidable."""
    segments: list[str] = []
    for para in (p.strip() for p in _PARA_RE.split(text)):
        if not para:
            continue
        if estimate_tokens(para) <= target:
            segments.append(para)
            continue
        for sentence in (s.strip() for s in _SENTENCE_RE.split(para)):
            if not sentence:
                continue
            if estimate_tokens(sentence) <= target:
                segments.append(sentence)
            else:
                segments.extend(_split_words(sentence, target))
    return segments


def _overlap_tail(buffer: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """The trailing segments of ``buffer`` summing to ~``overlap_tokens``, for chunk overlap."""
    if overlap_tokens <= 0:
        return [], 0
    tail: list[str] = []
    total = 0
    for seg in reversed(buffer):
        seg_tok = estimate_tokens(seg)
        if tail and total + seg_tok > overlap_tokens:
            break
        tail.insert(0, seg)
        total += seg_tok
    return tail, total


def chunk_units(
    units: list[ExtractedUnit],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk each unit independently, carrying its locator onto every produced chunk."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    chunks: list[Chunk] = []
    for unit in units:
        buffer: list[str] = []
        buf_tokens = 0
        for seg in _segments(unit.text, target_tokens):
            seg_tokens = estimate_tokens(seg)
            if buffer and buf_tokens + seg_tokens > target_tokens:
                chunks.append(Chunk(text="\n\n".join(buffer), locator=unit.locator))
                buffer, buf_tokens = _overlap_tail(buffer, overlap_tokens)
            buffer.append(seg)
            buf_tokens += seg_tokens
        if buffer:
            chunks.append(Chunk(text="\n\n".join(buffer), locator=unit.locator))
    return chunks
