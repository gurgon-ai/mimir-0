"""Text extraction for document ingestion (DESIGN §8).

Resolves an extractor by file extension and returns a list of ``ExtractedUnit`` — a span of
text plus a **locator** (page, section, …) that becomes provenance downstream. Plain text and
markdown are handled in core with zero dependencies; PDF lives behind the optional
``[documents]`` extra so the runtime contract (Python + SQLite, nothing else) holds for anyone
who doesn't need it.

A missing optional extractor fails loud with an install instruction — never a silent skip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import IngestError

log = logging.getLogger("mimir.documents")

_TEXT_EXTS = {".txt", ".text", ""}
_MARKDOWN_EXTS = {".md", ".markdown", ".mdown"}
_PDF_EXTS = {".pdf"}

# ATX markdown heading, e.g. "## Section title".
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(slots=True)
class ExtractedUnit:
    """A span of text with a human-readable locator (becomes part of provenance)."""

    text: str
    locator: str  # e.g. "p.3", a section heading, or "" for an un-located whole file


def extract(path: Path) -> list[ExtractedUnit]:
    """Extract a document into located units. Raises ``IngestError`` on unsupported types."""
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS:
        return _extract_text(path)
    if ext in _MARKDOWN_EXTS:
        return _extract_markdown(path)
    if ext in _PDF_EXTS:
        return _extract_pdf(path)
    raise IngestError(
        f"unsupported document type {ext!r} for {path.name}. Supported: .txt, .md in core; "
        f".pdf via the optional extra (pip install 'mimir-0[documents]')."
    )


def _read_text(path: Path) -> str:
    # Documents are best-effort text; undecodable bytes are replaced rather than crashing the
    # ingest. This is a convenience layer, not core memory integrity.
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "�" in text:
        log.warning("extract: %s had non-UTF-8 bytes; replaced during decode", path.name)
    return text


def _extract_text(path: Path) -> list[ExtractedUnit]:
    text = _read_text(path)
    return [ExtractedUnit(text=text, locator="")] if text.strip() else []


def _extract_markdown(path: Path) -> list[ExtractedUnit]:
    """Split markdown into sections by ATX headings; the heading becomes the locator."""
    text = _read_text(path)
    units: list[ExtractedUnit] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            units.append(ExtractedUnit(text=body, locator=current_heading))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            buffer = [line]  # keep the heading line in the section body for context
        else:
            buffer.append(line)
    flush()
    return units


def _extract_pdf(path: Path) -> list[ExtractedUnit]:
    try:
        import pypdf
    except ImportError as exc:
        raise IngestError(
            "PDF ingestion needs the optional extra. Install it with: "
            "pip install 'mimir-0[documents]'"
        ) from exc

    reader = pypdf.PdfReader(str(path))
    units: list[ExtractedUnit] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            units.append(ExtractedUnit(text=page_text, locator=f"p.{i + 1}"))
    if not units:
        log.warning("extract: %s yielded no extractable text (scanned/image PDF?)", path.name)
    return units
