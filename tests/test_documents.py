"""Executable spec for document extraction and chunking (v0.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimir.documents.chunk import Chunk, chunk_units
from mimir.documents.extract import ExtractedUnit, extract
from mimir.errors import IngestError


def test_extract_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello world\n\nsecond paragraph", encoding="utf-8")
    units = extract(f)
    assert len(units) == 1
    assert units[0].locator == ""
    assert "second paragraph" in units[0].text


def test_extract_markdown_splits_on_headings(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(
        "intro line\n\n# First\nalpha body\n\n## Second\nbeta body\n",
        encoding="utf-8",
    )
    units = extract(f)
    locators = [u.locator for u in units]
    assert "First" in locators and "Second" in locators
    first = next(u for u in units if u.locator == "First")
    assert "alpha body" in first.text
    # intro before the first heading is kept with an empty locator
    assert any(u.locator == "" and "intro line" in u.text for u in units)


def test_extract_unsupported_type_fails_loud(tmp_path: Path) -> None:
    f = tmp_path / "thing.xyz"
    f.write_text("data", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported document type"):
        extract(f)


def test_extract_docx_splits_on_headings(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")  # the [documents] extra (python-docx); skip if absent
    f = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_heading("First", level=1)
    d.add_paragraph("alpha body")
    d.add_heading("Second", level=2)
    d.add_paragraph("beta body")
    d.save(str(f))
    units = extract(f)
    locators = [u.locator for u in units]
    assert "First" in locators and "Second" in locators
    first = next(u for u in units if u.locator == "First")
    assert "alpha body" in first.text


def test_extract_docx_reads_table_cells(tmp_path: Path) -> None:
    """A table-structured .docx (e.g. a safety matrix) must not be lost — cell text is extracted,
    in document order, under the heading that precedes the table."""
    docx = pytest.importorskip("docx")
    f = tmp_path / "matrix.docx"
    d = docx.Document()
    d.add_heading("Hazards", level=1)
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Aerial lift"
    table.cell(0, 1).text = "Wear a harness"
    table.cell(1, 0).text = "Chemical handling"
    table.cell(1, 1).text = "Use ventilation"
    d.save(str(f))
    units = extract(f)
    text = "\n".join(u.text for u in units)
    # Every cell's content survives (the old paragraphs-only extractor dropped all of it).
    for cell in ("Aerial lift", "Wear a harness", "Chemical handling", "Use ventilation"):
        assert cell in text
    # The table lands under its preceding heading, not in a headingless limbo.
    hazards = next(u for u in units if u.locator == "Hazards")
    assert "Aerial lift" in hazards.text


def test_chunk_preserves_locator_and_bounds_size() -> None:
    # ~120 paragraphs of a few tokens each → must split into several chunks.
    body = "\n\n".join(f"paragraph number {i} has a little content here" for i in range(120))
    units = [ExtractedUnit(text=body, locator="p.1")]
    chunks = chunk_units(units, target_tokens=64, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.locator == "p.1" for c in chunks)
    # no chunk wildly exceeds the target (allow slack for the last packed segment)
    from mimir.context.sections import estimate_tokens

    assert all(estimate_tokens(c.text) <= 64 * 2 for c in chunks)


def test_chunk_overlap_shares_content() -> None:
    body = "\n\n".join(f"sentence segment {i}" for i in range(40))
    units = [ExtractedUnit(text=body, locator="")]
    no_overlap = chunk_units(units, target_tokens=32, overlap_tokens=0)
    with_overlap = chunk_units(units, target_tokens=32, overlap_tokens=12)
    # overlap carries trailing context forward, so it produces at least as many chunks
    assert len(with_overlap) >= len(no_overlap) > 1
    # the start of a later chunk repeats the tail of the previous one
    assert isinstance(with_overlap[1], Chunk)
    prev_tail = no_overlap[0].text.splitlines()[-1]
    assert prev_tail in with_overlap[1].text


def test_empty_document_yields_no_units(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("   \n\n  ", encoding="utf-8")
    assert extract(f) == []
