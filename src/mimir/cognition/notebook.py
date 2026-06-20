"""Notebook — lossless, name-addressable working memory (docs/EXTENSIBILITY.md).

Mimir's memory store is content-addressed, auto-managed, and *lossy* (it decays, dedups, archives —
correctly). A notebook is the opposite: a **named, lossless, self-curated** markdown document the
model writes to, re-reads, revises section by section, and grooms itself. *Memory is what it knows;
a notebook is what it's working on.*

The key general mechanism is **read = RAG re-trigger**: a cold re-read runs the note text back
through ordinary recall, so a note reconnects to current memory instead of being an orphaned snippet
("why did I write this?"). Sections are ``##``-delimited and individually addressable. Stored in
SQLite via the storage gateway (one row, markdown in a TEXT column) — not loose files — so it honors
the gateway law and survives restart cleanly. Pure functions; the brain owns the dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..context.sections import ContextSource, Section, SectionTier, estimate_tokens
from ..embed.base import Embedder
from ..errors import NotebookError
from ..retrieval.hybrid import retrieve
from ..storage.gateway import StorageGateway
from ..storage.models import Memory, MemoryKind
from ..storage.repo import (
    delete_notebook,
    get_notebook,
    list_memories,
    list_notebooks,
    rename_notebook,
    upsert_notebook,
)
from .tools import Tool

SELF = "__self__"
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ASSOC_TOP_K = 6


@dataclass(slots=True)
class NotebookMeta:
    """The catalog view of a notebook — enough for the ambient index, without the body."""

    title: str
    owner: str
    section_titles: list[str]
    size: int
    updated_at: float


def _slug(owner: str, title: str) -> str:
    base = _SLUG_RE.sub("-", f"{owner}-{title}".lower()).strip("-")
    return base or "notebook"


def _sections(body_md: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, block)`` pairs on ``##`` headings. Text before the first
    heading is returned under the empty-string heading. Each block keeps its own ``## heading``."""
    matches = list(_SECTION_RE.finditer(body_md))
    if not matches:
        return [("", body_md)] if body_md.strip() else []
    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = body_md[: matches[0].start()].strip()
        if preamble:
            out.append(("", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body_md)
        out.append((m.group(1).strip(), body_md[m.start() : end].rstrip()))
    return out


def meta(storage: StorageGateway, owner: str = SELF) -> list[NotebookMeta]:
    """The owner's notebooks as catalog metadata (newest first)."""
    return [
        NotebookMeta(title=nb.title, owner=nb.owner,
                     section_titles=[h for h, _ in _sections(nb.body_md) if h],
                     size=len(nb.body_md), updated_at=nb.updated_at)
        for nb in list_notebooks(storage, owner)
    ]


def index(storage: StorageGateway, owner: str = SELF) -> str:
    """A compact text catalog for ambient prompt injection — titles + section titles + size, never
    bodies. Empty string if the owner has no notebooks."""
    items = meta(storage, owner)
    if not items:
        return ""
    lines = []
    for m in items:
        secs = f" — sections: {', '.join(m.section_titles)}" if m.section_titles else ""
        lines.append(f"- {m.title} ({m.size} chars){secs}")
    return "Your notebooks (use the notebook tool to read/edit):\n" + "\n".join(lines)


def read(storage: StorageGateway, title: str, section: str | None = None, owner: str = SELF) -> str:
    """The whole notebook's markdown, or one ``##`` section. Empty string if absent."""
    nb = get_notebook(storage, owner, title)
    if nb is None:
        return ""
    if section is None:
        return nb.body_md
    for heading, block in _sections(nb.body_md):
        if heading.lower() == section.lower():
            return block
    return ""


def write(storage: StorageGateway, title: str, body_md: str, owner: str = SELF,
          *, soft_cap: int | None = None) -> str:
    """Create or replace a notebook by name. ``soft_cap`` (for ``__self__`` only) raises
    ``NotebookError`` when creating a NEW notebook past the cap — surfaced, never silently dropped,
    so the model grooms (rename/merge/delete) rather than hoards. Returns the notebook_id."""
    if (soft_cap is not None and owner == SELF
            and get_notebook(storage, owner, title) is None
            and len(list_notebooks(storage, owner)) >= soft_cap):
        raise NotebookError(
            f"notebook soft cap reached ({soft_cap}); rename, merge, or delete one before creating "
            f"{title!r} (your notebooks: {', '.join(m.title for m in meta(storage, owner))})")
    return upsert_notebook(storage, notebook_id=_slug(owner, title), owner=owner, title=title,
                           body_md=body_md)


def append(storage: StorageGateway, title: str, text: str, owner: str = SELF,
           *, soft_cap: int | None = None) -> str:
    """Append text to the end of a notebook (creating it if absent)."""
    existing = read(storage, title, owner=owner)
    body = f"{existing.rstrip()}\n\n{text.strip()}" if existing.strip() else text.strip()
    return write(storage, title, body, owner, soft_cap=soft_cap)


def edit(storage: StorageGateway, title: str, section: str, new_text: str,
         owner: str = SELF) -> str:
    """Replace one ``##`` section in place (preserving the rest + ordering); add the section if it's
    not present. Raises ``NotebookError`` if the notebook itself doesn't exist."""
    nb = get_notebook(storage, owner, title)
    if nb is None:
        raise NotebookError(f"no notebook {title!r} to edit")
    block = f"## {section}\n{new_text.strip()}"
    parts = _sections(nb.body_md)
    rebuilt, replaced = [], False
    for heading, existing in parts:
        if heading.lower() == section.lower():
            rebuilt.append(block)
            replaced = True
        else:
            rebuilt.append(existing)
    if not replaced:
        rebuilt.append(block)
    return write(storage, title, "\n\n".join(rebuilt), owner)


def rename(storage: StorageGateway, title: str, new_title: str, owner: str = SELF) -> bool:
    return rename_notebook(storage, owner, title, new_title)


def delete(storage: StorageGateway, title: str, owner: str = SELF) -> bool:
    return delete_notebook(storage, owner, title)


def associated_memories(
    storage: StorageGateway, embedder: Embedder, text: str, *, k: int = _ASSOC_TOP_K,
) -> list[Memory]:
    """The current memories most relevant to ``text`` — the recall a notebook re-read reconnects."""
    if not text.strip():
        return []
    vec = embedder.embed(text)
    candidates = list_memories(storage, user=None, kind=MemoryKind.MEMORY)
    return [s.memory for s in retrieve(text, vec, candidates, top_k=k)]


def read_with_memory(
    storage: StorageGateway, embedder: Embedder, title: str, section: str | None = None,
    owner: str = SELF,
) -> tuple[str, list[Memory]]:
    """Read a notebook (or section) AND the live memories it reconnects to — "a real notebook," not
    an orphaned clipping. The default the tool/burst-author use when reading."""
    passage = read(storage, title, section, owner)
    return passage, associated_memories(storage, embedder, passage)


# -- the connector wiring: a context source (sensory port) + a tool (motor port) ------

def make_index_source(storage: StorageGateway) -> ContextSource:
    """The ambient ``[notebooks]`` catalog section — titles + sections only, never bodies, so the
    model always knows what notebooks exist (the Tool pulls bodies on demand). Low/ambient tier."""

    class _NotebookIndexSource:
        name = "notebooks"
        tier = SectionTier.LOW
        budget_tokens = 400

        def build(self, query: str, user: str | None) -> Section | None:
            text = index(storage, SELF)
            if not text:
                return None
            toks = estimate_tokens(text)
            return Section(name="notebooks", title=text, body="", tier=SectionTier.LOW,
                           requested_tokens=toks, admitted_tokens=toks)

    return _NotebookIndexSource()


def make_notebook_tool(
    storage: StorageGateway, embedder: Embedder, *,
    soft_cap: int | None = None, read_rag: bool = True,
) -> Tool:
    """A `notebook` tool exposing list/read/write/append/edit/rename/delete over the assistant's own
    notebooks. **Non-actuating** (read/think/note — its own store, not the world), so it's safe
    under the no-hands rule (`state_changing=False`). Reads re-trigger recall when ``read_rag``."""

    def _handle(args: dict, ctx: object) -> str:
        op = str(args.get("op", "")).strip().lower()
        if op in ("", "list", "index"):
            return index(storage, SELF) or "(no notebooks yet)"
        title = str(args.get("title", "")).strip()
        if not title:
            return "error: 'title' is required for that op"
        if op == "read":
            if read_rag:
                passage, mems = read_with_memory(storage, embedder, title, args.get("section"))
                if not passage:
                    return f"(no notebook {title!r})"
                related = "\n".join(f"- {m.text}" for m in mems[:4])
                return passage + (f"\n\n(related from memory:\n{related})" if related else "")
            return read(storage, title, args.get("section")) or f"(no notebook {title!r})"
        if op == "write":
            write(storage, title, str(args.get("body", "")), soft_cap=soft_cap)
            return f"wrote notebook {title!r}"
        if op == "append":
            append(storage, title, str(args.get("text", "")), soft_cap=soft_cap)
            return f"appended to {title!r}"
        if op == "edit":
            edit(storage, title, str(args.get("section", "")), str(args.get("text", "")))
            return f"edited section of {title!r}"
        if op == "rename":
            ok = rename(storage, title, str(args.get("new_title", "")))
            return f"renamed {title!r}" if ok else "rename failed (target exists or source missing)"
        if op == "delete":
            return f"deleted {title!r}" if delete(storage, title) else f"(no notebook {title!r})"
        return f"error: unknown op {op!r} (list/read/write/append/edit/rename/delete)"

    return Tool(
        name="notebook",
        description=("read/write your own lossless notebooks (markdown, ## sections). args: {op: "
                     "list|read|write|append|edit|rename|delete, title, body?, text?, section?, "
                     "new_title?}"),
        handler=_handle,
        schema={"op": {"required": True}},
        state_changing=False,  # your own notes, not the world — safe under the no-hands rule
        keywords=("note", "jot", "notebook", "write down", "remember"),
    )
