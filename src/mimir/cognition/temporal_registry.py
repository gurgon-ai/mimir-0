"""The Temporal Registry — STATE vs NARRATIVE (docs/EXTENSIBILITY.md).

Mimir's memory store is NARRATIVE: it accumulates in mixed tense ("planning to do X" … "X is
underway" … "X is done"), all coexisting and ranked by relevance/recency/tier — *not by which one is
currently true*. So a status question can surface the older, higher-salience *planning* memory and
answer as if a long-finished thing is still upcoming. Confidence/salience decoupling doesn't fix it:
the stale memory isn't false in general — it was true when said; it's just **superseded as current
state.**

This is a small, **authoritative, dated, status-tagged** ledger of milestones — what is true *now* —
consulted for STATE and used as the authority that **reconciles** stale-state memories in the sleep
pass. It lives in its own table, so it's inherently exempt from memory decay/archival.

The reconcile guard is **distinctive tokens**: it only acts on a memory that shares a proper-noun /
number / rare token with a milestone — never a generic English word — so it can't clobber an
unrelated memory on a word like "system". Deterministic, no model call. Pure functions; the brain
owns the dependencies and wires reconcile into the sleep cycle.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from ..storage.gateway import StorageGateway
from ..storage.models import MemoryKind, Milestone
from ..storage.repo import (
    get_milestone,
    get_milestone_by_title,
    list_memories,
    list_milestones,
    update_memory,
    upsert_milestone,
)
from .tools import Tool

log = logging.getLogger("mimir.temporal_registry")

STATUSES = ("planned", "in_progress", "done", "abandoned")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w.\-]*")
# Future/planned framing — a memory with one of these, sharing a distinctive token with a *done*
# milestone, is asserting as upcoming what is in fact finished → superseded as current state.
_FUTURE_MARKERS = (
    "will ", "going to", "gonna", "plan to", "planning to", "planning on", "intend to",
    "about to", "upcoming", "soon", "we'll", "i'll", "next week", "next month", "hope to",
)
# Generic words a milestone must NOT reconcile on (would clobber unrelated memories). Distinctive
# tokens already exclude short/common words; this is the explicit belt for tech-generic nouns.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "will", "are", "was",
    "system", "systems", "project", "projects", "thing", "things", "stuff", "setup", "config",
    "update", "updated", "status", "current", "state", "done", "plan", "plans", "work", "working",
})
_DEMOTE_SALIENCE = 0.05   # push a superseded planning note toward deprioritization + archival
_PROTECT_FLOOR = 0.1      # keep a current-state memory above the archive floor this cycle


@dataclass(slots=True)
class ReconcileReport:
    examined: int = 0
    demoted: int = 0
    protected: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"examined": self.examined, "demoted": self.demoted, "protected": self.protected}


def _slug(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-") or "milestone"


def distinctive_tokens(text: str) -> list[str]:
    """Proper-noun / number / rare tokens — the safe keys reconcile may act on. A token qualifies if
    it has a digit, is long (≥8), or is Capitalized — and is ≥4 chars and not a generic stopword."""
    out: set[str] = set()
    for w in _TOKEN_RE.findall(text):
        low = w.lower()
        if len(w) < 4 or low in _STOPWORDS:
            continue
        if any(c.isdigit() for c in w) or len(w) >= 8 or w[0].isupper():
            out.add(low)
    return sorted(out)


def record_milestone(
    storage: StorageGateway, title: str, statement: str, status: str, *,
    occurred_at: float | None = None, is_current_config: bool = False,
    provenance: str = "stated", confidence: float = 0.9,
) -> str:
    """Upsert a milestone by title (a planned→in_progress→done progression updates one row in place,
    preserving its id + created_at). Distinctive tokens are re-extracted from title + statement."""
    if status not in STATUSES:
        raise ValueError(f"unknown milestone status {status!r}; one of {STATUSES}")
    now = time.time()
    occurred = occurred_at
    if occurred is None and status in ("in_progress", "done"):
        occurred = now
    existing = get_milestone_by_title(storage, title)
    mid = existing.milestone_id if existing else _slug(title)
    created = existing.created_at if existing else now
    upsert_milestone(storage, Milestone(
        milestone_id=mid, title=title, statement=statement, status=status,
        is_current_config=is_current_config, occurred_at=occurred, superseded_by=None,
        distinctive_tokens=distinctive_tokens(f"{title} {statement}"),
        provenance=provenance, confidence=confidence, created_at=created, updated_at=now,
    ))
    return mid


def set_status(storage: StorageGateway, milestone_id: str, status: str,
               *, occurred_at: float | None = None) -> None:
    m = get_milestone(storage, milestone_id)
    if m is None:
        return
    m.status = status
    m.occurred_at = occurred_at if occurred_at is not None else time.time()
    m.updated_at = time.time()
    upsert_milestone(storage, m)


def current_config(storage: StorageGateway) -> list[Milestone]:
    """The live current-configuration milestones (is_current_config, not superseded)."""
    return list_milestones(storage, current_only=True)


def timeline(storage: StorageGateway, limit: int = 12) -> list[Milestone]:
    return list_milestones(storage, statuses=("in_progress", "done", "planned"))[:limit]


def _fmt_date(occurred_at: float | None) -> str:
    if not occurred_at:
        return ""
    return time.strftime(" (%Y-%m-%d)", time.gmtime(occurred_at))


def timeline_text(storage: StorageGateway, limit: int = 12) -> str:
    """The authoritative current-state block for the top of the prompt — dated milestone lines.
    Empty string if the registry is empty."""
    items = timeline(storage, limit)
    if not items:
        return ""
    lines = [f"- {m.title} — {m.status.upper()}{_fmt_date(m.occurred_at)}. {m.statement}"
             for m in items]
    return ("The current state of things (authoritative — prefer this over older planning notes "
            "when answering 'what's the status of …'):\n" + "\n".join(lines))


def current_config_statements(storage: StorageGateway) -> list[str]:
    """Current-config statements to pin into the self-model ('how am I set up')."""
    return [m.statement for m in current_config(storage)]


def _memory_tokens(text: str) -> set[str]:
    return {w.lower() for w in _TOKEN_RE.findall(text)}


def _is_future_framed(text: str) -> bool:
    low = f" {text.lower()} "
    return any(marker in low for marker in _FUTURE_MARKERS)


def reconcile(storage: StorageGateway, *, dry_run: bool = False) -> ReconcileReport:
    """The authority pass (sleep cycle, deterministic). For each live done/in-progress/current
    milestone: **demote** non-milestone memories that share a distinctive token AND frame the thing
    as still upcoming (a *done* milestone supersedes them); **protect** an agreeing current-state
    memory from archival by lifting it above the floor. Never deletes; logs counts (no silent cap).
    """
    report = ReconcileReport()
    actionable = [m for m in list_milestones(storage)
                  if m.status in ("done", "in_progress") or m.is_current_config]
    if not actionable:
        return report
    memories = [m for m in list_memories(storage, kind=MemoryKind.MEMORY) if not m.archived]
    acted: set[int] = set()
    for ms in actionable:
        toks = set(ms.distinctive_tokens)
        if not toks:
            continue
        for mem in memories:
            if mem.id is None or mem.id in acted or not (toks & _memory_tokens(mem.text)):
                continue
            report.examined += 1
            if ms.status == "done" and _is_future_framed(mem.text):
                report.demoted += 1
                acted.add(mem.id)
                if not dry_run:
                    meta = dict(mem.meta)
                    meta["superseded_by_milestone"] = ms.milestone_id
                    update_memory(storage, mem.id, salience=_DEMOTE_SALIENCE, meta=meta)
            elif not _is_future_framed(mem.text) and mem.salience < _PROTECT_FLOOR:
                report.protected += 1
                acted.add(mem.id)
                if not dry_run:
                    meta = dict(mem.meta)
                    meta["confirmed_by_milestone"] = ms.milestone_id
                    update_memory(storage, mem.id, salience=_PROTECT_FLOOR, meta=meta)
    if report.examined:
        log.info("reconcile: examined=%d demoted=%d protected=%d",
                 report.examined, report.demoted, report.protected)
    return report


def make_timeline_source(storage: StorageGateway):
    """Kept for symmetry/testing; the brain injects the timeline as a top-attention section directly
    (it's authoritative current state, not an ambient connector section)."""
    return timeline_text(storage)


def make_milestone_tool(storage: StorageGateway) -> Tool:
    """A `record_milestone` tool so the model can log a durable state change the operator states
    ("we finished the migration today"). Non-actuating (it writes the brain's own STATE ledger, not
    the world), so safe under the no-hands rule (`state_changing=False`)."""

    def _handle(args: dict, ctx: object) -> str:
        title = str(args.get("title", "")).strip()
        statement = str(args.get("statement", "")).strip()
        status = str(args.get("status", "done")).strip().lower()
        if not title or not statement:
            return "error: 'title' and 'statement' are required"
        if status not in STATUSES:
            return f"error: status must be one of {STATUSES}"
        mid = record_milestone(storage, title, statement, status,
                               is_current_config=bool(args.get("is_current_config", False)),
                               provenance="stated")
        return f"recorded milestone {title!r} as {status} (id {mid})"

    return Tool(
        name="record_milestone",
        description=("record a durable STATE change (a milestone the operator states is "
                     "planned/in_progress/done). args: {title, statement, status, current_config}"),
        handler=_handle,
        schema={"title": {"required": True}, "statement": {"required": True}},
        state_changing=False,  # writes the brain's own state ledger, not the world
        keywords=("milestone", "finished", "completed", "done", "underway", "from now on"),
    )
