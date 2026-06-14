"""Temporal narratives — the system's sense of "what happened yesterday / last week / last month".

A stripped, universal extraction of the home AI's hierarchical journal. Three tiers, each compressed
from the one below:

- **daily**   — one first-person entry per day, from that day's material (the running summary +
  recent exchanges + the facts learned that day). Retained: the 10 most recent.
- **weekly**  — daily entries older than 3 days are compressed into a weekly summary. Retained: 5.
- **monthly** — weekly summaries beyond the 5 most recent are compressed into a monthly narrative.
  Retained: 13 (about a year of context).

The compression is **lossy by design** — details fade, patterns persist, like human memory. The most
recent entries from each tier are injected as a `[Recent history:]` section so a turn weeks later
still has the shape of what came before, without dragging the raw transcript.

Generic by construction: the sources are the conversation itself and what was learned from it — no
domain integrations. Generation is off the hot path (the consolidation/sleep pass) and uses the
``reasoning`` role, like the self-model and working-memory synthesis.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from ..model.gateway import ModelGateway
from ..prompts import (
    NARRATIVE_DAILY_SYSTEM,
    NARRATIVE_MONTHLY_SYSTEM,
    NARRATIVE_WEEKLY_SYSTEM,
)
from ..storage.gateway import StorageGateway
from ..storage.models import MemoryKind
from ..storage.repo import (
    get_narrative,
    list_memories,
    list_narratives,
    prune_narratives,
    save_narrative,
)
from .working_memory import current_working_memory

log = logging.getLogger("mimir.narratives")

DAILY_RETENTION = 10
WEEKLY_RETENTION = 5
MONTHLY_RETENTION = 13

_MIN_NARRATIVE_CHARS = 15  # below this the generation is treated as a failure, not stored
_MIN_DAILY_MATERIAL = 30   # don't journal an empty day
_FACTS_LIMIT = 100         # cap the facts fed into one daily entry


def _generate(model: ModelGateway, system: str, material: str) -> str | None:
    """One narrative generation through the reasoning role. ``None`` if empty/too short."""
    try:
        out = model.chat("reasoning", [
            {"role": "system", "content": system},
            {"role": "user", "content": material},
        ]).strip()
    except Exception as exc:  # off the hot path — a failed narrative is logged, never fatal (§10)
        log.warning("narratives: generation failed: %s", exc)
        return None
    return out if len(out) >= _MIN_NARRATIVE_CHARS else None


def _gather_daily_material(storage: StorageGateway, cutoff_ts: float) -> tuple[str, int]:
    """The day's raw material: the running summary + recent exchanges, plus the facts learned since
    ``cutoff_ts``. Returns ``(material, source_count)``. Generic — no domain sources."""
    parts: list[str] = []
    wm = current_working_memory(storage)
    if wm:
        parts.append("Running summary and recent exchanges:\n" + wm)
    facts = [
        m for m in list_memories(storage, kind=MemoryKind.MEMORY)
        if m.created_at and m.created_at >= cutoff_ts
    ]
    if facts:
        lines = "\n".join(f"- {' '.join(m.text.split())}" for m in facts[:_FACTS_LIMIT])
        parts.append("Facts learned this period:\n" + lines)
    return "\n\n".join(parts), (1 if wm else 0) + len(facts)


def generate_daily(
    model: ModelGateway, storage: StorageGateway, *, now: _dt.datetime
) -> str | None:
    """Generate today's daily narrative (idempotent — returns today's existing entry if present)."""
    period = now.strftime("%Y-%m-%d")
    existing = get_narrative(storage, "daily", period)
    if existing is not None:
        return existing
    material, source_count = _gather_daily_material(storage, now.timestamp() - 86400)
    if len(material) < _MIN_DAILY_MATERIAL:
        return None  # nothing meaningful happened — don't write an empty journal
    narrative = _generate(model, NARRATIVE_DAILY_SYSTEM, material)
    if narrative is None:
        return None
    save_narrative(storage, scope="daily", period=period, narrative=narrative,
                   source_count=source_count, created_at=now.timestamp())
    log.info("narratives: wrote daily %s (%d chars)", period, len(narrative))
    return narrative


def _compress(
    model: ModelGateway, storage: StorageGateway, *, from_scope: str, to_scope: str,
    skip_recent: int, take: int, min_take: int, system: str,
    now: _dt.datetime, prune_from: int,
) -> str | None:
    """Shared roll-up: compress the older entries of ``from_scope`` into one ``to_scope`` entry.

    Skips the ``skip_recent`` newest (they stay at the finer grain), takes up to ``take`` of the
    rest, needs at least ``min_take`` to bother, and is idempotent on the period. After writing,
    prunes ``from_scope`` to ``prune_from`` so finished entries don't accumulate."""
    rows = list_narratives(storage, from_scope)  # newest first
    pool = rows[skip_recent:skip_recent + take]
    if len(pool) < min_take:
        return None
    period = f"{pool[0]['period']}_to_{pool[-1]['period']}"
    existing = get_narrative(storage, to_scope, period)
    if existing is not None:
        return existing
    material = "\n\n".join(f"- {r['period']}: {r['narrative']}" for r in pool)
    narrative = _generate(model, system, material)
    if narrative is None:
        return None
    save_narrative(storage, scope=to_scope, period=period, narrative=narrative,
                   source_count=len(pool), created_at=now.timestamp())
    prune_narratives(storage, from_scope, prune_from)
    log.info("narratives: compressed %d %s → %s %s", len(pool), from_scope, to_scope, period)
    return narrative


def compress_weekly(
    model: ModelGateway, storage: StorageGateway, *, now: _dt.datetime
) -> str | None:
    """Compress daily entries older than the 3 newest into a weekly summary (needs ≥7 dailies)."""
    if len(list_narratives(storage, "daily")) < 7:
        return None
    return _compress(model, storage, from_scope="daily", to_scope="weekly",
                     skip_recent=3, take=7, min_take=4, system=NARRATIVE_WEEKLY_SYSTEM,
                     now=now, prune_from=DAILY_RETENTION)


def compress_monthly(
    model: ModelGateway, storage: StorageGateway, *, now: _dt.datetime
) -> str | None:
    """Compress weekly summaries beyond the 5 newest into a monthly narrative (needs ≥5 + 2)."""
    if len(list_narratives(storage, "weekly")) < 5:
        return None
    result = _compress(model, storage, from_scope="weekly", to_scope="monthly",
                       skip_recent=5, take=12, min_take=2, system=NARRATIVE_MONTHLY_SYSTEM,
                       now=now, prune_from=WEEKLY_RETENTION)
    prune_narratives(storage, "monthly", MONTHLY_RETENTION)
    return result


def run_narrative_cycle(
    model: ModelGateway, storage: StorageGateway, *, now: _dt.datetime
) -> dict[str, Any]:
    """Daily generation + weekly/monthly roll-up — the consolidation-time entry point (off path)."""
    return {
        "daily": generate_daily(model, storage, now=now),
        "weekly": compress_weekly(model, storage, now=now),
        "monthly": compress_monthly(model, storage, now=now),
    }


def recent_narratives(storage: StorageGateway) -> dict[str, list[dict[str, Any]]]:
    """The most recent entries per tier for injection: daily×3, weekly×2, monthly×1."""
    return {
        "daily": list_narratives(storage, "daily")[:3],
        "weekly": list_narratives(storage, "weekly")[:2],
        "monthly": list_narratives(storage, "monthly")[:1],
    }


def render_recent_history(storage: StorageGateway) -> str | None:
    """The ``[Recent history:]`` section body — coarsest first (monthly → weekly → daily), so the
    model reads the long arc then the recent detail. ``None`` if no narratives exist yet."""
    recent = recent_narratives(storage)
    lines: list[str] = []
    for scope, label in (("monthly", "This past while"), ("weekly", "Recent weeks"),
                         ("daily", "Lately")):
        for entry in reversed(recent[scope]):  # oldest → newest within a tier
            lines.append(f"{label} ({entry['period']}): {entry['narrative']}")
    return "\n".join(lines) if lines else None
