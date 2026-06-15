"""Working memory: rolling, cross-session salient context (DESIGN §3a, §3e).

Two parts, together giving "recency + compression":

- **recency** — the last few turn exchanges, kept raw and capped (``EXCHANGE`` rows). They
  persist across sessions, so a restart doesn't forget what was just being discussed. No model
  call — pure recency, available immediately.
- **compression** — periodically the reasoning model folds the accumulated exchanges (plus the
  previous summary) into a compact rolling summary (one ``WORKING_MEMORY`` row, latest wins), and
  the folded exchanges are cleared. This bounds the context and keeps the gist without the bulk.

What gets injected each turn is the summary followed by the most recent raw exchanges — older
context compressed, newest context verbatim. This is generic: it summarizes the conversation
itself, nothing domain-specific.
"""

from __future__ import annotations

import logging

from ..model.gateway import ModelGateway
from ..prompts import WORKING_MEMORY_SYSTEM
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import count_memories, delete_memories, prune_kind, recent_by_kind, save_memory

log = logging.getLogger("mimir.working_memory")

MAX_EXCHANGES = 20    # hard cap on the recency log (safety net if compression is disabled/lagging)
DISPLAY_RECENT = 6    # how many raw exchanges to show in the prompt
FOLD_THRESHOLD = 10   # fold once this many raw exchanges have accumulated
KEEP_RECENT = 4       # raw exchanges kept verbatim after a fold (the rest become summary)


def record_exchange(
    storage: StorageGateway, *, user: str | None, user_text: str, reply: str
) -> None:
    """Append one turn to the recency log and prune it back to ``MAX_EXCHANGES``."""
    speaker = user or "user"
    text = f"{speaker}: {user_text}\nyou: {reply}"
    save_memory(
        storage,
        Memory(
            text=text,
            kind=MemoryKind.EXCHANGE,
            evidence_tier=EvidenceTier.CONVERSATION,
            confidence=0.5,
            salience=1.0,
            embedding=None,
            provenance="conversation",
            user=user,
        ),
    )
    prune_kind(storage, MemoryKind.EXCHANGE, MAX_EXCHANGES)


def recent_exchanges(storage: StorageGateway, limit: int) -> list[Memory]:
    """The most recent exchanges in chronological order (oldest → newest)."""
    rows = recent_by_kind(storage, MemoryKind.EXCHANGE, limit=limit)  # newest first
    return list(reversed(rows))


def latest_working_memory(storage: StorageGateway) -> Memory | None:
    rows = recent_by_kind(storage, MemoryKind.WORKING_MEMORY, limit=1)
    return rows[0] if rows else None


def synthesize_working_memory(
    model: ModelGateway, storage: StorageGateway, *,
    fold_threshold: int = FOLD_THRESHOLD, keep_recent: int = KEEP_RECENT,
) -> Memory | None:
    """Fold the OLDEST accumulated exchanges (and the prior summary) into a fresh rolling summary,
    keeping the most recent ``keep_recent`` raw. A no-op until ``fold_threshold`` have accumulated.

    This is the rolling-compression scheme: recent turns stay verbatim, older ones become a short
    couple-paragraph summary that itself folds into the next one — older material compressed harder
    each pass. Returns the new summary, or ``None`` if there wasn't enough to fold. Off-hot-path; a
    failure here must never break the turn.
    """
    if fold_threshold <= 0:
        return None  # compression disabled — recency-only working memory
    exchanges = recent_exchanges(storage, MAX_EXCHANGES)  # oldest → newest
    if len(exchanges) < fold_threshold:
        return None  # not enough has built up yet — wait

    to_fold = exchanges[: -keep_recent] if keep_recent > 0 else exchanges
    if not to_fold:
        return None

    prior = latest_working_memory(storage)
    brief_parts: list[str] = []
    if prior:
        brief_parts.append(f"Previous working memory:\n{prior.text}")
    brief_parts.append("Older exchanges to fold in:\n" + "\n\n".join(e.text for e in to_fold))
    brief = "\n\n".join(brief_parts)

    summary = model.chat(
        "reasoning",
        [
            {"role": "system", "content": WORKING_MEMORY_SYSTEM},
            {"role": "user", "content": brief},
        ],
    ).strip()

    mem = Memory(
        text=summary,
        kind=MemoryKind.WORKING_MEMORY,
        evidence_tier=EvidenceTier.INFERRED,
        confidence=0.5,
        salience=1.0,
        embedding=None,  # fetched by recency, not similarity
        provenance="working-memory synthesis",
        user=None,
    )
    save_memory(storage, mem)
    # Drop only the folded (oldest) exchanges — the most recent `keep_recent` stay raw.
    delete_memories(storage, [e.id for e in to_fold if e.id is not None])
    log.info("working-memory: folded %d oldest exchange(s), kept %d raw",
             len(to_fold), len(exchanges) - len(to_fold))
    return mem


def exchange_count(storage: StorageGateway) -> int:
    """How many raw exchanges are currently buffered (drives the count-based fold trigger)."""
    return count_memories(storage, kind=MemoryKind.EXCHANGE)


def current_working_memory(storage: StorageGateway) -> str | None:
    """Compose the working-memory section: the rolling summary + the most recent raw exchanges."""
    summary = latest_working_memory(storage)
    recent = recent_exchanges(storage, DISPLAY_RECENT)
    parts: list[str] = []
    if summary:
        parts.append(summary.text)
    if recent:
        parts.append("Most recent exchanges:\n" + "\n\n".join(e.text for e in recent))
    return "\n\n".join(parts) if parts else None
