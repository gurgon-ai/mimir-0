"""The self-model: a synthesized, evolving, always-on identity (DESIGN §3a).

The self-model is the system's *authored identity* — but not a fixed persona. It is
re-synthesized from the system's own operational history: how much it knows, across which
evidence tiers, who it serves, and what it has recently reflected on. The reasoning model writes
a short first-person self-description grounded ONLY in those facts; it is stored and injected at
the top of every prompt (DESIGN §3e), so identity evolves from experience rather than a static
string.

Crucially **generic**: every signal here is universal operational metadata about the store
itself. Nothing deployment- or domain-specific leaks in — that is the seed identity's job (config)
and a registered context source's job, never the core self-model's.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from ..model.gateway import ModelGateway
from ..prompts import SELF_MODEL_SYSTEM
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import count_memories, list_memories, recent_by_kind, save_memory

log = logging.getLogger("mimir.self_model")

_RECENT_REFLECTIONS = 3


@dataclass(slots=True)
class SelfSignals:
    """Generic operational signals the self-model is grounded in."""

    total_memories: int = 0
    documents: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    distinct_users: int = 0
    reflections: int = 0
    recent_reflections: list[str] = field(default_factory=list)


def gather_signals(storage: StorageGateway) -> SelfSignals:
    """Read universal stats about the knowledge store — nothing domain-specific."""
    memories = list_memories(storage, kind=MemoryKind.MEMORY)
    tier_counts = Counter(m.evidence_tier.key for m in memories)
    users = {m.user for m in memories if m.user}
    recent = recent_by_kind(storage, MemoryKind.SENTINEL_NOTE, limit=_RECENT_REFLECTIONS)
    return SelfSignals(
        total_memories=len(memories),
        documents=tier_counts.get(EvidenceTier.DOCUMENT.key, 0),
        tier_counts=dict(tier_counts),
        distinct_users=len(users),
        reflections=count_memories(storage, kind=MemoryKind.SENTINEL_NOTE),
        recent_reflections=[m.text for m in recent],
    )


def build_brief(signals: SelfSignals) -> str:
    """Render the signals into a compact, factual brief for the synthesizer."""
    summary = (
        f"I currently hold {signals.total_memories} memor"
        f"{'y' if signals.total_memories == 1 else 'ies'} "
        f"({signals.documents} from documents), have made {signals.reflections} reflective note"
        f"{'' if signals.reflections == 1 else 's'}, and have interacted with "
        f"{signals.distinct_users} user{'' if signals.distinct_users == 1 else 's'}."
    )
    lines = [summary]
    if signals.tier_counts:
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(signals.tier_counts.items()))
        lines.append(f"Evidence tiers held: {breakdown}.")
    if signals.recent_reflections:
        lines.append("Recent reflections:")
        lines.extend(f"- {note}" for note in signals.recent_reflections)
    return "\n".join(lines)


def synthesize_self_model(
    model: ModelGateway, storage: StorageGateway
) -> Memory:
    """Synthesize a fresh self-description from current signals and persist it.

    Returns the stored self-model memory. Raised exceptions are the caller's to handle off the
    hot path (a self-model failure must never break the turn loop).
    """
    brief = build_brief(gather_signals(storage))
    text = model.chat(
        "reasoning",
        [
            {"role": "system", "content": SELF_MODEL_SYSTEM},
            {"role": "user", "content": brief},
        ],
    ).strip()

    mem = Memory(
        text=text,
        kind=MemoryKind.SELF_MODEL,
        evidence_tier=EvidenceTier.INFERRED,  # a synthesis, not a stated fact
        confidence=0.5,
        salience=1.0,
        embedding=None,  # the self-model is fetched by recency, not similarity
        provenance="self-model synthesis",
        user=None,  # the system's identity is shared, not scoped to a speaker
    )
    save_memory(storage, mem)
    log.info("self-model: synthesized a fresh identity from operational history")
    return mem
