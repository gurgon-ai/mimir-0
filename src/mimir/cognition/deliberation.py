"""Self-directed deliberation: the system argues its own open questions in sleep (DESIGN §5a).

Consolidation settles the clear-cut cases (functional contradictions, exact/near duplicates). What's
left are genuine *tensions* worth reasoning about — and this is where the inner council stops being
something you only invoke by hand and becomes self-initiated: during the sleep cycle the system
**surfaces its own conflicts and submits them to the council** for adversarial reasoning.

Two deterministic sources, neither of which consolidation resolves:

- **Graph tensions** — a subject with two+ objects under the *same non-functional* relation
  (e.g. "wants X" vs "wants Y"). Functional relations like "lives in" are left to consolidation,
  which resolves them newest-wins; non-functional ones are real judgment calls.
- **Divergent near-duplicates** — memory pairs similar enough to be about the same thing but not so
  similar that consolidation merged them (a cosine *tension band*), whose text actually differs.

A **curator** then picks the few most worth arguing (an LLM ranks them; a deterministic weight order
is the fallback when no model answers), and each goes to the council. The module is pure surfacing +
curation; the brain wires in the council call, the seen-conflict memory, and persistence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..embed.base import Embedder, cosine
from ..model.gateway import ModelGateway
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, MemoryKind
from ..storage.repo import browse_triples, list_memories
from .sleep import FUNCTIONAL_RELATIONS, NEAR_DUP_COSINE

log = logging.getLogger("mimir.deliberation")

# Memory pairs in this cosine band are "about the same thing" yet were not merged by consolidation
# (which merges at >= NEAR_DUP_COSINE) — close enough to be a tension, far enough to be a real one.
TENSION_COSINE_LO = 0.84
_MAX_MEMORIES_SCANNED = 200  # cap the O(n^2) near-duplicate scan to the most salient memories

# Tiers the council should NOT argue over: DOCUMENT chunks are reference material whose overlapping
# windows look like near-duplicates but aren't beliefs in tension; INFERRED is the system's OWN
# output (inner-life musings, prior verdicts) — arguing it just loops on itself. Real tensions live
# among what someone *stated* (primary/trusted/conversation/peer).
NON_BELIEF_TIERS = frozenset({EvidenceTier.DOCUMENT, EvidenceTier.INFERRED})

_CURATOR_SYSTEM = (
    "You triage a list of open questions/tensions found in an AI's own memory. Choose the few most "
    "worth careful adversarial reasoning — genuine disagreements or consequential judgments, not "
    "trivia with an obvious answer. Reply with ONLY the chosen item numbers, comma-separated."
)


@dataclass(slots=True)
class Conflict:
    """A surfaced tension: a stable ``key`` (for cross-night dedup), the ``question`` the council
    argues, and a ``weight`` used to rank when no curator model is available."""

    key: str
    question: str
    weight: float


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _graph_conflicts(storage: StorageGateway) -> list[Conflict]:
    triples = browse_triples(storage, limit=10_000)
    groups: dict[tuple[str, str, str], list] = {}
    for triple in triples:
        relation = _norm(triple.relation)
        if relation in FUNCTIONAL_RELATIONS:
            continue  # consolidation owns functional contradictions (newest-wins) — not a debate
        groups.setdefault((_norm(triple.subject), relation, triple.user or ""), []).append(triple)

    conflicts: list[Conflict] = []
    for (subject, relation, user), group in groups.items():
        objects = sorted({t.object.strip() for t in group if t.object.strip()})
        if len(objects) < 2:
            continue  # no disagreement
        listing = "; ".join(f"{relation} {o}" for o in objects)
        question = (
            f"My records disagree about {group[0].subject}: {listing}. "
            f"Which holds — or can they be reconciled — and why?"
        )
        weight = float(len(objects)) + sum(t.confidence for t in group)
        conflicts.append(Conflict(key=f"graph:{subject}|{relation}|{user}", question=question,
                                  weight=weight))
    return conflicts


def _memory_conflicts(storage: StorageGateway, embedder: Embedder | None) -> list[Conflict]:
    if embedder is None or not embedder.mode.is_semantic:
        # Only real (endpoint) embeddings are semantic; lexical-hash cosine here would be noise.
        return []
    memories = [
        m for m in list_memories(storage, user=None, kind=MemoryKind.MEMORY)
        if m.embedding and not m.archived and m.evidence_tier not in NON_BELIEF_TIERS
    ]
    memories.sort(key=lambda m: m.salience, reverse=True)
    memories = memories[:_MAX_MEMORIES_SCANNED]
    conflicts: list[Conflict] = []
    for i, a in enumerate(memories):
        for b in memories[i + 1:]:
            if a.user != b.user or _norm(a.text) == _norm(b.text):
                continue
            sim = cosine(a.embedding, b.embedding)
            if TENSION_COSINE_LO <= sim < NEAR_DUP_COSINE:
                lo, hi = sorted((a.id or 0, b.id or 0))
                question = (
                    f"Two of my memories pull against each other: (A) {a.text!r} vs "
                    f"(B) {b.text!r}. Which holds, do both, or is one stale? Reconcile them."
                )
                conflicts.append(Conflict(key=f"mem:{lo}|{hi}", question=question,
                                          weight=sim + (a.salience + b.salience) / 2.0))
    return conflicts


def surface_conflicts(
    storage: StorageGateway, *, embedder: Embedder | None = None
) -> list[Conflict]:
    """Surface graph tensions + divergent near-duplicates (highest-weight first), deterministic."""
    conflicts = _graph_conflicts(storage) + _memory_conflicts(storage, embedder)
    conflicts.sort(key=lambda c: c.weight, reverse=True)
    return conflicts


def _parse_indices(text: str, n: int) -> list[int]:
    seen: list[int] = []
    for match in re.findall(r"\d+", text):
        idx = int(match)
        if 0 <= idx < n and idx not in seen:
            seen.append(idx)
    return seen


def curate(model: ModelGateway, conflicts: list[Conflict], *, limit: int) -> list[Conflict]:
    """Pick the ``limit`` conflicts most worth arguing. Hybrid: an LLM ranks; if it errors or
    returns nothing usable, fall back to the deterministic weight order. Never raises."""
    ranked = sorted(conflicts, key=lambda c: c.weight, reverse=True)
    if len(ranked) <= limit:
        return ranked
    try:
        listing = "\n".join(f"{i}. {c.question}" for i, c in enumerate(ranked))
        resp = model.chat(
            "reasoning",
            [
                {"role": "system", "content": _CURATOR_SYSTEM},
                {"role": "user", "content": f"Pick the {limit} most worth it:\n\n{listing}"},
            ],
        )
        chosen = [ranked[i] for i in _parse_indices(resp, len(ranked))[:limit]]
        if chosen:
            return chosen
    except Exception as exc:  # curator is an optimization — degrade to weight order, never fail
        log.warning("deliberation: curator model failed (%s); using weight order", exc)
    return ranked[:limit]
