"""Sleep / consolidation: memory that maintains itself (DESIGN §1, §5).

A batch maintenance pass over the store — mostly deterministic, no model call — that keeps memory
healthy without a human curating it:

- **dedup** — exact and near-duplicate (cosine) memories merged into the best-sourced survivor,
  summing access counts.
- **decay** — salience decays with disuse (drives forgetting); **confidence decays only for
  low-tier provisionals**, never authority-tier facts — so a true-but-unused fact loses salience
  but not truth (DESIGN §3c: don't let "haven't used it lately" masquerade as "probably false").
- **archive** — only *low-salience provisional* memories are archived (excluded from active recall,
  kept in the store; archiving ≠ disbelieving). High-confidence facts are never archived, which
  avoids the death spiral the design warns about.
- **contradiction resolution** — over the entity graph: when a *functional* relation (one whose
  subject has a single value, e.g. "lives in") has conflicting objects, the newest wins and the
  stale edges are dropped. Deliberately conservative — non-functional relations are left alone, so
  "likes tea" and "likes coffee" are never treated as a contradiction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..embed.base import cosine
from ..storage.gateway import StorageGateway
from ..storage.models import Memory, MemoryKind, Triple
from ..storage.repo import (
    apply_decay,
    archive_memories,
    browse_triples,
    bump_memory,
    delete_memories,
    delete_triples,
    list_memories,
)

log = logging.getLogger("mimir.sleep")

_SECONDS_PER_DAY = 86_400.0
SALIENCE_HALF_LIFE_DAYS = 30.0
CONFIDENCE_HALF_LIFE_DAYS = 120.0  # gentler; only applied to decaying tiers
ARCHIVE_SALIENCE_FLOOR = 0.05
ARCHIVE_CONFIDENCE_CEILING = 0.6  # only provisional (low-confidence) memories are archived
NEAR_DUP_COSINE = 0.97

# Relations treated as functional (a subject has at most one value). Only these are eligible for
# contradiction resolution — kept tight on purpose to avoid clobbering many-valued relations.
FUNCTIONAL_RELATIONS = frozenset(
    {"is", "lives in", "located in", "is located in", "born in", "is from"}
)


@dataclass(slots=True)
class SleepReport:
    deduped: int = 0
    decayed: int = 0
    archived: int = 0
    contradictions_resolved: int = 0

    @property
    def total_changes(self) -> int:
        return self.deduped + self.decayed + self.archived + self.contradictions_resolved


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def consolidate(storage: StorageGateway, *, now: float | None = None) -> SleepReport:
    """Run a full consolidation pass and return what changed. Safe to run any time; idempotent-ish.

    A second run with no new activity is a near no-op: duplicates are gone, decay re-converges,
    archived stay archived. Never advances any external bookmark (DESIGN §10 governor fail-safe).
    """
    clock = time.time() if now is None else now
    report = SleepReport()

    report.deduped = _dedup(storage)
    memories = list_memories(storage, user=None, kind=MemoryKind.MEMORY)
    report.decayed = _decay(storage, memories, clock)  # mutates salience/confidence in-memory
    report.archived = _archive(storage, memories)
    report.contradictions_resolved = _resolve_contradictions(storage)

    log.info(
        "sleep: deduped=%d decayed=%d archived=%d contradictions=%d",
        report.deduped,
        report.decayed,
        report.archived,
        report.contradictions_resolved,
    )
    return report


def _dedup(storage: StorageGateway) -> int:
    memories = list_memories(storage, user=None, kind=MemoryKind.MEMORY)
    removed = 0

    # Exact duplicates: same normalized text and user.
    groups: dict[tuple[str, str | None], list[Memory]] = {}
    for mem in memories:
        groups.setdefault((_norm(mem.text), mem.user), []).append(mem)
    survivors: list[Memory] = []
    for group in groups.values():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        keeper = max(
            group, key=lambda m: (m.confidence, m.evidence_tier.multiplier, m.created_at)
        )
        losers = [m for m in group if m is not keeper]
        delete_memories(storage, [m.id for m in losers if m.id is not None])
        if keeper.id is not None:
            bump_memory(
                storage,
                keeper.id,
                access_count=sum(m.access_count for m in group),
                salience=max(m.salience for m in group),
            )
        removed += len(losers)
        survivors.append(keeper)

    removed += _near_dedup(storage, [m for m in survivors if m.embedding])
    return removed


def _near_dedup(storage: StorageGateway, mems: list[Memory]) -> int:
    removed = 0
    dropped: set[int] = set()
    for i, a in enumerate(mems):
        if a.id in dropped:
            continue
        for b in mems[i + 1 :]:
            if b.id in dropped or a.user != b.user:
                continue
            if cosine(a.embedding, b.embedding) >= NEAR_DUP_COSINE:
                keeper, loser = (
                    (a, b)
                    if (a.confidence, a.evidence_tier.multiplier)
                    >= (b.confidence, b.evidence_tier.multiplier)
                    else (b, a)
                )
                if loser.id is not None:
                    delete_memories(storage, [loser.id])
                    dropped.add(loser.id)
                if keeper.id is not None:
                    bump_memory(
                        storage,
                        keeper.id,
                        access_count=keeper.access_count + loser.access_count,
                        salience=max(keeper.salience, loser.salience),
                    )
                removed += 1
    return removed


def _decay(storage: StorageGateway, memories: list[Memory], now: float) -> int:
    updates: list[tuple[float, float, int]] = []
    for mem in memories:
        if mem.id is None:
            continue
        age_days = max(0.0, (now - mem.last_accessed) / _SECONDS_PER_DAY)
        new_salience = mem.salience * (0.5 ** (age_days / SALIENCE_HALF_LIFE_DAYS))
        new_confidence = mem.confidence
        if mem.evidence_tier.decays:
            new_confidence = mem.confidence * (0.5 ** (age_days / CONFIDENCE_HALF_LIFE_DAYS))
        if abs(new_salience - mem.salience) > 1e-6 or abs(new_confidence - mem.confidence) > 1e-6:
            updates.append((round(new_salience, 6), round(new_confidence, 6), mem.id))
            mem.salience, mem.confidence = new_salience, new_confidence  # for the archive decision
    apply_decay(storage, updates)
    return len(updates)


def _archive(storage: StorageGateway, memories: list[Memory]) -> int:
    # Only low-salience *provisional* memories — never archive a confident fact for going unused.
    ids = [
        mem.id
        for mem in memories
        if mem.id is not None
        and mem.salience < ARCHIVE_SALIENCE_FLOOR
        and mem.confidence < ARCHIVE_CONFIDENCE_CEILING
    ]
    return archive_memories(storage, ids)


def _resolve_contradictions(storage: StorageGateway) -> int:
    triples = browse_triples(storage, limit=10_000)
    groups: dict[tuple[str, str, str], list[Triple]] = {}
    for triple in triples:
        relation = _norm(triple.relation)
        if relation not in FUNCTIONAL_RELATIONS:
            continue  # only functional relations can contradict
        groups.setdefault((_norm(triple.subject), relation, triple.user or ""), []).append(triple)

    to_delete: list[int] = []
    for group in groups.values():
        if len({_norm(t.object) for t in group}) <= 1:
            continue  # all agree — no contradiction
        keeper = max(group, key=lambda t: (t.confidence, t.created_at))  # newest/best wins
        to_delete.extend(t.id for t in group if t is not keeper and t.id is not None)
    delete_triples(storage, to_delete)
    return len(to_delete)
