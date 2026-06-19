"""Sleep / consolidation: memory that maintains itself (DESIGN §1, §5).

A batch maintenance pass over the store — mostly deterministic, no model call — that keeps memory
healthy without a human curating it:

- **dedup** — exact and near-duplicate (cosine) memories merged into the best-sourced survivor,
  summing access counts.
- **decay** — salience decays with disuse (drives forgetting), and **faster for the decaying tiers**
  (conversational + inferred) than for authority/document facts, so peer chatter and self-generated
  rumination go dormant within weeks while a primary-user fact lingers for months. **Confidence
  decays only for those same low tiers**, never authority facts — so a true-but-unused fact loses
  salience but not truth (DESIGN §3c: don't let "haven't used it lately" masquerade as "probably
  false").
- **archive** — only *decaying-tier* memories (conversation/inferred) that have decayed below the
  salience floor are archived (excluded from active recall, kept in the store; archiving ≠
  disbelieving — confidence is preserved). Authority-tier and document facts are never archived for
  disuse, which avoids the death spiral the design warns about.
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
    prune_forum_threads,
    prune_kind,
)

log = logging.getLogger("mimir.sleep")

_SECONDS_PER_DAY = 86_400.0
SALIENCE_HALF_LIFE_DAYS = 30.0
# The decaying tiers (conversation/inferred — peer chatter + self-generated rumination) lose
# salience faster, going dormant in weeks of disuse instead of months. This is what makes the
# salience axis actually distil: low-value provisional content fades and archives; authority stays.
PROVISIONAL_SALIENCE_HALF_LIFE_DAYS = 10.0
CONFIDENCE_HALF_LIFE_DAYS = 120.0  # gentler; only applied to decaying tiers
ARCHIVE_SALIENCE_FLOOR = 0.05
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
    pruned: int = 0  # stale single-latest-wins rows (working-memory/self-model versions) tidied
    forum_pruned: int = 0  # old council deliberations trimmed from the browsable forum

    @property
    def total_changes(self) -> int:
        return (self.deduped + self.decayed + self.archived
                + self.contradictions_resolved + self.pruned + self.forum_pruned)


# Working-memory, self-model, and sentinel-note rows accumulate one-per-turn/synthesis but only the
# latest of each is ever injected (fetched by recency) and they're separate kinds (never recalled) —
# so old versions are pure dead weight. Keep a handful for introspection; sleep prunes the rest.
WORKING_MEMORY_KEEP = 2
SELF_MODEL_KEEP = 3
SENTINEL_NOTE_KEEP = 10  # the "reflections" view shows recent ones; older are unused dead weight
# The council forum is a browsable history of deliberations; each verdict is also a recallable
# memory, so the forum can be recency-bounded like every other aux store without losing knowledge.
FORUM_THREAD_KEEP = 200


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
    report.pruned = (
        prune_kind(storage, MemoryKind.WORKING_MEMORY, WORKING_MEMORY_KEEP)
        + prune_kind(storage, MemoryKind.SELF_MODEL, SELF_MODEL_KEEP)
        + prune_kind(storage, MemoryKind.SENTINEL_NOTE, SENTINEL_NOTE_KEEP)
    )
    report.forum_pruned = prune_forum_threads(storage, FORUM_THREAD_KEEP)

    log.info(
        "sleep: deduped=%d decayed=%d archived=%d contradictions=%d pruned=%d forum_pruned=%d",
        report.deduped,
        report.decayed,
        report.archived,
        report.contradictions_resolved,
        report.pruned,
        report.forum_pruned,
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
        half_life = (PROVISIONAL_SALIENCE_HALF_LIFE_DAYS if mem.evidence_tier.decays
                     else SALIENCE_HALF_LIFE_DAYS)
        new_salience = mem.salience * (0.5 ** (age_days / half_life))
        new_confidence = mem.confidence
        if mem.evidence_tier.decays:
            new_confidence = mem.confidence * (0.5 ** (age_days / CONFIDENCE_HALF_LIFE_DAYS))
        if abs(new_salience - mem.salience) > 1e-6 or abs(new_confidence - mem.confidence) > 1e-6:
            updates.append((round(new_salience, 6), round(new_confidence, 6), mem.id))
            mem.salience, mem.confidence = new_salience, new_confidence  # for the archive decision
    apply_decay(storage, updates)
    return len(updates)


def _archive(storage: StorageGateway, memories: list[Memory]) -> int:
    # Only decaying-tier memories (conversation/inferred) that have faded below the salience floor.
    # Never archive an authority-tier or document fact for going unused (no death spiral); archiving
    # preserves confidence — it removes from active recall, it does not disbelieve (DESIGN §3c).
    ids = [
        mem.id
        for mem in memories
        if mem.id is not None
        and mem.salience < ARCHIVE_SALIENCE_FLOOR
        and mem.evidence_tier.decays
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
