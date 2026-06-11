"""The entity graph: connected knowledge via subject–relation–object triples (DESIGN §3a).

A typed retrieval layer distinct from flat memory recall. Where memory answers "what was said,"
the graph answers "what's *connected*." Triples are extracted at bake time and stored deduped;
at retrieval, the entities mentioned in a turn seed a 1–2 hop traversal, and the connected edges
are injected as their own attributed section.

Entity matching is deliberately simple for v0.1: an entity node matches a query if it appears in
it (case-insensitive, length ≥ 3 to avoid noise). That is enough to make the graph *function* —
NER-grade entity linking is a later refinement, not the point of the layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..storage.gateway import StorageGateway
from ..storage.models import Triple
from ..storage.repo import all_entities, save_triple, traverse_from_entities

log = logging.getLogger("mimir.graph")

_MIN_ENTITY_LEN = 3


def store_triples(
    storage: StorageGateway,
    raw_triples: Sequence[Sequence[str]],
    *,
    user: str | None,
    provenance: str = "conversation",
    confidence: float = 0.8,
) -> int:
    """Persist extracted ``[subject, relation, object]`` triples. Returns the count newly stored."""
    stored = 0
    for raw in raw_triples:
        if len(raw) != 3:
            continue
        subject, relation, obj = (str(p).strip() for p in raw)
        if not (subject and relation and obj):
            continue
        new_id = save_triple(
            storage,
            Triple(
                subject=subject,
                relation=relation,
                object=obj,
                user=user,
                provenance=provenance,
                confidence=confidence,
            ),
        )
        if new_id:  # 0 means it was a duplicate (ignored)
            stored += 1
    if stored:
        log.info("graph: stored %d new triple(s)", stored)
    return stored


def _seed_entities(storage: StorageGateway, query: str) -> list[str]:
    """Entity nodes that appear in the query (case-insensitive substring, length ≥ 3)."""
    ql = query.lower()
    return [e for e in all_entities(storage) if len(e) >= _MIN_ENTITY_LEN and e.lower() in ql]


def retrieve_connected(
    storage: StorageGateway,
    query: str,
    *,
    hops: int = 2,
    max_facts: int = 8,
    user: str | None = None,
) -> list[Triple]:
    """Triples connected (within ``hops``) to the query's entities, best-confidence first.

    Seeds on the query's entities, then expands hop by hop to their neighbours, deduping and
    capping at ``max_facts``. Returns ``[]`` if the query names no known entity.
    """
    seeds = _seed_entities(storage, query)
    if not seeds:
        return []

    seen_ids: set[int] = set()
    result: list[Triple] = []
    visited: set[str] = {e.lower() for e in seeds}
    frontier = seeds

    for _hop in range(max(1, hops)):
        triples = traverse_from_entities(storage, frontier, user=user, limit=max_facts * 3)
        next_frontier: list[str] = []
        for triple in triples:
            if triple.id is not None and triple.id not in seen_ids:
                seen_ids.add(triple.id)
                result.append(triple)
                if len(result) >= max_facts:
                    return result
            for entity in (triple.subject, triple.object):
                if entity.lower() not in visited:
                    visited.add(entity.lower())
                    next_frontier.append(entity)
        if not next_frontier:
            break
        frontier = next_frontier

    return result[:max_facts]


def render_triples(triples: list[Triple]) -> list[str]:
    """Render triples as readable connected facts for the prompt."""
    return [t.render() for t in triples]
