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
from typing import Any

from ..storage.gateway import StorageGateway
from ..storage.models import MemoryKind, Triple
from ..storage.repo import (
    all_entities,
    browse_triples,
    list_memories,
    save_triple,
    traverse_from_entities,
)

log = logging.getLogger("mimir.graph")

_MIN_ENTITY_LEN = 3


def build_graph_map(
    storage: StorageGateway, *, memory_limit: int = 60, triple_limit: int = 300
) -> dict[str, list[dict[str, Any]]]:
    """A node/link map for the visual memory graph (DESIGN §3a).

    Nodes are **memory blobs** (the salient, non-archived memories — clickable/editable) plus the
    **entities** from the triple graph; links are the relation edges (entity—relation→entity) and a
    light "mentions" edge from a memory to any entity whose name appears in its text. Capped to the
    most salient memories so the graph stays legible. Pure read — the UI lays it out.
    """
    mems = [m for m in list_memories(storage, kind=MemoryKind.MEMORY) if not m.archived]
    mems.sort(key=lambda m: -(m.salience or 0.0))
    mems = mems[:memory_limit]
    triples = browse_triples(storage, limit=triple_limit)

    entity_id: dict[str, str] = {}  # lowercased name → node id
    entity_label: dict[str, str] = {}
    for t in triples:
        for name in (t.subject, t.object):
            low = name.lower()
            if len(low) >= _MIN_ENTITY_LEN:
                entity_id.setdefault(low, f"e:{low}")
                entity_label.setdefault(low, name)

    nodes: list[dict[str, Any]] = []
    for m in mems:
        text = " ".join(m.text.split())
        nodes.append({
            "id": f"m{m.id}", "type": "memory", "mid": m.id,
            "label": text[:42] + ("…" if len(text) > 42 else ""), "text": text,
            "tier": m.evidence_tier.key, "salience": round(m.salience, 3),
            "provenance": m.provenance,
        })
    for low, nid in entity_id.items():
        nodes.append({"id": nid, "type": "entity", "label": entity_label[low]})

    links: list[dict[str, Any]] = []
    for t in triples:
        s, o = t.subject.lower(), t.object.lower()
        if s in entity_id and o in entity_id and s != o:
            links.append({"source": entity_id[s], "target": entity_id[o], "label": t.relation})
    for m in mems:
        low_text = m.text.lower()
        for low, nid in entity_id.items():
            if low in low_text:
                links.append({"source": f"m{m.id}", "target": nid, "label": "mentions"})

    return {"nodes": nodes, "links": links}


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
