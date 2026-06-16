"""Baking: extract durable facts from a turn and persist them as attributed memories.

This is the write half of the §6 loop. The model (``bake`` role) extracts candidate facts;
**Mimir**, not the model, assigns the evidence tier — by *how the fact was sourced*, which the
model cannot know (DESIGN §3b). Each fact is embedded and written through the storage gateway.

Extraction is faithful-or-silent-but-logged: if the model returns something we can't parse, we
do NOT guess. We log the downgrade loudly and bake nothing this turn (an explicit, logged
downgrade — never a bare swallow; DESIGN §10).
"""

from __future__ import annotations

import json
import logging

from ..embed.base import Embedder
from ..model.gateway import ModelGateway
from ..prompts import BAKE_SYSTEM
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import save_memory
from .graph import store_triples

log = logging.getLogger("mimir.bake")


def _parse_bake(raw: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse ``{"facts": [...], "triples": [[s,r,o]]}`` from the reply. ``None`` if unparseable."""
    text = raw.strip()
    # Tolerate a model that wraps JSON in prose: take the outermost brace span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw_facts = data.get("facts")
    facts = (
        [str(f).strip() for f in raw_facts if str(f).strip()]
        if isinstance(raw_facts, list)
        else []
    )
    raw_triples = data.get("triples")
    triples: list[list[str]] = []
    if isinstance(raw_triples, list):
        for item in raw_triples:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append([str(p).strip() for p in item])
    return facts, triples


def _tier_and_provenance(
    user: str | None, primary_user: str | None, trusted_users: list[str] | None = None,
) -> tuple[EvidenceTier, str]:
    """Map *who said it* → an evidence tier (DESIGN §3b). A server-side trust policy: the caller
    declares the speaker (``user``), the config decides how much that speaker is believed — not the
    caller, so an open/exposed API can't let anyone launder claims into top-tier memory.

    - ``primary_user`` is the operator → ``STATED_BY_PRIMARY_USER`` (1.30).
    - ``trusted_users`` are additional believed identities → ``STATED_BY_TRUSTED`` (1.20).
    - any other *named* speaker (an unrecognized caller, a peer AI, a guest) → ``CONVERSATION``:
      attributed to them, but not treated as established fact.
    - ``user is None`` (unattributed call) → ``CONVERSATION``.

    Zero-config convenience: with NO policy set at all (no primary, no trusted list), the lone
    speaker IS treated as the primary — so a simple single-user build-your-own-UI just works.
    """
    trusted = trusted_users or ()
    if user is None:
        return EvidenceTier.CONVERSATION, "stated in conversation"
    if user == primary_user or (primary_user is None and not trusted):
        return EvidenceTier.STATED_BY_PRIMARY_USER, f"stated by {user}"
    if user in trusted:
        return EvidenceTier.STATED_BY_TRUSTED, f"stated by {user}"
    return EvidenceTier.CONVERSATION, f"stated by {user}"


def bake(
    model: ModelGateway,
    storage: StorageGateway,
    embedder: Embedder,
    *,
    turn_text: str,
    user: str | None,
    primary_user: str | None,
    trusted_users: list[str] | None = None,
) -> list[Memory]:
    """Extract, attribute, embed, and store durable facts from this turn's user text.

    Returns the memories actually written (possibly empty). Never raises on a model that
    misbehaves — it logs and bakes nothing, so a bad extraction can't break the turn.
    """
    raw = model.chat(
        "bake",
        [{"role": "system", "content": BAKE_SYSTEM}, {"role": "user", "content": turn_text}],
    )
    parsed = _parse_bake(raw)
    if parsed is None:
        log.warning(
            "bake: could not parse model reply; baking nothing this turn. "
            "Raw reply (truncated): %r",
            raw[:200],
        )
        return []
    facts, triples = parsed

    tier, provenance = _tier_and_provenance(user, primary_user, trusted_users)
    if triples:
        store_triples(storage, triples, user=user, provenance=provenance)
    stored: list[Memory] = []
    for fact in facts:
        mem = Memory(
            text=fact,
            kind=MemoryKind.MEMORY,
            evidence_tier=tier,
            confidence=0.9,  # the user stated it as true
            salience=1.0,  # freshly relevant
            embedding=embedder.embed(fact),
            provenance=provenance,
            user=user,
        )
        save_memory(storage, mem)
        stored.append(mem)
    if stored:
        log.info("bake: stored %d memor%s", len(stored), "y" if len(stored) == 1 else "ies")
    return stored
