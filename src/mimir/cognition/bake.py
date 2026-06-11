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

log = logging.getLogger("mimir.bake")


def _extract_facts(raw: str) -> list[str] | None:
    """Parse ``{"facts": [...]}`` out of the model's reply. ``None`` means unparseable."""
    text = raw.strip()
    # Tolerate a model that wraps JSON in prose: take the outermost brace span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return None
    return [str(f).strip() for f in facts if str(f).strip()]


def _tier_and_provenance(user: str | None, primary_user: str | None) -> tuple[EvidenceTier, str]:
    """Assign the evidence tier from the source of the statement (DESIGN §3b)."""
    if user is None:
        return EvidenceTier.CONVERSATION, "stated in conversation"
    # Single-user mode (no configured primary): the one speaker IS the primary user.
    if primary_user is None or user == primary_user:
        return EvidenceTier.STATED_BY_PRIMARY_USER, f"stated by {user}"
    return EvidenceTier.STATED_BY_TRUSTED, f"stated by {user}"


def bake(
    model: ModelGateway,
    storage: StorageGateway,
    embedder: Embedder,
    *,
    turn_text: str,
    user: str | None,
    primary_user: str | None,
) -> list[Memory]:
    """Extract, attribute, embed, and store durable facts from this turn's user text.

    Returns the memories actually written (possibly empty). Never raises on a model that
    misbehaves — it logs and bakes nothing, so a bad extraction can't break the turn.
    """
    raw = model.chat(
        "bake",
        [{"role": "system", "content": BAKE_SYSTEM}, {"role": "user", "content": turn_text}],
    )
    facts = _extract_facts(raw)
    if facts is None:
        log.warning(
            "bake: could not parse facts from model reply; baking nothing this turn. "
            "Raw reply (truncated): %r",
            raw[:200],
        )
        return []

    tier, provenance = _tier_and_provenance(user, primary_user)
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
