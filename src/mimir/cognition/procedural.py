"""Procedural memory: learned reasoning habits as trigger → procedure (DESIGN §3a).

Where the memory and graph layers hold *facts*, this layer holds *methods* — "when a situation
like X comes up, here's how to handle it." A procedure's trigger is embedded; when a turn matches
it (cosine + structural keyword overlap, nudged by how *proven* the habit is via its use count),
the procedure is injected as guidance.

v0.1 creates procedures by explicit teaching (``learn_procedure``); the retrieval/injection
mechanism is the durable core. Auto-extraction of habits from successful turns is a later layer on
top — not the procedural memory itself. Generic: triggers and procedures are whatever is taught.
"""

from __future__ import annotations

import logging
import re

from ..embed.base import Embedder, cosine
from ..storage.gateway import StorageGateway
from ..storage.models import Procedure
from ..storage.repo import list_procedures, save_procedure

log = logging.getLogger("mimir.procedural")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_MATCH = 0.3  # a trigger must be at least this relevant to a turn before its procedure fires

# Function words carry no trigger signal — drop them so a shared "the"/"a" can't fake a match.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "for", "to", "of", "is", "are", "was", "were", "be", "you", "me",
        "i", "it", "its", "my", "your", "can", "could", "please", "do", "does", "did", "what",
        "when", "where", "why", "how", "with", "in", "on", "at", "about", "and", "or", "this",
        "that", "would", "should", "give", "get",
    }
)


def learn_procedure(
    storage: StorageGateway,
    embedder: Embedder,
    *,
    trigger: str,
    procedure: str,
    user: str | None = None,
    confidence: float = 0.7,
) -> Procedure:
    """Teach a procedure: embed its trigger and store it."""
    proc = Procedure(
        trigger=trigger.strip(),
        procedure=procedure.strip(),
        trigger_embedding=embedder.embed(trigger),
        user=user,
        confidence=confidence,
    )
    save_procedure(storage, proc)
    log.info("procedural: learned a habit for trigger %r", proc.trigger[:60])
    return proc


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def retrieve_procedures(
    storage: StorageGateway,
    embedder: Embedder,
    query: str,
    *,
    top_k: int = 3,
    min_match: float = MIN_MATCH,
    user: str | None = None,
) -> list[Procedure]:
    """Procedures whose trigger matches the turn, best-first. cosine + keyword, boosted by uses."""
    procedures = list_procedures(storage, user=user)
    if not procedures:
        return []
    query_vec = embedder.embed(query)
    query_tokens = _tokens(query)

    scored: list[tuple[float, Procedure]] = []
    for proc in procedures:
        vec = max(0.0, cosine(query_vec, proc.trigger_embedding)) if query_vec else 0.0
        trigger_tokens = _tokens(proc.trigger)
        keyword = (
            len(query_tokens & trigger_tokens) / len(trigger_tokens) if trigger_tokens else 0.0
        )
        # Either signal can fire the match: strong keyword overlap (works in bootstrap mode) OR
        # strong cosine (works in endpoint mode). Truth ≠ relevance, so we take the stronger.
        base = max(vec, keyword)
        if base < min_match:
            continue
        # A proven habit (used often) surfaces a little more readily — the 'structural' signal.
        score = base * (1.0 + min(0.5, 0.05 * proc.uses))
        scored.append((score, proc))

    scored.sort(key=lambda s: s[0], reverse=True)
    return [proc for _, proc in scored[:top_k]]


def render_procedures(procedures: list[Procedure]) -> list[str]:
    """Render procedures as guidance lines for the prompt."""
    return [f"When {p.trigger}: {p.procedure}" for p in procedures]
