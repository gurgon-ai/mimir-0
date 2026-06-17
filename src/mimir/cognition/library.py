"""The Library layer (docs/LIBRARY.md): claim extraction + cited retrieval over the claim spine.

The DB holds short, atomic **claims**, each citing its source document + locator. These pure
functions distil a source passage into claims and rank claims for a turn; the brain wires
storage/model/embedder and attaches each claim's citation. Source documents (ground truth) and the
Markdown composites (the fuzzy understanding) live elsewhere — this is the cited spine between.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..embed.base import cosine
from ..prompts import CLAIM_EXTRACTION_SYSTEM
from ..storage.models import LibraryClaim

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# A claim must be genuinely on-topic to surface (a notch above the memory floor — claims are short
# and lexically overlap broadly).
MIN_RELEVANCE = 0.08
_MAX_UNIT_CHARS = 6000  # cap the passage handed to the model per extraction call


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def extract_claims(chat: Callable[[list[dict[str, str]]], str], unit_text: str) -> list[str]:
    """Distil a source passage into atomic claim sentences (the injected ``chat`` makes one call).
    Lenient JSON parse; ``[]`` on anything unparseable. The brain attaches the unit's locator."""
    reply = chat([
        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
        {"role": "user", "content": (unit_text or "")[:_MAX_UNIT_CHARS]},
    ]) or ""
    return _parse_claims(reply)


def _parse_claims(reply: str) -> list[str]:
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            return [str(c).strip() for c in data["claims"] if str(c).strip()]
    return []


@dataclass(slots=True)
class ScoredClaim:
    claim: LibraryClaim
    score: float


def retrieve_claims(
    query: str, query_vec: list[float] | None, claims: list[LibraryClaim], *, top_k: int
) -> list[ScoredClaim]:
    """Rank claims for ``query`` by keyword overlap + vector cosine. Degraded (no query vector) →
    keyword only, so it still works without embeddings."""
    q = _tokens(query)
    scored: list[ScoredClaim] = []
    for claim in claims:
        ct = _tokens(claim.text)
        kw = len(q & ct) / len(q) if q and ct else 0.0
        if query_vec and claim.embedding:
            score = 0.5 * kw + 0.5 * max(0.0, cosine(query_vec, claim.embedding))
        else:
            score = kw
        if score >= MIN_RELEVANCE:
            scored.append(ScoredClaim(claim=claim, score=score))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


def render_claims(scored: list[ScoredClaim], titles: dict[int, str]) -> str:
    """Format cited claims for the Library section: ``- <fact> [<title>, <locator>]``. ``titles``
    maps a claim's ``document_id`` → its source title, so every fact carries where it came from."""
    lines: list[str] = []
    for s in scored:
        c = s.claim
        cite = ", ".join(x for x in (titles.get(c.document_id, ""), c.locator) if x)
        lines.append(f"- {c.text}" + (f" [{cite}]" if cite else ""))
    return "\n".join(lines)
