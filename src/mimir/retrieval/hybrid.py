"""Hybrid retrieval: keyword overlap + vector cosine, tier-weighted, salience-aware.

This is the retrieval discipline for the ``memory`` layer (DESIGN §3a). It blends two signals
so it never collapses when one is weak:

- **keyword** overlap — always available, the floor that keeps recall working in degraded
  (no-vector) mode.
- **vector** cosine — added when both the query and the memory carry embeddings.

The blend is then nudged by the **evidence tier** (a gentle tie-breaker — better-sourced
facts win at equal relevance, DESIGN §3b) and by **salience** (relevance-now; a memory that
has been useful lately surfaces a little more readily, DESIGN §3c). Confidence is deliberately
*not* a retrieval factor: truth ≠ relevance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..embed.base import cosine
from ..storage.models import Memory

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Below this blended relevance a memory is treated as not a real hit — it neither gets
# injected nor counts as a "source" for the uncertainty gate (DESIGN §3d).
MIN_RELEVANCE = 0.05

_KEYWORD_WEIGHT = 0.5
_VECTOR_WEIGHT = 0.5

# Function words carry no topical signal, so counting them as keyword overlap produces
# false matches (e.g. an unrelated question sharing only "is"/"the"/"of"). We drop them
# from *keyword* scoring; they still contribute to vector similarity, which is fine.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "have", "has", "had", "of", "to", "in", "on", "at",
        "for", "with", "and", "or", "but", "if", "as", "by", "from", "that", "this",
        "these", "those", "it", "its", "i", "you", "he", "she", "they", "we", "me",
        "my", "your", "his", "her", "their", "our", "what", "who", "whom", "whose",
        "when", "where", "why", "how", "which", "can", "could", "will", "would",
        "should", "about", "so", "not", "no", "yes",
    }
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _keyword_score(query_tokens: set[str], text: str) -> float:
    """Overlap coefficient of query tokens found in the memory text, in [0, 1]."""
    if not query_tokens:
        return 0.0
    mem_tokens = _tokens(text)
    if not mem_tokens:
        return 0.0
    overlap = len(query_tokens & mem_tokens)
    return overlap / len(query_tokens)


@dataclass(slots=True)
class ScoredMemory:
    memory: Memory
    score: float
    keyword: float
    vector: float


def retrieve(
    query: str,
    query_vec: list[float] | None,
    memories: list[Memory],
    *,
    top_k: int,
) -> list[ScoredMemory]:
    """Rank ``memories`` for ``query`` and return the top ``top_k`` above ``MIN_RELEVANCE``.

    ``query_vec`` is ``None`` in degraded mode; retrieval then leans entirely on keywords.
    Ties are broken by salience then recency so the ordering is stable and sensible.
    """
    query_tokens = _tokens(query)
    scored: list[ScoredMemory] = []
    for mem in memories:
        kw = _keyword_score(query_tokens, mem.text)
        vec = max(0.0, cosine(query_vec, mem.embedding)) if query_vec else 0.0

        if query_vec and mem.embedding:
            base = _KEYWORD_WEIGHT * kw + _VECTOR_WEIGHT * vec
        else:
            # No usable vector signal → keyword carries the whole score (degraded path).
            base = kw

        # Gentle adjustments: tier breaks ties, salience reflects relevance-now.
        relevance = base * mem.evidence_tier.multiplier * (0.7 + 0.3 * mem.salience)
        if relevance >= MIN_RELEVANCE:
            scored.append(ScoredMemory(memory=mem, score=relevance, keyword=kw, vector=vec))

    scored.sort(
        key=lambda s: (s.score, s.memory.salience, s.memory.created_at),
        reverse=True,
    )
    return scored[:top_k]
