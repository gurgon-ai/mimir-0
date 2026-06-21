"""The embedding seam and its three honest modes.

The kickoff decision (see ``docs/SETUP.md``): a bundled zero-dependency default, with a
configurable endpoint that fully replaces it, and a keyword-only degraded fall-through.
Three modes, named honestly so no one mistakes the cheap bootstrap path for poor memory:

- **bootstrap** — a pure-stdlib *locality-hashing* embedder. Deterministic, offline, zero
  deps. NOT semantic search; it captures lexical/character overlap, nothing more. It exists
  so Mimir boots and recalls on literally Python + SQLite.
- **endpoint** — a real embeddings model reached through the model gateway. When configured,
  it *replaces* bootstrap entirely. This is the recommended path for real use.
- **degraded** — no vectors at all. Retrieval falls back to keyword overlap. For environments
  where even the bootstrap vector path is unwanted.

The active mode is surfaced loudly (startup log + ``build_context()`` introspection), per the
fail-loud doctrine (DESIGN §10): a quiet "it works, but badly" is exactly the silence we guard
against.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol, runtime_checkable


class EmbeddingMode(Enum):
    """Which retrieval-vector strategy is live. Reported, never hidden."""

    BOOTSTRAP = "bootstrap"
    ENDPOINT = "endpoint"
    DEGRADED = "degraded"

    @property
    def is_semantic(self) -> bool:
        """True only for ``endpoint`` — the only mode that does real semantic embedding."""
        return self is EmbeddingMode.ENDPOINT

    def banner(self) -> str:
        """A one-line, honest description for logs and introspection."""
        return {
            EmbeddingMode.BOOTSTRAP: (
                "bootstrap (stdlib locality-hashing; lexical overlap only, NOT semantic "
                "search — configure [roles.embed] for real semantic recall)"
            ),
            EmbeddingMode.ENDPOINT: "endpoint (real semantic embeddings via the model gateway)",
            EmbeddingMode.DEGRADED: "degraded (no vectors; keyword-only retrieval)",
        }[self]


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a retrieval vector — or decline to (degraded)."""

    @property
    def mode(self) -> EmbeddingMode:
        """The embedder's mode (bootstrap/endpoint/degraded). A property so both plain-attribute
        and @property implementers satisfy the protocol (e.g. the ResilientEmbedder wrapper)."""
        ...

    def embed(self, text: str) -> list[float] | None:
        """Return a vector for ``text``, or ``None`` if this embedder produces no vectors."""
        ...


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either side is missing or zero-length.

    Length-mismatched vectors return 0.0 rather than raising — a mismatch means the
    store was embedded under a different model than the query, which the caller handles
    as 'no semantic signal' (and which the embed-mode banner makes visible).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
