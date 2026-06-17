"""The non-bootstrap embedders: endpoint (real semantics) and null (degraded).

These complete the three-mode picture from ``base.py``. Both are thin: the endpoint embedder
defers to the model gateway's ``embed`` role; the null embedder produces no vectors so retrieval
falls back to keyword overlap.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ..model.gateway import ModelGateway
from .base import Embedder, EmbeddingMode

log = logging.getLogger("mimir.embed")


class EndpointEmbedder:
    """Real semantic embeddings via the configured ``embed`` role on the model gateway."""

    mode = EmbeddingMode.ENDPOINT

    def __init__(self, model: ModelGateway, role: str = "embed") -> None:
        self._model = model
        self._role = role

    def embed(self, text: str) -> list[float] | None:
        vectors = self._model.embed(self._role, [text])
        return vectors[0] if vectors else None


class NullEmbedder:
    """No vectors at all — the degraded, keyword-only path."""

    mode = EmbeddingMode.DEGRADED

    def embed(self, text: str) -> list[float] | None:
        return None


class ResilientEmbedder:
    """Wrap an embedder so a backend *outage* degrades loudly to keyword recall instead of crashing
    the turn (DESIGN §10: fail loud, keep working). If the wrapped embedder raises (e.g. the embed
    node is down, or its model isn't installed), this logs a throttled warning — captured into the
    error ring, so it surfaces in the turn's context and the Mind tab — and returns ``None``, which
    every caller already treats as the keyword-only path. ``mode`` delegates to the inner embedder,
    so this is a *transient* degradation: semantic recall, dedup, and conflict surfacing resume on
    their own the moment the backend returns. Not a silent backend swap — the store is the same and
    the failure is announced; it just stops one downed node from taking the whole brain down."""

    def __init__(
        self, inner: Embedder, *, clock: Callable[[], float] = time.monotonic,
        warn_every_s: float = 60.0,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._warn_every_s = warn_every_s
        self._last_warn = -1e9

    @property
    def mode(self) -> EmbeddingMode:
        return self._inner.mode

    def embed(self, text: str) -> list[float] | None:
        try:
            return self._inner.embed(text)
        except Exception as exc:  # an embedding-backend outage must not crash the turn (§10)
            now = self._clock()
            if now - self._last_warn >= self._warn_every_s:  # throttle: loud, not a spew
                log.warning("embeddings unavailable — degrading to keyword recall: %s", exc)
                self._last_warn = now
            return None
