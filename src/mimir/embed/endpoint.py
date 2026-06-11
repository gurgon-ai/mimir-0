"""The non-bootstrap embedders: endpoint (real semantics) and null (degraded).

These complete the three-mode picture from ``base.py``. Both are thin: the endpoint embedder
defers to the model gateway's ``embed`` role; the null embedder produces no vectors so retrieval
falls back to keyword overlap.
"""

from __future__ import annotations

from ..model.gateway import ModelGateway
from .base import EmbeddingMode


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
