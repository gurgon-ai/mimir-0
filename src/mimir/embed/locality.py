"""The bootstrap embedder: deterministic stdlib *locality hashing*.

This is the zero-dependency default that lets Mimir boot and recall on literally
Python + SQLite. It is the "hashing trick": text is reduced to lexical features
(word unigrams + intra-word character trigrams), each feature is hashed to a bucket and a
sign with a **stable** hash, and the signed weights are accumulated and L2-normalized.

Two things it is NOT, stated plainly so the code never oversells itself:

- It is **not semantic**. "car" and "automobile" share no features and land far apart.
  It captures lexical and sub-word overlap, which is enough to make hybrid retrieval
  *function* offline, not enough to call it semantic search.
- It must use a **stable** hash. Python's builtin ``hash()`` for ``str`` is salted per
  process, which would make embeddings non-reproducible across runs — a silent
  correctness bug. We use ``hashlib.blake2b`` so a given text always maps to the same
  vector, on any process, on any machine.
"""

from __future__ import annotations

import hashlib
import re

from .base import EmbeddingMode

_WORD_RE = re.compile(r"[a-z0-9]+")

# Feature weights: whole words carry more signal than character trigrams.
_UNIGRAM_WEIGHT = 1.0
_TRIGRAM_WEIGHT = 0.5


def _stable_bucket_sign(feature: str, dim: int) -> tuple[int, float]:
    """Map a feature string to a (bucket, sign) pair, reproducibly across processes."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    h = int.from_bytes(digest, "little")
    bucket = h % dim
    sign = 1.0 if (h >> 63) & 1 else -1.0
    return bucket, sign


def _features(text: str) -> list[tuple[str, float]]:
    feats: list[tuple[str, float]] = []
    for word in _WORD_RE.findall(text.lower()):
        feats.append((f"w:{word}", _UNIGRAM_WEIGHT))
        if len(word) >= 3:
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                feats.append((f"t:{padded[i:i + 3]}", _TRIGRAM_WEIGHT))
    return feats


class LocalityHashEmbedder:
    """A deterministic, dependency-free embedder for the bootstrap path.

    ``dim`` is the vector size; 256 is plenty for the bootstrap role and keeps the BLOB
    small. The output is L2-normalized so cosine similarity behaves.
    """

    mode = EmbeddingMode.BOOTSTRAP

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature, weight in _features(text):
            bucket, sign = _stable_bucket_sign(feature, self.dim)
            vec[bucket] += sign * weight
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return vec  # empty/symbol-only text → zero vector; cosine() treats it as no signal
        return [v / norm for v in vec]
