"""Executable spec for the three embedding modes and their honest reporting."""

from __future__ import annotations

import hashlib

from mimir.brain import Mimir
from mimir.config import Config
from mimir.embed.base import EmbeddingMode, cosine
from mimir.embed.endpoint import EndpointEmbedder, NullEmbedder
from mimir.embed.locality import LocalityHashEmbedder


def test_bootstrap_is_deterministic_and_stable_hashed() -> None:
    a = LocalityHashEmbedder(dim=256)
    b = LocalityHashEmbedder(dim=256)
    v1 = a.embed("My favorite color is teal")
    v2 = b.embed("My favorite color is teal")
    assert v1 == v2  # deterministic across instances

    # Guard against accidental use of the salted builtin hash(): the bucket for a known
    # feature must match an independent stable computation (blake2b), proving reproducibility.
    digest = hashlib.blake2b(b"w:teal", digest_size=8).digest()
    expected_bucket = int.from_bytes(digest, "little") % 256
    assert v1[expected_bucket] != 0.0


def test_bootstrap_captures_lexical_overlap_not_semantics() -> None:
    e = LocalityHashEmbedder(dim=256)
    teal1 = e.embed("my favorite color is teal")
    teal2 = e.embed("teal is my favorite color")
    france = e.embed("the capital of france is paris")
    # Lexical overlap ranks higher than an unrelated sentence...
    assert cosine(teal1, teal2) > cosine(teal1, france)
    # ...but it is NOT semantic: a synonym with no shared tokens is essentially unrelated.
    automobile = e.embed("i drive an automobile")
    car = e.embed("i own a car")
    assert cosine(automobile, car) < 0.5


def test_modes_report_honestly() -> None:
    assert EmbeddingMode.ENDPOINT.is_semantic
    assert not EmbeddingMode.BOOTSTRAP.is_semantic
    assert "NOT semantic" in EmbeddingMode.BOOTSTRAP.banner()
    assert "keyword-only" in EmbeddingMode.DEGRADED.banner()


def test_cosine_handles_missing_and_mismatched() -> None:
    assert cosine(None, [1.0]) == 0.0
    assert cosine([1.0, 2.0], [1.0]) == 0.0  # length mismatch → no signal
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_endpoint_mode_uses_provider(mock_config: Config) -> None:
    from mimir.config import RoleSpec

    mock_config.embed_mode = EmbeddingMode.ENDPOINT
    mock_config.roles["embed"] = RoleSpec(model="mock")
    with Mimir(mock_config) as m:
        assert isinstance(m._embedder, EndpointEmbedder)
        vec = m._embedder.embed("hello world")
        assert vec is not None and len(vec) == 64  # the mock's embedding dim


def test_degraded_mode_produces_no_vectors(mock_config: Config) -> None:
    mock_config.embed_mode = EmbeddingMode.DEGRADED
    with Mimir(mock_config) as m:
        assert isinstance(m._embedder, NullEmbedder)
        assert m._embedder.embed("anything") is None
