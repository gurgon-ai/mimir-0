"""Shared fixtures: a temp DB, a mock-backed config, and a ready Mimir."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mimir.brain import Mimir
from mimir.config import Config, ProviderSpec, RoleSpec
from mimir.embed.base import EmbeddingMode


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "mimir.db")


@pytest.fixture
def mock_config(db_path: str) -> Config:
    role = RoleSpec(model="mock")
    return Config(
        storage_path=db_path,
        roles={"chat": role, "bake": role, "reasoning": role},
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )


@pytest.fixture
def brain(mock_config: Config) -> Iterator[Mimir]:
    m = Mimir(mock_config)
    try:
        yield m
    finally:
        m.close()
