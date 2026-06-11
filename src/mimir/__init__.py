"""Mimir 0 — a local-first cognition core.

Typed, evidence-tiered, provenance-tracked memory assembled into prompts with explicit
epistemics. Import ``Mimir``, hand it a config, call ``.turn()``. See ``DESIGN.md`` for the
architecture and ``docs/SETUP.md`` to get running.
"""

from __future__ import annotations

from .brain import Mimir, TurnResult, build_provider, make_embedder
from .cognition.ingest import IngestResult
from .config import Config, ProviderSpec, RoleSpec, load_config
from .context.build import ContextBundle, build_context
from .embed.base import EmbeddingMode
from .errors import (
    ConfigError,
    ContextBudgetError,
    IngestError,
    MigrationError,
    MimirError,
    ModelGatewayError,
    ProviderError,
    SchemaError,
    SelfTestError,
    StorageError,
)
from .storage.models import EvidenceTier, Memory, MemoryKind

__version__ = "0.0.1"

__all__ = [
    "Mimir",
    "TurnResult",
    "IngestResult",
    "Config",
    "RoleSpec",
    "ProviderSpec",
    "load_config",
    "build_provider",
    "make_embedder",
    "build_context",
    "ContextBundle",
    "EmbeddingMode",
    "Memory",
    "MemoryKind",
    "EvidenceTier",
    "MimirError",
    "ConfigError",
    "IngestError",
    "SchemaError",
    "MigrationError",
    "StorageError",
    "ProviderError",
    "ModelGatewayError",
    "ContextBudgetError",
    "SelfTestError",
    "__version__",
]
