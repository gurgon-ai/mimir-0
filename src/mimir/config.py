"""Configuration: the single place role→model mapping and tuned params live (DESIGN §4).

Model choice is *never* hardcoded. The brain reads roles from here. A ``mimir.toml`` is
parsed with stdlib ``tomllib`` (no dependency), but ``Config`` is also a plain dataclass so
tests and embedders can build one in code without a file.

Validation fails loud (DESIGN §10): a missing required role, an ``endpoint`` embed mode with
no ``[roles.embed]``, or an unknown provider type raises ``ConfigError`` with an instruction —
never a silent default that changes behavior behind the user's back.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embed.base import EmbeddingMode
from .errors import ConfigError
from .prompts import DEFAULT_IDENTITY

# Roles the v0 spine cannot run without. `embed` is required only in endpoint mode.
REQUIRED_ROLES = ("chat", "bake", "reasoning")


@dataclass(slots=True)
class RoleSpec:
    """A cognitive role's bound model and its tuned parameters (DESIGN §4)."""

    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderSpec:
    """How to construct the provider behind the model gateway."""

    type: str  # "ollama" | "mock"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    storage_path: str
    roles: dict[str, RoleSpec]
    provider: ProviderSpec
    identity: str = DEFAULT_IDENTITY
    # The owner whose statements earn the top evidence tier. If None, Mimir runs in
    # single-user mode and treats whoever speaks as the primary user (DESIGN §3b).
    primary_user: str | None = None
    embed_mode: EmbeddingMode = EmbeddingMode.BOOTSTRAP
    embed_dim: int = 256
    # Per chat-turn token budget for the assembled prompt (context accounting, DESIGN §10).
    context_budget_tokens: int = 4096
    # How often (in turns) the self-model is re-synthesized off the hot path. 0 disables it
    # (the seed identity is then the whole self-model). The first turn always seeds one.
    self_model_refresh_every: int = 5

    def validate(self) -> None:
        """Raise ``ConfigError`` loud on any unrunnable configuration."""
        missing = [r for r in REQUIRED_ROLES if r not in self.roles]
        if missing:
            raise ConfigError(
                f"config is missing required role(s): {missing}. Add a [roles.<name>] "
                f"table with a `model` for each of {list(REQUIRED_ROLES)}."
            )
        if self.embed_mode is EmbeddingMode.ENDPOINT and "embed" not in self.roles:
            raise ConfigError(
                "embeddings.mode = 'endpoint' requires a [roles.embed] table naming the "
                "embeddings model. Add one, or switch to embeddings.mode = 'bootstrap'."
            )
        if self.embed_dim <= 0:
            raise ConfigError(f"embeddings.dim must be positive, got {self.embed_dim}")


def _parse_roles(raw: dict[str, Any]) -> dict[str, RoleSpec]:
    roles: dict[str, RoleSpec] = {}
    for name, table in raw.items():
        if not isinstance(table, dict):
            raise ConfigError(f"[roles.{name}] must be a table")
        if "model" not in table:
            raise ConfigError(f"[roles.{name}] is missing a `model` entry")
        params = {k: v for k, v in table.items() if k != "model"}
        roles[name] = RoleSpec(model=str(table["model"]), params=params)
    return roles


def load_config(path: str | Path) -> Config:
    """Load and validate a ``mimir.toml``. Raises ``ConfigError`` on any problem."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"config file not found: {p}. See docs/SETUP.md and mimir.toml.example to "
            f"create one."
        )
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {p}: {exc}") from exc

    storage = raw.get("storage", {})
    if "path" not in storage:
        raise ConfigError("config is missing [storage] path = \"...\"")

    embeddings = raw.get("embeddings", {})
    mode_key = str(embeddings.get("mode", "bootstrap"))
    try:
        embed_mode = EmbeddingMode(mode_key)
    except ValueError as exc:
        valid = ", ".join(m.value for m in EmbeddingMode)
        raise ConfigError(
            f"unknown embeddings.mode {mode_key!r}; valid values: {valid}"
        ) from exc

    provider_raw = raw.get("provider")
    if not provider_raw or "type" not in provider_raw:
        raise ConfigError('config is missing [provider] type = "ollama" (or "mock")')
    provider = ProviderSpec(
        type=str(provider_raw["type"]),
        options={k: v for k, v in provider_raw.items() if k != "type"},
    )

    identity_raw = raw.get("identity", {})
    config = Config(
        storage_path=str(storage["path"]),
        roles=_parse_roles(raw.get("roles", {})),
        provider=provider,
        identity=str(identity_raw.get("text", DEFAULT_IDENTITY)),
        primary_user=(
            str(identity_raw["primary_user"]) if "primary_user" in identity_raw else None
        ),
        embed_mode=embed_mode,
        embed_dim=int(embeddings.get("dim", 256)),
        context_budget_tokens=int(raw.get("context", {}).get("budget_tokens", 4096)),
        self_model_refresh_every=int(
            raw.get("self_model", {}).get("refresh_every", 5)
        ),
    )
    config.validate()
    return config
