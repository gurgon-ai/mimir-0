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

# Sentinel a role's `model` takes when the user wants automatic selection (DESIGN §4): the brain
# resolves it from the fleet (measured-best > approved-family heuristic > any reachable model). A
# role with no `model` entry defaults to this — "as automatic as possible, but configurable."
AUTO_MODEL = "auto"


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
class BackendConfig:
    """The distributed Ollama fleet (DESIGN §5). Nodes need zero setup — just ``ollama serve``.

    Discovery = localhost + explicit ``nodes`` + (when ``lan_backend``) a subnet scan of :11434.
    """

    lan_backend: bool = False  # scan the LAN for Ollama nodes
    subnet: str | None = None  # CIDR to scan; None → auto-detect the local /24
    nodes: list[str] = field(default_factory=list)  # explicit node hosts/urls
    scan_timeout_s: float = 0.5
    scan_concurrency: int = 64
    refresh_interval_s: float = 60.0  # active health/inventory refresh; 0 disables the prober
    # Speed-aware routing (DESIGN §5). Node speed is learned PASSIVELY from real calls — no wasted
    # synthetic calls — and only topped up by a rare idle heartbeat so quiet nodes don't go stale.
    idle_probe_interval_s: float = 1800.0  # idle latency heartbeat (s); 0 disables it. Real traffic
                                           # is the primary signal; this is a 30-min top-up,
                                           # decoupled from the faster health refresh above.
    latency_alpha: float = 0.3  # EWMA weight on the newest sample (higher = tracks current load)
    # Only the user knows their hardware + latency tolerance, so these are user knobs (UI fields):
    max_model_size_b: float = 30.0  # don't benchmark/route models bigger than this (params B)
    min_model_size_b: float = 0.0   # don't benchmark/route models SMALLER than this; 0 = off. On
                                    # capable hardware, keeps a tiny model that scores 'high enough'
                                    # and wins on latency from out-competing a bigger, better one.
    max_latency_s: float = 0.0      # routing latency target; 0 = off. Slower models are excluded.
    # Context window for ALL benchmark calls — set explicitly so Ollama doesn't silently fall back
    # to its tiny 2048 default and truncate the layered epistemic prompts (which would invalidate
    # the tier-deference gauntlet: the high-tier fact could be cut off). This is the OPERATIONAL
    # window: qualify at the size you deploy at, or the warm KV cache rebuilds on the first real
    # turn and the benchmark's latencies become lies. Held consistent across every benchmarked model
    # so qualification is fair. The long-context probe sizes its haystack to ~60% of this, so it
    # tests the window you'll actually run. Default 24576 (24k) — a proven operational window for a
    # RAG + compression system; a fraction of models' 128k–256k theoretical max, which you neither
    # need nor want (KV-cache cost + "lost in the middle"). Continuity comes from curated RAG memory
    # and compression, not a giant raw window.
    benchmark_num_ctx: int = 24576


@dataclass(slots=True)
class Config:
    storage_path: str
    roles: dict[str, RoleSpec]
    provider: ProviderSpec
    # When set, the model gateway is built from a discovered/declared Ollama fleet instead of the
    # single ``provider``. ``provider`` is then just the adapter type (ollama).
    backend: BackendConfig | None = None
    identity: str = DEFAULT_IDENTITY
    # The owner whose statements earn the top evidence tier. If None, Mimir runs in
    # single-user mode and treats whoever speaks as the primary user (DESIGN §3b).
    primary_user: str | None = None
    # Foundational identity anchors (name/operator/location/purpose), set declaratively here
    # for non-interactive deployments. Re-established (upserted) at boot. The interview sets
    # the same anchors interactively. See cognition/identity.py.
    identity_anchors: dict[str, Any] = field(default_factory=dict)
    embed_mode: EmbeddingMode = EmbeddingMode.BOOTSTRAP
    embed_dim: int = 256
    # Per chat-turn token budget for the assembled prompt (context accounting, DESIGN §10).
    context_budget_tokens: int = 4096
    # How often (in turns) the self-model is re-synthesized off the hot path. 0 disables it
    # (the seed identity is then the whole self-model). The first turn always seeds one.
    self_model_refresh_every: int = 5
    # How often (in turns) working memory's rolling summary is re-synthesized (folding the
    # accumulated exchanges). 0 disables compression (recency-only working memory).
    working_memory_refresh_every: int = 4
    # Entity-graph traversal: how many hops from the query's entities, and the max connected
    # facts to inject. hops=0 disables graph retrieval (triples are still extracted/stored).
    graph_hops: int = 2
    graph_max_facts: int = 8
    # How often (in turns) consolidation (sleep) runs off the hot path. 0 = manual only
    # (call brain.sleep() or the web UI button / scheduler).
    sleep_every: int = 0
    # Procedural memory: how many matching procedures to inject, and the minimum trigger match.
    procedural_top_k: int = 3
    procedural_min_match: float = 0.3

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
        # A missing or "auto" model means automatic selection from the fleet (DESIGN §4).
        model = str(table.get("model", AUTO_MODEL))
        params = {k: v for k, v in table.items() if k != "model"}
        roles[name] = RoleSpec(model=model, params=params)
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

    backend_raw = raw.get("backend")
    backend: BackendConfig | None = None
    if backend_raw:
        backend = BackendConfig(
            lan_backend=bool(backend_raw.get("lan_backend", False)),
            subnet=str(backend_raw["subnet"]) if "subnet" in backend_raw else None,
            nodes=[str(n) for n in backend_raw.get("nodes", [])],
            scan_timeout_s=float(backend_raw.get("scan_timeout_s", 0.5)),
            scan_concurrency=int(backend_raw.get("scan_concurrency", 64)),
            refresh_interval_s=float(backend_raw.get("refresh_interval_s", 60.0)),
            idle_probe_interval_s=float(backend_raw.get("idle_probe_interval_s", 1800.0)),
            latency_alpha=float(backend_raw.get("latency_alpha", 0.3)),
            max_model_size_b=float(backend_raw.get("max_model_size_b", 30.0)),
            min_model_size_b=float(backend_raw.get("min_model_size_b", 0.0)),
            max_latency_s=float(backend_raw.get("max_latency_s", 0.0)),
            benchmark_num_ctx=int(backend_raw.get("benchmark_num_ctx", 8192)),
        )

    provider_raw = raw.get("provider")
    if provider_raw and "type" in provider_raw:
        provider = ProviderSpec(
            type=str(provider_raw["type"]),
            options={k: v for k, v in provider_raw.items() if k != "type"},
        )
    elif backend is not None:
        provider = ProviderSpec(type="ollama")  # a fleet is Ollama nodes by definition
    else:
        raise ConfigError('config is missing [provider] type = "ollama" (or "mock")')

    identity_raw = raw.get("identity", {})
    anchor_keys = (
        "name", "operator", "location", "purpose", "values", "scope", "boundaries", "voice"
    )
    config = Config(
        storage_path=str(storage["path"]),
        roles=_parse_roles(raw.get("roles", {})),
        provider=provider,
        backend=backend,
        identity=str(identity_raw.get("text", DEFAULT_IDENTITY)),
        primary_user=(
            str(identity_raw["primary_user"]) if "primary_user" in identity_raw else None
        ),
        identity_anchors={
            k: str(identity_raw[k]) for k in anchor_keys if k in identity_raw
        },
        embed_mode=embed_mode,
        embed_dim=int(embeddings.get("dim", 256)),
        context_budget_tokens=int(raw.get("context", {}).get("budget_tokens", 4096)),
        self_model_refresh_every=int(
            raw.get("self_model", {}).get("refresh_every", 5)
        ),
        working_memory_refresh_every=int(
            raw.get("working_memory", {}).get("refresh_every", 4)
        ),
        graph_hops=int(raw.get("entity_graph", {}).get("hops", 2)),
        graph_max_facts=int(raw.get("entity_graph", {}).get("max_facts", 8)),
        sleep_every=int(raw.get("sleep", {}).get("every", 0)),
        procedural_top_k=int(raw.get("procedural", {}).get("top_k", 3)),
        procedural_min_match=float(raw.get("procedural", {}).get("min_match", 0.3)),
    )
    config.validate()
    return config
