"""Configuration: the single place role→model mapping and tuned params live (DESIGN §4).

Model choice is *never* hardcoded. The brain reads roles from here. A ``mimir.toml`` is
parsed with stdlib ``tomllib`` (no dependency), but ``Config`` is also a plain dataclass so
tests and embedders can build one in code without a file.

Validation fails loud (DESIGN §10): a missing required role, an ``endpoint`` embed mode with
no ``[roles.embed]``, or an unknown provider type raises ``ConfigError`` with an instruction —
never a silent default that changes behavior behind the user's back.
"""

from __future__ import annotations

import os
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
class WikiConfig:
    """An optional offline-reference source: a Kiwix server over a ZIM (DESIGN §9). Zero Python
    dependency — Mimir talks to ``kiwix-serve`` with stdlib HTTP, like any other local endpoint. The
    user downloads any ZIM (Wikipedia nopic, a medical wiki, top-50k, …), runs ``kiwix-serve`` over
    it, and points ``url`` + ``book`` here; it's queried live and injected as an attributed section.
    """

    enabled: bool = False
    url: str = "http://localhost:8080"   # the kiwix-serve base URL
    book: str = ""                       # the ZIM's book name (as kiwix-serve lists it)
    max_articles: int = 2                # how many top hits to inject per turn
    max_chars: int = 800                 # chars of each article's lead text
    timeout_s: float = 2.0               # hard cap so a slow/missing wiki never stalls a turn


@dataclass(slots=True)
class Config:
    storage_path: str
    roles: dict[str, RoleSpec]
    provider: ProviderSpec
    # When set, the model gateway is built from a discovered/declared Ollama fleet instead of the
    # single ``provider``. ``provider`` is then just the adapter type (ollama).
    backend: BackendConfig | None = None
    # Optional offline encyclopedia (Kiwix/ZIM over HTTP) — a live, attributed reference layer.
    wiki: WikiConfig | None = None
    identity: str = DEFAULT_IDENTITY
    # The owner whose statements earn the top evidence tier. If None (and no trusted_users), Mimir
    # runs in single-user mode and treats whoever speaks as the primary user (DESIGN §3b).
    primary_user: str | None = None
    # Additional believed identities → STATED_BY_TRUSTED. Any *other* named speaker (an unknown API
    # caller, a peer system, a guest) is attributed but baked at CONVERSATION tier, not as fact —
    # the server-side trust policy, so an exposed API can't self-assert trust. Empty = no extra.
    trusted_users: list[str] = field(default_factory=list)
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
    # Working memory's rolling compression: once ``fold_threshold`` raw exchanges have accumulated,
    # fold the oldest ones into the rolling summary (the previous summary folded in too — compressed
    # harder the further back), keeping the most recent ``keep_recent`` raw. 0 disables compression.
    working_memory_fold_threshold: int = 10
    working_memory_keep_recent: int = 4
    # Deprecated turn-cadence trigger (superseded by the count-based fold above); kept for compat.
    working_memory_refresh_every: int = 4
    # Bidirectional (output-triggered) RAG (DESIGN §5a): after the model replies, retrieve memory
    # relevant to *its own reply* (in the burst) and surface it into the next turn — so a thread the
    # model itself opened gets grounded, not just what the user asked. False = off.
    output_rag_enabled: bool = True
    output_rag_top_k: int = 3
    # Entity-graph traversal: how many hops from the query's entities, and the max connected
    # facts to inject. hops=0 disables graph retrieval (triples are still extracted/stored).
    graph_hops: int = 2
    graph_max_facts: int = 8
    # How often (in turns) consolidation (sleep) runs off the hot path. 0 = manual only
    # (call brain.sleep() or the web UI button / scheduler). Superseded for most setups by the
    # wall-clock sleep window below — with streaming chat on a slow machine the post-turn burst
    # never gets real idle time, so heavy maintenance wants its own quiet window (DESIGN §5a).
    sleep_every: int = 0
    # The wall-clock sleep cycle (DESIGN §5a): a nightly maintenance window when nobody's around.
    # A daemon checks every ``sleep_check_interval_s``; inside the window (and not already done
    # today, and not mid-turn) it runs consolidation + narratives phase-by-phase, skipping any phase
    # that won't fit the time left, with catch-up before noon if the window was missed. Manual too.
    sleep_enabled: bool = True             # the window scheduler; False = manual/turn-cadence only
    sleep_window_start: str = "02:00"      # local HH:MM the window opens
    sleep_window_end: str = "06:00"        # local HH:MM it closes (may cross midnight, e.g. 23:00)
    sleep_check_interval_s: float = 900.0  # how often the daemon checks the clock (15 min)
    # Self-directed deliberation (DESIGN §5a): during sleep, the system surfaces its own conflicts
    # (graph tensions + divergent near-duplicates) and submits them to the inner council for
    # adversarial reasoning, storing the verdicts. A sleep phase; also a manual trigger.
    deliberation_enabled: bool = True
    deliberation_limit: int = 3  # max conflicts argued per cycle (each is several model calls)
    # Self-observability (DESIGN §10): surface recent errors into the turn's context so the model
    # knows when it's degraded ("I've had an error"), and digest them in the nightly cycle.
    surface_errors: bool = True
    error_context_window_s: float = 1800.0  # an error this recent shows in context (30 min)
    error_context_max: int = 5              # max errors shown in the context section
    # Self-knowledge: a doc describing what the system is and how it works, baked into memory in the
    # nightly cycle (content-hashed, so it re-embeds only when the doc changes) so it can answer
    # about itself. Empty disables. Path is relative to the working directory (the repo root).
    self_knowledge_doc: str | None = "README.md"
    # Web server / integration API (DESIGN §8: a brain with endpoints, no built-in hands).
    # ``api_token`` (or the env var named by ``[server] api_token_env``, default MIMIR_API_TOKEN,
    # which wins) gates every ``/api/*`` route via a Bearer check; unset = open (localhost dev). Run
    # two instances on one box? Give each its own ``api_token_env`` so their tokens don't collide.
    # ``cors_origins`` = browser origins allowed to call it (``["*"]`` for any); empty = same site.
    api_token: str | None = None
    cors_origins: list[str] = field(default_factory=list)
    # When a token is set it always guards REMOTE callers, but the local browser UI (same machine,
    # 127.0.0.1) is exempt by default — so a fresh run isn't blocked by a token wall. Flip this on
    # to require the token for the local UI too (e.g. a shared box, or behind a reverse proxy where
    # every request looks local — then enable this or have the proxy do auth).
    secure_ui: bool = False
    # Procedural memory: how many matching procedures to inject, and the minimum trigger match.
    procedural_top_k: int = 3
    procedural_min_match: float = 0.3
    # Temporal grounding (DESIGN §3e): the system's clock + calendar awareness, injected each turn
    # so it can answer relative-time questions ("how long ago?", "what season?"). An IANA timezone
    # (e.g. "America/Vancouver"); None = the host's local zone. Hemisphere flips the season dates —
    # universal, no place baked into core. Both are locale knobs, not deployment secrets.
    timezone: str | None = None
    hemisphere: str = "north"  # "north" or "south" — which way the seasons run

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
        for label, value in (("start", self.sleep_window_start), ("end", self.sleep_window_end)):
            try:
                h, m = (int(p) for p in value.split(":"))
            except ValueError:
                raise ConfigError(f"sleep.window_{label} must be 'HH:MM', got {value!r}") from None
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ConfigError(f"sleep.window_{label} out of range: {value!r}")


def _as_str_list(value: Any) -> list[str]:
    """Accept a string or a list in config and normalize to a list of strings (CORS origins)."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


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
            # Default must match BackendConfig.benchmark_num_ctx (24576) + docs — the proven
            # operational window. (Can't reference the field default: BackendConfig is slotted.)
            benchmark_num_ctx=int(backend_raw.get("benchmark_num_ctx", 24576)),
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

    wiki_raw = raw.get("wiki")
    wiki: WikiConfig | None = None
    if wiki_raw:
        wiki = WikiConfig(
            enabled=bool(wiki_raw.get("enabled", True)),  # presence of the block implies intent
            url=str(wiki_raw.get("url", "http://localhost:8080")).rstrip("/"),
            book=str(wiki_raw.get("book", "")),
            max_articles=int(wiki_raw.get("max_articles", 2)),
            max_chars=int(wiki_raw.get("max_chars", 800)),
            timeout_s=float(wiki_raw.get("timeout_s", 2.0)),
        )

    identity_raw = raw.get("identity", {})
    anchor_keys = (
        "name", "operator", "location", "purpose", "values", "scope", "boundaries", "voice"
    )
    config = Config(
        storage_path=str(storage["path"]),
        roles=_parse_roles(raw.get("roles", {})),
        provider=provider,
        backend=backend,
        wiki=wiki,
        identity=str(identity_raw.get("text", DEFAULT_IDENTITY)),
        primary_user=(
            str(identity_raw["primary_user"]) if "primary_user" in identity_raw else None
        ),
        trusted_users=_as_str_list(identity_raw.get("trusted_users", [])),
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
        working_memory_fold_threshold=int(
            raw.get("working_memory", {}).get("fold_threshold", 10)
        ),
        working_memory_keep_recent=int(
            raw.get("working_memory", {}).get("keep_recent", 4)
        ),
        output_rag_enabled=bool(raw.get("output_rag", {}).get("enabled", True)),
        output_rag_top_k=int(raw.get("output_rag", {}).get("top_k", 3)),
        graph_hops=int(raw.get("entity_graph", {}).get("hops", 2)),
        graph_max_facts=int(raw.get("entity_graph", {}).get("max_facts", 8)),
        sleep_every=int(raw.get("sleep", {}).get("every", 0)),
        sleep_enabled=bool(raw.get("sleep", {}).get("enabled", True)),
        sleep_window_start=str(raw.get("sleep", {}).get("window_start", "02:00")),
        sleep_window_end=str(raw.get("sleep", {}).get("window_end", "06:00")),
        sleep_check_interval_s=float(raw.get("sleep", {}).get("check_interval_s", 900.0)),
        deliberation_enabled=bool(raw.get("deliberation", {}).get("enabled", True)),
        deliberation_limit=int(raw.get("deliberation", {}).get("limit", 3)),
        surface_errors=bool(raw.get("diagnostics", {}).get("surface_errors", True)),
        error_context_window_s=float(
            raw.get("diagnostics", {}).get("error_context_window_s", 1800.0)
        ),
        error_context_max=int(raw.get("diagnostics", {}).get("error_context_max", 5)),
        self_knowledge_doc=(raw.get("self_knowledge", {}).get("doc", "README.md") or None),
        # Token resolution: the env var named by `api_token_env` (default MIMIR_API_TOKEN) wins over
        # the config value, so secrets needn't live in the file. Running two instances on one box?
        # Point each at its own var (e.g. api_token_env = "MIMIR0_TOKEN") so one's MIMIR_API_TOKEN
        # doesn't bleed into the other.
        api_token=(
            os.environ.get(raw.get("server", {}).get("api_token_env", "MIMIR_API_TOKEN"))
            or raw.get("server", {}).get("api_token")
            or None
        ),
        cors_origins=_as_str_list(raw.get("server", {}).get("cors_origins", [])),
        secure_ui=bool(raw.get("server", {}).get("secure_ui", False)),
        procedural_top_k=int(raw.get("procedural", {}).get("top_k", 3)),
        procedural_min_match=float(raw.get("procedural", {}).get("min_match", 0.3)),
        timezone=(str(raw["locale"]["timezone"]) if raw.get("locale", {}).get("timezone")
                  else None),
        hemisphere=str(raw.get("locale", {}).get("hemisphere", "north")),
    )
    config.validate()
    return config
