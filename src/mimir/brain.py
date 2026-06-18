"""``Mimir`` — the facade. Import it, hand it a config, call ``.turn()`` (DESIGN §1).

This is where the spine is wired into the §6 loop:

    turn(text, user)
      → embed the query → recall via the storage gateway → build_context() (assemble)
      → model (chat) → reply
      → record access · bake new facts (storage gateway) · signal the burst worker (async)

Everything routes through the two gateways. Post-response cognition (sentinel, self-model, working
memory, sleep/narratives) is scheduled through the **burst worker** (DESIGN §5a) — priority-ordered,
slot-capped, interruptible — and runs in the idle window after the reply. The next turn settles the
burst before assembling, so the latest note/identity is ready, without making the reply wait on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .cognition.bake import bake, normalize_speaker_kind
from .cognition.benchmark import FleetBenchmarkResult, ModelBenchmark
from .cognition.benchmark import benchmark_fleet as _benchmark_fleet
from .cognition.benchmark import complete_speed_matrix as _complete_speed_matrix
from .cognition.burst import BurstResult, BurstWorker, ResponseContext
from .cognition.citations import citation_warning, unverified_citations
from .cognition.council import CouncilResult, deliberate
from .cognition.deliberation import curate, surface_conflicts
from .cognition.epistemics import EpistemicResult, run_epistemics
from .cognition.fleet import (
    ROLE_NEEDS,
    FleetScanResult,
    council_roster,
    fleet_model_pool,
    fleet_report,
    placement_matrix,
    recommend_roles,
    resolve_auto_model,
    roster_for,
    scan_fleet,
)
from .cognition.graph import build_graph_map, render_triples, retrieve_connected
from .cognition.identity import (
    current_anchors,
    establish_identity,
    pending_questions,
    render_anchors,
)
from .cognition.ingest import (
    SUPPORTED_SUFFIXES,
    IngestResult,
    ingest_document,
    list_documents,
)
from .cognition.inner_life import (
    Stimulus,
    compose_thought,
    gather_stimuli,
    pick_stimulus,
    should_think,
)
from .cognition.library import compose_page, extract_claims, render_claims, retrieve_claims
from .cognition.narratives import render_recent_history, run_narrative_cycle
from .cognition.onboarding import (
    onboarding_profile,
    pending_onboarding,
    record_answer,
)
from .cognition.procedural import learn_procedure, render_procedures, retrieve_procedures
from .cognition.self_model import synthesize_self_model
from .cognition.sentinel import run_sentinel
from .cognition.sleep import SleepReport, consolidate
from .cognition.sleep_cycle import CycleReport, in_window, run_cycle
from .cognition.sleep_cycle import Phase as SleepPhase
from .cognition.temporal import (
    answer_time_query,
    gap_insight,
    local_now,
    resolve_timezone,
    time_prefix,
)
from .cognition.wiki import WikiSource
from .cognition.working_memory import (
    current_working_memory,
    exchange_count,
    record_exchange,
    synthesize_working_memory,
)
from .config import AUTO_MODEL, BackendConfig, Config, ProviderSpec, RoleSpec, load_config
from .context.build import ContextBundle, build_context
from .diagnostics import install_error_capture, render_errors
from .documents.extract import extract
from .embed.base import Embedder, EmbeddingMode, cosine
from .embed.endpoint import EndpointEmbedder, NullEmbedder, ResilientEmbedder
from .embed.locality import LocalityHashEmbedder
from .errors import ConfigError, IngestError, StorageError
from .model.discovery import discover_node_urls
from .model.gateway import ModelGateway
from .model.pool import ProviderPool
from .model.provider import Provider, is_embedding_model
from .model.providers.mock import MockProvider
from .model.providers.ollama import OllamaProvider
from .prompts import DOC_SUMMARY_SYSTEM
from .retrieval.hybrid import retrieve
from .sanitize import StreamTagStripper, strip_epistemic_tags
from .storage.gateway import StorageGateway
from .storage.models import (
    EvidenceTier,
    LibraryClaim,
    LibraryDocument,
    LibraryPage,
    Memory,
    MemoryKind,
    Procedure,
)
from .storage.repo import (
    add_forum_post,
    browse_memories,
    bump_procedure_uses,
    claims_for_document,
    claims_for_page,
    delete_by_source,
    delete_forum_post,
    delete_forum_thread,
    delete_library_document,
    delete_library_page,
    delete_memory,
    disabled_models,
    disabled_nodes,
    get_forum_thread,
    get_library_page,
    interaction_history,
    kv_get,
    kv_set,
    last_conversation_meta,
    latest_self_model,
    latest_sentinel_note,
    list_catalogue,
    list_forum_threads,
    list_library_claims,
    list_library_documents,
    list_library_pages,
    list_memories,
    list_procedures,
    list_sessions,
    pages_for_claims,
    recent_conversation,
    record_access,
    record_conversation_turn,
    record_interaction,
    reembed_claims,
    reembed_memories,
    reembed_procedures,
    replace_document_claims,
    retier_by_provenance,
    save_memory,
    set_forum_thread_status,
    set_model_enabled,
    set_node_enabled,
    set_page_claims,
    update_catalogue_speed,
    update_memory,
    upsert_library_document,
    upsert_library_page,
)

log = logging.getLogger("mimir")

# How many memories the knowledge section may draw on per turn (pre-budget). The token budget caps
# what's actually admitted, so this is just the candidate pool; document ingestion put hundreds of
# chunks alongside personal memories, so 6 starved recall. Hardening (adaptive top-k, SQL-side
# prefiltering) is a later session; v0 keeps it a simple constant.
DEFAULT_TOP_K = 10

# The idle latency heartbeat (DESIGN §5): a short generation forcing a real-length reply, so the
# timed call reflects throughput, not just round-trip. Kept rare — real traffic is the main signal.
_PROBE_PROMPT = [{"role": "user", "content": "In one or two sentences, say you are online."}]
_PROBE_TIMEOUT_S = 20.0
_PROBE_PREDICT = 64  # cap the probe generation — long enough to time throughput, still cheap
_FALLBACK_DEPTH = 4  # how many acceptable models deep a role's fallback chain runs (best first)
_HISTORY_TURNS = 6   # recent exchanges replayed to the model as real messages (continuity)
_SESSION_GAP_S = 6 * 3600  # a fresh boot continues the last conversation only if it's this recent


def _latency_staleness(info: dict[str, object] | None) -> float:
    """How overdue a (node, model) is for a probe: ``inf`` if never measured, else its age in secs.
    The idle heartbeat probes the highest-staleness model per node, so coverage rotates fairly."""
    if info is None or not info.get("samples"):
        return float("inf")
    age = info.get("age_s")
    return float(age) if age is not None else float("inf")


@dataclass(slots=True)
class TurnResult:
    """What a turn produced: the reply, the assembled context (introspectable), and bakes.

    ``library_sources`` lists the composite pages behind the cited claims this turn surfaced
    (``[{page_id, title}]``) — so the UI can offer 'load the source this drew on' after a reply."""

    reply: str
    context: ContextBundle
    baked: list[Memory]
    library_sources: list[dict[str, Any]] = field(default_factory=list)


def build_provider(spec: ProviderSpec) -> Provider:
    """Construct the provider named in config. Fails loud on an unknown type."""
    if spec.type == "mock":
        return MockProvider()
    if spec.type == "ollama":
        return OllamaProvider(host=str(spec.options.get("host", "http://localhost:11434")))
    raise ConfigError(
        f"unknown provider type {spec.type!r}; supported: 'ollama', 'mock'. See docs/SETUP.md."
    )


def build_fleet_pool(backend: BackendConfig) -> ProviderPool:
    """Discover Ollama nodes and build a model-aware pool over them (DESIGN §5).

    Nodes need zero setup — just ``ollama serve``. Discovery = localhost + declared nodes +
    (when ``lan_backend``) a subnet scan.
    """
    urls = discover_node_urls(backend)
    endpoints: list[tuple[str, Provider]] = [(url, OllamaProvider(url)) for url in urls]
    return ProviderPool(endpoints, latency_alpha=backend.latency_alpha)


def make_embedder(config: Config, model: ModelGateway) -> Embedder:
    """Pick the embedder for the configured mode (DESIGN kickoff decision; see docs/SETUP.md)."""
    if config.embed_mode is EmbeddingMode.BOOTSTRAP:
        return LocalityHashEmbedder(dim=config.embed_dim)
    if config.embed_mode is EmbeddingMode.ENDPOINT:
        # Wrapped so a downed embed node degrades to keyword recall (loud), not a crashed turn.
        return ResilientEmbedder(EndpointEmbedder(model))
    return NullEmbedder()


class Mimir:
    """The cognition core. One instance owns one store and one provider."""

    def __init__(self, config: Config, *, provider: Provider | None = None) -> None:
        config.validate()
        self.config = config
        # Capture WARNING+ off the `mimir` logger into a ring, so the system can see its own recent
        # failures — surfaced into context each turn and digested in the sleep cycle (DESIGN §10).
        self._errors = install_error_capture()
        # Roles the user left to automatic selection (model = "auto" or omitted) — resolved from
        # the fleet once inventory is available, and re-resolved on rescan (DESIGN §4).
        self._auto_roles = {r for r, s in config.roles.items() if s.model == AUTO_MODEL}
        self._storage = StorageGateway(config.storage_path)
        if config.backend is not None and provider is None:
            # A discovered/declared Ollama fleet: build a model-aware pool, then inventory it and
            # start the active-health prober IN THE BACKGROUND — so a slow node can't block boot
            # (or the web server). Routing is optimistic until the first inventory lands (§5).
            pool = build_fleet_pool(config.backend)
            self._model = ModelGateway(pool, config.roles)
            threading.Thread(
                target=self._init_fleet,
                args=(config.backend.refresh_interval_s,),
                name="mimir-fleet-init",
                daemon=True,
            ).start()
            log.info("Mimir online | fleet: discovered nodes; inventorying in background")
        else:
            self._model = ModelGateway(provider or build_provider(config.provider), config.roles)
        self._embedder: Embedder = make_embedder(config, self._model)
        # Fail-loud visibility: announce the active embedding mode so no one mistakes the
        # cheap bootstrap path for poor memory (DESIGN §10; kickoff decision).
        log.info("Mimir online | embeddings: %s", self._embedder.mode.banner())

        self._last_sentinel_error: BaseException | None = None
        self._turn_count = 0
        self._session_id: str | None = None  # the current conversation; resolved lazily on turn 1
        # Optional offline encyclopedia (Kiwix/ZIM over HTTP) — a live, attributed reference layer.
        wcfg = config.wiki
        self._wiki = (
            WikiSource(url=wcfg.url, book=wcfg.book, max_articles=wcfg.max_articles,
                       max_chars=wcfg.max_chars, timeout_s=wcfg.timeout_s)
            if wcfg and wcfg.enabled and wcfg.book else None
        )
        self._turn_active = False  # True while a turn is generating — background yields to it (§5a)
        # The burst worker: all post-response cognition (sentinel, self-model, working memory,
        # sleep) is scheduled through it — priority-ordered, slot-capped, interruptible — instead of
        # N raw threads (DESIGN §5a). Runs in the idle window after a reply; the next turn settles.
        self._burst = BurstWorker(is_busy=lambda: self._turn_active)
        self._register_burst_tasks()
        self._burst.start()
        self._stop_idle = threading.Event()  # signals the idle latency heartbeat to stop
        self._idle_prober: threading.Thread | None = None
        # The wall-clock sleep cycle: heavy maintenance in its own quiet window, since streaming +
        # slow machines starve the post-turn burst of real idle time (DESIGN §5a).
        self._stop_sleep = threading.Event()
        self._sleep_scheduler: threading.Thread | None = None
        self._start_sleep_scheduler()
        # The live inner life: a low-frequency idle loop that thinks between turns (DESIGN §5a).
        # OFF by default (it spends idle compute), routed off the chat model, yields to live turns.
        # Reads effective settings each tick so the UI toggle/cadence take effect without a restart.
        self._last_turn_at = 0.0      # wall-clock end of last turn (the inner-life idle floor)
        self._last_thought_at = 0.0   # wall-clock of last inner-life thought (the cadence gate)
        self._last_thought_kind: str | None = None
        self._last_escalation_at = 0.0  # wall-clock of last inner-life→council escalation
        self._stop_inner = threading.Event()
        self._inner_life: threading.Thread | None = None
        self._start_inner_life()

        # Establish any identity anchors declared in config (idempotent upsert at boot), so a
        # non-interactive deployment is grounded without running the interactive interview.
        if config.identity_anchors:
            establish_identity(self._storage, config.identity_anchors)

        # Apply any per-node / per-model vetoes to the pool up front, so a disabled box or model is
        # never routed to (the model veto re-resolves a role that points at a disabled model).
        self._model.set_disabled_nodes(disabled_nodes(self._storage))
        self._model.set_disabled_models(disabled_models(self._storage))

        # Resolve `auto` roles now for the local/single-provider path (inventory is ready). The
        # fleet path resolves in _init_fleet once its background inventory lands; until then the
        # gateway stop-gaps `auto` to any reachable model so turns never fail (DESIGN §4).
        self._resolve_auto_roles()
        # Re-apply persisted manual role pins (survive restart; override config + auto). DESIGN §4.
        self._restore_role_pins()

    def _init_fleet(self, refresh_interval_s: float) -> None:
        """Inventory the fleet once, then start the active-health prober (off the boot path)."""
        try:
            self._model.refresh_inventory()
        except Exception as exc:  # never let fleet init crash the brain
            log.warning("fleet: initial inventory failed: %s", exc)
        try:
            self._resolve_auto_roles()
            self._seed_latency_from_catalogue()  # route informed by prior qualification from turn 1
        except StorageError:  # background init raced a close() — the store is gone; just stop (§10)
            log.info("fleet: init aborted — storage closed (shutting down)")
            return
        self._model.start_prober(refresh_interval_s)
        self._start_idle_prober()

    def _resolve_auto_roles(self) -> dict[str, str]:
        """Bind every `auto` role to a concrete model from the current fleet (DESIGN §4).

        No-op (returns ``{}``) until at least one model is reachable; safe to call repeatedly —
        re-resolution simply re-picks the current best, so a freshly benchmarked or newly
        disabled model is reflected on the next scan.
        """
        if not self._auto_roles:
            return {}
        available = set(self._model.available_models())
        if not available:
            return {}
        disabled = disabled_models(self._storage)
        vetoed_nodes = disabled_nodes(self._storage)
        resolved: dict[str, str] = {}
        for role in self._auto_roles:
            if role == "embed":
                continue  # embeddings need a STABLE, remembered choice — see _resolve_embed_model
            model = resolve_auto_model(
                self._storage, role, available=available, disabled=disabled
            )
            if model is not None:
                self._model.set_role_model(role, model)
                resolved[role] = model
            # The ranked fallback chain: the role's acceptable models, best first, so a
            # heterogeneous fleet still serves the role across nodes (DESIGN §4/§5). Empty until a
            # benchmark exists (roster_for needs scores); the single resolve above carries routing
            # until then. Filtered to reachable models so the chain lists nothing no live node runs.
            if role in ROLE_NEEDS:
                chain = [
                    m["model"]
                    for m in roster_for(
                        self._storage, role, n=_FALLBACK_DEPTH,
                        disabled=disabled, disabled_nodes=vetoed_nodes,
                    )
                    if m["model"] in available
                ]
                self._model.set_role_fallbacks(role, chain)
        if resolved:
            log.info("fleet: auto-resolved role(s) %s", resolved)
        self._resolve_embed_model()  # embed is auto-discovered + remembered separately
        return resolved

    _EMBED_MODEL_KEY = "embed_model"

    def _resolve_embed_model(self) -> None:
        """Auto-discover + REMEMBER the embedding model when ``[roles.embed] = "auto"`` (DESIGN §4).

        Embeddings are special: different models produce different vector spaces, so the choice must
        be STABLE — silently flipping models would make old and new embeddings incomparable. So:
        prefer the remembered model; the first time, pick one deterministically and persist it; if
        the remembered model isn't currently reachable, stay pinned to it (the resilient embedder
        degrades to keyword recall until it returns) rather than switch to an incompatible one. A
        pinned (non-``auto``) embed role is respected untouched."""
        if "embed" not in self._auto_roles:
            return
        embed_models = sorted(m for m in self._model.available_models() if is_embedding_model(m))
        remembered = kv_get(self._storage, self._EMBED_MODEL_KEY)
        if remembered:
            self._model.set_role_model("embed", remembered)
            if embed_models and remembered not in embed_models:
                log.warning(
                    "embed: remembered model %r not currently reachable (available: %s); staying "
                    "pinned to preserve the vector space — pull it back, or clear the %r kv to "
                    "re-pick (and re-embed).", remembered, embed_models, self._EMBED_MODEL_KEY,
                )
            return
        if not embed_models:
            return  # nothing discovered yet; the gateway's sorted stop-gap covers call-time
        chosen = embed_models[0]
        self._model.set_role_model("embed", chosen)
        kv_set(self._storage, self._EMBED_MODEL_KEY, chosen)
        log.info("embed: auto-discovered %r (remembered for vector-space stability)", chosen)

    # -- live node speed / health (speed-aware routing, DESIGN §5) ---------------------

    def _seed_latency_from_catalogue(self) -> None:
        """Prime the pool's live latency from the catalogue's ``return_time`` so routing starts
        informed, not cold. Real traffic overrides a seed for any pair it touches."""
        seeds = {
            (e.node, e.model): e.return_time
            for e in list_catalogue(self._storage)
            if e.return_time is not None
        }
        if seeds:
            self._model.seed_latency(seeds)

    def _start_idle_prober(self) -> None:
        """Start the rare idle latency heartbeat (DESIGN §5). Real traffic is the primary signal; it
        only tops up nodes that have gone quiet — a long interval, never on a busy node."""
        interval = self.config.backend.idle_probe_interval_s if self.config.backend else 0.0
        if interval <= 0 or self._idle_prober is not None:
            return

        def _loop() -> None:
            while not self._stop_idle.wait(interval):
                try:
                    self._idle_latency_probe()
                except Exception as exc:  # the heartbeat must never die (DESIGN §10)
                    log.warning("fleet: idle latency probe failed: %s", exc)

        # Start before publishing the handle, so close() can never join an unstarted thread (the
        # prober is launched from the background fleet-init thread, racing a quick close()).
        prober = threading.Thread(target=_loop, name="mimir-idle-prober", daemon=True)
        prober.start()
        self._idle_prober = prober

    def _idle_latency_probe(self) -> None:
        """One heartbeat pass: probe the stalest model on each idle node, then persist live speed.

        Bounded to one probe per idle node per cycle (the model with the oldest/absent sample, so
        coverage rotates), at the operational ``num_ctx`` so it doesn't trigger a KV-cache reload.
        """
        nodes = set(self._model.idle_nodes())
        if not nodes:
            return
        snapshot = self._model.latency_snapshot()
        by_node: dict[str, list[str]] = {}
        for e in list_catalogue(self._storage):
            if e.node in nodes and "embed" not in e.model.lower():
                by_node.setdefault(e.node, []).append(e.model)
        ctx = self.config.backend.benchmark_num_ctx if self.config.backend else 24576
        params = {"num_ctx": ctx, "max_tokens": _PROBE_PREDICT, "__timeout_s__": _PROBE_TIMEOUT_S}
        for node, models in by_node.items():
            target = max(models, key=lambda m: _latency_staleness(snapshot.get((node, m))))
            self._model.probe_latency(node, target, _PROBE_PROMPT, params)
        self._persist_live_latency()

    def _persist_live_latency(self) -> None:
        """Write the live, real-traffic latency back to the catalogue so the placement matrix and
        leaderboard reflect current speed (not the frozen benchmark). Only measured pairs are
        written — a mere seed never overwrites the qualification number it came from."""
        for (node, model), info in self._model.latency_snapshot().items():
            if info.get("samples", 0) and info.get("return_time") is not None:
                update_catalogue_speed(self._storage, node, model, float(info["return_time"]))

    def node_health(self) -> dict[str, Any]:
        """Live fleet health for introspection/UI: pool stats (reachable/saturated/load + fastest
        per-node speed) plus the full per-(node, model) live latency snapshot (DESIGN §5)."""
        snapshot = self._model.latency_snapshot()
        return {
            "pool": self._model.get_stats(),
            "latency": {f"{node} · {model}": info for (node, model), info in snapshot.items()},
        }

    @classmethod
    def from_config(cls, path: str) -> Mimir:
        """Construct from a ``mimir.toml`` path."""
        return cls(load_config(path))

    # -- temporal grounding (DESIGN §3e) ----------------------------------------------

    def _time_context(self) -> str:
        """The clock/calendar line injected each turn, in the configured zone + hemisphere."""
        return time_prefix(local_now(self._tz()), self.config.hemisphere)

    def maybe_time_answer(self, text: str) -> str | None:
        """A direct, model-free answer to an explicit time/date/season question, or ``None``.

        The deterministic intercept (DESIGN §3e) — exposed so a host (CLI, server, voice) can short-
        circuit before a model call. ``turn`` uses it automatically."""
        return answer_time_query(text, local_now(self._tz()), self.config.hemisphere)

    def _temporal_awareness(self, user: str | None, now_ts: float) -> str | None:
        """A deterministic 'you've been away longer than usual' note from the interaction log, or
        ``None``. Computed from PRIOR interactions (call before logging the current turn)."""
        history = interaction_history(self._storage, user=user)
        return gap_insight(history, now_ts)

    def _recent_history(self) -> str | None:
        """The temporal-narrative arc (month → week → lately) for the prompt, or ``None``."""
        return render_recent_history(self._storage)

    def wiki_status(self) -> dict[str, Any]:
        """Whether the offline encyclopedia is configured and reachable (for the UI status line)."""
        if self._wiki is None:
            return {"enabled": False}
        return {"enabled": True, **self._wiki.status()}

    def _wiki_context(self, text: str) -> str | None:
        """A live reference lookup from the offline encyclopedia, or ``None`` (disabled / smalltalk
        / no hit / unreachable). Skips trivial turns so a greeting doesn't trigger a search."""
        if self._wiki is None:
            return None
        if len(text.split()) < 3 and not text.rstrip().endswith("?"):
            return None
        return self._wiki.context(text)

    def _history_messages(self, user: str | None, session_id: str) -> list[dict[str, str]]:
        """The current session's recent exchanges as real chat messages, so the model has genuine
        continuity (not just summarized text) — scoped to this conversation so a new one starts
        clean (DESIGN §3a)."""
        msgs: list[dict[str, str]] = []
        for turn in recent_conversation(
            self._storage, user=user, limit=_HISTORY_TURNS, session_id=session_id
        ):
            msgs.append({"role": "user", "content": turn["user_text"]})
            msgs.append({"role": "assistant", "content": turn["reply"]})
        return msgs

    def _resolve_session(self) -> str:
        """The current conversation's id. On the first turn it continues the last conversation if it
        was recent, else starts a new one; explicit new/resume pin it thereafter (DESIGN §3a)."""
        if self._session_id is None:
            meta = last_conversation_meta(self._storage)
            if (meta and meta["session_id"]
                    and (time.time() - meta["created_at"]) < _SESSION_GAP_S):
                self._session_id = str(meta["session_id"])
            else:
                self._session_id = self._new_session_id()
        return self._session_id

    @staticmethod
    def _new_session_id() -> str:
        return f"s{int(time.time() * 1000)}"

    def start_new_session(self) -> str:
        """Begin a fresh conversation — subsequent turns won't carry the prior one's context."""
        self._session_id = self._new_session_id()
        return self._session_id

    def resume_session(self, session_id: str) -> None:
        """Continue an earlier conversation — its recent turns replay to the model again."""
        self._session_id = session_id

    def sessions(self, *, user: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Past conversations (most recent first) with a one-line summary — for the dropdown."""
        return list_sessions(self._storage, user=user, limit=limit)

    def history(
        self, *, user: str | None = None, limit: int = 50, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The durable conversation log, oldest→newest — restored by the UI. ``session_id`` scopes
        it to one conversation (§3a)."""
        return recent_conversation(self._storage, user=user, limit=limit, session_id=session_id)

    # -- the memory graph (visual review/edit; DESIGN §3a) -----------------------------

    def graph_map(self, *, memory_limit: int = 60) -> dict[str, Any]:
        """Nodes (memory blobs + entities) and links (relations + mentions) for the visual graph."""
        return build_graph_map(self._storage, memory_limit=memory_limit)

    def edit_memory(
        self, memory_id: int, *, text: str | None = None, salience: float | None = None
    ) -> Memory | None:
        """Edit a memory's text/salience (the graph editor) and return it, or ``None`` if gone."""
        update_memory(self._storage, memory_id, text=text, salience=salience)
        from .storage.repo import get_memory
        return get_memory(self._storage, memory_id)

    def forget_memory(self, memory_id: int) -> None:
        """Permanently delete a memory (the graph editor's remove)."""
        delete_memory(self._storage, memory_id)

    def retier_speaker(
        self, speaker: str, tier: EvidenceTier = EvidenceTier.CONVERSATION
    ) -> int:
        """Re-tier memories baked from ``speaker`` (provenance ``stated by <speaker>``) to ``tier``.

        Maintenance for when a speaker was ingested at the wrong trust level — e.g. a peer AI baked
        as ``stated_by_primary_user`` before ``[identity] primary_user`` was set. Default drops them
        to ``conversation`` (attributed, but not believed as fact). Returns rows changed."""
        n = retier_by_provenance(self._storage, f"stated by {speaker}", tier)
        log.info("retier: moved %d memory(ies) from %r to tier %s", n, speaker, tier.key)
        return n

    def generate_narratives(self) -> dict[str, Any]:
        """Run the temporal-narrative cycle now (daily entry + weekly/monthly roll-up), sync.

        Normally runs off the hot path in the consolidation pass; this is the explicit hook (DESIGN
        §3a/§3e). Idempotent per period — re-running the same day reuses today's entry."""
        return run_narrative_cycle(
            self._model, self._storage, now=local_now(self._tz())
        )

    # -- the turn ---------------------------------------------------------------------

    def turn(self, text: str, user: str | None = None, *,
             speaker_kind: str = "human", loaded_pages: list[int] | None = None,
             deep_read: bool = False, include_memory: bool = True,
             include_library: bool = True, include_wiki: bool = True) -> TurnResult:
        # ``speaker_kind`` ("human"/"ai_peer") is the caller's declaration of what kind of speaker
        # this is; validate it up front so a bad value fails the turn cleanly (DESIGN §3b).
        normalize_speaker_kind(speaker_kind)
        # Settle the previous turn's burst (sentinel note, self-model) before we assemble — so the
        # prompt reflects the latest reflection and identity (DESIGN §5a).
        self._burst.wait_idle()
        self._turn_count += 1

        # Temporal awareness: read the gap from PRIOR interactions, then log this turn (DESIGN §3e).
        now = time.time()
        awareness = self._temporal_awareness(user, now)
        record_interaction(self._storage, now, user)
        sid = self._resolve_session()

        # 0. Deterministic time-query intercept — answer "what day/season is it" with no model call
        #    (DESIGN §3e). Still recorded as an exchange (continuity), nothing to bake/reflect on.
        intercept = self.maybe_time_answer(text)
        if intercept is not None:
            bundle = build_context(
                query=text, user=user, identity=self.config.identity, retrieved=[],
                sentinel_note=None, embed_mode=self._embedder.mode,
                budget_tokens=self.config.context_budget_tokens,
                time_context=self._time_context(), now_ts=now,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=intercept)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=intercept,
                                     session_id=sid)
            return TurnResult(reply=intercept, context=bundle, baked=[])

        self._turn_active = True  # foreground in progress — the burst yields to it (§5a)
        try:
            # Background notes the prior burst surfaced, to carry into this reply.
            notes = self._burst.drain_surfaces()

            # 1. Recall: embed the query, pull candidates, rank them. Inner-life musings are split
            #    out of the knowledge recall and may surface as a framed, tentative note (§5a).
            query_vec = self._embedder.embed(text)
            disabled_docs = self._disabled_documents()  # per-doc "include in context" toggles
            candidates = list_memories(self._storage, user=user, kind=MemoryKind.MEMORY,
                                       exclude_sources=disabled_docs)
            candidates, il_note = self._surface_inner_life(candidates, text, query_vec)
            if il_note and include_memory:
                notes = notes + [il_note]
            candidates = self._filter_layers(candidates, include_memory, include_library)
            retrieved = retrieve(text, query_vec, candidates, top_k=DEFAULT_TOP_K)
            note = latest_sentinel_note(self._storage, user)
            self_knowledge = self._compose_self_knowledge()
            working_memory = current_working_memory(self._storage)
            graph_facts = self._connected_facts(text, user)
            procedures = self._matching_procedures(text, user)
            if include_library:
                lib_text, lib_refs, lib_count = self._library_gist(text, query_vec, disabled_docs)
                pages = self._pages_to_load(loaded_pages, deep_read, lib_refs)
                library = self._merge_loaded_library(lib_text, pages)
                lib_count += len(pages)  # full pages (loaded or deep-read) are strong grounding too
                if self.config.library_model_fetch and lib_refs:
                    library = self._with_fetch_hint(library, lib_refs)  # let the model open a page
            else:
                lib_refs, lib_count, library = [], 0, None
            wiki = self._wiki_context(text) if include_wiki else None

            # 2. Assemble the epistemic prompt.
            bundle = build_context(
                query=text,
                user=user,
                identity=self.config.identity,
                retrieved=retrieved,
                sentinel_note=note,
                embed_mode=self._embedder.mode,
                budget_tokens=self.config.context_budget_tokens,
                self_knowledge=self_knowledge,
                working_memory=working_memory,
                graph_facts=graph_facts,
                procedures=procedures,
                time_context=self._time_context(),
                temporal_awareness=awareness,
                recent_history=self._recent_history(),
                background_notes="\n".join(notes) if notes else None,
                wiki_context=wiki,
                library=library,
                library_count=lib_count,
                system_health=self._error_context(),
                now_ts=now,
            )

            # 3. Generate the reply through the model gateway. Strip any internal epistemic tags the
            #    model echoed (small models mimic the [tier=...; source=...] style) before it lands.
            reply = strip_epistemic_tags(
                self._model.chat(
                    "chat",
                    [
                        {"role": "system", "content": bundle.prompt},
                        *self._history_messages(user, sid),
                        {"role": "user", "content": text},
                    ],
                )
            )
            # 3b. Model-driven Library fetch (opt-in): if the model asked to open a page, load it
            #     and answer again with the detail in hand (docs/LIBRARY.md Phase 2).
            reply = self._maybe_model_fetch(reply, bundle, user, sid, text)
            # 3c. Citation guard: flag any source the reply cited that we don't actually hold (§10).
            reply += self._citation_note(reply)

            # 4. Side effects through the storage gateway: relevance bookkeeping + bake.
            record_access(self._storage, bundle.retrieved_ids)
            baked = bake(
                self._model,
                self._storage,
                self._embedder,
                turn_text=text,
                user=user,
                primary_user=self.config.primary_user,
                trusted_users=self.config.trusted_users,
                peer_agents=self.config.peer_agents,
                speaker_kind=speaker_kind,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=reply)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=reply,
                                     session_id=sid)
        finally:
            self._turn_active = False
            self._last_turn_at = time.time()  # start the inner-life idle floor (DESIGN §5a)

        # 5. Fire the burst window: sentinel + any due self-model/working-memory/sleep, scheduled
        #    and run off the hot path (DESIGN §5a). The next turn settles it.
        self._burst.signal(ResponseContext(
            user_text=text, reply=reply, user=user, turn_index=self._turn_count
        ))
        return TurnResult(reply=reply, context=bundle, baked=baked, library_sources=lib_refs)

    def turn_stream(
        self, text: str, user: str | None = None, *, speaker_kind: str = "human",
        loaded_pages: list[int] | None = None, deep_read: bool = False,
        include_memory: bool = True, include_library: bool = True, include_wiki: bool = True,
    ) -> Generator[str, None, dict[str, Any]]:
        """Like ``turn`` but yields the reply token-by-token; returns the introspection dict.

        The side effects (record access, bake, sentinel, self-model) run after the stream
        completes, so a fully-consumed stream behaves exactly like ``turn``. If the consumer
        abandons the stream early, the turn is treated as interrupted — nothing is baked. The
        generator's *return value* (via ``StopIteration.value``) is ``context.introspect()``.
        """
        normalize_speaker_kind(speaker_kind)  # validate up front (DESIGN §3b), mirroring `turn`
        self._burst.wait_idle()
        self._turn_count += 1

        # Temporal awareness + the deterministic time intercept (DESIGN §3e), mirroring `turn`.
        now = time.time()
        awareness = self._temporal_awareness(user, now)
        record_interaction(self._storage, now, user)
        sid = self._resolve_session()
        intercept = self.maybe_time_answer(text)
        if intercept is not None:
            bundle = build_context(
                query=text, user=user, identity=self.config.identity, retrieved=[],
                sentinel_note=None, embed_mode=self._embedder.mode,
                budget_tokens=self.config.context_budget_tokens,
                time_context=self._time_context(), now_ts=now,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=intercept)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=intercept,
                                     session_id=sid)
            yield intercept
            return bundle.introspect()

        self._turn_active = True  # foreground in progress — the burst yields to it (§5a)
        try:
            notes = self._burst.drain_surfaces()
            query_vec = self._embedder.embed(text)
            disabled_docs = self._disabled_documents()  # per-doc "include in context" toggles
            candidates = list_memories(self._storage, user=user, kind=MemoryKind.MEMORY,
                                       exclude_sources=disabled_docs)
            candidates, il_note = self._surface_inner_life(candidates, text, query_vec)
            if il_note and include_memory:
                notes = notes + [il_note]
            candidates = self._filter_layers(candidates, include_memory, include_library)
            retrieved = retrieve(text, query_vec, candidates, top_k=DEFAULT_TOP_K)
            note = latest_sentinel_note(self._storage, user)
            self_knowledge = self._compose_self_knowledge()
            working_memory = current_working_memory(self._storage)
            graph_facts = self._connected_facts(text, user)
            procedures = self._matching_procedures(text, user)
            if include_library:
                lib_text, lib_refs, lib_count = self._library_gist(text, query_vec, disabled_docs)
                pages = self._pages_to_load(loaded_pages, deep_read, lib_refs)
                library = self._merge_loaded_library(lib_text, pages)
                lib_count += len(pages)  # full pages (loaded or deep-read) are strong grounding too
            else:
                lib_refs, lib_count, library = [], 0, None
            wiki = self._wiki_context(text) if include_wiki else None
            bundle = build_context(
                query=text,
                user=user,
                identity=self.config.identity,
                retrieved=retrieved,
                sentinel_note=note,
                embed_mode=self._embedder.mode,
                budget_tokens=self.config.context_budget_tokens,
                self_knowledge=self_knowledge,
                working_memory=working_memory,
                graph_facts=graph_facts,
                procedures=procedures,
                time_context=self._time_context(),
                temporal_awareness=awareness,
                recent_history=self._recent_history(),
                background_notes="\n".join(notes) if notes else None,
                wiki_context=wiki,
                library=library,
                library_count=lib_count,
                system_health=self._error_context(),
                now_ts=now,
            )
            # Stream the reply (honoring an opt-in model fetch of a full library page); internal
            # epistemic tags are stripped as we go (a tag may straddle deltas) inside the helper.
            reply = yield from self._stream_chat_with_fetch(bundle, lib_refs, user, sid, text)
            # Citation guard: append a fail-loud note (and stream it) if a cited source is unknown.
            note = self._citation_note(reply)
            if note:
                yield note
                reply += note

            record_access(self._storage, bundle.retrieved_ids)
            bake(
                self._model,
                self._storage,
                self._embedder,
                turn_text=text,
                user=user,
                primary_user=self.config.primary_user,
                trusted_users=self.config.trusted_users,
                peer_agents=self.config.peer_agents,
                speaker_kind=speaker_kind,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=reply)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=reply,
                                     session_id=sid)
        finally:
            self._turn_active = False
            self._last_turn_at = time.time()  # start the inner-life idle floor (DESIGN §5a)

        self._burst.signal(ResponseContext(
            user_text=text, reply=reply, user=user, turn_index=self._turn_count
        ))
        intro = bundle.introspect()
        intro["library_sources"] = lib_refs
        return intro

    def _connected_facts(self, query: str, user: str | None) -> list[str]:
        """Connected facts from the entity graph for this turn (empty if disabled or no match)."""
        if self.config.graph_hops <= 0:
            return []
        triples = retrieve_connected(
            self._storage,
            query,
            hops=self.config.graph_hops,
            max_facts=self.config.graph_max_facts,
            user=user,
        )
        return render_triples(triples)

    def learn_procedure(
        self, trigger: str, procedure: str, *, user: str | None = None, confidence: float = 0.7
    ) -> Procedure:
        """Teach a reasoning habit: when ``trigger`` applies, follow ``procedure`` (DESIGN §3a)."""
        return learn_procedure(
            self._storage,
            self._embedder,
            trigger=trigger,
            procedure=procedure,
            user=user,
            confidence=confidence,
        )

    def _matching_procedures(self, query: str, user: str | None) -> list[str]:
        """Procedures whose trigger matches this turn; bumps their use count (relevance signal)."""
        procedures = retrieve_procedures(
            self._storage,
            self._embedder,
            query,
            top_k=self.config.procedural_top_k,
            min_match=self.config.procedural_min_match,
            user=user,
        )
        bump_procedure_uses(self._storage, [p.id for p in procedures if p.id is not None])
        return render_procedures(procedures)

    # -- document ingestion (v0.1) ----------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        *,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
    ) -> IngestResult:
        """Ingest a document (.txt/.md in core; .pdf via the [documents] extra).

        The file is extracted, chunked, embedded, and stored as document-tier knowledge. Its
        chunks are then recalled like any other memory on subsequent turns, attributed to the
        file and locator (page/section). Re-ingesting the same path replaces its prior chunks.
        """
        return ingest_document(
            self._storage,
            self._embedder,
            path=path,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

    # -- identity ---------------------------------------------------------------------

    def establish_identity(self, answers: dict[str, str]) -> dict[str, str]:
        """Record foundational identity anchors (name/operator/location/purpose, …).

        The interactive interview (``python -m mimir.interview``) and config both flow through
        here. Returns the full anchor set after the update. Anchors ground the always-on
        self-model from the first boot, before any history exists.
        """
        return establish_identity(self._storage, answers)

    def identity_anchors(self) -> dict[str, str]:
        """The established identity anchors as a ``{key: value}`` map."""
        return current_anchors(self._storage)

    def pending_identity_questions(self) -> list[tuple[str, str]]:
        """The ``(key, question)`` pairs the interview still needs answered."""
        return pending_questions(self._storage)

    # -- the seeding interview (onboarding, DESIGN §9) ---------------------------------

    def onboarding_profile(self) -> list[dict[str, Any]]:
        """The seeding interview as the editable 'one place': every question + its current answer.

        These are the operator's highest-provenance facts (``stated_by_primary_user``,
        ``provenance="onboarding"``) — the orientation everything else builds on. Re-runnable and
        editable any time (see ``record_onboarding_answer``)."""
        return onboarding_profile(self._storage)

    def pending_onboarding(self) -> list[dict[str, str]]:
        """The interview questions still unanswered — drives the strip and the first-run prompt."""
        return pending_onboarding(self._storage)

    def record_onboarding_answer(self, key: str, answer: str) -> Memory | None:
        """Store/update one interview answer as a top-tier onboarding fact (blank clears it).

        Captured model-free and persisted immediately (crash-safe; the doc's §2 law), mirroring
        name/operator/location into the always-on identity anchors. Returns the stored memory, or
        ``None`` if cleared/unknown."""
        return record_answer(
            self._storage, self._embedder,
            key=key, answer=answer, primary_user=self.config.primary_user,
        )

    def _compose_self_knowledge(self) -> str | None:
        """The self-model section body: identity anchors (verbatim) + synthesized self-model.

        Anchors go first and verbatim so foundational facts (name, purpose) are reliably present;
        the synthesized paragraph adds the evolving narrative grounded in operational history.
        """
        anchors_text = render_anchors(current_anchors(self._storage))
        self_model = latest_self_model(self._storage)
        parts = [p for p in (anchors_text, self_model.text if self_model else None) if p]
        return "\n\n".join(parts) if parts else None

    # -- self-model -------------------------------------------------------------------

    def refresh_self_model(self) -> Memory:
        """Synthesize a fresh self-model now (synchronous) and return it.

        The self-model is the system's always-on identity, authored from its own operational
        history (DESIGN §3a). It is normally refreshed automatically off the hot path; this is
        the explicit hook.
        """
        return synthesize_self_model(self._model, self._storage)

    # -- working memory ---------------------------------------------------------------

    def refresh_working_memory(self) -> Memory | None:
        """Fold the accumulated exchanges into the rolling working-memory summary now (sync)."""
        return synthesize_working_memory(self._model, self._storage)

    # -- burst tasks: the post-response cognition the worker schedules (DESIGN §5a) ---

    def _register_burst_tasks(self) -> None:
        """Register the standard post-response work on the burst worker. The sentinel is user-class
        (runs first, every turn — its note must be ready for the next turn); the rest are autonomous
        and fire on their cadence (self-model, working memory, sleep+narratives)."""
        self._burst.register("sentinel", self._sentinel_task, base_priority=5.0,
                             user_requested=True)
        self._burst.register("self_model", self._self_model_task, base_priority=30.0,
                             trigger=lambda ctx: self._due("self_model"))
        self._burst.register("working_memory", self._working_memory_task, base_priority=25.0,
                             trigger=lambda ctx: self._due("working_memory"))
        self._burst.register("output_rag", self._output_rag_task, base_priority=20.0,
                             trigger=lambda ctx: self.config.output_rag_enabled)
        self._burst.register("sleep", self._sleep_task, base_priority=60.0,
                             trigger=lambda ctx: self._due("sleep"))

    def _due(self, which: str) -> bool:
        """Whether a cadence task is due this turn (preserving the prior per-task schedules)."""
        if which == "self_model":
            every = self.config.self_model_refresh_every
            return every > 0 and (self._turn_count == 1 or self._turn_count % every == 0)
        if which == "working_memory":
            # Count-based: fold once enough raw exchanges have accumulated (after the model has
            # streamed its reply and the user is composing — a few seconds, off the hot path).
            threshold = self.config.working_memory_fold_threshold
            return threshold > 0 and exchange_count(self._storage) >= threshold
        if which == "sleep":
            every = self.config.sleep_every
            return every > 0 and self._turn_count % every == 0
        return False

    def _sentinel_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            self._last_sentinel_error = None
            try:
                run_sentinel(self._model, self._storage, user=ctx.user,
                             turn_text=ctx.user_text, reply=ctx.reply)
            except BaseException as exc:  # logged downgrade — the turn is already done (§10)
                self._last_sentinel_error = exc
                log.error("sentinel failed (off the hot path; turn unaffected): %s", exc,
                          exc_info=True)
            return BurstResult()
        return run

    def _self_model_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            try:
                synthesize_self_model(self._model, self._storage)
            except BaseException as exc:
                log.error("self-model refresh failed (turn unaffected): %s", exc, exc_info=True)
            return BurstResult()
        return run

    def _working_memory_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            try:
                synthesize_working_memory(
                    self._model, self._storage,
                    fold_threshold=self.config.working_memory_fold_threshold,
                    keep_recent=self.config.working_memory_keep_recent,
                )
            except BaseException as exc:
                log.error("working-memory refresh failed (turn unaffected): %s", exc, exc_info=True)
            return BurstResult()
        return run

    def _output_rag(self, reply: str, user: str | None) -> str | None:
        """Bidirectional RAG (DESIGN §5a): retrieve memory relevant to the model's OWN reply, to
        surface into the next turn — so a thread the model itself opened gets grounded. Excludes the
        facts just baked from this very reply (they'd be a redundant echo). Returns a surface, or
        ``None``. Off the hot path (burst); a failure must never break the turn."""
        text = (reply or "").strip()
        if len(text.split()) < 4:
            return None  # a trivial reply — nothing worth a retrieval pass
        query_vec = self._embedder.embed(text)
        now = time.time()
        candidates = [
            m for m in list_memories(self._storage, user=user, kind=MemoryKind.MEMORY)
            if now - m.created_at > 10  # skip what was just baked from this turn (an echo)
        ]
        scored = retrieve(text, query_vec, candidates, top_k=max(1, self.config.output_rag_top_k))
        if not scored:
            return None
        lines = "\n".join(f"- {s.memory.text}" for s in scored)
        return f"Possibly relevant from memory, on what you last said:\n{lines}"

    def _output_rag_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            try:
                note = self._output_rag(ctx.reply, ctx.user)
            except BaseException as exc:
                log.error("output-rag failed (turn unaffected): %s", exc, exc_info=True)
                return BurstResult()
            return BurstResult(surface=note) if note else BurstResult()
        return run

    def _sleep_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            try:
                consolidate(self._storage)
                run_narrative_cycle(
                    self._model, self._storage, now=local_now(self._tz())
                )
            except BaseException as exc:
                log.error("consolidation failed (turn unaffected): %s", exc, exc_info=True)
            return BurstResult()
        return run

    # -- sleep / consolidation --------------------------------------------------------

    def sleep(self) -> SleepReport:
        """Run a consolidation pass now (dedup, decay, archive, contradiction resolution) + write
        the period's temporal narratives (daily entry, weekly/monthly roll-up; DESIGN §3a/§3e)."""
        report = consolidate(self._storage)
        try:
            self.generate_narratives()
        except Exception as exc:  # narratives are enrichment — never fail consolidation (§10)
            log.error("narrative cycle failed during sleep (consolidation unaffected): %s", exc)
        return report

    # -- runtime settings (UI-editable; override config defaults) ---------------------

    _SETTINGS_KEY = "settings"

    def _settings_defaults(self) -> dict[str, Any]:
        """The settable keys and their config-supplied defaults. Config is the headless default;
        these overrides are the live, UI-set preference (stored in kv, so no TOML mutation)."""
        return {
            "timezone": self.config.timezone,                    # IANA name, or None = host local
            "sleep_enabled": self.config.sleep_enabled,
            "sleep_window_start": self.config.sleep_window_start,
            "sleep_window_end": self.config.sleep_window_end,
            "deliberation_enabled": self.config.deliberation_enabled,
            "inner_life_enabled": self.config.inner_life_enabled,
            "inner_life_cadence_s": self.config.inner_life_cadence_s,
        }

    def _overrides(self) -> dict[str, Any]:
        raw = kv_get(self._storage, self._SETTINGS_KEY)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):  # corrupt blob — fall back to config defaults, never crash
            return {}

    def settings(self) -> dict[str, Any]:
        """Effective settings (override → config default) plus the list of keys the user has set."""
        defaults = self._settings_defaults()
        overrides = self._overrides()
        effective = {k: overrides.get(k, d) for k, d in defaults.items()}
        effective["overridden"] = sorted(k for k in overrides if k in defaults)
        return effective

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Validate + persist setting overrides (loud on a bad value); return new effective.
        The sleep scheduler reads these each tick, so a change takes effect without a restart."""
        defaults = self._settings_defaults()
        overrides = self._overrides()
        for key, val in changes.items():
            if key not in defaults:
                raise ConfigError(f"unknown setting: {key!r}")
            if key in ("sleep_window_start", "sleep_window_end"):
                h, m = (int(p) for p in str(val).split(":"))  # raises ValueError → caught upstream
                if not (0 <= h < 24 and 0 <= m < 60):
                    raise ConfigError(f"{key} out of range: {val!r}")
                val = f"{h:02d}:{m:02d}"
            elif key == "timezone":
                val = str(val) if val else None
                known = val and (self._tz_resolves(val) or val in set(self.available_timezones()))
                if val and not known:
                    raise ConfigError(
                        f"unknown timezone: {val!r} (install the optional `tzdata` package "
                        f"for full IANA support)"
                    )
            elif key in ("sleep_enabled", "deliberation_enabled", "inner_life_enabled"):
                val = bool(val)
            elif key == "inner_life_cadence_s":
                val = max(30.0, float(val))  # floor at 30s so a typo can't hammer the fleet
            overrides[key] = val
        kv_set(self._storage, self._SETTINGS_KEY, json.dumps(overrides))
        return self.settings()

    def _tz(self) -> str | None:
        """The effective timezone for all wall-clock reads (override → config)."""
        return self._overrides().get("timezone", self.config.timezone)

    @staticmethod
    def _tz_resolves(tz: str | None) -> bool:
        """Whether ``tz`` resolves to a real zone. UTC offsets always do (pure arithmetic); IANA
        names need the OS tz db or the `tzdata` extra. ``None`` = host-local, always fine."""
        return not tz or resolve_timezone(tz) is not None

    def _effective_window(self) -> tuple[bool, str, str]:
        """(enabled, window_start, window_end) for the sleep cycle, override → config."""
        o = self._overrides()
        return (
            bool(o.get("sleep_enabled", self.config.sleep_enabled)),
            o.get("sleep_window_start", self.config.sleep_window_start),
            o.get("sleep_window_end", self.config.sleep_window_end),
        )

    # Fixed UTC offsets always work (no tz database); they lead the picker so there's a zero-dep way
    # to pin a zone even without `tzdata`. (Offsets don't track DST — fine for a sleep window.)
    _UTC_OFFSETS = ["UTC"] + [
        f"UTC{sign}{h:02d}:00" for sign in ("-", "+") for h in range(1, 13)
    ]

    def available_timezones(self) -> list[str]:
        """Zones for the UI picker: UTC offsets first (always available), then IANA names from the
        system tz db / `tzdata` extra if present, else a small curated IANA fallback."""
        iana: list[str] = []
        try:
            from zoneinfo import available_timezones
            iana = sorted(available_timezones())
        except Exception:  # pragma: no cover - platform without tzdata
            iana = []
        if not iana:
            iana = [
                "America/Vancouver", "America/Los_Angeles", "America/Denver", "America/Chicago",
                "America/New_York", "America/Toronto", "America/Sao_Paulo", "Europe/London",
                "Europe/Paris", "Europe/Berlin", "Europe/Moscow", "Asia/Dubai", "Asia/Kolkata",
                "Asia/Singapore", "Asia/Shanghai", "Asia/Tokyo", "Australia/Sydney",
                "Pacific/Auckland",
            ]
        return self._UTC_OFFSETS + iana

    # -- the wall-clock sleep cycle (DESIGN §5a) --------------------------------------

    _SLEEP_STATE_KEY = "sleep_cycle"
    _last_sleep_report: SleepReport | None = None

    def _sleep_phases(self) -> list[SleepPhase]:
        """The maintenance phases, in order. Consolidation is fast (mostly deterministic, no model);
        narratives make LLM calls, so it needs a bigger slice on a slow box."""
        def _consolidate() -> SleepReport:
            self._last_sleep_report = consolidate(self._storage)
            return self._last_sleep_report

        return [
            SleepPhase("consolidate", min_minutes=2.0, run=_consolidate),
            SleepPhase("self_knowledge", min_minutes=1.0, run=self.bake_self_knowledge),
            SleepPhase("documents", min_minutes=2.0, run=self.ingest_pending_documents),
            SleepPhase("library", min_minutes=3.0, run=self.ingest_pending_library),
            SleepPhase("deliberate", min_minutes=15.0, run=self.deliberate_open_questions),
            SleepPhase("narratives", min_minutes=10.0, run=self.generate_narratives),
            SleepPhase("health", min_minutes=1.0, run=self.digest_errors),  # cheap, no model
        ]

    def _load_sleep_state(self) -> dict:
        raw = kv_get(self._storage, self._SLEEP_STATE_KEY)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (ValueError, TypeError):  # corrupt checkpoint — start fresh, don't crash the cycle
            return {}

    def _save_sleep_state(self, state: dict) -> None:
        kv_set(self._storage, self._SLEEP_STATE_KEY, json.dumps(state))

    def run_sleep_cycle(self, force: bool = False) -> CycleReport:
        """Run the windowed sleep cycle now. ``force=True`` ignores the window + once-a-day guard —
        the manual "run sleep" path. Otherwise honours the effective window and phase budgets."""
        _enabled, start, end = self._effective_window()
        return run_cycle(
            self._sleep_phases(),
            clock=lambda: local_now(self._tz()),
            window_start=start,
            window_end=end,
            load_state=self._load_sleep_state,
            save_state=self._save_sleep_state,
            is_busy=lambda: self._turn_active,
            force=force,
        )

    def sleep_cycle_status(self) -> dict[str, Any]:
        """Effective window + today's checkpoint, for the UI's sleep panel."""
        state = self._load_sleep_state()
        enabled, start, end = self._effective_window()
        now = local_now(self._tz())
        return {
            "enabled": enabled,
            "window_start": start,
            "window_end": end,
            "in_window": in_window(now, start, end),
            "now_local": now.strftime("%Y-%m-%d %H:%M"),
            "utc_offset": now.strftime("%z") or "",  # the offset actually in use (e.g. -0700)
            "timezone": self._tz(),
            "timezone_active": self._tz_resolves(self._tz()),  # False → set but unresolved → host
            "last_cycle_date": state.get("date"),
            "completed": state.get("completed", False),
            "phases": state.get("phases", {}),
        }

    def _start_sleep_scheduler(self) -> None:
        """Daemon: every check interval, if enabled + inside the window (and not done today, not
        mid-turn) run the cycle. Catch-up before noon covers a window missed to an off host.

        Always started (unless the interval is disabled): it reads *effective* settings each tick,
        so the UI toggle/window take effect live without a restart."""
        if self._sleep_scheduler is not None or self.config.sleep_check_interval_s <= 0:
            return
        interval = max(60.0, self.config.sleep_check_interval_s)

        def _loop() -> None:
            log.info("sleep: scheduler started (every %.0fs)", interval)
            while not self._stop_sleep.wait(timeout=interval):
                try:
                    enabled, start, end = self._effective_window()
                    if not enabled:
                        continue
                    now = local_now(self._tz())
                    state = self._load_sleep_state()
                    done_today = (state.get("date") == now.strftime("%Y-%m-%d")
                                  and state.get("completed"))
                    if done_today or self._turn_active:
                        continue
                    catch_up = (not in_window(now, start, end)) and now.hour < 12
                    if in_window(now, start, end) or catch_up:
                        if catch_up:
                            log.info("sleep: window missed; catch-up at %s", now.strftime("%H:%M"))
                        self.run_sleep_cycle()
                except StorageError:  # raced a close() — the store is gone; stop quietly (§10)
                    return
                except Exception as exc:  # the scheduler must never die on a transient error (§10)
                    log.error("sleep: scheduler tick failed: %s", exc, exc_info=True)

        scheduler = threading.Thread(target=_loop, name="mimir-sleep-cycle", daemon=True)
        scheduler.start()
        self._sleep_scheduler = scheduler

    # -- the live inner life: low-frequency idle thinking (DESIGN §5a) -----------------

    def _inner_life_enabled(self) -> bool:
        return bool(self._overrides().get("inner_life_enabled", self.config.inner_life_enabled))

    def _inner_life_cadence(self) -> float:
        val = self._overrides().get("inner_life_cadence_s", self.config.inner_life_cadence_s)
        try:
            return max(30.0, float(val))
        except (TypeError, ValueError):
            return self.config.inner_life_cadence_s

    def _pool_degraded(self) -> bool:
        """Whether the fleet is in no shape to spend idle compute — nothing reachable."""
        try:
            return int(self.pool_health().get("nodes_up", 0)) <= 0
        except Exception:  # health is best-effort; assume OK rather than block forever (§10)
            return False

    def _inner_life_chat(self, messages: list[dict[str, str]]) -> str:
        """Route an inner-life reflection OFF the chat model: the loose ``background`` model if the
        fleet has qualified one, else the reasoning role. Never the warm chat model (keeps turns
        fast and avoids an expensive reload of the identity-bearing model)."""
        name = self.background_model()
        if name:
            return self._model.chat_with_model(name, messages)
        return self._model.chat("reasoning", messages)

    def run_inner_life_tick(self, *, force: bool = False) -> dict[str, Any]:
        """One inner-life cycle: gate → pick a stimulus → compose one cheap reflection → store it as
        a low-confidence, decaying memory (it 'earns its way' back via recall). The daemon calls it
        on its cadence; ``force`` (manual 'think now') bypasses the enable/cadence/idle/health gates
        but still yields to a live turn. Returns a small report; never raises (§10)."""
        now = time.time()
        if force:
            if self._turn_active:
                return {"ran": False, "reason": "turn in flight"}
        else:
            ok, reason = should_think(
                enabled=self._inner_life_enabled(),
                turn_active=self._turn_active,
                degraded=self._pool_degraded(),
                now=now,
                last_turn_at=self._last_turn_at,
                last_thought_at=self._last_thought_at,
                cadence_s=self._inner_life_cadence(),
                idle_floor_s=self.config.inner_life_idle_floor_s,
            )
            if not ok:
                return {"ran": False, "reason": reason}
        try:
            # 'Think now' (force) stays a quick solo musing; the autonomous loop may escalate a
            # fresh conflict to the full council (a daytime forum thread, see _should_escalate).
            return self._do_inner_life(now, allow_escalation=not force)
        except BaseException as exc:  # idle musing must never destabilise the process (§10)
            log.error("inner life: tick failed: %s", exc, exc_info=True)
            return {"ran": False, "reason": f"error: {exc}"}

    def _do_inner_life(self, now: float, *, allow_escalation: bool = True) -> dict[str, Any]:
        errors = [str(r.get("message", "")) for r in self.recent_errors(limit=1)]
        wm = current_working_memory(self._storage) or ""
        stimuli = gather_stimuli(
            self._storage, embedder=self._embedder,
            recent_errors=errors, working_memory_text=wm,
        )
        stim = pick_stimulus(stimuli, avoid_kind=self._last_thought_kind)
        if stim is None:
            return {"ran": False, "reason": "no stimulus"}
        # A genuine, fresh tension occasionally goes to the full council instead of a solo musing —
        # so the inner life feeds the forum during the day, not just the nightly sleep pass (§5a).
        if allow_escalation and stim.kind == "conflict" \
                and self._should_escalate_to_council(stim, now):
            return self._escalate_to_council(stim, now)
        thought = compose_thought(self._inner_life_chat, stim)
        if not thought:
            return {"ran": False, "reason": "empty thought"}
        vec = self._embedder.embed(thought)
        if vec is None:
            # Embedding backend is down — skip rather than store an un-deduplicatable musing that
            # would pile up during the outage (the embedder already logged the degradation).
            return {"ran": False, "reason": "embeddings unavailable"}
        if self._is_duplicate_musing(thought, vec):
            # Don't pile up near-verbatim repeats — the over-retention distillation guards against.
            self._last_thought_at = now  # still spent the cadence; don't immediately retry
            return {"ran": False, "reason": "duplicate musing"}
        mem = Memory(
            text=thought,
            kind=MemoryKind.MEMORY,
            evidence_tier=EvidenceTier.INFERRED,
            confidence=0.3,   # a musing, not a fact — low belief; decays unless reinforced
            salience=0.25,    # starts faint: fades and archives in weeks unless recall revives it
            embedding=vec,
            provenance="inner life",
        )
        save_memory(self._storage, mem)
        self._last_thought_at = now
        self._last_thought_kind = stim.kind
        log.info("inner life: mused on %s (mem %s)", stim.kind, mem.id)
        return {"ran": True, "kind": stim.kind, "thought": thought, "memory_id": mem.id}

    _COUNCIL_ESCALATION_COOLDOWN_S = 3600.0  # at most ~one inner-life-driven council per hour

    def _should_escalate_to_council(self, stim: Stimulus, now: float) -> bool:
        """Whether this idle cycle should convene the council on ``stim`` instead of musing solo.
        Gated so the (expensive) council is a daytime *trickle*: a genuine **conflict** stimulus,
        the self-directed council enabled, a healthy fleet, an hourly cooldown, and the conflict not
        already argued (shares the sleep seen-set, so neither re-litigates the other)."""
        if stim.kind != "conflict" or not self._deliberation_enabled() or self._pool_degraded():
            return False
        if now - self._last_escalation_at < self._COUNCIL_ESCALATION_COOLDOWN_S:
            return False
        return stim.key not in self._load_deliberated()

    def _escalate_to_council(self, stim: Stimulus, now: float) -> dict[str, Any]:
        """Run a full council deliberation on a fresh conflict the idle loop surfaced — persisting a
        forum thread + verdict — and record it in the shared seen-set so the nightly pass skips it.
        """
        try:
            result = deliberate(
                self._model, self._storage, self._embedder,
                question=stim.prompt, provenance="inner life",
            )
        except Exception as exc:  # a bad council run never destabilises the idle loop (§10)
            log.error("inner life: council escalation failed on %r: %s", stim.key, exc)
            self._last_escalation_at = now  # back off before retrying
            return {"ran": False, "reason": f"council error: {exc}"}
        seen = self._load_deliberated()
        seen[stim.key] = local_now(self._tz()).strftime("%Y-%m-%d")
        self._prune_and_save_deliberated(seen)
        self._last_escalation_at = now
        self._last_thought_at = now
        self._last_thought_kind = "conflict"
        log.info("inner life: escalated a conflict to the council (thread %s)", result.thread_id)
        return {"ran": True, "kind": "conflict", "escalated": True,
                "thread_id": result.thread_id, "verdict": result.verdict}

    _MUSING_DUP_COSINE = 0.95

    def _is_duplicate_musing(self, text: str, vec: list[float] | None) -> bool:
        """True if this musing is near-identical to a recent one, so the inner life doesn't accrue
        verbatim repeats. Exact lexical match always counts; cosine only when embeddings are
        semantic (endpoint mode) — a hash embedder's cosine would be noise."""
        norm = " ".join(text.lower().split())
        recent = [
            m for m in list_memories(self._storage, user=None, kind=MemoryKind.MEMORY)
            if (m.provenance or "") == "inner life"
        ]
        recent.sort(key=lambda m: m.created_at, reverse=True)
        semantic = vec is not None and self._embedder.mode.is_semantic
        for m in recent[:20]:
            if " ".join((m.text or "").lower().split()) == norm:
                return True
            if semantic and m.embedding and cosine(vec, m.embedding) >= self._MUSING_DUP_COSINE:
                return True
        return False

    _INNER_LIFE_SURFACE_MIN_RELEVANCE = 0.2  # only a genuinely on-topic musing earns a surface

    def _surface_inner_life(
        self, candidates: list[Memory], query: str, query_vec: list[float] | None,
    ) -> tuple[list[Memory], str | None]:
        """Inner life, Slice 2 (DESIGN §5a): a musing is a framed *reflection*, not a knowledge row.
        Split inner-life memories OUT of the knowledge recall, and if the most relevant one clears
        the bar this turn, return it as a single tentative background note — it 'earns its way in',
        framed as the system's own idle thought, never force-injected and never as fact.
        Returns ``(knowledge_candidates, note_or_None)``."""
        musings = [m for m in candidates if (m.provenance or "") == "inner life"]
        if not musings:
            return candidates, None
        knowledge = [m for m in candidates if (m.provenance or "") != "inner life"]
        top = retrieve(query, query_vec, musings, top_k=1)
        if top and top[0].score >= self._INNER_LIFE_SURFACE_MIN_RELEVANCE:
            return knowledge, (
                f"While idle earlier, I'd found myself thinking: {top[0].memory.text} "
                "(my own tentative reflection — weigh it as such, not as established fact)."
            )
        return knowledge, None

    def recent_thoughts(self, *, limit: int = 12) -> list[dict[str, Any]]:
        """Recent inner-life musings (provenance ``"inner life"``), newest first — for the UI's
        Mind tab. These don't sit in the knowledge recall; this is the window onto them."""
        thoughts = [
            m for m in list_memories(self._storage, user=None, kind=MemoryKind.MEMORY)
            if (m.provenance or "") == "inner life"
        ]
        thoughts.sort(key=lambda m: m.created_at, reverse=True)
        return [
            {"text": m.text, "created_at": m.created_at,
             "salience": round(m.salience, 3), "archived": bool(m.archived)}
            for m in thoughts[: max(1, limit)]
        ]

    def _start_inner_life(self) -> None:
        """Daemon: every check interval run one inner-life tick (which self-gates on
        enable/cadence/idle/health). Always started so the UI toggle takes effect live; the tick is
        a cheap no-op while disabled."""
        if self._inner_life is not None or self.config.inner_life_check_interval_s <= 0:
            return
        interval = max(5.0, self.config.inner_life_check_interval_s)

        def _loop() -> None:
            log.info("inner life: idle loop started (checks every %.0fs)", interval)
            while not self._stop_inner.wait(timeout=interval):
                try:
                    self.run_inner_life_tick()
                except StorageError:  # raced a close() — the store is gone; stop quietly (§10)
                    return
                except Exception as exc:  # the loop must never die on a transient error (§10)
                    log.error("inner life: loop tick failed: %s", exc, exc_info=True)

        thread = threading.Thread(target=_loop, name="mimir-inner-life", daemon=True)
        thread.start()
        self._inner_life = thread

    # -- self-observability: recent errors into context + a nightly digest (DESIGN §10) ----

    _HEALTH_DIGEST_KEY = "health_digest"

    def recent_errors(self, *, limit: int = 10, min_level: str = "WARNING") -> list[dict[str, Any]]:
        """Recent captured errors (WARNING+), oldest-first — for the UI and introspection."""
        return [r.as_dict() for r in self._errors.recent(limit=limit, min_level=min_level)]

    @staticmethod
    def _short_node(name: str) -> str:
        """A compact node label: the last IP octet (``…189``) for a URL, else the bare name."""
        host = name.split("//", 1)[-1].split(":", 1)[0]
        return f"…{host.rsplit('.', 1)[-1]}" if host.count(".") == 3 else (host or name)

    def pool_health(self) -> dict[str, Any]:
        """Backend pool health for the UI: nodes up, down, saturated, and per-node speeds."""
        stats = self._model.get_stats()
        endpoints = list(stats.get("endpoints", []))
        return {
            "nodes": len(endpoints),
            "nodes_up": int(stats.get("nodes_up", 0)),
            "down": list(stats.get("down", [])),
            "saturated": stats.get("saturated", {}) or {},
            "latency": stats.get("latency", {}) or {},  # node → fastest known s/turn
        }

    def _backend_health_line(self) -> tuple[str, bool] | None:
        """``(line, degraded)`` — a one-line backend summary (nodes up/down, saturation, per-node
        speeds) plus whether it's degraded. ``None`` for a single local provider (no fleet)."""
        h = self.pool_health()
        if h["nodes"] <= 1:
            return None  # a single local provider — no fleet health worth narrating
        degraded = bool(h["down"]) or bool(h["saturated"])
        parts = [f"Backend: {h['nodes_up']}/{h['nodes']} nodes up"]
        if h["down"]:
            parts.append("down: " + ", ".join(self._short_node(n) for n in h["down"]))
        if h["saturated"]:
            parts.append("saturated: " + ", ".join(
                f"{self._short_node(n)} ({s:.0f}s)" for n, s in h["saturated"].items()))
        if h["latency"]:
            speeds = sorted(h["latency"].items(), key=lambda kv: kv[1])
            parts.append(
                "speeds: " + ", ".join(f"{self._short_node(n)} {t:.1f}s" for n, t in speeds))
        return "; ".join(parts) + ".", degraded

    def _error_context(self) -> str | None:
        """The system-health block for this turn's prompt: recent errors + backend pool health.

        Errors inside the recency window (so a fixed problem fades), capped; plus a one-line backend
        summary when there's a fleet that's degraded or there are errors to report alongside.
        ``None`` when there's nothing worth saying — observability without noise (DESIGN §10)."""
        if not self.config.surface_errors:
            return None
        blocks: list[str] = []
        recent = self._errors.within(
            self.config.error_context_window_s, time.time(), limit=self.config.error_context_max
        )
        if recent:
            blocks.append(render_errors(recent))
        backend = self._backend_health_line()
        # Surface backend status when it's degraded (a node down/saturated) or alongside errors —
        # not on every healthy turn, to keep the prompt quiet when all is well.
        if backend and (backend[1] or recent):
            blocks.append(backend[0])
        return "\n".join(blocks) if blocks else None

    def digest_errors(self) -> dict[str, Any]:
        """The sleep cycle's health pass: summarize the period's errors and record the digest (kv),
        so the nightly cycle 'reviews' what went wrong and it survives a restart. Returns it.
        """
        counts = self._errors.counts()
        samples = [r.as_dict() for r in self._errors.recent(limit=10, min_level="WARNING")]
        digest = {
            "date": local_now(self._tz()).strftime("%Y-%m-%d"),
            "counts": counts,
            "total": sum(counts.values()),
            "samples": samples,
            "generated_at": time.time(),
        }
        kv_set(self._storage, self._HEALTH_DIGEST_KEY, json.dumps(digest))
        log.info("sleep: health digest — %d issue(s) logged this session (%s)",
                 digest["total"],
                 ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none")
        return digest

    def reembed(self) -> dict[str, int]:
        """Re-embed every stored vector (memories + library claims + procedure triggers) with the
        CURRENT embed model, so the whole store shares one vector space.

        Embeddings of the SAME dimension produced by DIFFERENT models are NOT comparable — switching
        embed models silently corrupts recall until the store is rebuilt. Run this once after any
        embed-model change. Best run with the live server stopped (exclusive DB writer); rows whose
        embed call fails are left untouched (counted under 'failed') so a partial outage is non-
        destructive. The embed model must be reachable, or every row 'fails' and nothing changes."""
        if self._embedder.mode is EmbeddingMode.DEGRADED:
            log.warning("reembed: embedder degraded (%s) — aborting; fix the embed backend first",
                        self._embedder.mode.banner())
            return {"memories": 0, "claims": 0, "procedures": 0, "failed": 0, "aborted": 1}
        counts = {"memories": 0, "claims": 0, "procedures": 0, "failed": 0}
        mem_updates: list[tuple[list[float] | None, int]] = []
        for m in browse_memories(self._storage, kind=MemoryKind.MEMORY, limit=1_000_000):
            if not (m.text and m.id):
                continue
            vec = self._embedder.embed(m.text)
            if vec is None:
                counts["failed"] += 1
                continue
            mem_updates.append((vec, m.id))
            counts["memories"] += 1
        reembed_memories(self._storage, mem_updates)

        claim_updates: list[tuple[list[float] | None, int]] = []
        for c in list_library_claims(self._storage):
            if not (c.text and c.id):
                continue
            vec = self._embedder.embed(c.text)
            if vec is None:
                counts["failed"] += 1
                continue
            claim_updates.append((vec, c.id))
            counts["claims"] += 1
        reembed_claims(self._storage, claim_updates)

        proc_updates: list[tuple[list[float] | None, int]] = []
        for p in list_procedures(self._storage):
            if not (p.trigger and p.id):
                continue
            vec = self._embedder.embed(p.trigger)
            if vec is None:
                counts["failed"] += 1
                continue
            proc_updates.append((vec, p.id))
            counts["procedures"] += 1
        reembed_procedures(self._storage, proc_updates)

        log.info("reembed (model=%s): %s", self._embedder.mode.banner(), counts)
        return counts

    _SELF_KNOWLEDGE_HASH_KEY = "self_knowledge_hash"

    def bake_self_knowledge(self, *, force: bool = False) -> dict[str, Any]:
        """Bake the self-knowledge doc (default README) into memory so the system can answer about
        what it is and how it works. Content-hashed: re-embeds only when the doc changed (or force).

        A sleep-cycle phase; also reachable via 'Run sleep now'. Recall (and the self-model, which
        reads the store) then draw on it. Fail-soft — never sinks the cycle (DESIGN §10)."""
        doc = self.config.self_knowledge_doc
        if not doc:
            return {"baked": False, "reason": "disabled"}
        path = Path(doc)
        if not path.is_file():
            log.warning("self-knowledge: doc not found, skipping: %s", path)
            return {"baked": False, "reason": "not found", "path": str(path)}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not force and kv_get(self._storage, self._SELF_KNOWLEDGE_HASH_KEY) == digest:
            return {"baked": False, "reason": "unchanged", "path": str(path)}
        try:
            result = ingest_document(self._storage, self._embedder, path=path)
        except IngestError as exc:
            log.error("self-knowledge: could not bake %s: %s", path, exc)
            return {"baked": False, "reason": str(exc), "path": str(path)}
        kv_set(self._storage, self._SELF_KNOWLEDGE_HASH_KEY, digest)
        log.info("self-knowledge: baked %s (%d chunk(s)) into memory", path.name,
                 result.chunks_written)
        return {"baked": True, "path": str(path), "chunks": result.chunks_written}

    # -- documents drop-folder → recallable knowledge + a small local "wiki" (DESIGN §8) ----

    _DOCS_LEDGER_KEY = "documents"

    def _docs_folder(self) -> Path | None:
        return Path(self.config.documents_folder) if self.config.documents_folder else None

    def _load_docs_ledger(self) -> dict[str, Any]:
        raw = kv_get(self._storage, self._DOCS_LEDGER_KEY)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    def _record_document(self, path: Path) -> IngestResult:
        """Ingest one document into recallable knowledge and record it in the wiki ledger. A changed
        file drops its stale summary (regenerated lazily in the idle pass)."""
        start = time.time()
        result = ingest_document(self._storage, self._embedder, path=path)
        ledger = self._load_docs_ledger()
        entry = ledger.get(result.source, {})
        entry.update(
            name=path.name, hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            chunks=result.chunks_written, ingested_at=time.time(),
            ingest_seconds=round(time.time() - start, 1),  # chunk+embed cost, for the UI
        )
        entry.pop("summary", None)  # content changed → old summary is stale
        ledger[result.source] = entry
        kv_set(self._storage, self._DOCS_LEDGER_KEY, json.dumps(ledger))
        return result

    def _summarize_document(self, path: Path) -> str | None:
        """A short 'wiki' summary of a document (one reasoning call). Best-effort: None on any
        failure (e.g. the model fleet is down) — the doc is still recallable from its chunks."""
        try:
            text = "\n".join(u.text for u in extract(path))[:6000]
            if not text.strip():
                return None
            summary = self._model.chat(
                "reasoning",
                [{"role": "system", "content": DOC_SUMMARY_SYSTEM},
                 {"role": "user", "content": text}],
            ).strip()
            return summary or None
        except Exception as exc:  # summarization is enrichment — never sink ingestion (§10)
            log.warning("documents: could not summarize %s: %s", path.name, exc)
            return None

    def upload_document(self, filename: str, data: bytes) -> dict[str, Any]:
        """Save an uploaded file into the drop folder and ingest it now (recallable immediately; its
        wiki summary is generated in the next idle pass). The 📎 button calls this."""
        folder = self._docs_folder()
        if folder is None:
            raise ConfigError("no [documents] folder configured")
        safe = Path(filename).name  # strip any path components from the client
        if not safe or Path(safe).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise IngestError(
                f"unsupported document {filename!r}; allowed: {sorted(SUPPORTED_SUFFIXES)}"
            )
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / safe
        dest.write_bytes(data)
        result = self._record_document(dest)
        log.info("documents: uploaded + ingested %s (%d chunk(s))", safe, result.chunks_written)
        return {"name": safe, "chunks": result.chunks_written}

    def ingest_pending_documents(self, *, force: bool = False) -> dict[str, Any]:
        """Idle pass over the drop folder: ingest new/changed docs and fill missing wiki summaries.
        A sleep phase; also a manual 'scan folder' trigger. Fail-soft per file (§10)."""
        folder = self._docs_folder()
        if folder is None:
            return {"folder": None, "ingested": [], "summarized": 0, "failed": [],
                    "unsupported": [], "forgotten": []}
        ingested: list[str] = []
        failed: list[dict[str, str]] = []  # {name, error} — surfaced in the UI, not swallowed (§10)
        summarized = 0
        for path in list_documents(folder):
            source = str(path.resolve())
            ledger = self._load_docs_ledger()
            entry = ledger.get(source, {})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if force or entry.get("hash") != digest:
                try:
                    self._record_document(path)
                    ingested.append(path.name)
                except Exception as exc:  # per-file isolation: one bad doc never sinks scan (§10)
                    log.warning("documents: skipping %s: %s", path.name, exc)
                    failed.append({"name": path.name, "error": str(exc)})
                    continue
            ledger = self._load_docs_ledger()  # re-read: _record_document rewrote it
            entry = ledger.get(source, {})
            if not entry.get("summary"):
                summary = self._summarize_document(path)
                if summary:
                    entry["summary"] = summary
                    ledger[source] = entry
                    kv_set(self._storage, self._DOCS_LEDGER_KEY, json.dumps(ledger))
                    summarized += 1
        # Files in the folder we won't touch (wrong type), so an odd drop isn't a silent skip.
        unsupported = [f.name for f in folder.iterdir() if f.is_file()
                       and not f.name.startswith(".")
                       and f.suffix.lower() not in SUPPORTED_SUFFIXES] if folder.is_dir() else []
        # Reverse cleanup: a previously-ingested file that's now gone gets fully forgotten (memories
        # + library + composite + ledger), so deleting a file from the folder self-heals on scan.
        forgotten: list[str] = []
        for src in list(self._load_docs_ledger()):
            if not Path(src).is_file():
                self.forget_document(src)
                forgotten.append(Path(src).name)
        if ingested or summarized:
            log.info("documents: ingested %d, summarized %d from %s",
                     len(ingested), summarized, folder)
        if failed:
            log.warning("documents: %d file(s) failed to ingest from %s", len(failed), folder)
        if forgotten:
            log.info("documents: forgot %d deleted file(s): %s", len(forgotten), forgotten)
        return {"folder": str(folder), "ingested": ingested, "summarized": summarized,
                "failed": failed, "unsupported": unsupported, "forgotten": forgotten}

    def documents(self) -> list[dict[str, Any]]:
        """The ingested-document 'wiki' for the UI: name, chunks, summary, when — newest first."""
        ledger = self._load_docs_ledger()
        items = [{"source": s, **e} for s, e in ledger.items()]
        items.sort(key=lambda d: d.get("ingested_at", 0.0), reverse=True)
        return items

    # -- the Library layer (docs/LIBRARY.md): source docs → cited claims (Phase 1b) --------

    def _library_source_folder(self) -> Path | None:
        # Ground truth = the documents drop folder, where the user left the files (in place).
        return Path(self.config.documents_folder) if self.config.documents_folder else None

    def _claim_chat(self, messages: list[dict[str, str]]) -> str:
        return self._model.chat("reasoning", messages)  # claim extraction, off the chat model

    def _library_compose_folder(self) -> Path | None:
        # Where the Markdown composites (the fuzzy understanding) are written — a separate tree.
        return Path(self.config.library_folder) if self.config.library_folder else None

    _LIBRARY_TIMINGS_KEY = "library_seconds"

    def _library_timings(self) -> dict[str, float]:
        """Per-document index time (seconds) from the last library pass — 'how long per doc'."""
        raw = kv_get(self._storage, self._LIBRARY_TIMINGS_KEY)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    def ingest_pending_library(self, *, force: bool = False) -> dict[str, Any]:
        """Idle pass over the source documents (ground truth, left in place): record each, distil it
        into cited claims, and compile a Markdown composite (the fuzzy understanding) from those
        claims. Drops a document whose file is gone (cascading its claims). A sleep phase; fail-soft
        per file (§10)."""
        folder = self._library_source_folder()
        if folder is None:
            return {"folder": None, "documents": [], "claims": 0, "composed": 0, "dropped": 0}
        existing = {d.path: d for d in list_library_documents(self._storage)}
        seen: set[str] = set()
        processed: list[str] = []
        total_claims = 0
        composed = 0
        timings = self._library_timings()  # path → seconds to index (claim-extract + compose)
        for path in list_documents(folder):
            try:
                data = path.read_bytes()
            except OSError:
                continue
            source = str(path.resolve())
            seen.add(source)
            digest = hashlib.sha256(data).hexdigest()
            prior = existing.get(source)
            doc_id = upsert_library_document(self._storage, LibraryDocument(
                path=source, filename=path.name, size_bytes=len(data),
                content_hash=digest, title=path.stem))
            if prior is not None and prior.content_hash == digest and not force:
                continue  # unchanged → keep its claims + composite
            try:
                units = extract(path)
            except IngestError as exc:
                log.warning("library: cannot read %s: %s", path.name, exc)
                continue
            start = time.time()
            claims: list[LibraryClaim] = []
            for unit in units:
                for text in extract_claims(self._claim_chat, unit.text):
                    claims.append(LibraryClaim(
                        document_id=doc_id, text=text, locator=unit.locator,
                        embedding=self._embedder.embed(text)))
            replace_document_claims(self._storage, doc_id, claims)
            total_claims += len(claims)
            processed.append(path.name)
            if self._compile_composite(doc_id, path.stem):
                composed += 1
            timings[source] = round(time.time() - start, 1)  # index cost for this doc (the UI)
        kv_set(self._storage, self._LIBRARY_TIMINGS_KEY,
               json.dumps({k: v for k, v in timings.items() if k in seen}))
        dropped = 0
        for path_str in existing:  # a source file that vanished → fully forget it (DB + memories)
            if path_str not in seen:
                self.forget_document(path_str)
                dropped += 1
        if processed or dropped:
            log.info("library: %d doc(s) → %d claim(s), %d composite(s), dropped %d",
                     len(processed), total_claims, composed, dropped)
        return {"folder": str(folder), "documents": processed, "claims": total_claims,
                "composed": composed, "dropped": dropped}

    def _resolve_source(self, source: str) -> str:
        """Map a UI identifier (resolved path, or a bare filename) to the canonical source key — the
        one shared by memories.source / library_documents.path / the ledger. Prefers a known key."""
        known = {d.path for d in list_library_documents(self._storage)}
        known |= set(self._load_docs_ledger())
        if source in known:
            return source
        for k in known:  # match by basename (the UI may show only the filename)
            if Path(k).name == source:
                return k
        folder = self._docs_folder()
        if folder is not None and (folder / source).exists():
            return str((folder / source).resolve())
        return str(Path(source).resolve()) if source else source

    def forget_document(self, source: str, *, delete_file: bool = False) -> dict[str, Any]:
        """Purge a document and everything derived from it: its document-tier memory chunks, its
        library document + cited claims, any composite page(s) (DB row + Markdown file) left
        orphaned, and the wiki ledger entry. Keyed by the shared source path. With ``delete_file``
        the source file is removed too, so an idle scan won't re-ingest it. Idempotent — forgetting
        an already-gone document is a clean no-op.

        The single primitive behind both cleanup directions: the Library 'delete' button calls it
        with ``delete_file=True``; the idle scans call it (no file delete) when a file has vanished.
        """
        src = self._resolve_source(source)
        # Composite page(s) derived from this doc — captured before delete cuts the claim links.
        page_ids: set[int] = set()
        doc = {d.path: d for d in list_library_documents(self._storage)}.get(src)
        if doc is not None and doc.id is not None:
            claim_ids = [c.id for c in claims_for_document(self._storage, doc.id) if c.id]
            for pids in pages_for_claims(self._storage, claim_ids).values():
                page_ids.update(pids)

        chunks = delete_by_source(self._storage, src)           # document-tier memory chunks
        lib_docs = delete_library_document(self._storage, src)  # library doc + claims + page links
        pages_removed = 0
        for pid in page_ids:  # delete composites that this removal left with no remaining claims
            if claims_for_page(self._storage, pid):
                continue
            page = get_library_page(self._storage, pid)
            if page and page.path:
                try:
                    Path(page.path).unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("forget: could not remove composite %s: %s", page.path, exc)
            delete_library_page(self._storage, pid)
            pages_removed += 1

        ledger = self._load_docs_ledger()  # drop the wiki ledger entry (keyed by the source)
        if src in ledger:
            del ledger[src]
            kv_set(self._storage, self._DOCS_LEDGER_KEY, json.dumps(ledger))

        file_deleted = False
        if delete_file:
            try:
                p = Path(src)
                if p.is_file():
                    p.unlink()
                    file_deleted = True
            except OSError as exc:
                log.warning("forget: could not delete source file %s: %s", src, exc)

        log.info("forget: %s — %d chunk(s), %d lib doc, %d page(s), file_deleted=%s",
                 Path(src).name, chunks, lib_docs, pages_removed, file_deleted)
        return {"source": src, "memory_chunks": chunks, "library_doc": lib_docs,
                "pages": pages_removed, "file_deleted": file_deleted}

    def _compile_composite(self, document_id: int, title: str) -> bool:
        """Synthesize a Markdown composite from a document's claims (the fuzzy understanding) and
        link it to those claims. Non-destructive: a hand-edited page (file hash ≠ last written) is
        left alone. Returns True if it wrote a page. Needs a library compose folder set."""
        folder = self._library_compose_folder()
        if folder is None:
            return False
        claims = claims_for_document(self._storage, document_id)
        if not claims:
            return False
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "page"
        dest = folder / f"{safe}.md"
        prior = next((p for p in list_library_pages(self._storage) if p.path == str(dest)), None)
        if dest.is_file() and prior is not None:
            on_disk = hashlib.sha256(dest.read_bytes()).hexdigest()
            if on_disk != prior.content_hash:
                log.info("library: %s was hand-edited; leaving the composite as-is", dest.name)
                return False
        summary, markdown = compose_page(self._claim_chat, title, [c.text for c in claims])
        if not markdown:
            return False
        folder.mkdir(parents=True, exist_ok=True)
        dest.write_text(markdown, encoding="utf-8")
        page_id = upsert_library_page(self._storage, LibraryPage(
            path=str(dest), title=title, summary=summary,
            content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest()))
        set_page_claims(self._storage, page_id, [c.id for c in claims if c.id is not None])
        return True

    def _known_source_labels(self) -> set[str]:
        """Every document/source name the system actually holds — library documents (filename +
        title) and the drop-folder wiki ledger — so the citation guard can tell a real citation from
        an invented one (DESIGN §10)."""
        labels: set[str] = set()
        for d in list_library_documents(self._storage):
            if d.filename:
                labels.add(d.filename)
            if d.title:
                labels.add(d.title)
        for entry in self._load_docs_ledger().values():
            name = entry.get("name") if isinstance(entry, dict) else None
            if name:
                labels.add(name)
        return labels

    # -- context-layer + per-document toggles (what the user wants in-context this turn) ------

    _DISABLED_DOCS_KEY = "disabled_documents"

    def _disabled_documents(self) -> set[str]:
        """Source paths the user toggled OFF in the Library ('include in context' unchecked). Their
        chunks + claims are excluded from recall — so an unselected book costs nothing to scan."""
        raw = kv_get(self._storage, self._DISABLED_DOCS_KEY)
        if not raw:
            return set()
        try:
            value = json.loads(raw)
            return set(value) if isinstance(value, list) else set()
        except (ValueError, TypeError):
            return set()

    def set_document_enabled(self, source: str, enabled: bool) -> dict[str, Any]:
        """Toggle a document's 'include in context'. Disabled docs are excluded from recall (their
        memory chunks and library claims) until re-enabled — data is kept, just not consulted."""
        src = self._resolve_source(source)
        disabled = self._disabled_documents()
        if enabled:
            disabled.discard(src)
        else:
            disabled.add(src)
        kv_set(self._storage, self._DISABLED_DOCS_KEY, json.dumps(sorted(disabled)))
        return {"source": src, "enabled": enabled}

    def _disabled_doc_ids(self, disabled_docs: set[str] | None) -> set[int] | None:
        """Map disabled source paths → library document ids (for SQL-side claim exclusion)."""
        if not disabled_docs:
            return None
        return {d.id for d in list_library_documents(self._storage)
                if d.id is not None and d.path in disabled_docs}

    def _filter_layers(
        self, candidates: list[Memory], include_memory: bool, include_library: bool
    ) -> list[Memory]:
        """Apply the per-turn layer toggles to the knowledge candidates: document-tier chunks belong
        to the 'document library' layer, everything else to the 'memory' layer."""
        if include_memory and include_library:
            return candidates
        out: list[Memory] = []
        for m in candidates:
            is_doc = m.evidence_tier is EvidenceTier.DOCUMENT
            if is_doc and not include_library:
                continue
            if not is_doc and not include_memory:
                continue
            out.append(m)
        return out

    def _citation_note(self, reply: str) -> str:
        """A fail-loud note if the reply cited a source the system doesn't hold ('' if none/off)."""
        if not self.config.library_citation_guard:
            return ""
        bad = unverified_citations(reply, self._known_source_labels())
        if bad:
            log.warning("citation guard: reply cited unknown source(s): %s", bad)
        return citation_warning(bad)

    def _library_gist(
        self, query: str, query_vec: list[float] | None, disabled_docs: set[str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], int]:
        """The Library section + the composite pages behind it. Returns ``(text, refs, count)`` —
        the cited claims most relevant to the turn (each shown with its source title + locator), the
        composite page(s) those claims belong to (``[{page_id, title}]``) for the after-reply chips,
        and how many claims surfaced (grounding signal for the uncertainty gate). ``disabled_docs``
        (per-doc 'include in context' toggles) are excluded at the SQL layer. ``(None, [], 0)`` if
        the layer is off or nothing is on-topic."""
        if self._library_source_folder() is None:
            return None, [], 0
        claims = list_library_claims(
            self._storage, exclude_doc_ids=self._disabled_doc_ids(disabled_docs))
        if not claims:
            return None, [], 0
        top = retrieve_claims(query, query_vec, claims,
                              top_k=max(1, self.config.library_claims_top_k))
        if not top:
            return None, [], 0
        titles = {d.id: d.title for d in list_library_documents(self._storage)}
        text = render_claims(top, titles) or None
        # Which composite page(s) those surfaced claims belong to → after-reply Load chips.
        claim_pages = pages_for_claims(self._storage, [c.claim.id for c in top if c.claim.id])
        page_ids = {pid for pids in claim_pages.values() for pid in pids}
        page_titles = {p.id: p.title for p in list_library_pages(self._storage)}
        refs = [{"page_id": pid, "title": page_titles.get(pid, "")} for pid in sorted(page_ids)]
        return text, refs, len(top)

    def _loaded_library_detail(self, page_ids: list[int]) -> str:
        """The full Markdown of the given composite pages, as one block (empty if none load)."""
        blocks: list[str] = []
        for pid in page_ids or []:
            page = self.library_page(int(pid))
            if page and page.get("markdown"):
                blocks.append(f"## {page['title']}\n{page['markdown']}")
        return "Full pages you've loaded:\n\n" + "\n\n".join(blocks) if blocks else ""

    def _merge_loaded_library(self, gist: str | None, loaded_pages: list[int] | None) -> str | None:
        """Append the full Markdown of any composite pages the user explicitly **loaded** (the Load
        button / 'active sources') to the Library section, so a pulled page is in this turn's
        context. Detail the user chose to spend the window on — beyond the always-on gist."""
        detail = self._loaded_library_detail(loaded_pages or [])
        if not detail:
            return gist
        return f"{gist}\n\n{detail}" if gist else detail

    def _pages_to_load(
        self, loaded_pages: list[int] | None, deep_read: bool, refs: list[dict[str, Any]]
    ) -> list[int]:
        """Composite pages whose FULL Markdown should go into this turn: the ones the user loaded
        (Load button), plus — when 'deep read' is on — the page(s) the surfaced claims came
        from, so the model gets the whole composite, not just the cited gist (deterministic)."""
        pages = list(dict.fromkeys(loaded_pages or []))  # de-duped, order preserved
        if deep_read:
            for r in refs:
                pid = r.get("page_id")
                if pid is not None and pid not in pages:
                    pages.append(pid)
        return pages

    _FETCH_RE = re.compile(r"<FETCH\s+id=['\"]?(\d+)['\"]?\s*/?>", re.IGNORECASE)

    def _fetch_hint_text(self, refs: list[dict[str, Any]]) -> str:
        """The Phase-2 fetch instruction + available page ids, so the model can open one."""
        listing = ", ".join(f"{r['page_id']}: {r['title']}" for r in refs if r.get("page_id"))
        return ("If you need the full source of any of these to answer, reply with exactly "
                f"<FETCH id=N> (and nothing else) — available pages: {listing}.")

    def _with_fetch_hint(self, library: str | None, refs: list[dict[str, Any]]) -> str:
        """Append the Phase-2 fetch instruction so the model can open a page."""
        hint = self._fetch_hint_text(refs)
        return f"{library}\n\n{hint}" if library else hint

    def _stream_chat_with_fetch(
        self, bundle: ContextBundle, refs: list[dict[str, Any]],
        user: str | None, sid: str | None, text: str,
    ) -> Generator[str, None, str]:
        """Stream the chat reply, honoring an opt-in model fetch of a full library page (Phase 2,
        streaming). If model-fetch is on and the model opens with ``<FETCH id=N>``, that marker is
        intercepted (never shown to the user), the page is loaded, and the FINAL answer is streamed
        with the page in context. Returns the full (tag-stripped) reply for storage/bake."""
        history = self._history_messages(user, sid)
        do_fetch = self.config.library_model_fetch and bool(refs)
        system = f"{bundle.prompt}\n\n{self._fetch_hint_text(refs)}" if do_fetch else bundle.prompt
        gen = self._model.chat_stream(
            "chat", [{"role": "system", "content": system}, *history,
                     {"role": "user", "content": text}])

        # Peek the opening: a fetch reply is *exactly* the marker, so a few non-whitespace chars
        # tell us whether to intercept it — we never stream the marker to the user.
        first_raw: list[str] = []
        is_fetch = False
        if do_fetch:
            for delta in gen:
                first_raw.append(delta)
                head = "".join(first_raw).lstrip()
                if len(head) >= 6:
                    is_fetch = head[:6].upper() == "<FETCH"
                    break
                if head and not "<FETCH".startswith(head.upper()):
                    break  # diverged from the marker prefix — it's a normal answer
            else:
                is_fetch = "".join(first_raw).lstrip()[:6].upper() == "<FETCH"

        stripper = StreamTagStripper()
        chunks: list[str] = []
        if is_fetch:
            full = "".join(first_raw) + "".join(gen)  # drain the (short) marker reply
            cap = max(1, self.config.library_max_fetches)
            ids = [int(m) for m in self._FETCH_RE.findall(full)][:cap]
            detail = self._loaded_library_detail(ids) if ids else ""
            log.info("library: model fetched page(s) %s (stream)", ids)
            system2 = f"{bundle.prompt}\n\n{detail}" if detail else bundle.prompt
            gen = self._model.chat_stream(
                "chat", [{"role": "system", "content": system2}, *history,
                         {"role": "user", "content": text}])
        else:
            for delta in first_raw:  # replay what we buffered while peeking, then continue
                clean = stripper.feed(delta)
                if clean:
                    chunks.append(clean)
                    yield clean
        for delta in gen:
            clean = stripper.feed(delta)
            if clean:
                chunks.append(clean)
                yield clean
        tail = stripper.flush()
        if tail:
            chunks.append(tail)
            yield tail
        return "".join(chunks)

    def _maybe_model_fetch(
        self, reply: str, bundle: ContextBundle, user: str | None, sid: str | None, text: str
    ) -> str:
        """Phase 2 (docs/LIBRARY.md): if model-fetch is on and the model asked to open page(s) with
        ``<FETCH id=N>``, load them and answer once more with the detail in hand. One bounded extra
        pass; the marker is stripped from the reply. Off by default (a deliberate 2nd call)."""
        if not self.config.library_model_fetch:
            return reply
        cap = max(1, self.config.library_max_fetches)
        ids = [int(m) for m in self._FETCH_RE.findall(reply)][:cap]
        if not ids:
            return reply
        detail = self._loaded_library_detail(ids)
        if not detail:
            return self._FETCH_RE.sub("", reply).strip()  # bad ids → just drop the marker
        log.info("library: model fetched page(s) %s", ids)
        reply2 = strip_epistemic_tags(self._model.chat(
            "chat",
            [
                {"role": "system", "content": f"{bundle.prompt}\n\n{detail}"},
                *self._history_messages(user, sid),
                {"role": "user", "content": text},
            ],
        ))
        return self._FETCH_RE.sub("", reply2).strip()

    def library_overview(self) -> dict[str, Any]:
        """The Library for the UI: source documents (with claim counts) + composite pages."""
        docs = list_library_documents(self._storage)
        counts = {d.id: len(claims_for_document(self._storage, d.id)) for d in docs}
        disabled = self._disabled_documents()
        timings = self._library_timings()
        return {
            "source_folder": self.config.documents_folder,
            "compose_folder": self.config.library_folder,
            "documents": [
                {"id": d.id, "filename": d.filename, "title": d.title, "path": d.path,
                 "size_bytes": d.size_bytes, "claims": counts.get(d.id, 0),
                 "ingested_at": d.ingested_at, "enabled": d.path not in disabled,
                 "index_seconds": timings.get(d.path)}
                for d in docs
            ],
            "pages": [
                {"id": p.id, "title": p.title, "summary": p.summary, "path": p.path,
                 "updated_at": p.updated_at}
                for p in list_library_pages(self._storage)
            ],
        }

    def library_page(self, page_id: int) -> dict[str, Any] | None:
        """A composite page with its full Markdown loaded on demand, plus its source citations (the
        claims it was composed from → their document + locator). The Load button / fetch path."""
        page = get_library_page(self._storage, page_id)
        if page is None:
            return None
        try:
            markdown = Path(page.path).read_text(encoding="utf-8")
        except OSError as exc:  # missing/renamed file → a noted gap, never a crash (§10)
            log.warning("library: cannot load composite %s: %s", page.path, exc)
            markdown = page.summary
        titles = {d.id: d.title for d in list_library_documents(self._storage)}
        citations = [
            {"text": c.text, "title": titles.get(c.document_id, ""), "locator": c.locator}
            for c in claims_for_page(self._storage, page_id)
        ]
        return {"id": page.id, "title": page.title, "summary": page.summary,
                "markdown": markdown, "citations": citations}

    def library_source(self, document_id: int) -> dict[str, Any] | None:
        """A source document's verbatim text loaded on demand from disk — ground truth, for quoting
        or checking a cited line."""
        doc = next((d for d in list_library_documents(self._storage) if d.id == document_id), None)
        if doc is None:
            return None
        try:
            text = "\n".join(u.text for u in extract(Path(doc.path)))
        except (IngestError, OSError) as exc:
            log.warning("library: cannot load source %s: %s", doc.path, exc)
            text = ""
        return {"id": doc.id, "filename": doc.filename, "title": doc.title, "text": text}

    def health_digest(self) -> dict[str, Any] | None:
        """The last nightly health digest (kv), or None if the sleep cycle hasn't run one yet."""
        raw = kv_get(self._storage, self._HEALTH_DIGEST_KEY)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    # -- self-directed deliberation (the council, on the system's own conflicts; DESIGN §5a) --

    _DELIB_SEEN_KEY = "deliberated"
    _DELIB_SEEN_TTL_DAYS = 30  # re-argue a conflict only if it's still around this long later

    def _deliberation_enabled(self) -> bool:
        return bool(self._overrides().get("deliberation_enabled", self.config.deliberation_enabled))

    def _load_deliberated(self) -> dict[str, str]:
        raw = kv_get(self._storage, self._DELIB_SEEN_KEY)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    def deliberate_open_questions(self, *, force: bool = False) -> dict[str, Any]:
        """Surface the system's own conflicts → curate → submit each to the council (DESIGN §5a).

        The sleep cycle calls this as a phase; ``/api/deliberate/run`` calls it with ``force``.
        Skips conflicts argued within the last ``_DELIB_SEEN_TTL_DAYS`` so it doesn't loop nightly.
        Returns a small report (what was argued + counts)."""
        if not force and not self._deliberation_enabled():
            return {"enabled": False, "ran": []}
        conflicts = surface_conflicts(self._storage, embedder=self._embedder)
        today = local_now(self._tz()).strftime("%Y-%m-%d")
        seen = self._load_deliberated()
        fresh = [c for c in conflicts if c.key not in seen]
        chosen = curate(self._model, fresh, limit=max(1, self.config.deliberation_limit))
        results: list[dict[str, Any]] = []
        for conflict in chosen:
            try:
                outcome = deliberate(
                    self._model, self._storage, self._embedder,
                    question=conflict.question, provenance="sleep deliberation",
                )
            except Exception as exc:  # one bad council run never sinks the phase (§10)
                log.error("deliberation: council failed on %r: %s", conflict.key, exc)
                continue
            seen[conflict.key] = today
            results.append({"question": conflict.question, "verdict": outcome.verdict,
                            "memory_id": outcome.memory_id})
        self._prune_and_save_deliberated(seen)
        log.info("deliberation: %d conflict(s) surfaced, %d fresh, %d argued",
                 len(conflicts), len(fresh), len(results))
        return {"enabled": True, "surfaced": len(conflicts), "fresh": len(fresh), "ran": results}

    def _prune_and_save_deliberated(self, seen: dict[str, str]) -> None:
        cutoff = local_now(self._tz()) - timedelta(days=self._DELIB_SEEN_TTL_DAYS)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        pruned = {k: d for k, d in seen.items() if d >= cutoff_str}
        kv_set(self._storage, self._DELIB_SEEN_KEY, json.dumps(pruned))

    def scan_fleet(self) -> FleetScanResult:
        """Inventory the model fleet (nodes + models) and rebuild the catalogue (DESIGN §5)."""
        self._model.refresh_inventory()
        result = scan_fleet(self._model, self._storage)
        self._resolve_auto_roles()  # a rescan may surface a better model for an auto role
        self._seed_latency_from_catalogue()  # pick up speed for any newly catalogued (node, model)
        return result

    def fleet_report(self) -> dict[str, Any]:
        """The fleet catalogue as a per-node summary, with per-role recommendations."""
        return fleet_report(self._storage)

    def placement_matrix(self) -> dict[str, Any]:
        """The per-node worker roster: every model on every node it runs on, this-node speed, role
        eligibility, and each node's champion/fastest. The display side of the speed-test matrix."""
        return placement_matrix(
            self._storage, disabled=disabled_models(self._storage),
            disabled_nodes=disabled_nodes(self._storage),
        )

    def council_roster(self, *, size: int = 5) -> dict[str, Any]:
        """The diverse adversarial-council roster (the 'second lineup') — a spread of model families
        for council / background reasoning, not the top-N ranking. Not latency-gated."""
        return council_roster(
            self._storage, size=size, disabled=disabled_models(self._storage),
            disabled_nodes=disabled_nodes(self._storage),
        )

    def fleet_recommendations(self) -> dict[str, Any]:
        """Best model per role from the benchmarked catalogue (recommend-only; DESIGN §4)."""
        return recommend_roles(
            self._storage, disabled=disabled_models(self._storage),
            disabled_nodes=disabled_nodes(self._storage),
        )

    # -- the harness query: staff a role from the qualified fleet (DESIGN §5a) ---------

    def roster_for(self, role: str, *, n: int = 1) -> list[dict[str, Any]]:
        """Staff a role from the qualified fleet — the brain harness's query into the catalogue.

        "Give me N models for role R", honouring the user's model/node vetoes. Pool roles
        (``council``) return a diversity-first spread of up to ``n`` models; every other role
        returns up to ``n`` role-eligible models, best first. ``[]`` if nothing qualifies yet (run a
        benchmark). This is the bridge the harness calls instead of a human reading a view.
        """
        return roster_for(
            self._storage, role, n=n, disabled=disabled_models(self._storage),
            disabled_nodes=disabled_nodes(self._storage),
        )

    def background_model(self) -> str | None:
        """The single best model for off-hot-path reasoning (the loose, non-discipline-gated
        ``background`` role) — what the harness routes background cognition to. ``None`` if nothing
        is benchmarked yet."""
        picks = self.roster_for("background", n=1)
        return picks[0]["model"] if picks else None

    def council_members(self, n: int = 5) -> list[str]:
        """The seated council — a diverse spread of up to ``n`` model names for adversarial
        deliberation (the second lineup). Convenience over ``roster_for('council', n=n)``."""
        return [m["model"] for m in self.roster_for("council", n=n)]

    def tournament_finals(self, keep: set[str]) -> dict[str, Any]:
        """Per-role champions among the kept finalists only — ``recommend_roles`` with everything
        not carried into the final round vetoed out. The qualifying tournament's last round."""
        others = {e.model for e in list_catalogue(self._storage) if e.model not in keep}
        return recommend_roles(
            self._storage, disabled=disabled_models(self._storage) | others,
            disabled_nodes=disabled_nodes(self._storage),
        )

    def set_model_enabled(self, model: str, enabled: bool) -> dict[str, str]:
        """Enable or disable a model for `auto` routing (a user's bias veto; DESIGN §4).

        Disabling a model excludes it from recommendations, `auto` resolution, AND live routing — a
        role pointing at it (even a manual pin or config default) re-resolves to the next-best
        enabled model. Returns the roles that moved as a result.
        """
        set_model_enabled(self._storage, model, enabled)
        self._model.set_disabled_models(disabled_models(self._storage))  # enforce at routing too
        return self._resolve_auto_roles()

    def set_node_enabled(self, node: str, enabled: bool) -> dict[str, str]:
        """Enable or disable a fleet node (a user's per-machine veto; DESIGN §5).

        A disabled node is excluded from routing immediately — even if reachable — and from
        qualification and recommendations. Re-resolves `auto` roles in case a role's fastest node
        changed. Returns the auto roles that moved as a result.
        """
        set_node_enabled(self._storage, node, enabled)
        self._model.set_disabled_nodes(disabled_nodes(self._storage))
        return self._resolve_auto_roles()

    def model_pool(self) -> dict[str, Any]:
        """The fleet's models with qualification, speed, size, nodes, and enable/disable state.

        The data behind the web UI's Model Pool tab: one row per distinct model, a ``passed`` flag
        (cleared the role-gating quality + discipline floor), and which roles it currently serves.
        """
        return fleet_model_pool(
            self._storage,
            disabled=disabled_models(self._storage),
            active_roles={r: s.model for r, s in self._model.roles_view().items()},
            auto_roles=self._auto_roles,
        )

    def set_role(self, role: str, model: str, node: str | None = None) -> dict[str, str]:
        """Pin a role to a specific model — a manual override of `auto` selection (DESIGN §4).

        Routing then uses exactly this model (its fallback chain is cleared — a pin is never
        substituted) and the role leaves the auto set, so a later rescan won't reassign it. An
        optional ``node`` also pins *where* it runs (e.g. an edge box, off the local beast); routing
        prefers that node and falls back only if it's down. Returns the full role→model map.
        """
        self._model.set_role_model(role, model, node)
        existing = self.config.roles.get(role)
        self.config.roles[role] = RoleSpec(model=model, params=existing.params if existing else {})
        self._auto_roles.discard(role)
        self._model.set_role_fallbacks(role, [])
        self._persist_role_pin(role, model, node)  # survive restart (DESIGN §4)
        log.info("role %r manually pinned to %s%s", role, model, f" on {node}" if node else "")
        return {r: s.model for r, s in self._model.roles_view().items()}

    _ROLE_PINS_KEY = "role_pins"

    def _persist_role_pin(self, role: str, model: str, node: str | None) -> None:
        raw = kv_get(self._storage, self._ROLE_PINS_KEY)
        try:
            pins = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            pins = {}
        pins[role] = {"model": model, "node": node}
        kv_set(self._storage, self._ROLE_PINS_KEY, json.dumps(pins))

    def _restore_role_pins(self) -> None:
        """Re-apply manual role pins saved by ``set_role`` so they survive a restart (they override
        config + auto). A pin whose model is now disabled is skipped (the gateway re-resolves)."""
        raw = kv_get(self._storage, self._ROLE_PINS_KEY)
        if not raw:
            return
        try:
            pins = json.loads(raw)
        except (ValueError, TypeError):
            return
        disabled = disabled_models(self._storage)
        for role, pin in (pins or {}).items():
            model = (pin or {}).get("model")
            if not model or model in disabled:
                continue
            self._model.set_role_model(role, model, (pin or {}).get("node"))
            self._auto_roles.discard(role)
            self._model.set_role_fallbacks(role, [])
            log.info("role %r restored to pinned %s", role, model)

    def role_nodes(self) -> dict[str, str]:
        """The per-role node pins (role → node) the gateway is honouring, for the UI."""
        return self._model.role_nodes()

    def _apply_role_recs(self, recs: dict[str, Any]) -> dict[str, str]:
        """Re-point the real roles (chat/bake/reasoning) at the given recommendations. Shared by
        'Apply best' and the tournament finals. Updates routing in memory — persist via toml."""
        applied: dict[str, str] = {}
        for role in ("chat", "bake", "reasoning"):
            rec = recs.get(role)
            if rec and role in self.config.roles:
                self._model.set_role_model(role, rec["model"])
                self.config.roles[role].model = rec["model"]
                applied[role] = rec["model"]
        if applied:
            log.info("fleet: applied role recommendations %s", applied)
        return applied

    def apply_recommendations(self) -> dict[str, str]:
        """Re-point the live roles at their recommended models. Returns what changed.

        Only the real roles (chat/bake/reasoning) are applied; tools/code are advisory until those
        features exist. Recommendations come only from benchmarked models, so this never routes to
        an untested model.
        """
        return self._apply_role_recs(
            recommend_roles(
                self._storage, disabled=disabled_models(self._storage),
                disabled_nodes=disabled_nodes(self._storage),
            )
        )

    def apply_finals(self, keep: set[str]) -> dict[str, str]:
        """Apply the tournament finals: re-point roles to the champions among the kept finalists."""
        return self._apply_role_recs(self.tournament_finals(keep))

    def benchmark_fleet(
        self,
        *,
        only_approved: bool = True,
        limit: int = 64,
        max_params_b: float | None = None,
        min_params_b: float | None = None,
        latency_budget_s: float | None = None,
        judge: bool = True,
        only_models: set[str] | None = None,
        framework: bool = True,
        persist: bool = True,
        progress: Callable[[int, int, str, float | None], None] | None = None,
        on_result: Callable[[ModelBenchmark, str], None] | None = None,
    ) -> FleetBenchmarkResult:
        """Scan + benchmark the fleet's models (speed + capability + coherence) (DESIGN §4).

        Re-scans first so the catalogue is current, then scores each distinct approved model
        **up to the user's size cap** (``[backend] max_model_size_b``, since only the user knows
        their hardware), smallest-first, and writes the results back. Expensive — many model calls —
        so it is on-demand; ``progress(i, total, model)`` lets a caller show live progress and the
        user stop it. ``max_params_b`` / ``latency_budget_s`` override the configured cap/latency
        (e.g. from the UI fields) for a single run.
        """
        cap = max_params_b if max_params_b is not None else (
            self.config.backend.max_model_size_b if self.config.backend else 30.0
        )
        floor = min_params_b if min_params_b is not None else (
            self.config.backend.min_model_size_b if self.config.backend else 0.0
        )
        budget = latency_budget_s if latency_budget_s is not None else (
            self.config.backend.max_latency_s if self.config.backend else 0.0
        )
        ctx = self.config.backend.benchmark_num_ctx if self.config.backend else 24576
        self.scan_fleet()
        return _benchmark_fleet(
            self._model,
            self._storage,
            only_approved=only_approved,
            limit=limit,
            max_params_b=cap,
            min_params_b=floor,
            judge=judge,
            latency_budget_s=budget,
            num_ctx=ctx,
            only_models=only_models,
            disabled_nodes=disabled_nodes(self._storage),
            framework=framework,
            persist=persist,
            progress=progress,
            on_result=on_result,
        )

    def complete_speed_matrix(
        self, *, only_models: set[str] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """The final time trial: speed-test acceptable models on the enabled nodes they're installed
        on but not yet timed on — completing the per-node placement matrix for the background pool.
        Reads the existing catalogue (does NOT rescan — that would wipe the quality scores)."""
        ctx = self.config.backend.benchmark_num_ctx if self.config.backend else 24576
        return _complete_speed_matrix(
            self._storage, num_ctx=ctx, disabled_nodes=disabled_nodes(self._storage),
            only_models=only_models, progress=progress,
        )

    def benchmark_council_pool(
        self,
        *,
        progress: Callable[[int, int, str, float | None], None] | None = None,
        on_result: Callable[[ModelBenchmark, str], None] | None = None,
    ) -> FleetBenchmarkResult:
        """Grade the COUNCIL pool: the models **above** the chat size cap, with the user-facing caps
        OFF (no upper size limit, no latency gate) — so the big/slow models a chat cap excludes get
        scored and can enter the council roster (the second lineup). Same full gauntlet as the main
        pool (``framework``/``judge`` on) so their quality is comparable.

        Like the speed-test, it **does NOT rescan** — that would wipe the existing quality scores;
        it grades the big models *in place* on the catalogue a prior tournament/benchmark built. So
        run a tournament first (to discover + score the main pool), then this to fill in the big
        council models without touching the rest.
        """
        cap = self.config.backend.max_model_size_b if self.config.backend else 30.0
        ctx = self.config.backend.benchmark_num_ctx if self.config.backend else 24576
        return _benchmark_fleet(
            self._model,
            self._storage,
            only_approved=False,   # council wants diversity — don't gate on the curated allowlist
            max_params_b=0.0,      # 0 = no upper size cap (the whole point of the second lineup)
            min_params_b=cap,      # only the models ABOVE the chat cap (the big pool)
            judge=True,
            latency_budget_s=0.0,  # 0 = no latency gate (capacity-bound, not latency-bound)
            num_ctx=ctx,
            disabled_nodes=disabled_nodes(self._storage),
            framework=True,
            persist=True,
            progress=progress,
            on_result=on_result,
        )

    def evaluate_epistemics(
        self, models: list[str] | None = None, *, samples: int = 3
    ) -> list[EpistemicResult]:
        """Measure how well models exploit the epistemic context framework (DESIGN §3).

        Runs the structured-vs-flat probes (tier deference, attribution, uncertainty) across the
        given models — or every reachable non-embedding model — and returns each model's scores
        and the framework ``lift``. Call-heavy; intended on-demand.
        """
        names = models or [m for m in self._model.available_models() if "embed" not in m.lower()]
        return run_epistemics(self._model, names, samples=samples)

    def deliberate(self, question: str, user: str | None = None) -> CouncilResult:
        """Convene the inner council on an open question — adversarial deliberation → a verdict.

        Personas spread across whatever models are installed; the verdict is stored as recallable
        understanding (DESIGN §0.4, §4, §5).
        """
        return deliberate(self._model, self._storage, self._embedder, question=question, user=user)

    # -- the council forum (browsable deliberations + housekeeping; DESIGN §5a) --------

    def forum_threads(self) -> list[dict[str, Any]]:
        """All forum threads (deliberations), newest first, with post counts."""
        return list_forum_threads(self._storage)

    def forum_thread(self, thread_id: int) -> dict[str, Any] | None:
        """One thread with its posts (persona positions, verdict, comments), or None."""
        return get_forum_thread(self._storage, thread_id)

    def forum_comment(self, thread_id: int, text: str, *, user: str | None = None) -> None:
        """Add a user comment to a thread (annotation only — not fed back into reasoning)."""
        add_forum_post(self._storage, thread_id=thread_id, author=user or "you",
                       kind="comment", content=text)

    def forum_set_status(self, thread_id: int, status: str) -> None:
        """Close or reopen a thread."""
        if status not in ("open", "closed"):
            raise ConfigError(f"bad thread status: {status!r}")
        set_forum_thread_status(self._storage, thread_id, status)

    def forum_delete_thread(self, thread_id: int) -> None:
        delete_forum_thread(self._storage, thread_id)

    def forum_delete_post(self, post_id: int) -> None:
        delete_forum_post(self._storage, post_id)

    def wait_for_sentinel(self) -> None:
        """Block until the most recent turn's burst (note, self-model) has drained (DESIGN §5a)."""
        self._burst.wait_idle()

    @property
    def last_sentinel_error(self) -> BaseException | None:
        """The exception from the last sentinel run, if it failed; else ``None``."""
        return self._last_sentinel_error

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:
        self._burst.stop()
        self._stop_idle.set()
        self._stop_sleep.set()
        self._stop_inner.set()
        if self._idle_prober is not None:
            self._idle_prober.join(timeout=5)
            self._idle_prober = None
        if self._sleep_scheduler is not None:
            self._sleep_scheduler.join(timeout=5)
            self._sleep_scheduler = None
        if self._inner_life is not None:
            self._inner_life.join(timeout=5)
            self._inner_life = None
        self._model.stop_prober()
        self._storage.close()

    def __enter__(self) -> Mimir:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
