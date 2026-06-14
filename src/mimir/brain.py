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

import logging
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cognition.bake import bake
from .cognition.benchmark import FleetBenchmarkResult, ModelBenchmark
from .cognition.benchmark import benchmark_fleet as _benchmark_fleet
from .cognition.benchmark import complete_speed_matrix as _complete_speed_matrix
from .cognition.burst import BurstResult, BurstWorker, ResponseContext
from .cognition.council import CouncilResult, deliberate
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
from .cognition.graph import render_triples, retrieve_connected
from .cognition.identity import (
    current_anchors,
    establish_identity,
    pending_questions,
    render_anchors,
)
from .cognition.ingest import IngestResult, ingest_document
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
from .cognition.temporal import answer_time_query, gap_insight, local_now, time_prefix
from .cognition.working_memory import (
    current_working_memory,
    record_exchange,
    synthesize_working_memory,
)
from .config import AUTO_MODEL, BackendConfig, Config, ProviderSpec, RoleSpec, load_config
from .context.build import ContextBundle, build_context
from .embed.base import Embedder, EmbeddingMode
from .embed.endpoint import EndpointEmbedder, NullEmbedder
from .embed.locality import LocalityHashEmbedder
from .errors import ConfigError, StorageError
from .model.discovery import discover_node_urls
from .model.gateway import ModelGateway
from .model.pool import ProviderPool
from .model.provider import Provider
from .model.providers.mock import MockProvider
from .model.providers.ollama import OllamaProvider
from .retrieval.hybrid import retrieve
from .sanitize import StreamTagStripper, strip_epistemic_tags
from .storage.gateway import StorageGateway
from .storage.models import Memory, MemoryKind, Procedure
from .storage.repo import (
    bump_procedure_uses,
    disabled_models,
    disabled_nodes,
    interaction_history,
    latest_self_model,
    latest_sentinel_note,
    list_catalogue,
    list_memories,
    recent_conversation,
    record_access,
    record_conversation_turn,
    record_interaction,
    set_model_enabled,
    set_node_enabled,
    update_catalogue_speed,
)

log = logging.getLogger("mimir")

# How many memories the knowledge section may draw on per turn (pre-budget). Hardening
# (adaptive top-k, SQL-side prefiltering) is a later session; v0 keeps it a simple constant.
DEFAULT_TOP_K = 6

# The idle latency heartbeat (DESIGN §5): a short generation forcing a real-length reply, so the
# timed call reflects throughput, not just round-trip. Kept rare — real traffic is the main signal.
_PROBE_PROMPT = [{"role": "user", "content": "In one or two sentences, say you are online."}]
_PROBE_TIMEOUT_S = 20.0
_PROBE_PREDICT = 64  # cap the probe generation — long enough to time throughput, still cheap
_FALLBACK_DEPTH = 4  # how many acceptable models deep a role's fallback chain runs (best first)
_HISTORY_TURNS = 6   # recent exchanges replayed to the model as real messages (continuity)


def _latency_staleness(info: dict[str, object] | None) -> float:
    """How overdue a (node, model) is for a probe: ``inf`` if never measured, else its age in secs.
    The idle heartbeat probes the highest-staleness model per node, so coverage rotates fairly."""
    if info is None or not info.get("samples"):
        return float("inf")
    age = info.get("age_s")
    return float(age) if age is not None else float("inf")


@dataclass(slots=True)
class TurnResult:
    """What a turn produced: the reply, the assembled context (introspectable), and bakes."""

    reply: str
    context: ContextBundle
    baked: list[Memory]


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
        return EndpointEmbedder(model)
    return NullEmbedder()


class Mimir:
    """The cognition core. One instance owns one store and one provider."""

    def __init__(self, config: Config, *, provider: Provider | None = None) -> None:
        config.validate()
        self.config = config
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
        self._turn_active = False  # True while a turn is generating — background yields to it (§5a)
        # The burst worker: all post-response cognition (sentinel, self-model, working memory,
        # sleep) is scheduled through it — priority-ordered, slot-capped, interruptible — instead of
        # N raw threads (DESIGN §5a). Runs in the idle window after a reply; the next turn settles.
        self._burst = BurstWorker(is_busy=lambda: self._turn_active)
        self._register_burst_tasks()
        self._burst.start()
        self._stop_idle = threading.Event()  # signals the idle latency heartbeat to stop
        self._idle_prober: threading.Thread | None = None

        # Establish any identity anchors declared in config (idempotent upsert at boot), so a
        # non-interactive deployment is grounded without running the interactive interview.
        if config.identity_anchors:
            establish_identity(self._storage, config.identity_anchors)

        # Apply any per-node vetoes to the pool up front, so a disabled box is never routed to.
        self._model.set_disabled_nodes(disabled_nodes(self._storage))

        # Resolve `auto` roles now for the local/single-provider path (inventory is ready). The
        # fleet path resolves in _init_fleet once its background inventory lands; until then the
        # gateway stop-gaps `auto` to any reachable model so turns never fail (DESIGN §4).
        self._resolve_auto_roles()

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
        return resolved

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
        return time_prefix(local_now(self.config.timezone), self.config.hemisphere)

    def maybe_time_answer(self, text: str) -> str | None:
        """A direct, model-free answer to an explicit time/date/season question, or ``None``.

        The deterministic intercept (DESIGN §3e) — exposed so a host (CLI, server, voice) can short-
        circuit before a model call. ``turn`` uses it automatically."""
        return answer_time_query(text, local_now(self.config.timezone), self.config.hemisphere)

    def _temporal_awareness(self, user: str | None, now_ts: float) -> str | None:
        """A deterministic 'you've been away longer than usual' note from the interaction log, or
        ``None``. Computed from PRIOR interactions (call before logging the current turn)."""
        history = interaction_history(self._storage, user=user)
        return gap_insight(history, now_ts)

    def _recent_history(self) -> str | None:
        """The temporal-narrative arc (month → week → lately) for the prompt, or ``None``."""
        return render_recent_history(self._storage)

    def _history_messages(self, user: str | None) -> list[dict[str, str]]:
        """Recent exchanges as real chat messages, so the model has genuine conversational
        continuity (not just summarized text) — the session-history replay (DESIGN §3a)."""
        msgs: list[dict[str, str]] = []
        for turn in recent_conversation(self._storage, user=user, limit=_HISTORY_TURNS):
            msgs.append({"role": "user", "content": turn["user_text"]})
            msgs.append({"role": "assistant", "content": turn["reply"]})
        return msgs

    def history(self, *, user: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """The durable conversation log, oldest→newest — what the UI restores on load (§3a)."""
        return recent_conversation(self._storage, user=user, limit=limit)

    def generate_narratives(self) -> dict[str, Any]:
        """Run the temporal-narrative cycle now (daily entry + weekly/monthly roll-up), sync.

        Normally runs off the hot path in the consolidation pass; this is the explicit hook (DESIGN
        §3a/§3e). Idempotent per period — re-running the same day reuses today's entry."""
        return run_narrative_cycle(
            self._model, self._storage, now=local_now(self.config.timezone)
        )

    # -- the turn ---------------------------------------------------------------------

    def turn(self, text: str, user: str | None = None) -> TurnResult:
        # Settle the previous turn's burst (sentinel note, self-model) before we assemble — so the
        # prompt reflects the latest reflection and identity (DESIGN §5a).
        self._burst.wait_idle()
        self._turn_count += 1

        # Temporal awareness: read the gap from PRIOR interactions, then log this turn (DESIGN §3e).
        now = time.time()
        awareness = self._temporal_awareness(user, now)
        record_interaction(self._storage, now, user)

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
            record_conversation_turn(self._storage, user=user, user_text=text, reply=intercept)
            return TurnResult(reply=intercept, context=bundle, baked=[])

        self._turn_active = True  # foreground in progress — the burst yields to it (§5a)
        try:
            # Background notes the prior burst surfaced, to carry into this reply.
            notes = self._burst.drain_surfaces()

            # 1. Recall: embed the query, pull candidates, rank them.
            query_vec = self._embedder.embed(text)
            candidates = list_memories(self._storage, user=user, kind=MemoryKind.MEMORY)
            retrieved = retrieve(text, query_vec, candidates, top_k=DEFAULT_TOP_K)
            note = latest_sentinel_note(self._storage, user)
            self_knowledge = self._compose_self_knowledge()
            working_memory = current_working_memory(self._storage)
            graph_facts = self._connected_facts(text, user)
            procedures = self._matching_procedures(text, user)

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
                now_ts=now,
            )

            # 3. Generate the reply through the model gateway. Strip any internal epistemic tags the
            #    model echoed (small models mimic the [tier=...; source=...] style) before it lands.
            reply = strip_epistemic_tags(
                self._model.chat(
                    "chat",
                    [
                        {"role": "system", "content": bundle.prompt},
                        *self._history_messages(user),
                        {"role": "user", "content": text},
                    ],
                )
            )

            # 4. Side effects through the storage gateway: relevance bookkeeping + bake.
            record_access(self._storage, bundle.retrieved_ids)
            baked = bake(
                self._model,
                self._storage,
                self._embedder,
                turn_text=text,
                user=user,
                primary_user=self.config.primary_user,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=reply)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=reply)
        finally:
            self._turn_active = False

        # 5. Fire the burst window: sentinel + any due self-model/working-memory/sleep, scheduled
        #    and run off the hot path (DESIGN §5a). The next turn settles it.
        self._burst.signal(ResponseContext(
            user_text=text, reply=reply, user=user, turn_index=self._turn_count
        ))
        return TurnResult(reply=reply, context=bundle, baked=baked)

    def turn_stream(
        self, text: str, user: str | None = None
    ) -> Generator[str, None, dict[str, Any]]:
        """Like ``turn`` but yields the reply token-by-token; returns the introspection dict.

        The side effects (record access, bake, sentinel, self-model) run after the stream
        completes, so a fully-consumed stream behaves exactly like ``turn``. If the consumer
        abandons the stream early, the turn is treated as interrupted — nothing is baked. The
        generator's *return value* (via ``StopIteration.value``) is ``context.introspect()``.
        """
        self._burst.wait_idle()
        self._turn_count += 1

        # Temporal awareness + the deterministic time intercept (DESIGN §3e), mirroring `turn`.
        now = time.time()
        awareness = self._temporal_awareness(user, now)
        record_interaction(self._storage, now, user)
        intercept = self.maybe_time_answer(text)
        if intercept is not None:
            bundle = build_context(
                query=text, user=user, identity=self.config.identity, retrieved=[],
                sentinel_note=None, embed_mode=self._embedder.mode,
                budget_tokens=self.config.context_budget_tokens,
                time_context=self._time_context(), now_ts=now,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=intercept)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=intercept)
            yield intercept
            return bundle.introspect()

        self._turn_active = True  # foreground in progress — the burst yields to it (§5a)
        try:
            notes = self._burst.drain_surfaces()
            query_vec = self._embedder.embed(text)
            candidates = list_memories(self._storage, user=user, kind=MemoryKind.MEMORY)
            retrieved = retrieve(text, query_vec, candidates, top_k=DEFAULT_TOP_K)
            note = latest_sentinel_note(self._storage, user)
            self_knowledge = self._compose_self_knowledge()
            working_memory = current_working_memory(self._storage)
            graph_facts = self._connected_facts(text, user)
            procedures = self._matching_procedures(text, user)
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
                now_ts=now,
            )
            messages = [
                {"role": "system", "content": bundle.prompt},
                *self._history_messages(user),
                {"role": "user", "content": text},
            ]

            # Strip internal epistemic tags as we stream, so neither the live display nor the stored
            # reply carries them (a tag may straddle two deltas, hence the stateful stripper).
            stripper = StreamTagStripper()
            chunks: list[str] = []
            for delta in self._model.chat_stream("chat", messages):
                clean = stripper.feed(delta)
                if clean:
                    chunks.append(clean)
                    yield clean
            tail = stripper.flush()
            if tail:
                chunks.append(tail)
                yield tail
            reply = "".join(chunks)

            record_access(self._storage, bundle.retrieved_ids)
            bake(
                self._model,
                self._storage,
                self._embedder,
                turn_text=text,
                user=user,
                primary_user=self.config.primary_user,
            )
            record_exchange(self._storage, user=user, user_text=text, reply=reply)
            record_conversation_turn(self._storage, user=user, user_text=text, reply=reply)
        finally:
            self._turn_active = False

        self._burst.signal(ResponseContext(
            user_text=text, reply=reply, user=user, turn_index=self._turn_count
        ))
        return bundle.introspect()

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
        self._burst.register("sleep", self._sleep_task, base_priority=60.0,
                             trigger=lambda ctx: self._due("sleep"))

    def _due(self, which: str) -> bool:
        """Whether a cadence task is due this turn (preserving the prior per-task schedules)."""
        if which == "self_model":
            every = self.config.self_model_refresh_every
            return every > 0 and (self._turn_count == 1 or self._turn_count % every == 0)
        if which == "working_memory":
            every = self.config.working_memory_refresh_every
            return every > 0 and self._turn_count % every == 0
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
                synthesize_working_memory(self._model, self._storage)
            except BaseException as exc:
                log.error("working-memory refresh failed (turn unaffected): %s", exc, exc_info=True)
            return BurstResult()
        return run

    def _sleep_task(self, ctx: ResponseContext) -> Callable[[], BurstResult]:
        def run() -> BurstResult:
            try:
                consolidate(self._storage)
                run_narrative_cycle(
                    self._model, self._storage, now=local_now(self.config.timezone)
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

        Disabling a model excludes it from every recommendation and `auto` resolution; if it was
        serving an auto role, that role is immediately re-resolved to the next-best model. Returns
        the auto roles that moved as a result.
        """
        set_model_enabled(self._storage, model, enabled)
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

    def set_role(self, role: str, model: str) -> dict[str, str]:
        """Pin a role to a specific model — a manual override of `auto` selection (DESIGN §4).

        Routing then uses exactly this model (its fallback chain is cleared — a pin is never
        substituted) and the role leaves the auto set, so a later rescan won't reassign it. Returns
        the full role→model map after the change.
        """
        self._model.set_role_model(role, model)
        existing = self.config.roles.get(role)
        self.config.roles[role] = RoleSpec(model=model, params=existing.params if existing else {})
        self._auto_roles.discard(role)
        self._model.set_role_fallbacks(role, [])
        log.info("role %r manually pinned to %s", role, model)
        return {r: s.model for r, s in self._model.roles_view().items()}

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
        if self._idle_prober is not None:
            self._idle_prober.join(timeout=5)
            self._idle_prober = None
        self._model.stop_prober()
        self._storage.close()

    def __enter__(self) -> Mimir:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
