"""``Mimir`` — the facade. Import it, hand it a config, call ``.turn()`` (DESIGN §1).

This is where the spine is wired into the §6 loop:

    turn(text, user)
      → embed the query → recall via the storage gateway → build_context() (assemble)
      → model (chat) → reply
      → record access · bake new facts (storage gateway) · fire the sentinel (async)

Everything routes through the two gateways. The sentinel runs off the hot path and its failure
cannot break the loop. The next turn joins the prior sentinel before assembling, so a note is
always ready when it matters — without making the reply wait on reflection.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cognition.bake import bake
from .cognition.benchmark import FleetBenchmarkResult
from .cognition.benchmark import benchmark_fleet as _benchmark_fleet
from .cognition.council import CouncilResult, deliberate
from .cognition.epistemics import EpistemicResult, run_epistemics
from .cognition.fleet import (
    FleetScanResult,
    fleet_model_pool,
    fleet_report,
    recommend_roles,
    resolve_auto_model,
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
from .cognition.procedural import learn_procedure, render_procedures, retrieve_procedures
from .cognition.self_model import synthesize_self_model
from .cognition.sentinel import run_sentinel
from .cognition.sleep import SleepReport, consolidate
from .cognition.working_memory import (
    current_working_memory,
    record_exchange,
    synthesize_working_memory,
)
from .config import AUTO_MODEL, BackendConfig, Config, ProviderSpec, load_config
from .context.build import ContextBundle, build_context
from .embed.base import Embedder, EmbeddingMode
from .embed.endpoint import EndpointEmbedder, NullEmbedder
from .embed.locality import LocalityHashEmbedder
from .errors import ConfigError
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
    latest_self_model,
    latest_sentinel_note,
    list_memories,
    record_access,
    set_model_enabled,
)

log = logging.getLogger("mimir")

# How many memories the knowledge section may draw on per turn (pre-budget). Hardening
# (adaptive top-k, SQL-side prefiltering) is a later session; v0 keeps it a simple constant.
DEFAULT_TOP_K = 6


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
    return ProviderPool(endpoints)


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

        self._pending: list[threading.Thread] = []
        self._last_sentinel_error: BaseException | None = None
        self._turn_count = 0

        # Establish any identity anchors declared in config (idempotent upsert at boot), so a
        # non-interactive deployment is grounded without running the interactive interview.
        if config.identity_anchors:
            establish_identity(self._storage, config.identity_anchors)

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
        self._resolve_auto_roles()
        self._model.start_prober(refresh_interval_s)

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
        resolved: dict[str, str] = {}
        for role in self._auto_roles:
            model = resolve_auto_model(
                self._storage, role, available=available, disabled=disabled
            )
            if model is not None:
                self._model.set_role_model(role, model)
                resolved[role] = model
        if resolved:
            log.info("fleet: auto-resolved role(s) %s", resolved)
        return resolved

    @classmethod
    def from_config(cls, path: str) -> Mimir:
        """Construct from a ``mimir.toml`` path."""
        return cls(load_config(path))

    # -- the turn ---------------------------------------------------------------------

    def turn(self, text: str, user: str | None = None) -> TurnResult:
        # Make sure the previous turn's background work (sentinel note, self-model) has landed
        # before we assemble — so the prompt reflects the latest reflection and identity.
        self._join_background()
        self._turn_count += 1

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
        )

        # 3. Generate the reply through the model gateway. Strip any internal epistemic tags the
        #    model echoed (small models mimic the [tier=...; source=...] style) before it lands.
        reply = strip_epistemic_tags(
            self._model.chat(
                "chat",
                [
                    {"role": "system", "content": bundle.prompt},
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

        # 5. Background cognition off the hot path: sentinel, self-model, working memory.
        self._spawn_sentinel(user=user, turn_text=text, reply=reply)
        self._maybe_refresh_self_model()
        self._maybe_refresh_working_memory()
        self._maybe_sleep()

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
        self._join_background()
        self._turn_count += 1

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
        )
        messages = [
            {"role": "system", "content": bundle.prompt},
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
        self._spawn_sentinel(user=user, turn_text=text, reply=reply)
        self._maybe_refresh_self_model()
        self._maybe_refresh_working_memory()
        self._maybe_sleep()
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

    def _maybe_refresh_self_model(self) -> None:
        every = self.config.self_model_refresh_every
        if every <= 0:
            return
        # Seed one on the first turn, then refresh on cadence as experience accumulates.
        if self._turn_count == 1 or self._turn_count % every == 0:

            def _run() -> None:
                try:
                    synthesize_self_model(self._model, self._storage)
                except BaseException as exc:  # logged downgrade — never touches the turn
                    log.error(
                        "self-model refresh failed (off the hot path; turn unaffected): %s",
                        exc,
                        exc_info=True,
                    )

            self._start_background("mimir-self-model", _run)

    # -- working memory ---------------------------------------------------------------

    def refresh_working_memory(self) -> Memory | None:
        """Fold the accumulated exchanges into the rolling working-memory summary now (sync)."""
        return synthesize_working_memory(self._model, self._storage)

    def _maybe_refresh_working_memory(self) -> None:
        every = self.config.working_memory_refresh_every
        if every <= 0:
            return
        if self._turn_count % every == 0:

            def _run() -> None:
                try:
                    synthesize_working_memory(self._model, self._storage)
                except BaseException as exc:  # logged downgrade — never touches the turn
                    log.error(
                        "working-memory refresh failed (off the hot path; turn unaffected): %s",
                        exc,
                        exc_info=True,
                    )

            self._start_background("mimir-working-memory", _run)

    # -- sleep / consolidation --------------------------------------------------------

    def sleep(self) -> SleepReport:
        """Run a consolidation pass now (dedup, decay, archive, contradiction resolution)."""
        return consolidate(self._storage)

    def scan_fleet(self) -> FleetScanResult:
        """Inventory the model fleet (nodes + models) and rebuild the catalogue (DESIGN §5)."""
        self._model.refresh_inventory()
        result = scan_fleet(self._model, self._storage)
        self._resolve_auto_roles()  # a rescan may surface a better model for an auto role
        return result

    def fleet_report(self) -> dict[str, Any]:
        """The fleet catalogue as a per-node summary, with per-role recommendations."""
        return fleet_report(self._storage)

    def fleet_recommendations(self) -> dict[str, Any]:
        """Best model per role from the benchmarked catalogue (recommend-only; DESIGN §4)."""
        return recommend_roles(self._storage, disabled=disabled_models(self._storage))

    def set_model_enabled(self, model: str, enabled: bool) -> dict[str, str]:
        """Enable or disable a model for `auto` routing (a user's bias veto; DESIGN §4).

        Disabling a model excludes it from every recommendation and `auto` resolution; if it was
        serving an auto role, that role is immediately re-resolved to the next-best model. Returns
        the auto roles that moved as a result.
        """
        set_model_enabled(self._storage, model, enabled)
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

    def apply_recommendations(self) -> dict[str, str]:
        """Re-point the live roles at their recommended models. Returns what changed.

        Only the real roles (chat/bake/reasoning) are applied; tools/code are advisory until those
        features exist. Recommendations come only from benchmarked models, so this never routes to
        an untested model. Updates routing in memory — persist by editing your ``mimir.toml``.
        """
        recs = recommend_roles(self._storage, disabled=disabled_models(self._storage))
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

    def benchmark_fleet(
        self,
        *,
        only_approved: bool = True,
        limit: int = 8,
        max_params_b: float = 30.0,
        judge: bool = True,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> FleetBenchmarkResult:
        """Scan + benchmark the fleet's models (speed + capability + coherence) (DESIGN §4).

        Re-scans first so the catalogue is current, then scores each distinct model (smallest-first,
        skipping models over ``max_params_b``) and writes the results back. Expensive — many model
        calls — so it is on-demand (this, the web UI, or cron). ``progress(i, total, model)`` is
        invoked before each model so a caller can show live progress on this multi-minute run.
        """
        self.scan_fleet()
        return _benchmark_fleet(
            self._model,
            self._storage,
            only_approved=only_approved,
            limit=limit,
            max_params_b=max_params_b,
            judge=judge,
            progress=progress,
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

    def _maybe_sleep(self) -> None:
        every = self.config.sleep_every
        if every <= 0 or self._turn_count % every != 0:
            return

        def _run() -> None:
            try:
                consolidate(self._storage)
            except BaseException as exc:  # logged downgrade — never touches the turn
                log.error(
                    "consolidation failed (off the hot path; turn unaffected): %s",
                    exc,
                    exc_info=True,
                )

        self._start_background("mimir-sleep", _run)

    # -- background plumbing ----------------------------------------------------------

    def _spawn_sentinel(self, *, user: str | None, turn_text: str, reply: str) -> None:
        self._last_sentinel_error = None

        def _run() -> None:
            try:
                run_sentinel(
                    self._model, self._storage, user=user, turn_text=turn_text, reply=reply
                )
            except BaseException as exc:  # logged downgrade — never touches the turn
                self._last_sentinel_error = exc
                log.error(
                    "sentinel failed (off the hot path; this turn is unaffected): %s",
                    exc,
                    exc_info=True,
                )

        self._start_background("mimir-sentinel", _run)

    def _start_background(self, name: str, work: Callable[[], None]) -> None:
        thread = threading.Thread(target=work, name=name, daemon=True)
        thread.start()
        self._pending.append(thread)

    def _join_background(self) -> None:
        pending, self._pending = self._pending, []
        for thread in pending:
            thread.join(timeout=30)

    def wait_for_sentinel(self) -> None:
        """Block until the most recent turn's background work (note, self-model) has landed."""
        self._join_background()

    @property
    def last_sentinel_error(self) -> BaseException | None:
        """The exception from the last sentinel run, if it failed; else ``None``."""
        return self._last_sentinel_error

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:
        self._join_background()
        self._model.stop_prober()
        self._storage.close()

    def __enter__(self) -> Mimir:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
