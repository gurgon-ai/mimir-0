# Changelog

All notable changes to Mimir 0. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Pre-1.0: the API and schema may change between releases.

## [Unreleased]

Nothing yet. The itemised history of every change in 0.2.0 (about 1,160 lines, kept for anyone who needs
the exact sequence) is in git: `git show v0.1.0..v0.2.0 -- CHANGELOG.md`, or the file as it stood at
commit fc519fe.

## [0.2.0] — 2026-10-31

Four months of work after the feature-complete cut, in five themes: a harder, honest model
qualification; epistemic integrity (who is believed, and why); knowledge ingestion (documents, a cited
library, an offline encyclopedia); autonomous background cognition (burst worker, sleep, inner life,
deliberation); and opening the brain to integrations (an authenticated API, four extension ports, a
much larger UI). Schema v9 → v24.

### Upgrading from 0.1.0
- **Re-clone or hard-reset.** The git history was rewritten (`git-filter-repo`) to drop two stray SQLite
  sidecar files; every earlier commit SHA changed.
- **Trust default changed.** A named speaker who is not `[identity] primary_user` or in `trusted_users`
  is now baked at the CONVERSATION tier, not trusted automatically. Existing stores can be re-tiered with
  `Mimir.retier_speaker()`. Peer agents should send `speaker_kind="ai_peer"`.
- `[working_memory] refresh_every` is deprecated; use `fold_threshold` (10) and `keep_recent` (4).
- The wall-clock sleep window is on by default (`[sleep] window_start` / `window_end`); the old
  `[sleep] every` interval defaults to off.
- `think` defaults to off and is sent as a top-level field; opt in per role.
- `[roles.embed] = "auto"` is the example default; after changing embed models run `--reembed`.
- Setting `[server] api_token` (or `MIMIR_API_TOKEN`) makes every `/api/*` call require a Bearer token.
- Context size: `backend.benchmark_num_ctx` still defaults to 24576 in config; the runtime Context-size
  preset (Small/Medium/Large/X-Large, default Medium = 8192 ctx / 4096 budget) overrides `num_ctx`,
  `context_budget_tokens` and `benchmark_num_ctx` together once set from the UI.
- Routing and recall defaults: `max_latency_s` is a hard cap at routing, never a capability filter;
  `max_model_size_b` defaults to 30; `DEFAULT_TOP_K` 6 → 10; `[library] claims_top_k` 5 → 8; output-RAG
  and the citation guard are on by default.

### Added (2026-10-31, from the production system's incident record)
- `cognition/generation.py`: a guard at every store boundary. Sentinel, self-model and council refuse
  to store an empty or placeholder generation; a council voice that failed is not answered,
  synthesised or filed as a position.
- Working memory is rebuilt from sources: the fold ARCHIVES the exchanges it summarises (kept out of
  recall) and writes the summary from the last `SOURCE_WINDOW` of them, never from its own previous
  text. A rejected fold saves nothing and archives nothing (it used to save an empty summary and
  delete the exchanges).
- `cognition/ledger.py`: the capability ledger. Every tool, burst task, sleep phase and idle loop is
  registered at boot and touched when it runs (kv-persisted); never-fired, stale and always-failing
  capabilities are findings in the system-health prompt section and the nightly digest;
  `Mimir.capability_report()`.
- Tools on the streaming path: `turn_stream` / `POST /api/turn/stream` offer tools, hold back the
  marker, run the round and stream the final answer; `actions` ride on `done.introspect`.
- `actions_followup` burst task: a refused or crashed tool call is put in front of the model on the
  next turn as an unkept promise; only status `ok` counts as done. A handler's `refused:` result is
  reported as `blocked`.
- `Mimir.in_conversation()` and `[cognition] chat_window_s` (300): the sleep scheduler and the idle
  latency prober stay clear for a window after the last turn, not just during one.
- `cognition/names.py` and `[identity] known_names` / `name_variants` / `snap_names`: heard or
  mistyped names snap to the household roster at the turn, before bake, graph or recall.
- Ollama: `keep_alive` is a top-level field a role may set (`-1` = always warm); an empty body with
  `done_reason == "length"` under thinking is retried once with thinking off; embeddings go in
  batches of 128 and are shape-checked (a short or `None` batch raises instead of blanking).
- `storage.models.is_system_generated`: the system's own musings, verdicts and reflections are kept
  out of a narrative's facts, the self-model's knowledge counts (reported apart) and output-RAG
  grounding.
- The operational `num_ctx` now reaches `chat_with_model` / `chat_on_node` (council and background
  calls), so no caller can force a warm model to reload with a different window.
- The time intercept counts by position: the question must open the turn (after a short lead-in).
- The notebook tool refuses an empty `write` (it would have wiped the note) and, with the milestone
  tool, refuses mutations from a peer AI; `record_milestone` accepts `current_config` and
  `is_current_config`.
- `recent_by_kind` / `count_memories` exclude archived rows unless asked; `prune_kind` prunes only
  live rows; `prune_archived` bounds the archived window.
- Public-clean pass: employer document names, LAN addresses, a machine nickname and private paths
  removed from code, tests and docs.

### Security
- Integration API auth: `[server] api_token` / `MIMIR_API_TOKEN` (renamable via `api_token_env`) gates
  `/api/*` with a constant-time Bearer check; `cors_origins` for browser front-ends on other origins.
- The local browser UI is token-exempt unless `[server] secure_ui = true`; `GET /api/health` is always open.
- Server-side trust policy: the config decides belief, never the caller. `primary_user` (1.30),
  `trusted_users` (1.20), any other human at CONVERSATION tier.
- Peer-AI source tier: `speaker_kind` on the turn API, `[identity] peer_agents`, a decaying
  `STATED_BY_PEER` tier (0.95); kind overrides identity, so a peer cannot reach a human tier by renaming itself.
- The tool dispatcher schema-validates and trust-gates state-changing actions; peers and guests cannot actuate.
- Added `SECURITY.md` (disclosure policy, scope, the not-hardened caveat). `CLAUDE.md` untracked from the public repo.

### Added
- Extensibility ports (`docs/EXTENSIBILITY.md`, `cognition/tools.py`): sensory `ContextSource`, motor `Tool`
  via `<TOOL …>` markers, backend `register_provider`, reflex `register_burst_task`; results in
  `TurnResult.actions` and the `actions` field of `/api/turn`.
- Notebook (schema v23, `cognition/notebook.py`, `[notebook]`): model-curated, lossless working memory; the
  `notebook` tool; reading a note re-runs recall (`read_with_memory`).
- Temporal Registry (schema v24, `cognition/temporal_registry.py`, `[temporal_registry]`): a dated,
  status-tagged milestone ledger (STATE) beside the narrative memory store; `[Timeline]` prompt section;
  `record_milestone` tool; a sleep-time reconcile that demotes memories it supersedes.
- Burst worker (`cognition/burst.py`): one prioritised, interruptible idle-window pool for all post-reply work.
- Wall-clock sleep cycle (`cognition/sleep_cycle.py`, `kv` table): phases skipped if they will not fit the
  window, checkpointed per day, catch-up before noon; `GET /api/sleep/status`.
- Inner life (`cognition/inner_life.py`, `[inner_life]`, off by default): idle reflections stored faint and
  surfaced only when relevant; fresh conflicts escalated to the council; `POST /api/inner_life/run`.
- Self-directed deliberation (`cognition/deliberation.py`, `POST /api/deliberate/run`) and a council forum
  (schema v20, `/api/forum`).
- Temporal grounding (`cognition/temporal.py`, `[locale]`): current-time line, relative-age tags on recalled
  facts, deterministic answers to plain time questions; interaction-gap baselines (v14); daily/weekly/monthly
  narratives (v15).
- Sessions and history (v16, v17): recent turns replayed as real messages; `/api/history`, `/api/sessions`,
  `/api/session`.
- Output-triggered RAG (`[output_rag]`, on by default) and opt-in two-pass draft-RAG (`[draft_rag]`).
- Self-knowledge bake (`[self_knowledge] doc`) and self-observability (`diagnostics.py`,
  `[diagnostics] surface_errors`, a pool-health context section).
- Documents drop folder and upload (`[documents]`): .txt/.md in core; .pdf/.docx via the `mimir-0[documents]`
  extra; images transcribed by the `vision` role; `/api/documents{,/upload,/scan}`.
- Library layer (migration 21, `docs/LIBRARY.md`): cited claims plus Markdown composites, Library tab, deep
  read, `[library] model_fetch`, `/api/library*`; `forget_document` purges every derived layer; per-layer turn
  toggles (`include_memory` / `include_library` / `include_wiki`).
- Kiwix/ZIM reference layer (`[wiki]`): stdlib HTTP, fail-open, `GET /api/wiki/status`.
- Citation guard (`cognition/citations.py`, on by default): flags citations to sources the system does not hold.
- `--reembed` / `Mimir.reembed()` rebuilds the vector store after an embed-model change.
- Benchmark dimensions `discipline`, `epistemics`, `reasoning`, `vision`; `chat` and `reasoning` roles gate on
  discipline and epistemics; `model = "auto"` role resolution; a recommended-models registry; `[roles.vision]`.
- Qualifying tournament (`/api/fleet/tournament/*`), qualify-new, skip-model, council-pool grading; per-node
  placement matrix (`/api/fleet/placement`); diversity-first council roster (`/api/fleet/council`); per-node
  veto, role pinning, ranked fallback chains; latency-aware routing (`model/latency.py`).
- Context-size presets (Small/Medium/Large/X-Large) setting `num_ctx`, `context_budget_tokens` and
  `benchmark_num_ctx` together.
- UI: streaming chat (`/api/turn/stream`), memory graph view, Sleep tab, Profile tab with a 19-question
  seeding interview (`/api/onboarding`), Models tab, merged Library tab.

### Changed
- Memory distillation: provisional tiers decay with a 10-day half-life; archival is tier-based; sentinel,
  working-memory and self-model rows are pruned.
- Working memory folds only the oldest turns (`fold_threshold`, `keep_recent`).
- Recommendation is a points total (quality, speed, size prior); speed is measured from decode throughput
  (Ollama `eval_count` / `eval_duration`), normalised to seconds per 256-token turn.
- The coherence benchmark dimension was removed (its column stays, NULL); verifiable dimensions run at temperature 0.
- Council personas are pinned one per node across the fleet (concurrency cap 16).
- Timezone accepts UTC offsets without `tzdata`; IANA names need the `mimir-0[timezone]` extra.

### Fixed
- Fleet: timeouts are not retried and the node is cooled down; "model not found" is non-transient;
  `ResilientEmbedder` falls back to keyword recall; boot does not block on inventory; Ctrl-C works mid-benchmark.
- Benchmark: per-node speed no longer overwritten model-wide; a failed probe records None, not 0.0 s; warmups
  use `num_predict=1`; a bad node fails over instead of hanging; every model up to `max_model_size_b` is tested.
- Role pins and "Apply best" survive restarts; disabled models drop out of routing and fallback chains.
- Embedding models without "embed" in the name are never used as chat or council models (`is_embedding_model()`).
- Epistemics: internal tags stripped by `mimir.sanitize` (stream-safe); the uncertainty flag is a directive;
  library claims count toward `source_count`; the self-model may not invent name, operator or location.
- Inner life and deliberation consider only stated beliefs (DOCUMENT and INFERRED tiers excluded).
- Documents: DOCX tables extracted (merged-cell bug fixed); scans report `failed` / `unsupported`; `force` re-ingest.
- UI: graph layout no longer explodes; columns scroll independently; a running tournament reattaches after refresh.

### Validation
- End-to-end on a live multi-node LAN Ollama fleet: document ingest → recall with provenance, a multi-family
  council verdict, clean consolidation, a full fleet benchmark. 501 tests; `python -m mimir.selftest` passes
  on the mock provider with no model installed. (523 tests as of the 2026-10-31 additions.)

## [0.1.0] — pre-alpha, feature-complete

The first feature-complete pre-alpha: the whole `DESIGN.md` architecture is implemented and
verified end-to-end against a live multi-node LAN. Still unhardened and untuned.

### The spine
- The §6 acceptance loop: boot empty → converse → bake a memory → a later turn recalls it with
  correct evidence tier and provenance via `build_context()` → the sentinel fires async and leaves
  a note. Runs as an automated self-test with a canary.
- Two chokepoint gateways: storage (single-writer thread, priority queue, batching, coalescing,
  retry-on-locked, flush) and model (provider pool with priority tiers, retry/backoff,
  transient-fail signaling, saturation breaker, failover).
- SQLite schema with versioned migrations (v1–v8), a startup schema check, and the fail-loud
  doctrine throughout (no silent swallow).

### Knowledge & epistemics
- Three-mode embeddings: stdlib bootstrap (locality hashing), endpoint, degraded — active mode
  reported loudly.
- Typed knowledge with evidence tiers + provenance, hybrid retrieval, and a deterministic
  uncertainty gate.
- Document ingestion (`ingest()`): text + markdown in core, PDF via the `[documents]` extra.
- Entity graph: subject–relation–object triples with 1–2 hop traversal.
- Working memory: rolling recency + periodic compression.
- Self-model: an evolving, generic identity synthesized from the store's own history, plus a
  re-runnable 8-anchor identity interview.
- Procedural memory: learned trigger → procedure habits.

### Async cognition
- Sentinel: a reflective pass that leaves a note for the next turn.
- Sleep / consolidation: dedup, salience/confidence decay (with the death-spiral guard), archival,
  and contradiction resolution.
- Inner council: adversarial deliberation across auto-discovered models, synthesized into a verdict.

### Distributed model fleet
- LAN auto-discovery of Ollama nodes (zero setup on the nodes), model-aware routing (a request
  goes only to a node that has the model), active health checks, and least-loaded selection.
- A persisted catalogue and benchmarking — a capability "IQ test" (talk / tools / code) plus a
  coherence vote by a panel of other models, guarded by a canary pair.
- Per-role recommendations from the benchmarked catalogue.

### Surface
- A zero-dependency stdlib reference web UI: streaming chat, the identity interview, mind / memory
  / graph / habits browsers, the inner council, document ingest, and the fleet (scan / benchmark /
  recommend).
- The library API, plus `python -m mimir.{selftest,interview,server}`.
