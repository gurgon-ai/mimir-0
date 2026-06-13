# Changelog

All notable changes to Mimir 0. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Pre-1.0: the API and schema may change between releases.

## [Unreleased]

First fixes from real single-machine + LAN use after the feature-complete cut.

### Fixed (benchmark hangs)
- **The latency cap didn't actually bound the benchmark — a 7s cap could still hang for minutes.**
  Three compounding causes: (1) the pre-gate **warmup was untimed** (120s ceiling) *and* generated
  freely, so a thinking model (e.g. qwen3) could reason for the full 120s during the load, before the
  cap was ever checked; (2) the scoring **battery calls** ran on the pool's production 120s socket
  timeout, which the cap never touched; (3) the pool **retried** that 120s timeout up to 3× on the
  same slow node (~6 min on one model). Now: warmups load with a **single token** (`num_predict=1`)
  so they can't reason; scoring calls carry a **tight per-call timeout** (~2× the latency budget,
  45–90s) and run with **no pool retries**, so a slow/wedged model fails fast and the round continues
  (it still fails *over* to another node that has the model — it just doesn't retry the slow one).
  New plumbing: a reserved `__timeout_s__` param the Ollama provider honours per call, and a
  `max_retries` override on the pool/gateway.
- **The latency cap must never cut *capability* — a model slow on one weak node may be excellent on
  another.** Capability is per-model (test it once, anywhere); latency is per-(model, node) and only
  governs routing. The benchmark no longer skips a model because it exceeds the user's cap; it only
  skips a node too slow to *test* on within a generous budget (`max(30s, cap)` per turn), records the
  per-node speed for routing, and the cap is applied at finals/routing — not as a quality filter.
  (This reverses an earlier "cap skips early" choice. Per-node *requeue* — try a faster node before
  giving up — and concurrent distribution across nodes are the next build.)
- **The latency cap's pre-gate measured the wrong thing, so it never skipped slow models.** The gate
  timed `"reply ok"` — one token, instant for anything — so a model that's snappy on a token but
  takes ~13s on a real turn sailed through a 7s cap, then crawled through the ~15-call battery
  (~160s total, none of it skipped). The pre-gate now times a **representative ~64-token generation**
  and normalizes it to **seconds per ~256-token turn** (the cap's actual units), so a 13s/turn model
  is skipped *before* the battery under a 7s cap. Per-node speed is now stored in the same normalized
  units, so routing's fastest-node pick reflects real per-turn latency.
- **A page refresh lost the whole tournament/benchmark view.** The resume logic only ran on the
  Fleet-tab click, so a fresh load never reconnected — the run kept going server-side but the UI
  forgot it. It now reattaches on page load (and tab-open), and shows a per-model elapsed timer so a
  slow model reads as grinding, not hung.

### Added
- **Per-node veto (schema v13).** Each discovered edge node can be toggled off in the Fleet tab —
  excluded from the pool's routing (with a fail-safe if *every* node is vetoed, so chat never
  hard-blocks), from qualification, and from recommendations, even if it's reachable. "Don't use that
  box, even though it's there." Mirrors the per-model enable/disable; `node_prefs` table + a
  `/api/fleet/node` endpoint.
- **The qualifying tournament — a staged, human-veto model knock-out** (Fleet tab → "🏆 Run
  qualifying tournament"). Built on the benchmark's new staging primitives (subset / triage /
  ephemeral): **Round 0 · Qualifying** scores the *whole* fleet fast and cheap (capabilities only,
  nothing saved) → you **untick** who shouldn't advance → **🥊 FIGHT → Round 1 · Gauntlet** re-tests
  only the survivors through the full framework qualification (reasoning + the epistemic layered/
  grounding/long-context probes, scores saved) → veto again → **Round 2 · Finals** champions each
  role *among your finalists only* (the veto wins over the global best), then Apply. Round 3 (vision)
  is reserved.
  The board takes over the chat pane; rounds run in the background with live progress and resume on
  tab-switch. New endpoints under `/api/fleet/tournament/{start,advance,apply,status}`.
- **A `reasoning` dimension in the benchmark (schema v12).** The old battery only tested *format*
  compliance (say `PONG`, return a weather JSON, write `def add`) — every competent model passed, so
  `quality` saturated near 1.0 and a fluent model that *couldn't actually solve anything* could sweep
  every role. The new dimension scores deterministic, regex-checkable **problems** (multi-step
  arithmetic, letter-counting, sequence completion, a code-trace, an instruction transform) — wrong
  answers fail however fluent the prose. The `chat`/`reasoning`/`code` roles now **gate** on it.
- **A chat-LLM epistemic qualifying round.** `score_epistemic_competence` now includes a big
  **layered, conflicting-tier gauntlet** (high-evidence section says "blue", a lower section says
  "red", buried in irrelevant filler → defer to the high tier under noise — the structured arm can,
  a flat blob can't), a **grounding floor** (recall a nonce that exists *only* in the provided
  context), and a **long-context needle** (a nonce planted in the middle of a ~2k+-token document —
  past Ollama's 2048 default, so it doubles as proof the `num_ctx` pin works). A model that can't
  follow a layered prompt, won't read context, or can't handle long input is barred from `chat`.
- **`backend.min_model_size_b` (a size floor; 0 = off).** The sibling of `max_model_size_b`: on
  capable hardware, an imperfect test lets a tiny model that scores "high enough" and wins on latency
  keep beating a bigger, genuinely-better one a second behind at the same score. The floor excludes
  models below it from scoring (and therefore from recommendations). Exposed as a UI scope field.
- **`backend.benchmark_num_ctx` (default 8192).** Ollama defaults `num_ctx` to a tiny 2048 unless
  told otherwise; the benchmark now pins an explicit, consistent context for every model so the
  layered prompts aren't silently truncated (which would cut off the high-tier fact). Raise to test
  longer context. The pre-gate warmup uses the same value, so a model loads once and stays warm.

### Changed
- **Benchmark latency now reflects a real turn, not a 3-token reply.** `return_time` was the mean
  wall-time of the battery's tiny calls, which can't tell a slow remote 12B from a snappy local 3B —
  so a big model looked "instant" and won even speed-weighted roles. It's now measured from one
  real-length generation, normalized to seconds per ~256-token turn.
- **Routing objective made explicit: the best-scoring model *for this system* that you're willing to
  wait for.** Latency is now a hard **cap** (`max_latency_s` excludes too-slow models before
  scoring), not a soft penalty. The `chat` role dropped its quality-minus-speed "balanced" formula
  for pure **quality-under-the-cap** (every role now ranks this way): within the cap a dominant model
  wins outright and speed only breaks ties — so a 26B that's a second behind a 4B at a higher score
  wins, because under the cap you've already decided the wait is worth it.
- **The benchmark scorecard is grouped by node/IP**, so it's obvious which machine each model runs
  on (LAN-only leftovers cluster under their IP instead of looking local), and gained a `Reason`
  column.
- **The Fleet tab leads with the tournament** as the recommended path; the manual
  Find / Benchmark / Apply buttons move below as the one-step-at-a-time equivalents.
- **Docs synced** — INFERENCE_ENGINE, DESIGN §4, README, and SETUP now describe the `reasoning`
  dimension, the epistemic gauntlet, real-turn latency, the size floor, `benchmark_num_ctx`, the
  quality-under-cap routing objective, and the qualifying tournament.

### Fixed
- **Benchmark timed cold model-loads, not warm performance.** With models swapping in and out of
  VRAM, every measurement included a one-time load cost — which inflated `return_time` *and* unfairly
  tripped the latency gate (a 26B that's fast warm but slow to load could be wrongly skipped). Now
  each model is **loaded with an untimed warmup call first**, then timed *warm* — for the latency
  probe/gate and the capability battery. A model that can't load within a 120s window is reported as
  unusably slow and skipped. Measurements now reflect steady-state, the way the model actually runs.
- **Thinking mode was never controlled (and couldn't be turned off).** All role params went into
  Ollama's `options`, but `think` is a *top-level* field — so thinking models thought by default
  (slow) and a `think` set in config was silently ignored. Now `think` defaults **off** (it slows
  generation and rarely improves output, per testing across models) and is sent top-level; opt in
  per role with `think = true` only where it helps (e.g. some models on tool selection). `think=false`
  is accepted by non-thinking models too, so it's safe everywhere.
- **Benchmark only ever tested the 8 smallest models — so a 4B "won" every role while the user's
  much better 26B model was never benchmarked at all.** The default capped the run at the 8 smallest
  approved models ≤30B, smallest-first, so mid/large models (gemma3:12b, gemma4:26b, …) were silently
  excluded and the recommendations were dominated by tiny models. Now the benchmark covers **all**
  approved models up to a **user-set size cap** (`[backend] max_model_size_b`, default 30B — only the
  user knows their hardware), and reports coverage ("benchmarked N of M; K skipped as too large") in
  the UI and logs. **Per-model latency timeout:** before the expensive battery, each model gets a
  trivial-prompt probe; one that exceeds the budget (`max_latency_s` if set, else a 30s default) is
  **skipped** instead of stalling the whole run — so a slow big model can no longer hang the
  benchmark (or hold the lock so a second run "doesn't work"). The UI now also shows the **scan
  phase** ("scanning the fleet…") instead of a blank "0/0", and reports too-slow skips.
  **Live scoreboard:** each model's scores stream into the Fleet area *as it finishes* (best-first
  table: quality + all dimensions + speed) — so the otherwise-idle UI fills with useful results
  during the run instead of just a counter (`benchmark_fleet` gained an `on_result` callback).
  **Scope fields on the Fleet tab:** "Max model size (B)" and "Max latency (s)" inputs (pre-filled
  from config) override the cap/latency for a run — no `mimir.toml` editing needed to control what
  the benchmark tests.
- **Benchmark looked frozen / gave no progress.** The fleet benchmark ran synchronously while
  holding the brain lock, so the entire web UI (including header polling) blocked for the multi-minute
  run, and nothing logged per model — it was impossible to tell a running benchmark from a broken one.
  Now: the run happens in a background thread, `/api/state` and a new `/api/fleet/benchmark/status`
  are lock-free so the page stays responsive, the UI polls and shows **"Benchmarking i/N: model…"**,
  and `benchmark_fleet` logs every model start/finish (`[i/N] model …`) so the log/console show life.
  Errors surface in the status instead of dying silently.
- **Identity drift in the self-model.** A small model (`gemma3:4b`) synthesizing the self-model could
  hallucinate a name not in the operator-established anchors (observed: anchor name `Mimir` but the
  synthesis wrote "I am Arthur"), creating a contradiction the chat model then adopted and inverted
  ("you serve Greg"). The synthesizer is now forbidden from stating or inventing the name, operator,
  or location — those are the verbatim anchors' job — and the identity section is framed as
  authoritative. (DESIGN §3e.)
- **Internal epistemic tags leaking into replies.** Small models absorbed the `[tier=…; source=…]`
  provenance style from the prompt and emitted it on their own sentences (even inventing
  `[tier=question]` / `[tier=focus]`). These are now stripped deterministically by `mimir.sanitize`,
  with a streaming-safe stripper so a tag split across stream deltas is still removed and no double
  space is left behind — applied to both the live SSE display and the stored exchange. (DESIGN §3b,
  §10.)
- **Boot no longer blocks on fleet inventory.** Initial LAN node discovery/inventory now runs in a
  background thread, so the web server starts listening immediately (~2s) instead of waiting on a
  full multi-node scan; a "Starting Mimir…" line prints at once.
- **Uncertainty flag no longer recited.** The §3d honesty flag was phrased as a statement
  ("grounded in only N sources") that models parroted verbatim into the reply — the same
  scaffolding-leak class as the tags. It is now a directive the model acts on (answer from what
  you know, name the gap, ask one question) and is told not to narrate its source count.

### Added
- **Recommended-models registry (inference engine, Phase A).** A curated, versioned data file
  (`cognition/recommended_models.toml`, loaded by `cognition/registry.py`) of families Mimir has
  tested — gemma/qwen/llama/phi/mistral/command-r/deepseek/granite/internlm, with per-role fitness,
  measured score floors, and `judge_ok` flags. `auto` routing now prefers a present
  recommended-for-the-role model **before any benchmark** (then approved-family, then any reachable),
  so a fresh user with both `gemma3:4b` and `gemma4:e4b` installed gets `gemma4:e4b` for chat, not the
  known-weak one — closing the worst out-of-box failure mode. Measured scores still override the
  registry once benchmarking runs. Not a whitelist: any installed model can still be measured and
  used. Spec: `docs/INFERENCE_ENGINE.md` §4.
- **Epistemic-competence experiment (`cognition/epistemics.py`).** Makes the core §3 thesis —
  typed/tiered/provenance context improves cognition over flat RAG — *measurable* per model. Each of
  three probes (tier deference, attribution, uncertainty) runs through the real `build_context()`
  (structured arm) and as a flat blob of the same facts (flat arm); `lift = structured − flat` is the
  framework's value. `brain.evaluate_epistemics(models, samples)` runs it across the fleet. Live
  cross-model finding: **positive lift for every model tested** — attribution is a universal win
  (impossible without provenance), the uncertainty gate most helps the weakest models, and
  tier-deference is model-dependent (gemma3:12b/gemma4:e4b defer perfectly; qwen3.5:9b ignores tiers).
- **`epistemics` is now a qualification-battery dimension.** The benchmark scores each model's
  structured-arm epistemic competence (does it exploit the framework?), and the identity-bearing
  roles (`chat`, `reasoning`) gate on **both** `discipline` and `epistemics` — so a model that
  ignores evidence tiers is barred from speaking as the system, just like one that leaks tags. New
  catalogue column (`epistemics`, schema v11); `ROLE_NEEDS` now lists multiple required capabilities
  per role. This is what keeps the framework from being handed to a model that won't use it.
- **Automatic model selection (`model = "auto"`).** A role's `model` can be pinned, set to `"auto"`,
  or omitted (→ auto). Auto resolves from the fleet by a strict hierarchy — **pin > measured-best
  (benchmarked + role-gated) > approved-family heuristic > any reachable model** — re-resolving on
  every rescan so a freshly benchmarked model is picked up. Users can **disable** a model (a bias
  veto) via `brain.set_model_enabled(...)` and it's skipped everywhere; the gateway stop-gaps an
  unresolved `auto` role to any reachable model so a turn never fails while the fleet is still
  inventorying. Default stays **local-only** (the LAN fleet is opt-in). New `model_prefs` table
  (schema v10) and a `brain.model_pool()` view (qualified ✓, speed, size, nodes, enabled, roles
  served) behind the Model Pool UI.
- **Model Pool tab in the web UI.** Lists every routable model with a ✓ if it passed the
  qualification gate, its size/quality/discipline/speed/nodes, and which roles it serves. A
  checkbox per model toggles it in or out of the automatic pool (the bias veto) — disabling a model
  serving an auto role re-routes that role live. Shows the backend mode (local vs LAN fleet) and the
  auto roles. New endpoints: `GET /api/fleet/pool`, `POST /api/fleet/model`.
- **`discipline` capability in the fleet IQ test.** The benchmark battery now scores a fourth
  dimension: does the model honor prohibitions, above all **not reproducing the internal
  `[tier=...; source=...]` scaffolding it is shown**. The probe replicates the *production* condition
  that actually triggers the leak — a tag-saturated recall block under the real soft "don't copy the
  tags" instruction — and samples it several times, scoring the fraction of bracket-free replies
  (leakage is probabilistic; consistency is the signal, per DESIGN §4). A weak single-tag prompt was
  too easy and missed the failure. The identity-bearing roles (`chat`, `reasoning`) gate on
  discipline, so the recommender refuses to route them to a fluent-but-leaky model — caught in
  qualification, not production. New catalogue column (`discipline`, schema v9). Validated live:
  `gemma3:4b` scores 0.25 (barred) while `gemma4:e2b`/`e4b`/`qwen3.5:9b` score 1.00 and `gemma3:12b`
  0.75.

### Validation
- End-to-end live run against a real LAN Ollama node (`gemma3:12b` for chat/reasoning,
  `gemma3:4b` for bake, `nomic-embed-text:v1.5` for embed): clean self-model synthesis (no
  hallucinated name), correct non-inverted identity ("I am Mimir, and I serve Greg"), no leaked
  tags or flag text, and a working bake → recall with attribution.
- Broader subsystem validation against the live 4-node fleet (43 models): document ingest →
  recall with provenance; the inner council deliberating across 5 distinct models with a coherent
  verdict; sleep/consolidation (salience decay) running clean; and a fleet benchmark with the
  coherence-judge canary passing and per-model quality/return-time scored.

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
