# Changelog

All notable changes to Mimir 0. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Pre-1.0: the API and schema may change between releases.

## [Unreleased]

First fixes from real single-machine + LAN use after the feature-complete cut.

### Fixed
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
