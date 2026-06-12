# Changelog

All notable changes to Mimir 0. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Pre-1.0: the API and schema may change between releases.

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
