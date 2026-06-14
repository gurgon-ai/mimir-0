# Mimir 0

**A local-first cognition core for evidence-aware memory and context assembly.**

Mimir 0 is a small Python library that gives a language model a memory that behaves like a
mind's, not a database's. Knowledge is **typed, provenance-tracked, and evidence-tiered**, and
it's assembled into the prompt with an explicit epistemic structure. You tell it something;
later it recalls that fact, cites where it came from, and tells you when it's reasoning from
thin evidence instead of confabulating.

> **Status: pre-alpha — feature-complete, actively evolving (snapshot 2026-06-13; subject to
> change).** The whole architecture in [`DESIGN.md`](DESIGN.md) is implemented and verified
> end-to-end against a live multi-node LAN: the acceptance loop, every typed knowledge layer, the
> async cognition, and the distributed model fleet. The **fleet qualification surface in particular
> is being actively built out and tuned** — the feature list below is a current snapshot, and APIs,
> schema, scores, and UI may shift between commits. It is **not yet hardened**. Setup lives in
> [`docs/SETUP.md`](docs/SETUP.md); see [`CHANGELOG.md`](CHANGELOG.md) for the running log.

## What's included vs. what you provide

This repo ships **only code** — everything in it is ours, under Apache-2.0. Anything that *runs a
model* is yours to install; Mimir talks to it over a local endpoint and never bundles it. That
keeps the repo fully distributable and your install footprint minimal.

| ✅ Included (in this repo) | 🔧 You provide (install yourself) |
|---|---|
| The library + reference web UI — **pure Python, zero runtime dependencies** | **[Ollama](https://ollama.com)** (or any chat/embeddings endpoint) — for real model inference |
| **SQLite** storage — bundled with Python; no install, no server, no daemon | **Open model(s)** — `ollama pull` whatever you like (each under its own license) |
| A deterministic **mock provider + stdlib embedder** so the core runs with *nothing* installed | *(optional)* `pypdf` — only for PDF ingestion (the `[documents]` extra) |

**Core runtime dependencies: none.** Python's standard library (including SQLite) is the entire
floor — the mock provider and bootstrap embedder run the full acceptance loop on Python alone. You
only need Ollama + a model when you want a real LLM conversation. Nothing here is something you
*can't* freely use or redistribute.

## Try it in 10 seconds (zero account, no model server)

```bash
pip install -e ".[dev]"
python -m mimir.selftest        # runs the whole loop on a deterministic mock provider
python examples/quickstart.py   # watch it bake a fact and recall it, attributed
```

No Ollama, no GPU, no network needed — Mimir ships a mock provider and a stdlib bootstrap
embedder so the core loop boots on literally Python + SQLite. For a real conversation with a
local model, see [`docs/SETUP.md`](docs/SETUP.md).

Prefer a browser? Point it at a local model and run the **reference web UI** (stdlib, zero deps —
chat, the identity interview, and document ingest):

```bash
python -m mimir.server --config mimir.toml   # → http://127.0.0.1:8765
```

## What makes it different

Most memory libraries store text in a vector blob and retrieve by similarity. Mimir 0 adds:

- **Typed knowledge layers** — facts, learned conclusions, and entity relationships live in
  separate stores with separate retrieval, not one flat index.
- **Evidence tiers + provenance** — every fact knows who said it and how reliable that source
  is. Injected context is attributed, never flattened into "you told me."
- **Confidence vs. salience, decoupled** — truth and relevance are separate axes. A fact
  doesn't become *false* just because it hasn't been used lately; it just becomes less *salient*.
- **An uncertainty gate** — when the system is reasoning from thin evidence, it says so and
  asks a clarifying question, instead of guessing confidently.
- **An async "second mind"** — a reflective pass reviews each turn and leaves a note for the next.

## What's inside

Every turn assembles an epistemic prompt — self-model → identity → persona → attributed knowledge
(memory + documents + entity graph) → learned procedures → working memory → sentinel note →
uncertainty gate — and routes it through two disciplined gateways. On top of that:

- **Document ingestion** — `ingest()` for text/markdown (PDF via the `[documents]` extra) into a
  document-tier layer with file/section provenance.
- **Entity graph** — subject–relation–object triples with 1–2 hop traversal.
- **Working memory & self-model** — rolling salient context, and an evolving generic identity
  seeded by a short **identity interview**.
- **Sleep / consolidation** — dedup, decay, archival, and contradiction resolution, so memory
  maintains itself.
- **Inner council** — adversarial deliberation across whatever models are installed.
- **Distributed fleet + qualifying tournament** — point it at your LAN and it discovers every
  `ollama serve` (zero setup on those machines), routes each request to a node that has the model,
  and **qualifies** them on a measured battery (talk, tools, code, reasoning, discipline, and an
  epistemic-framework gauntlet — tier-deference under noise, context grounding, long-context recall
  that scales with your deployment window). Run it as a staged, human-veto **tournament** that
  narrows the fleet round by round. Qualification is **distributed and concurrent** (one worker per
  node), and it qualifies at your **operational context window**, not a toy one. Then:
  - a **per-node placement matrix** — every model on every node it runs on, that node's measured
    speed, and each node's **winner** (best quality, speed breaking ties) and ⚡ fastest;
  - a **diversity-first "second lineup"** — an adversarial council roster that favours a *spread of
    model families* over raw ranking (different families fail differently), graded with the
    user-facing size/latency caps **off** so the big, slow, brilliant models a chat cap excludes
    still earn a council seat;
  - a **self-explaining leaderboard** — it shows *why* a model is barred from a role
    (`discipline 0.25 < 0.50`), not just who won, and frames scores as *operational fitness for this
    system on this hardware* — best **for you**, not "best model in the world."

  Run the brain on a Raspberry Pi and borrow GPUs over the network.
- **Reference web UI** — a zero-dependency stdlib server for all of the above.

## Runtime contract

Runs on **Python + SQLite + one chat endpoint + one embeddings endpoint.** No GPU requirement,
no cloud, no peripherals. Bring a local model (Ollama, etc.) or an API — it's provider-agnostic,
and it can pool several local inference nodes across a LAN for distributed local inference.

## Contributing

The bar for core is high and the discipline is specific (zero runtime deps, the two gateways are
law, fail loud, keep core layers generic). See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).

---

*Mimir 0 is the general, reusable cognition core extracted and rebuilt from a larger private
home-AI system. None of the original's hardware integrations or personal data come with it —
just the memory and reasoning architecture.*
