# Mimir 0

[![ci](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml/badge.svg)](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A local-first cognition core for evidence-aware memory and context assembly.**

Mimir 0 is a small Python library that gives a language model a memory that behaves like a
mind's, not a database's. Knowledge is **typed, provenance-tracked, and evidence-tiered**, and
it's assembled into the prompt with an explicit epistemic structure. You tell it something;
later it recalls that fact, cites where it came from, and tells you when it's reasoning from
thin evidence instead of confabulating.

> **Status: pre-alpha — feature-rich, actively evolving (snapshot 2026-06-20; subject to
> change).** The whole architecture in [`DESIGN.md`](DESIGN.md) is implemented and verified
> end-to-end against a live multi-node LAN: the acceptance loop, every typed knowledge layer, the
> async cognition, and the distributed model fleet. On top of the spine, the **highest-leverage
> thinking layers** from the larger private home-AI have been extracted public-clean — temporal
> grounding, hierarchical memory narratives, the idle-window burst worker, durable session history,
> a visual memory graph, a self-curated **notebook**, and a **STATE-vs-NARRATIVE timeline** — over a
> set of **typed connector ports** ("hands") for tools, context, and backends, alongside the **fleet
> qualification** surface, which is still being actively tuned. The feature list below is a current
> snapshot; APIs, schema, scores, and UI may shift between commits, and it is **not yet hardened**.
> Setup lives in [`docs/SETUP.md`](docs/SETUP.md); see [`CHANGELOG.md`](CHANGELOG.md) for the log.

## What's included vs. what you provide

This repo ships **only code** — everything in it is ours, under Apache-2.0. Anything that *runs a
model* is yours to install; Mimir talks to it over a local endpoint and never bundles it. That
keeps the repo fully distributable and your install footprint minimal.

| ✅ Included (in this repo) | 🔧 You provide (install yourself) |
|---|---|
| The library + reference web UI — **pure Python, zero runtime dependencies** | **[Ollama](https://ollama.com)** (or any chat/embeddings endpoint) — for real model inference |
| **SQLite** storage — bundled with Python; no install, no server, no daemon | **Open model(s)** — `ollama pull` whatever you like (each under its own license) |
| A deterministic **mock provider + stdlib embedder** so the core runs with *nothing* installed | *(optional)* `pypdf` + `python-docx` — only for PDF/DOCX ingestion (the `[documents]` extra) |
| The wiki integration — **pure stdlib HTTP**, no library | *(optional)* **[Kiwix](https://kiwix.org)** `kiwix-serve` + any **ZIM** — only if you want offline Wikipedia as a reference layer |
| Temporal grounding falls back to the **host's local clock** | *(optional)* `tzdata` (the `[timezone]` extra) — only to set an explicit IANA timezone on a host without a tz database (e.g. Windows) |

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

**Recommended models** — *empirical, not prescriptive.* Mimir is **model-agnostic**: it benchmarks
whatever fleet you install and picks the best per role automatically, so the right answer is always
"measure your own." The list below is simply **what scored well on my hardware** (edge nodes hanging
off an RTX box) — a good first `ollama pull`, never a requirement, and your mileage will differ. On
**small / edge hardware**, **`gemma4:e2b`** was the standout here — fast, vision-capable, the
strongest small-model epistemics I measured. Mid-range: `gemma3:12b` or `qwen2.5:14b`. On a capable
box, `gemma4:26b` and `qwen3.5:27b` topped my board. Pull anything; let the benchmark decide.

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

Every turn assembles an epistemic prompt — self-model → **current state (timeline)** → identity →
persona → **the current moment** → attributed knowledge (memory + documents + entity graph, **each
fact tagged with its age**) → learned procedures → **recent history (journal)** → working memory →
**temporal awareness** → background notes → sentinel note → uncertainty gate — and routes it through
two disciplined gateways.
On top of that:

- **Document ingestion + a local "wiki"** — `ingest()` a file by path, **upload with the 📎** by the
  chat box, or just **drop files into a `[documents] folder`**. Idle time ingests new/changed files
  into a document-tier layer (file/section provenance) and writes a short summary of each — a small
  browsable wiki the model draws on. Text/markdown work in core; **PDF + DOCX need the optional
  extra** (`pip install 'mimir-0[documents]'` — pulls `pypdf` + `python-docx`; a missing extra fails
  loud with that instruction, the scan never silently skips). Extraction is text-only (no OCR).
- **Offline encyclopedia (optional)** — point a `[wiki]` block at a local **Kiwix server** over any
  **ZIM** (Wikipedia nopic, a medical wiki, top-50k, …) and the model gets a live, attributed
  reference layer — **zero Python dependency** (stdlib HTTP, like talking to Ollama), nothing to
  ingest, fail-open.
- **Draft-RAG (optional, two-pass recall)** — a per-turn chat toggle: the model writes a short
  *draft* answer first, memory is re-retrieved against that draft (it names what the reply is *about*,
  which the user's wording alone can miss), and the new hits fold into the prompt the real answer is
  generated from. It does **two LLM calls per turn, so replies are slower** — off by default, with a
  one-click warning. (`[draft_rag] enabled` for library callers; `turn(draft_rag=True)`.)
- **Entity graph** — subject–relation–object triples with 1–2 hop traversal.
- **Working memory & self-model** — rolling salient context (folds the oldest exchanges into a
  short summary, keeps the most recent verbatim), and an evolving generic identity seeded by the
  **seeding interview** (a re-runnable, ~12-essential + 7-optional get-to-know-you whose answers
  become the operator's highest-provenance orienting facts).
- **Self-knowledge** — it bakes its own docs (the README, by default) into memory in the nightly
  cycle, so it can answer about what it is and how it works, grounded in its own documentation.
- **Temporal grounding** — an always-on clock/calendar sense (date, season, "3 days ago" on recalled
  facts), a zero-cost intercept for plain time questions, and an awareness baseline that notices when
  you've been away longer than usual *for your own rhythm*.
- **Temporal narratives** — a hierarchical daily → weekly → monthly journal, lossy by design (details
  fade, patterns persist), written off the hot path and injected as recent history.
- **The burst worker** — post-response cognition (sentinel, self-model, working memory, sleep,
  narratives) scheduled into the idle window after each reply: pent-up-demand priority, interruptible,
  with results that surface into the next turn. Includes **bidirectional (output-triggered) RAG** —
  it retrieves memory relevant to the model's *own reply* and grounds the next turn with it, so a
  thread the model opened isn't dropped (DESIGN §5a).
- **Self-observability** — fail-loud, but also fail-*aware*: it captures its own recent errors and
  surfaces them (plus backend-fleet health: nodes up/down, per-node speeds) into the turn's context
  and the Mind tab, so the model knows when it's degraded — and the nightly cycle digests them
  (DESIGN §10).
- **Session history** — a durable, restorable conversation log; the web UI switches between past
  conversations and the model replays the active one for real continuity.
- **Visual memory graph** — the chat pane flips to a drifting "galaxy" of memory blobs + entities
  (foundational facts brightest and central); click any blob to review/edit it.
- **Sleep / consolidation** — dedup, decay, archival, and contradiction resolution, so memory
  maintains itself. Runs in a user-set **nightly window** (with phase budgeting, same-night resume,
  and catch-up) because streaming chat on a slow machine leaves the post-turn window too short for
  heavy work — plus a "run sleep now" button any time (DESIGN §5a).
- **Inner council** — adversarial deliberation across whatever models are installed. Convene it on a
  question yourself, or let it run **self-directed during sleep**: the system surfaces its own
  conflicts (graph tensions, divergent memories), a curator picks the few worth arguing, and the
  verdicts become recallable understanding (DESIGN §5a). Personas **fan across the whole fleet** (one
  node each, in parallel), and every debate persists to a **forum** view (toggle over the chat, like
  the graph) you can read, comment on, and keep house in.
- **Live inner life** — between conversations it can **think on its own**: on a gentle, you-set
  cadence it reflects on a recent exchange, a memory, a tension in what it knows, or an error it hit,
  and keeps the thought as a low-confidence note that resurfaces only if it later turns out relevant.
  It runs *off* the chat model and yields the instant you type, but it does use spare compute — so
  it's **off by default**; flip it on and set the pace in the Sleep tab (DESIGN §5a).
- **Distributed fleet + qualifying tournament** — point it at your LAN and it discovers every
  `ollama serve` (zero setup on those machines), routes each request to a node that has the model,
  and **qualifies** them on a measured battery (talk, tools, code, reasoning, discipline, and an
  epistemic-framework gauntlet — tier-deference under noise, context grounding, long-context recall
  that scales with your deployment window). Run it as a staged, human-veto **tournament** that
  narrows the fleet round by round. Qualification is **distributed and concurrent** (one worker per
  node), and it qualifies at your **operational context window**, not a toy one — a **Context-size
  slider** (Small → X-Large) sets that window *and* how much memory you recall, together. Then:
  - a **per-node placement matrix** — every model on every node it runs on, that node's measured
    speed, and each node's **winner** (best quality, speed breaking ties) and ⚡ fastest. The
    tournament board groups results **by machine** and shows *every machine tested* — including the
    ones where a model failed or timed out (a failed test is still a result), so no node is hidden;
  - a **diversity-first "second lineup"** — an adversarial council roster that favours a *spread of
    model families* over raw ranking (different families fail differently), graded with the
    user-facing size/latency caps **off** so the big, slow, brilliant models a chat cap excludes
    still earn a council seat;
  - a **self-explaining leaderboard** — it shows *why* a model is barred from a role
    (`discipline 0.25 < 0.50`), not just who won, and frames scores as *operational fitness for this
    system on this hardware* — best **for you**, not "best model in the world."

  Run the brain on a Raspberry Pi and borrow GPUs over the network.
- **Notebook** — lossless, name-addressable working memory the model curates itself (markdown with
  `##` sections), distinct from the lossy memory store: *memory is what it knows; a notebook is what
  it's working on.* Never decayed or deduped; a re-read **re-triggers recall** so the note reconnects
  to live memory instead of becoming an orphaned clipping.
- **Timeline (STATE vs NARRATIVE)** — memory accumulates in mixed tense ("planning X" … "X is done"),
  so a status question can surface a stale *planning* note. The **Temporal Registry** is the separate
  STATE axis: a small, dated, status-tagged ledger of milestones (what's true *now*) that injects a
  high-attention timeline, pins your current setup into the self-model, and during sleep **reconciles**
  memories it supersedes — guarded so it never touches a memory sharing only a generic word.
- **Connector ports ("hands")** — the brain ships four typed extension seams so you attach
  capabilities without forking: **tools** the model invokes mid-turn through one trust-gated dispatcher
  (a peer/guest can't actuate), **context sources** that fold typed sections into the prompt,
  **backends** (any chat/embeddings provider), and **background reflexes**. The notebook and timeline
  above are the two reference connectors built entirely on these ports. See
  [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md).
- **Reference web UI + integration API** — a zero-dependency stdlib server for all of the above, and
  a documented, optionally token-authenticated HTTP API ([`docs/API.md`](docs/API.md)). Mimir is a
  *brain with endpoints, no built-in hands*: bring your own IO (voice, avatar, Home Assistant, an
  agent framework — or a relay where two Mimirs talk to each other).

## Integration API — a brain with endpoints, no built-in hands

Mimir ships **no IO of its own** (no voice, avatar, Home Assistant, social) — on purpose. You drive
it through a small, stable surface and build whatever front-end you want on top: a voice loop, an
avatar, a home-assistant bridge, an agent framework, or a relay where **two Mimirs talk to each
other**. It *can* grow hands on your terms — register **tools** on the motor port and the model
invokes them mid-turn through one trust-gated dispatcher, reported back in the turn's `actions` (see
[`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md)). Full contract in [`docs/API.md`](docs/API.md);
the essentials:

**In Python (the cleanest path):**
```python
from mimir import Mimir
m = Mimir.from_config("mimir.toml")
print(m.turn("My garlic goes in around October.", user="alex").reply)
print(m.turn("When do I plant the garlic?", user="alex").reply)   # recalls it, attributed
```

**Over HTTP** — `python -m mimir.server --config mimir.toml` serves the UI *and* the API on one port:
```bash
curl -s http://127.0.0.1:8765/api/turn \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "user": "alex"}'
# → {"reply": "...", "introspect": {context accounting: sources, tiers, tokens}}
```
- **`user` is the speaker's identity** and **`speaker_kind` (`human`/`ai_peer`) is its kind** — the
  seam for multi-speaker and agent-to-agent. The server, not the caller, decides how much each is
  *believed*: `[identity] primary_user` → top tier, `trusted_users` → trusted, any other human →
  conversation tier, and a **peer AI** (`speaker_kind="ai_peer"` or `peer_agents`) → a *lower*
  `stated_by_peer` tier, attributed and marked AI-sourced — so one agent's hallucination (or two
  agents echoing each other) can't be mistaken for fact. An exposed endpoint can't launder claims
  into trusted memory, and a peer can't reach a human tier by renaming itself.
- **`POST /api/turn/stream`** streams the reply token-by-token (Server-Sent Events) for low-latency
  voice/chat front-ends.
- **`GET /api/health`** — instant, unauthenticated liveness (`{ok, busy, embed_mode, nodes_up}`).

**Security (opt-in, off by default):** set `[server] api_token` (or the env var named by
`api_token_env`, default `MIMIR_API_TOKEN`) and every `/api/*` route requires
`Authorization: Bearer <token>`. The **local browser UI is exempt by default** so a fresh run is
never blocked — the token guards *remote/integration* callers; `[server] secure_ui = true` requires
it locally too. `[server] cors_origins` allows browser front-ends on other origins.

**Two Mimirs hanging out:** give each instance the same API, then a tiny relay loops one's reply into
the other's `POST /api/turn` (tagged with its `user` name). Each remembers the other as a peer — and,
per the trust policy above, won't take the other's hallucinations as gospel. A worked relay example
is in [`docs/API.md`](docs/API.md).

## Runtime contract

Runs on **Python + SQLite + one chat endpoint + one embeddings endpoint.** No GPU requirement,
no cloud, no peripherals. Bring a local model (Ollama, etc.) or an API — it's provider-agnostic,
and it can pool several local inference nodes across a LAN for distributed local inference.

## Security

Pre-alpha and not yet hardened; the reference server is for localhost / a trusted LAN, not a public
service. Responsible disclosure and the security model are in [`SECURITY.md`](SECURITY.md).

## Contributing

The bar for core is high and the discipline is specific (zero runtime deps, the two gateways are
law, fail loud, keep core layers generic). See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).

---

*Mimir 0 is the general, reusable cognition core extracted and rebuilt from a larger private
home-AI system. None of the original's hardware integrations or personal data come with it —
just the memory and reasoning architecture.*
