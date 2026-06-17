# Mimir 0 — Setup & Configuration

This is the **user-facing** guide: everything a new human (or a fresh AI session) needs to go
from a clone to a running cognition core. It is maintained from day one and updated in lockstep
with any change that touches setup or config — if something here is stale, that's a bug.

> For the architecture and the "why", read `DESIGN.md`. For build conventions, read `CLAUDE.md`.
> This document is only about **getting it running**.

---

## 1. The runtime contract (what Mimir needs to exist)

Mimir 0 runs on **exactly this and nothing else**:

- **Python 3.11+**
- **SQLite** — bundled with Python's stdlib; nothing to install
- **one chat endpoint** and **one embeddings endpoint** — behind a provider; both optional to
  *start* (see the zero-account path below)

Core has **zero third-party runtime dependencies**. If a step below asks you to install more than
Python to boot the core loop, it's wrong.

---

## 2. Install

```bash
git clone https://github.com/gurgon-ai/mimir-0
cd mimir-0
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` pulls in `pytest`, `ruff`, and `mypy`. Drop it (`pip install -e .`) for a runtime-only
install — core needs nothing else.

---

## 3. Prove it works in 10 seconds (zero account, no model server)

Mimir ships a deterministic **mock provider** and a **bootstrap embedder**, so the full §6
acceptance loop — boot → bake a memory → recall it with provenance → sentinel leaves a note —
runs with **no Ollama, no GPU, no network, no account**:

```bash
python -m mimir.selftest
```

You should see `self-test PASSED`. That is the entire cognition spine breathing. Then watch it
hold a two-turn conversation and print exactly what went into the prompt:

```bash
python examples/quickstart.py
```

This is the place to start. It needs nothing but Python.

---

## 3b. Talk to Mimir in a browser (the web UI)

The core is a library; the **reference web UI** is a thin adapter built on Python's stdlib
`http.server` — **no extra dependencies, no Node, no build step**. It's where a human chats (with
streaming + a thinking indicator), switches between past conversations, browses the **memory graph**,
runs the seeding interview, and ingests documents.

```bash
python -m mimir.server --config mimir.toml          # then open http://127.0.0.1:8765
# options: --host 0.0.0.0  --port 8765  --log-file mimir.log
```

The server logs to the console **and** to a rotating file (`mimir.log` by default, 5 MB × 3
backups) so a long run leaves a reviewable trail — point `--log-file` elsewhere, or pass
`--log-file ""` for console only.

The **chat pane** streams replies (each shows its source count and embedding mode, and flags thin
evidence) with a conversation dropdown on top (**Restore** a past conversation or **+ New** one) and a
**🕸 Graph** toggle that flips it to a drifting galaxy of memory blobs + entities — click any blob to
review/edit it. The side tabs cover **Profile** (the seeding interview / your orienting facts),
**Mind**, **Memories**, **Graph**, **Habits**, **Council**, **Fleet** (role assignment → qualify →
models), and **Docs**. It binds to localhost by default; it's a reference adapter, not a hardened
public service — put a reverse proxy in front if you expose it.

Under the hood it's a small JSON API (`/api/turn` + `/api/turn/stream`, `/api/identity`,
`/api/onboarding`, `/api/sessions` + `/api/history`, `/api/graph/map` + `/api/memory`, `/api/fleet/*`,
`/api/ingest`, `/api/state`), so you can build your own front-end against the same endpoints.

## 4. The three embedding modes (decide before real use)

Embeddings are "just another role." Mimir has **three honest modes** — pick one in config. The
active mode is always printed at startup and exposed in `build_context(...).introspect()`, so you
always know which one you're on.

| Mode | What it is | When to use |
|---|---|---|
| **bootstrap** *(default)* | A pure-stdlib *locality-hashing* embedder. Deterministic, offline, zero deps. **Lexical overlap only — NOT semantic search.** | First boot, tests, CI, "does it run." |
| **endpoint** | A real embeddings model via the model gateway. **Replaces bootstrap entirely.** | **Recommended for real use** — actual semantic recall. |
| **degraded** | No vectors at all; keyword-only retrieval. | Environments where you want no vector path. |

> ⚠️ The bootstrap embedder exists so Mimir boots on *literally* Python + SQLite. It matches words,
> not meaning — "car" and "automobile" look unrelated to it. Don't judge Mimir's memory by it.
> For real recall, configure **endpoint** mode (next section).

---

## 5. Real conversation with a local model (Ollama)

The reference provider talks to a local [Ollama](https://ollama.com) server using only stdlib
HTTP — no extra Python deps.

### 5.1 Install Ollama and pull models

1. Install Ollama from <https://ollama.com/download> and start it (it serves on
   `http://localhost:11434`).
2. Pull a chat model and an embeddings model. A solid, modest starting point:

   ```bash
   ollama pull llama3.1:8b        # chat / bake / reasoning (instruction-following)
   ollama pull nomic-embed-text   # embeddings (endpoint mode)
   ```

   These are **examples, not requirements.** Any instruction-following chat model works for the
   `chat`/`bake`/`reasoning` roles; any embedding model works for `embed`. Bigger/smaller models
   trade quality for speed and VRAM. See `DESIGN.md` §4 on roles.

   > ⚠️ **In endpoint mode the embedding model must stay reachable** — every turn embeds the query
   > and what it stores, so recall depends on it. The `[roles.embed]` name (and tag) must match a
   > model that's actually installed; `nomic-embed-text` and `nomic-embed-text:v1.5` are *different*
   > tags to Ollama, so pin the one you pulled. **On a LAN fleet**, make sure the embed model lives
   > on a node that's reliably up (ideally pull it on more than one) — if *no reachable node* has it,
   > Mimir doesn't crash: it **degrades loudly to keyword-only recall** (logged, and shown in the
   > Mind tab's system-health panel) and resumes semantic recall on its own when the model returns.
   > **Keep one embedding model fixed for the life of a store** — switching models changes the vector
   > space, so old and new embeddings stop being comparable until you re-embed.

   **Recommended models (a starting point, not a whitelist).** Mimir ships a curated, versioned
   registry (`src/mimir/cognition/recommended_models.toml`) of families it has tested — currently
   **gemma** (gemma4 e2b/e4b, gemma3:12b — *not* gemma3:4b), **qwen** (2.5/3/3.5), **llama** (3.x),
   **phi**, **mistral**, **command-r**, **deepseek**, **granite**, **internlm**. If you set a role to
   `model = "auto"`, Mimir prefers a present recommended model out of the box (so it won't land on a
   known-weak one), and after you benchmark, measured scores take over. Running a **variety of
   families** also unlocks stronger multi-family adversarial reasoning (the inner council). The full
   rationale is in [`INFERENCE_ENGINE.md`](INFERENCE_ENGINE.md).

   > **On model size and identity.** The `chat` and `reasoning` roles carry Mimir's *identity* — its
   > self-model and how it speaks as itself. Very small models (≈4B) are unreliable here: in testing,
   > a 4B model hallucinated its own name and mimicked the prompt's internal tag style. Mimir guards
   > both deterministically (the synthesizer can't invent a name; internal tags are stripped from
   > output), but for faithful identity prefer **≥12B** for `chat`/`reasoning` (e.g. `gemma3:12b`,
   > `qwen2.5:14b`). A 4B model is fine for `bake` (extraction). If you run a fleet, let
   > `brain.benchmark_fleet()` + `apply_recommendations()` pick capable models per role.

### 5.2 Write your `mimir.toml`

Copy the template and edit it:

```bash
cp mimir.toml.example mimir.toml
```

The template is fully commented. The essentials:

- `[storage] path` — where the SQLite brain lives (gitignored; never commit it).
- `[provider] type = "ollama"` and `host`.
- `[embeddings] mode = "endpoint"` to use the real embedder (or `"bootstrap"` to stay offline).
- `[roles.chat|bake|reasoning|embed]` — one `model` per role, plus tuned params. **Keep `num_ctx`
  identical across roles that share a warm model**, or Ollama reloads the model (slow) — see
  `DESIGN.md` §4.
  - **What you must provide:** whatever chat LLM(s) you like for `chat`/`bake`/`reasoning`, **and at
    least one embedding model** for `embed` (when `[embeddings] mode = "endpoint"`). Set
    `model = "auto"` and Mimir **discovers and routes to** what's installed — chat roles pick the
    best-benchmarked model, and `embed` discovers an embedding model and **remembers the choice** so
    the vector space stays stable. Pull the same embedding model on every node so recall survives any
    one node going down.

### 5.3 Run it

```python
from mimir import Mimir

brain = Mimir.from_config("mimir.toml")
print(brain.turn("My favorite color is teal.", user="alex").reply)
print(brain.turn("What's my favorite color?", user="alex").reply)
brain.close()
```

The second turn recalls the first, attributed to its source and evidence tier.

---

## 5a. Establishing identity (the init interview)

A fresh Mimir has no history, so its self-model starts thin. Give it a foundational identity via
**eight universal anchors**: `name`, `operator`, `location`, `purpose`, `values`, `scope`,
`boundaries`, `voice`. (The first four are who/where/why; the rest set the operating frame —
principles, responsibilities, hard limits, and tone — that the system can't derive on its own.)

Interactively (Mimir asks only the anchors it still needs):

```bash
python -m mimir.interview --config mimir.toml
# re-run any time to update existing answers:
python -m mimir.interview --config mimir.toml --revise
```

Or declaratively in `mimir.toml` (auto-established at boot — good for headless deployments):

```toml
[identity]
name = "Mimir"
operator = "your household or team"
location = "a home server"
purpose = "to remember, reflect, and assist"
values = "honesty, privacy, attributing knowledge to its source, admitting uncertainty"
scope = "household memory, documents, and reasoning; not financial or medical advice"
boundaries = "never fabricate facts, never expose private data outside the home"
voice = "concise, plainspoken, evidence-backed"
```

These anchors are injected verbatim at the top of the always-on self-model every turn, so the
foundational facts are reliably present, and they also seed the self-model's evolving narrative.
The interview is re-runnable (`--revise` re-asks all, Enter keeps each current value). From code,
drive your own with `brain.pending_identity_questions()` and `brain.establish_identity({...})`.

## 5b. Ingesting documents (v0.1)

Give Mimir documents to recall from. Plain text and markdown work with **no extra dependencies**;
PDF needs the optional extra.

```python
from mimir import Mimir

brain = Mimir.from_config("mimir.toml")
brain.ingest("notes/handbook.md")      # .txt and .md work in core
brain.ingest("research/paper.pdf")     # .pdf needs: pip install 'mimir-0[documents]'

print(brain.turn("What does the handbook say about onboarding?", user="alex").reply)
brain.close()
```

What happens: the file is **extracted** (markdown splits on headings; PDF splits by page),
**chunked** with overlap (carrying each section/page as provenance), **embedded**, and stored as
`document`-tier memories. On a later turn they are recalled like any other knowledge, attributed
to the file and locator (e.g. `handbook.md:Onboarding`, `paper.pdf:p.4`). Re-ingesting the same
path **replaces** its previous chunks rather than duplicating them.

- Install PDF support: `pip install 'mimir-0[documents]'` (pulls `pypdf`). Without it, ingesting a
  `.pdf` fails loud with that instruction — it never silently skips.
- Recall quality on documents depends on the embedding mode (§4). Bootstrap matches words; for
  semantic recall over documents, use **endpoint** mode.

## 5c. Offline encyclopedia (Kiwix/ZIM) — optional, zero extra deps

Point Mimir at an offline copy of Wikipedia (or any **ZIM**) and it becomes a **live reference
layer**: each turn's query is searched and the top articles are injected as an attributed
`Reference — … (Wikipedia)` section, so the model can answer from a whole encyclopedia without you
ingesting anything. There is **no Python dependency** — Mimir talks to a Kiwix server over stdlib
HTTP, exactly like it talks to Ollama.

1. **Get the binary + a ZIM.** Download `kiwix-tools` (a single static `kiwix-serve` binary; builds
   exist for Linux/macOS/Windows **and ARM/Raspberry Pi**) and any ZIM from
   [library.kiwix.org](https://library.kiwix.org) — pick your size: Wikipedia **nopic** is a great
   default; there are medical wikis, a top-50k slice, Wiktionary, and more.
2. **Serve it** (no install, no Mimir code on that box): `kiwix-serve --port 8080 wikipedia_en_nopic.zim`
3. **Point Mimir at it** — add a `[wiki]` block to `mimir.toml`:

```toml
[wiki]
url = "http://localhost:8080"   # the kiwix-serve base URL
book = "wikipedia_en_nopic"     # the ZIM's book name as kiwix-serve lists it (its homepage shows it)
max_articles = 2                # top hits injected per turn
max_chars = 800                 # chars of lead text per article
timeout_s = 2.0                 # hard cap so a slow/missing wiki never stalls a turn
```

It's fully optional and **fail-open**: omit the block to disable, and a missing or slow server just
yields no reference section — it never errors or stalls a turn. Trivial turns (greetings) skip the
lookup. (Want to point at the raw `.zim` file instead of running a server? That needs the GPL
`libzim` library + Python 3.14's stdlib zstd or a zstd dep, so we keep the dependency-free HTTP path
as the default.)

## 5d. Distributed inference — pool several machines (the fleet)

Don't have one powerful machine? Pool several modest ones. Every computer that runs `ollama serve`
becomes a worker — **with zero setup on it** (no Mimir code, no agent, just Ollama). Mimir
discovers them, catalogues their models, and routes each request to a node that actually has the
model.

```toml
[backend]
lan_backend = true             # scan the LAN for Ollama nodes (localhost is ALWAYS included)
# subnet = "192.168.1.0/24"    # omit to auto-detect your local /24
# nodes = ["192.168.1.50:11434"]  # optional explicit nodes (always included)
refresh_interval_s = 60        # active health/inventory refresh (reachable? which models?)
idle_probe_interval_s = 1800   # LIVE SPEED: node latency is learned passively from real calls (no
                               # wasted synthetic calls); this rare idle heartbeat (default 30 min,
                               # 0 = off) only tops up nodes that have gone quiet so their speed
                               # estimate doesn't go stale. Routing sends each call to the healthiest,
                               # fastest node (expected wait = latency × load), and writes live speed
                               # back to the placement matrix.
latency_alpha = 0.3            # EWMA weight on the newest latency sample (higher tracks load faster)
max_model_size_b = 30          # only YOU know your hardware: benchmark/route models up to this many
                               # billion params (raise on a big GPU, lower on a Pi). Bigger = skipped.
min_model_size_b = 0           # ...and a FLOOR; 0 = off. On capable hardware, stops a tiny model
                               # that scores 'high enough' and wins on latency from out-competing a
                               # bigger, genuinely-better one the imperfect test can't separate.
max_latency_s = 0              # latency target in seconds; 0 = off. When set, it also becomes the
                               # benchmark's per-model timeout: a model is loaded into VRAM first
                               # (untimed warmup), then timed WARM — so a model whose *warm* trivial
                               # call exceeds this is skipped. When off, a 30s default applies so one
                               # slow model can't stall the run. (Routing-exclusion: next.)
benchmark_num_ctx = 24576      # OPERATIONAL context window for ALL benchmark calls — qualify at the
                               # size you deploy at. Ollama defaults to a tiny 2048, too small for the
                               # layered epistemic prompts (it would silently truncate the high-tier
                               # fact and break the tier-deference test). Held consistent across every
                               # model so qualification is fair AND so it matches production — a
                               # different window in deployment rebuilds the warm KV cache on the first
                               # real turn, making the benchmark's latencies lies. The long-context
                               # needle probe SIZES ITS HAYSTACK to ~60% of this, so it tests the window
                               # you'll actually run, catching models fine at 8k but "lost in the
                               # middle" of a big context. 24576 (24k) is a proven window for a
                               # RAG + compression system — a fraction of models' 128k–256k max, which
                               # you neither need nor want (KV-cache cost + attention degradation).
                               # Continuity comes from curated RAG memory + compression, not a giant
                               # raw window. Lower it only if your edge nodes can't hold 24k (then the
                               # placement matrix records which nodes can serve which models at it).
```

On boot Mimir scans the subnet for `:11434`, inventories each node's models (family, weight,
quantization), and from then on a call for `qwen2.5:14b` goes to whichever node has it. Health is
checked actively; a node that drops off is routed around. Run **a variety of model families** across
your machines — it makes the inner council genuinely diverse and gives the fleet more to route to.

**Edge / Raspberry Pi recipe:** the brain is just Python + SQLite, so run it on a tiny box. Set
`lan_backend = true`, use bootstrap or an endpoint embedder, and point nothing at localhost — the
Pi holds the *memory* while your gaming PC / Mac / server does the *inference* over the LAN.

```bash
# on the Pi (Python 3.11+; the Pi runs NO Ollama itself)
git clone <repo> mimir-0 && cd mimir-0
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # core only — zero deps
cp mimir.toml.example mimir.toml        # set [backend] lan_backend = true, roles, embeddings=bootstrap
python -m mimir.selftest                # sanity (mock, offline)
python -m mimir.server --config mimir.toml --host 0.0.0.0   # browse from another machine
```

The Pi discovers your LAN's Ollama nodes and routes every model call to them. After a
`brain.benchmark_fleet()`, `apply_recommendations()` re-points each role at the best model the
fleet can serve.

Inspect or refresh the catalogue from `brain.scan_fleet()` / `brain.fleet_report()`, or the web
UI's **Fleet** tab.

**Benchmark your models** (`brain.benchmark_fleet()` or the Fleet tab's *Benchmark* button) to fill
the catalogue's `quality` and `return_time`. Each model runs a capability battery — *talk*
(instruction following), *tools* (emit a valid tool call), *code* (write parseable code), *reasoning*
(solve a problem with one checkable answer), *discipline* (don't leak the internal `[tier=…]` tags),
and *epistemics* (use Mimir's tiered/provenance context — defer to the high tier under noise, recall
context-only facts, handle long context) — plus a *coherence* pass scored by a panel of your other
models, guarded by a canary (the judges must rank a known-good answer above a garbled one, or
coherence is skipped). It's call-heavy, so run it on-demand. Recommended models are
instruction-following families — `gemma`, `qwen`, `llama`, `mistral`, `phi`, `command-r`, `deepseek`;
running a **variety** of families is ideal.

**Or run the qualifying tournament** (Fleet tab → *🏆 Run qualifying tournament*) — a staged,
human-veto narrowing instead of one big pass. **Round 0 · Qualifying** scores the whole fleet fast
and cheap; you untick who shouldn't advance; **Round 1 · Gauntlet** puts the survivors through the
full framework qualification; **Round 2 · Finals** points each role at its champion *among your
finalists* (your veto beats the global best). The board takes over the chat pane and resumes if you
switch tabs. The size/latency scope fields (`min_model_size_b`, `max_model_size_b`, `max_latency_s`)
apply to the whole run. This is the recommended path; *Find / Benchmark / Apply* remain as the manual,
one-step-at-a-time equivalents.

**Assign roles manually** (Fleet tab → *Role assignment*, or `brain.set_role(role, model)`). Each role
has a dropdown: leave it on **auto** (the system routes to the best-qualified model and re-picks on a
rescan) or choose a specific model to **pin** it — a pin is honoured exactly and never substituted or
re-assigned. The Fleet tab now also holds the full **model pool** (enable/disable each model, see which
roles it serves or is barred from) — Fleet and Models were merged into one view since they share so
much.

## 6. Configuration reference

```toml
[storage]
path = "data/mimir.db"          # required; SQLite file (created on first run)

[identity]
text = "You are Mimir..."       # optional; the always-on self-model / persona
primary_user = "alex"           # optional; this user's statements earn the top evidence tier.
                                # Omit for single-user mode (whoever speaks is treated as primary).
trusted_users = ["sam"]         # optional; also believed (trusted tier). Any OTHER named human
                                # (unknown API caller, guest) is attributed but baked at conversation
                                # tier, not as fact — the server-side trust policy.
peer_agents = ["mimir-home"]    # optional; identities known to be peer AIs. Their input bakes at
                                # stated_by_peer (below human conversation), marked AI-sourced — even
                                # if named above. Callers can also self-declare per turn with the
                                # API field speaker_kind="ai_peer" (default "human").

[embeddings]
mode = "bootstrap"              # "bootstrap" | "endpoint" | "degraded"
dim = 256                       # bootstrap vector size (ignored in endpoint mode)

[context]
budget_tokens = 4096            # per-turn prompt budget for assembly + accounting

[self_model]
refresh_every = 5               # turns between self-model re-synthesis; 0 disables (seed only)

[working_memory]                # rolling compression: fold oldest exchanges, keep recent ones raw
fold_threshold = 10             # fold once this many raw exchanges accumulate; 0 disables compression
keep_recent = 4                 # raw exchanges kept verbatim after a fold (the rest become summary)
# refresh_every = 4             # deprecated turn-cadence trigger (superseded by fold_threshold)

[entity_graph]
hops = 2                        # how far to traverse from a turn's entities (1–2); 0 disables
max_facts = 8                   # max connected facts injected per turn

[sleep]                         # the sleep cycle: consolidation + narratives in a nightly window
enabled = true                  # the wall-clock scheduler (recommended on)
window_start = "02:00"          # local HH:MM the maintenance window opens
window_end = "06:00"            # local HH:MM it closes (may cross midnight, e.g. "23:00"→"06:00")
check_interval_s = 900          # how often the daemon checks the clock (15 min)
every = 0                       # legacy turn-cadence path (superseded by the window); 0 = off

[output_rag]                    # bidirectional RAG: ground the next turn in memory about the LAST reply
enabled = true                  # after a reply, retrieve memory relevant to the model's own words
top_k = 3                       # how many related memories to surface into the next turn

[deliberation]                  # during sleep, argue the system's own conflicts via the inner council
enabled = true                  # also toggle live in the Sleep tab; "Deliberate now" triggers it manually
limit = 3                       # max conflicts argued per cycle (each is several model calls)

[inner_life]                    # the live inner life: think on its own during idle time (DESIGN §5a)
enabled = false                 # OFF by default — it uses spare compute; toggle live in the Sleep tab
cadence_s = 300                 # at most one thought this often (~5 min); set lower if you have GPU to spare
idle_floor_s = 30               # stay quiet at least this long after a turn before musing
check_interval_s = 20           # how often the daemon wakes to check (it self-gates on the cadence)

[server]                        # reference web server / integration API — see docs/API.md
# api_token = "..."             # require Authorization: Bearer <token> on /api/* (env var wins)
# api_token_env = "MIMIR_API_TOKEN"  # which env var holds the token; give co-located instances their own
# cors_origins = ["*"]          # browser origins allowed to call the API; empty = same-origin only
# secure_ui = false             # the LOCAL UI is token-exempt by default (first run isn't blocked);
                                # set true to require the token for the local UI too

[self_knowledge]                # bake a doc describing the system into memory each nightly cycle
doc = "README.md"               # path (relative to the working dir); empty/unset disables it
                                # → recallable DOCUMENT-tier memory; "what are you?" answers from it

[diagnostics]                   # self-observability: the system sees its own recent errors (§10)
surface_errors = true           # inject a "recent errors" block into the turn's context
error_context_window_s = 1800   # an error this recent (seconds) shows in context; older ones fade
error_context_max = 5           # max errors shown in that block

[procedural]
top_k = 3                       # max matching learned habits injected per turn
min_match = 0.3                 # minimum trigger relevance before a habit fires

[locale]                        # temporal grounding — clock/calendar sense injected each turn
# timezone = "America/Vancouver"  # omit to use the host's local clock (recommended — correct when
                                  # the machine runs in your timezone). Also settable live in the
                                  # web UI's Sleep tab. Two ways to set it explicitly:
                                  #   • a UTC offset — "UTC", "UTC-08:00", "GMT+5:30" — works with
                                  #     zero extra deps (no DST tracking; fine for a sleep window);
                                  #   • an IANA name — "America/Vancouver" — needs the OS tz database
                                  #     or `pip install 'mimir-0[timezone]'`. Without it, Mimir uses
                                  #     host-local and says so (it never silently ignores the setting).
hemisphere = "north"            # "north" | "south" — which way the seasons run

[provider]
type = "ollama"                 # "ollama" | "mock"
host = "http://localhost:11434" # ollama only

# One table per cognitive role. `model` is a model name, or "auto" (or omit it) to let Mimir
# pick from the fleet — measured-best if benchmarked, else an approved-family model, re-chosen on
# each scan. A pin always wins; disable models you distrust from the web UI (or
# brain.set_model_enabled(...)) and `auto` skips them. Everything else is passed to the provider as
# tuned params (temperature, num_ctx, max_tokens → Ollama's num_predict, ...). `think` defaults
# OFF (thinking mode slows generation and rarely helps); set `think = true` on a role only where
# it earns its cost.
[roles.chat]
model = "llama3.1:8b"   # or: model = "auto"
temperature = 0.7
num_ctx = 8192

[roles.bake]
model = "llama3.1:8b"
temperature = 0.0               # faithful extraction; no creativity

[roles.reasoning]               # the sentinel / deliberation role
model = "llama3.1:8b"
temperature = 0.3

[roles.embed]                   # REQUIRED when embeddings.mode = "endpoint" — install ≥1 embed model
model = "auto"                   # discovers an installed embedding model + remembers it (stable
                                 # vector space); or pin an exact name/tag. Same model on every node.
```

Misconfiguration **fails loud** with an instruction — a missing required role, an `endpoint` embed
mode with no `[roles.embed]`, or an unknown provider type raises a clear error at boot. Mimir never
silently falls back to a different store or a different mode.

---

## 7. Verify your install

```bash
python -m pytest -q          # the full spec, incl. the §6 acceptance loop (mock provider)
python -m mimir.selftest     # the runtime self-test + canary
python -m ruff check .       # lint
python -m mypy src/mimir     # types (strict)
```

All four should pass. The acceptance loop and self-test use the mock provider, so they need no
model server.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not reach Ollama at ...` | Ollama isn't running, or `host` is wrong. Start Ollama; check `ollama list`. |
| Recall feels weak / misses synonyms | You're probably on **bootstrap** embeddings (lexical only). Switch `[embeddings] mode = "endpoint"` and configure `[roles.embed]`. The startup log prints the active mode. |
| `config is missing required role(s)` | Add `[roles.chat]`, `[roles.bake]`, `[roles.reasoning]` (each with a `model`). |
| `embeddings.mode = 'endpoint' requires a [roles.embed] table` | Add `[roles.embed]`, or switch to `mode = "bootstrap"`. |
| Slow first response per model | Ollama is loading the model into VRAM. Keep `num_ctx` consistent across roles sharing a model to avoid reloads. |
| `found a 'memories' table but no schema_version marker` | You pointed Mimir at a non-Mimir or corrupt DB. Use a fresh `[storage] path`. |
| Mimir greets with the wrong name, or inverts who serves whom | A too-small `reasoning` model hallucinated its self-model. Use **≥12B** for `chat`/`reasoning` (§5.1) and let the self-model re-synthesize (it refreshes on the `[self_model] refresh_every` cadence). Your `[identity]` anchors are not the problem. |
| Internal `[tier=…; source=…]` tags appear in replies | A small model echoing the prompt's tag style. They're stripped automatically — if you still see them, you're on **old code**; restart the server. A larger `chat` model stops producing them at the source. |
