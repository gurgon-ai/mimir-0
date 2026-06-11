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

## 6. Configuration reference

```toml
[storage]
path = "data/mimir.db"          # required; SQLite file (created on first run)

[identity]
text = "You are Mimir..."       # optional; the always-on self-model / persona
primary_user = "alex"           # optional; this user's statements earn the top evidence tier.
                                # Omit for single-user mode (whoever speaks is treated as primary).

[embeddings]
mode = "bootstrap"              # "bootstrap" | "endpoint" | "degraded"
dim = 256                       # bootstrap vector size (ignored in endpoint mode)

[context]
budget_tokens = 4096            # per-turn prompt budget for assembly + accounting

[self_model]
refresh_every = 5               # turns between self-model re-synthesis; 0 disables (seed only)

[provider]
type = "ollama"                 # "ollama" | "mock"
host = "http://localhost:11434" # ollama only

# One table per cognitive role. `model` is required; everything else is passed to the provider
# as tuned params (temperature, num_ctx, max_tokens → Ollama's num_predict, ...).
[roles.chat]
model = "llama3.1:8b"
temperature = 0.7
num_ctx = 8192

[roles.bake]
model = "llama3.1:8b"
temperature = 0.0               # faithful extraction; no creativity

[roles.reasoning]               # the sentinel / deliberation role
model = "llama3.1:8b"
temperature = 0.3

[roles.embed]                   # required only when embeddings.mode = "endpoint"
model = "nomic-embed-text"
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
