# Mimir 0

[![ci](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml/badge.svg)](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Long-term memory and turn orchestration for a local language model, in pure Python with SQLite and
no runtime dependencies.

A bare local model forgets everything past its context window, can't tell something a user told it from
something it made up, and confidently answers questions it has no evidence for. Mimir-0 sits between your
application and the model and fixes those three things. Every fact it stores records who said it and how
far that source is trusted. Every prompt is assembled from typed stores under a token budget, with each
fact tagged by source and age. When the evidence is thin, the prompt says so, and the model is told to ask
rather than guess.

## What that looks like

Three speakers state the same kind of fact. The server, not the caller, decides how much each is believed:

```python
m.turn("The garlic goes in around October.", user="alex")                 # alex is the primary user
m.turn("The garlic goes in around March.", user="sam")                    # another human
m.turn("The garlic goes in around June.", user="relay", speaker_kind="ai_peer")
```

```
text                                    evidence_tier            provenance
The garlic goes in around October.      stated_by_primary_user   stated by alex
The garlic goes in around March.        conversation             stated by sam
The garlic goes in around June.         stated_by_peer           stated by peer AI relay
```

On a later turn, recalled facts reach the model with their provenance attached, so it can attribute
instead of flattening everyone into "you told me":

```
<RECALL>
- The garlic goes in around October. [tier=stated_by_primary_user; source=stated by alex; just now]
</RECALL>
```

That output is from a real run on the built-in mock provider. Try it with no model installed:

```bash
pip install -e ".[dev]"
python -m mimir.selftest        # the full loop, deterministic, offline
python examples/quickstart.py   # bake a fact, recall it, print what went into the prompt
```

## How a turn works

1. **Names are normalised.** A misheard or mistyped name ("Katelyn" for "Caitlin") is snapped to a known one
   before anything is stored or searched, from configured variants and a conservative phonetic match.
   Fixing it here is the only place that works: once two spellings are in memory, every lookup finds half.
2. **Retrieval runs per store.** Stated facts, conclusions the system reached itself, document chunks,
   cited library claims and entity relationships are separate stores with separate retrieval. Each hit
   carries its evidence tier, which acts as a gentle multiplier (1.30 for the primary user down to 0.90
   for the system's own inferences): it breaks ties between equally relevant facts and never overrides
   relevance.
3. **The prompt is assembled under a budget.** Sections (self-model, identity, time, knowledge, graph,
   working memory, notes) are rendered with a priority tier. When the budget is tight, low-priority
   sections are truncated first, and truncating a high-priority one is reported as a warning.
   `TurnResult.context.introspect()` shows exactly what was requested, what was admitted and what was cut.
4. **The uncertainty gate.** If the knowledge that made it in is thin for the question asked, the prompt
   says so and tells the model to ask instead of answering.
5. **The model is called through one gateway.** The gateway keeps `num_ctx` identical for every call to a
   given model, sets `keep_alive` per role, retries an empty "length" reply once with thinking off, and
   routes to whichever node on the LAN has the model loaded.
6. **Tools run on both paths.** The model can emit tool calls in streaming and non-streaming replies; the
   results go back for one more round, and the reply carries an `actions` list of what ran and how it went.
7. **Facts are baked.** New statements are extracted and stored with the tier the server's trust policy
   assigns. A guard at every store boundary refuses empty output, placeholders and error strings, so a
   failed generation never becomes a memory.
8. **Background work runs after the reply.** A burst worker runs post-turn tasks in priority order: a short
   reflection for the next turn, the self-model, working-memory folding, and a check that any tool call
   that failed is put back in front of the model. It yields the moment the next turn arrives, and stays
   out of the way for a configurable window after a turn, not only while one is running.

Between conversations, a nightly sleep cycle deduplicates, decays salience, archives, resolves
contradictions, ingests documents and writes daily, weekly and monthly summaries. An optional idle loop
reflects on recent exchanges and errors and stores the results at low salience.

## Things it handles that you would otherwise find out the hard way

Each of these is a failure that is silent in a naive implementation. The measurements are in
[`research/field_notes_local_model_cognition.md`](research/field_notes_local_model_cognition.md).

| Problem | What the code does |
|---|---|
| Ollama reloads the model whenever `num_ctx` changes, discarding the KV cache | One `num_ctx` per model on every call, including background and council calls |
| With thinking on, a model can spend its whole budget reasoning and return an empty body with `done_reason: "length"` | Retries once with thinking off instead of treating it as "no answer" |
| `/api/embed` returns nothing useful past ~256 inputs, with no error | Batches of 128, and every response is shape-checked |
| A rolling summary that reads its own previous version can never drop a claim that was corrected | Working memory archives what it folds and rebuilds the summary from the archived turns every time |
| An empty or placeholder generation gets stored, then the source it "summarised" is deleted | `require_generation()` at every store boundary; a fold that gets nothing back keeps its sources |
| "What time is it" matches inside "I lost track of what time it was yesterday" | Intercepts trigger on position: the question has to open the turn |
| A tool call that was refused or crashed gets counted as done | The reply's `actions` carry status; failed calls are surfaced on the next turn |
| The system's own reflections get recalled later as if someone had stated them | System-generated provenance is recallable but excluded from summaries' facts and the self-model's counts |
| A feature stops running and nothing anywhere changes | A capability ledger records the last run of every tool, task, sleep phase and loop, and reports anything that never ran, went stale, or keeps failing |

## Using it

From Python:

```python
from mimir import Mimir
m = Mimir.from_config("mimir.toml")
r = m.turn("When do I plant the garlic?", user="alex")
print(r.reply, r.actions, r.context.introspect())
```

Over HTTP, with the reference server (`python -m mimir.server --config mimir.toml`, then
http://127.0.0.1:8765 for a web UI):

```bash
curl -s http://127.0.0.1:8765/api/turn -H "Content-Type: application/json" \
  -d '{"text": "hello", "user": "alex"}'
# {"reply": "...", "actions": [...], "introspect": {...}}
```

`POST /api/turn/stream` streams tokens over Server-Sent Events. Set `[server] api_token` to require a
Bearer token on every route except `/api/health`. Full contract: [`docs/API.md`](docs/API.md).

For a real model, install [Ollama](https://ollama.com), pull a chat model and `nomic-embed-text`, and
follow [`docs/SETUP.md`](docs/SETUP.md). Mimir benchmarks the models you have and picks one per role
(chat, bake, reasoning, vision), so the right model is whichever measures best on your machine.

## Extending it

Mimir has no IO of its own: no voice, no avatar, no home automation. You add capabilities through four
ports without forking:

- **Tools**: functions the model can call.
- **Context sources**: extra prompt sections, budgeted like the built-in ones.
- **Backends**: model providers and embedders.
- **Reflexes**: background tasks registered on the burst worker or the sleep cycle.

The notebook and the milestone ledger in core are written as ordinary plugins on these ports, to serve as
examples. See [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md).

## Also in the box

- **Documents and a cited library.** Drop text, Markdown, PDF or DOCX into a folder. Claims extracted from
  them keep their citations, and a guard flags a citation to a source the system doesn't hold.
- **An offline encyclopedia.** Point it at a local Kiwix server and it gets an attributed reference layer.
- **A multi-node model fleet.** It discovers every `ollama serve` on the LAN, routes each request to a node
  that has the model, and qualifies models on a measured battery at your real context size.
- **A council.** Hard questions can be argued by several models or personas before an answer is settled;
  a voice that fails to answer is dropped, never synthesised as an argument.
- **A milestone ledger.** Memory accumulates in mixed tense ("planning X", "X is done"). A separate dated
  ledger holds what is true now, and the sleep cycle demotes memories it supersedes.

## Status

0.2.0, pre-alpha. Everything in [`DESIGN.md`](DESIGN.md) is implemented and tested (523 tests, mypy
strict, CI on Python 3.11 to 3.13). The API and schema can still change between releases; the
[`CHANGELOG`](CHANGELOG.md) lists what each release changed and what an upgrade needs. The reference
server is for localhost or a trusted LAN, not the internet; see [`SECURITY.md`](SECURITY.md).

Requires Python 3.11+, SQLite, one chat endpoint and one embeddings endpoint. No GPU requirement and no
cloud dependency.

## Research

[`research/`](research/) has the measurements behind the table above, and a preregistered experiment on
what makes a model cross a boundary with its tools (not stories about confinement; one sentence of
autonomy in the system prompt took it from 1–2% to 60%).

## Contributing and license

Core stays dependency-free, every model call goes through the model gateway and every write through the
storage gateway, and failures are loud. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
[Apache-2.0](LICENSE).
