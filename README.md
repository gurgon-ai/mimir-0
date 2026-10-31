# Mimir 0

[![ci](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml/badge.svg)](https://github.com/gurgon-ai/mimir-0/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A local-first cognition core: a memory that behaves like a mind's, not a database's, and the
machinery around a language model that a bare model lacks.**

Mimir 0 is a pure-Python library (standard library and SQLite only; no runtime dependencies) that
gives a local language model typed, provenance-tracked, evidence-tiered memory; a prompt assembled
with an explicit epistemic structure; a post-reply "burst" of background cognition; a nightly sleep
cycle; an inner life that thinks between conversations; and a ledger that notices when any part of
it stops working. You tell it something; later it recalls the fact, says who told it and how far it
trusts them, and says plainly when it is reasoning from thin evidence instead of guessing.

It was extracted, public-clean, from a larger private home AI that has run a private site for a
year. The lessons that system paid for are in [`research/`](research/), and most of them are now
mechanisms in this core (see *What it learned the hard way* below).

**Status: 0.2.0, pre-alpha.** Everything in [`DESIGN.md`](DESIGN.md) is implemented and runs end to
end on a single machine and on a multi-node LAN. The API, schema and UI can still change between
releases, and it is not hardened for exposure beyond localhost or a trusted LAN. The
[`CHANGELOG`](CHANGELOG.md) is honest about what each release changed and what it means for an upgrade.

## Research

[`research/`](research/) holds twenty-five measured field notes from a year of running a local-model
mind, a preregistered tool-choice experiment with its data (narrative framing did not move an agent's
tool choice; system-prompt autonomy moved it from about 1 % to 60 %), and the system's own
self-description checked line by line against its records. See [`research/README.md`](research/README.md).

## Try it in ten seconds

```bash
pip install -e ".[dev]"
python -m mimir.selftest        # the whole loop on a deterministic mock provider: no model, no network
python examples/quickstart.py   # bake a fact, recall it attributed
```

For a real conversation with a local model, install [Ollama](https://ollama.com), pull a model, and
follow [`docs/SETUP.md`](docs/SETUP.md). The reference web UI (chat, the identity interview, documents,
the memory graph, the sleep and model tabs) is one command:

```bash
python -m mimir.server --config mimir.toml   # → http://127.0.0.1:8765
```

Mimir is model-agnostic. It benchmarks whatever models you install and picks the best per role, so the
right recommendation is "measure your own". On the hardware this was built on, `gemma4:e2b` was the
standout small model and `gemma4:26b` / `qwen3.5:27b` the best large ones; treat that as a first pull,
not a requirement.

## What makes it different

- **Typed knowledge layers.** Stated facts, learned conclusions, document chunks, cited library claims
  and entity relationships live in separate stores with separate retrieval, not one flat index.
- **Evidence tiers and provenance.** Every fact knows who said it and how much that source is
  believed, and the *server* decides belief, never the caller: the primary user at the top, trusted
  users next, any other human at conversation tier, a peer AI below that and marked as AI-sourced. An
  exposed endpoint cannot launder a claim into trusted memory.
- **Confidence and salience, decoupled.** Truth and relevance are separate axes. A fact does not
  become false because it has not been used lately; it becomes less salient.
- **An uncertainty gate.** When the system is reasoning from thin evidence it says so and asks,
  instead of answering confidently.
- **State beside narrative.** Memory accumulates in mixed tense ("planning X", "X is done"). A
  separate, dated milestone ledger says what is true *now*, and during sleep it demotes the memories
  it supersedes.
- **A guard at every store boundary.** Nothing the model failed to produce (an empty body, a
  placeholder, an error string) is ever stored, folded on or filed as an argument.
- **It can feel a part of itself stop.** Every tool, background task, sleep phase and idle loop is
  registered in a capability ledger and touched when it runs. Anything that never fires, goes stale,
  or fires and always fails becomes a finding in the prompt and the nightly digest.

## What is inside

Every turn assembles an epistemic prompt: self-model → current state (timeline) → identity → the
current moment → attributed knowledge (memory, documents, library, graph; each fact tagged with its
age) → learned procedures → recent history → working memory → temporal awareness → background notes →
sentinel note → uncertainty gate, routed through two disciplined gateways (model and storage).

Around it:

- **The burst worker.** After each reply, the idle window runs a prioritised, interruptible pool of
  post-response work: the sentinel reflection, the self-model, working memory, output-triggered RAG,
  and a follow-up that puts any tool call that failed back in front of the model as an unkept promise.
- **Sleep.** A wall-clock nightly window for consolidation (dedup, decay, archival, contradiction
  resolution), self-knowledge (it bakes its own README), document ingest, narratives (daily → weekly →
  monthly, lossy by design), self-directed council deliberation, and the health digest.
- **Inner life.** Between conversations, on a cadence you set, it reflects on a recent exchange, a
  memory, a tension in what it knows or an error it hit, and keeps the thought as a faint note that
  resurfaces only if it later proves relevant. Off by default; it spends idle compute.
- **The inner council.** Adversarial deliberation across whatever models you have installed, personas
  fanned one per node, every debate kept in a browsable forum. A voice that fails is not an argument
  and is never synthesised or filed as one.
- **Working memory rebuilt from sources.** Recent turns stay verbatim; older ones are folded into a
  summary that is rebuilt from the archived exchanges every time, never from its own previous text.
- **Documents and a library.** Drop files in a folder or upload them; text and Markdown in core, PDF
  and DOCX behind an extra, images via a vision role. A three-tier library (documents → cited claims →
  composites) with a citation guard that flags a citation to a source the system does not hold.
- **An offline encyclopedia.** Point it at a local Kiwix server over any ZIM and it gets a live,
  attributed reference layer over stdlib HTTP.
- **A distributed, qualified model fleet.** It discovers every `ollama serve` on your LAN, routes
  each request to a node that has the model, and qualifies models on a measured battery (talk, tools,
  code, reasoning, discipline, epistemics, vision) at your operational context size, with a staged,
  human-veto tournament, a per-node placement matrix and a diversity-first council roster.
- **Names at the door.** A heard or mistyped name is snapped to the household's known names before
  memory, graph or recall see it, from confirmed variants and a conservative phonetic match.
- **Notebook, sessions, graph, streaming.** A lossless self-curated notebook; durable, restorable
  conversation sessions; a visual memory graph; token-by-token streaming with tools on both paths.
- **Four extension ports.** Tools, context sources, backends and background reflexes, so you attach
  capabilities without forking. The notebook and the milestone ledger are the two reference
  connectors. See [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md).

## What it learned the hard way

Each of these came from an incident in the production system and is now a mechanism here; the
measurements are in [`research/field_notes_local_model_cognition.md`](research/field_notes_local_model_cognition.md).

- One `num_ctx` on every call to a model, including council and background calls; a different value
  forces a full reload.
- An empty body with `done_reason == "length"` under thinking is not "no answer"; it is retried once
  with thinking off.
- Embeddings go to Ollama in batches of 128 and are shape-checked, because past about 256 inputs the
  endpoint returns nothing useful with no error.
- A rolling summary that reads its own last version cannot forget a corrected claim; it is rebuilt from
  sources.
- Intercepts count by position: a time question must open the turn, not merely contain a trigger.
- A tool that was refused or crashed is a promise still unkept, not a call that was "made".
- The system's own musings, verdicts and reflections are recallable but are not facts anyone stated;
  they are kept out of narratives' "facts", the self-model's knowledge counts and output grounding.
- Background work stays out of the way for a window after the last turn, not just while a turn is in
  flight; someone who pauses to think is still in the conversation.

## Integration API

Mimir ships no IO of its own (no voice, avatar, home automation) on purpose. Drive it from Python or
over HTTP and build any front end on top; register tools on the motor port when you want hands. Full
contract in [`docs/API.md`](docs/API.md).

```python
from mimir import Mimir
m = Mimir.from_config("mimir.toml")
print(m.turn("My garlic goes in around October.", user="alex").reply)
print(m.turn("When do I plant the garlic?", user="alex").reply)   # recalled, attributed
```

```bash
curl -s http://127.0.0.1:8765/api/turn -H "Content-Type: application/json" \
  -d '{"text": "hello", "user": "alex"}'
# → {"reply": "...", "actions": [...], "introspect": {sources, tiers, tokens}}
```

- `user` is the speaker's identity and `speaker_kind` (`human` / `ai_peer`) its kind; the server's
  trust policy decides how much each is believed.
- `POST /api/turn/stream` streams tokens over Server-Sent Events, runs the tool round, and carries
  `actions` in the final `done` event.
- `GET /api/health` is an instant, unauthenticated liveness check. Set `[server] api_token` to require
  a Bearer token on every other `/api/*` route.

Two Mimirs can talk to each other through a tiny relay; each remembers the other as a peer and, by
the trust policy, does not take the other's hallucinations as fact. A worked example is in the API doc.

## Runtime contract

Python 3.11+, SQLite, one chat endpoint, one embeddings endpoint. No GPU requirement, no cloud, no
peripherals. Provider-agnostic; pools several local inference nodes across a LAN when you have them.

## Security

Pre-alpha and not hardened; the reference server is for localhost or a trusted LAN, not a public
service. Disclosure policy and the security model are in [`SECURITY.md`](SECURITY.md).

## Contributing

The bar for core is high and the discipline is specific: zero runtime dependencies, the two gateways
are law, fail loud, keep core layers generic, and every store boundary guards what it stores. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE). Research write-ups under `research/` are CC BY 4.0 where stated.

---

*Mimir 0 is the general, reusable cognition core extracted and rebuilt from a larger private home-AI
system. None of the original's hardware integrations or personal data come with it; the memory and
reasoning architecture, and what running it taught, do.*
