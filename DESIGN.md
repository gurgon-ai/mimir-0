# Mimir 0 — Design

> Founding design document. The architecture is specified here; the code is built spine-first
> against it. Status: **design phase / pre-alpha.**

---

## 0. What this is

The open-source landscape is full of "agent frameworks" and "RAG libraries." Mimir 0 is neither.
It is a **cognition substrate**: a small library that treats knowledge the way a mind does, not
the way a database does.

> Most memory libraries store text in a vector blob and retrieve by similarity. Mimir 0 treats
> knowledge as **typed, evidence-tiered, provenance-tracked beliefs that decay, consolidate during
> "sleep," and get adversarially self-reviewed** — wired into the prompt with an explicit epistemic
> structure.

The defensible ideas are all in cognition:

1. **Disciplined epistemics in prompt assembly** — typed knowledge layers, evidence tiers,
   confidence/salience decoupling (truth ≠ relevance), provenance tagging, an uncertainty gate.
   *This is the differentiator.*
2. **An async second mind** — a reflective pass reviews each turn and leaves a note for the next.
3. **Sleep / consolidation** — dedup, contradiction resolution, hygiene. Memory that maintains itself.
4. **An inner council** — adversarial multi-perspective deliberation over open questions.
5. **Local-first distributed inference** — a pool of inference nodes, health-checked and priority-routed.
6. **Two disciplined gateways** — one chokepoint for all model traffic, one for all storage writes —
   plus an attention governor that makes background cognition yield to the user.

---

## 1. Library, not framework

You don't subclass-and-implement abstract methods. You import it, hand it a provider, a storage
path, and an identity config, and call `.turn()`. It grows through **extension points** — register
a tool, a context source, a provider — not through inheritance scaffolding.

```python
brain = Mimir(config="mimir.toml")          # provider, storage, identity, knowledge layers
reply = brain.turn("how's the project looking?", user="alex")
#   → assemble epistemic context → route → model → sentinel(async) → returns reply
#   → bake / working-memory side effects happen through the storage gateway
```

---

## 2. The runtime contract (the law)

Mimir 0 runs on **exactly this and nothing else**:

- **Python** (3.11+)
- **SQLite** (stdlib — zero external database)
- **one chat/completions endpoint** (behind the provider abstraction)
- **one embeddings endpoint** (behind the provider abstraction)

No GPU assumption. No cloud. No peripherals. No accounts. If it needs more than this to boot and
hold a grounded conversation, the contract is broken. Every design decision flows from this line.

---

## 3. The epistemic model (the core)

Knowledge is not a flat vector store — it is a set of **typed layers**, each with its own retrieval
discipline and its own slot in the prompt.

### 3a. Typed knowledge layers
| Layer | Holds | Retrieval |
|---|---|---|
| **Memory** | What happened / what someone said (facts, events) | hybrid keyword + embedding |
| **Documents** | Ingested files (notes, docs, books) — chunked, with file/page provenance | hybrid + `document` evidence-tier |
| **Understanding** | What the system learned or concluded (syntheses) | cosine |
| **Entity graph** | What's *connected* (subject–relation–object triples) | graph traversal (1–2 hop) |
| **Working memory** | Rolling cross-session salient context | recency + compression |
| **Self-model** | The system's authored identity | always-on |
| **Procedural** | Learned reasoning habits (trigger → procedure) | cosine + structural |

Each layer is **separate by design** — understanding never competes with memory for injection slots;
they live in different prompt sections with different framing. Additional layers (a compiled doc
library, etc.) register as further typed sources.

### 3b. Evidence tiers (truth provenance)
Every memory carries an `evidence_tier` assigned at write time by *how it was sourced* — e.g.
`stated_by_primary_user` > `stated_by_trusted` > `document` / `multi_source` > `conversation` >
`inferred`. The tier becomes a gentle retrieval multiplier (at equal relevance, better-sourced
facts win) and an explicit provenance tag in the prompt, so the model attributes correctly instead
of flattening everyone's knowledge into "you told me." That tag is an *internal* prompt convention:
it is deterministically stripped from the user-facing reply (see §10), so the scaffolding never
leaks into output.

### 3c. Confidence / salience decoupling (the foundational idea)
Two **separate** axes — conflating them is the bug this design exists to avoid:
- **confidence** = "is it TRUE?" — does *not* decay from disuse. Only low-tier, uncorroborated
  provisionals decay; authority-tier and corroborated facts never do.
- **salience** = "is it RELEVANT now?" — decays over time, bumps on access. Drives forgetting/archival.

**Access frequency measures relevance, not truth.** Archiving ≠ disbelieving — a resurfaced memory
is still trusted. The failure this prevents: a true fact buried below the injection floor because it
wasn't accessed lately, decaying further because it was never injected — a death spiral. Don't let
"haven't used it lately" masquerade as "probably false."

### 3d. Uncertainty gate (deterministic, zero model cost)
After context assembly, count how many layers produced substantive content. If a real question drew
from ≤1 source, inject an explicit honesty flag: *say what you don't know, name the gap, ask a
clarifying question.* Pure pipeline introspection — no model call. The mechanical antidote to
confident hallucination.

### 3e. The assembly contract — `build_context()`
This is the heart. Given a turn + user, it produces an ordered, budgeted prompt: self-model →
identity/persona → typed knowledge sections (each capped) → goals → working memory → sentinel note
(high-attention end slot) → uncertainty flag. It is the single point where "universal" must be kept
strictly separate from any deployment-specific context — the core ships the universal sections;
everything else is a *registered context source*.

The self-model section leads with the operator-established identity anchors **verbatim** — these are
authoritative. Any synthesized self-narrative is grounded in operational history and must **not**
restate or override those anchors: a synthesis that invents or changes the system's name is a
grounding failure, not a stylistic one, so the synthesizer is forbidden from stating the name,
operator, or location at all (those are the anchors' job).

**This is measured, not just asserted.** The epistemic-competence experiment
(`cognition/epistemics.py`) is the executable spec for §3b/§3d: it runs each model's facts through
the *real* `build_context()` (the **structured** arm) and as a flat blob of the same facts (the
**flat** arm), so `lift = structured − flat` is the framework's measured value per model. Three
probes — *tier deference* (defer to the higher-tier of two contradicting facts), *attribution*
(name the source, which lives only in provenance), *uncertainty* (hedge when evidence is thin, not
confabulate). Cross-model runs show a **positive lift for every model**: **attribution is a
universal win** (impossible without provenance), the **uncertainty gate most helps the weakest
models**, and **tier-deference is model-dependent** — some models exploit evidence tiers, some
ignore them, which is itself a qualification signal.

---

## 4. Model-agnostic by role

Mimir 0 has cognitive **roles**, not hardcoded models. Model choice lives in config (one entry per
role), and the brain reads it. Test which models can do a job; don't assert it.

| Role | Job | What it needs |
|---|---|---|
| `chat` | live conversation | instruction-following, fits context budget |
| `reasoning` | sentinel + deliberation | clean **structured output**, reasoning |
| `bake` | memory extraction | faithful extraction, no hallucination |
| `embed` | embeddings | an embedding model |
| `background` | off-the-record reasoning | reasoning competence — **not** discipline-gated |
| `council` | adversarial pool | reasoning + **diversity** (a spread of families) |

**The second lineup (`background`, `council`).** These staff cognition that never speaks *as* the
assistant — off-hot-path reasoning, inner deliberation — so they are deliberately **not**
discipline/epistemics-gated: a capable model that "leaks" the identity is fine here, and the big/slow
models a chat latency cap excludes are prime members. Both clear a reasoning-competence floor only.
`background` resolves to a single best; `council` resolves to a **diverse pool** (families before
depth — different families fail differently, so a council of five distinct families beats five of one;
see *Council = auto-discovery* below). The brain harness staffs itself by **querying the roster** —
`roster_for(role, n)` ("give me N models for role R") — honouring the same model/node vetoes as every
other pick, rather than a human reading a view. The role gate is one predicate (`_bar_reason`), so the
seated roster can never disagree with the eligibility the leaderboard renders.

Each role entry carries its tuned params (context window, temperature, output budget). A per-model
constraint worth knowing: the context-window size must stay consistent across callers of the same
warm model, or you trigger an expensive reload — so it belongs in config, not code.

**Automatic selection (`model = "auto"`).** A role's model may be pinned, or left to the system: a
`model` of `"auto"` (or omitted) makes the brain choose from the fleet — *as automatic as possible,
but configurable.* Resolution is a strict hierarchy, each level vetoing models the user has disabled
and anything the pool can't currently reach:

1. **explicit pin** — a named model always wins; the system never overrides the operator;
2. **measured-best** — the benchmarked, role-gated recommendation (quality + the discipline floor
   for identity roles), so it future-proofs itself as new models are benchmarked;
3. **approved-family heuristic** — before any benchmark, a curated-family model near the role's
   ideal size (approved models win the first round);
4. **any reachable model** — last resort, so `auto` always yields something runnable.

A user who distrusts a model **disables** it (a bias veto) and resolution skips it everywhere. The
default is **local-only** — the LAN fleet is opt-in (`[backend] lan_backend`), never polled unless
asked. This serves the spread of users — one powerful machine, an edge node borrowing LAN GPUs, or
no Ollama at all — from a sensible zero-config default, with every level overridable.

### Council = auto-discovery, scales to the machine
The council uses **no fixed model list**. On start it asks the provider pool what's actually
installed, filters to eligible models, and assigns personas across them. **1 eligible model** →
single-model council, persona diversity via distinct system prompts. **N models** → N genuinely
different minds. Diversity is emergent from the hardware, not a config chore.

### The qualification battery — three filters, cheapest first
"Could model X do the job as well as Y?" — don't guess; gate, judge, then watch.

1. **Deterministic gate** (mechanical, zero judge cost): does output parse as valid JSON against the
   role schema? Required fields, sane types? **Consistency** across K runs (5/5 vs 3/5 is a real
   score). **Latency** against a per-role ceiling — too slow for `chat` may still qualify for a
   nightly role. **Latency** is timed from one *real-length* generation (normalized to seconds per
   ~256-token turn), so a slow remote model can't masquerade as instant on a 3-token reply. A model
   that fails here never costs a judge call. The gate scores six capability dimensions — *talk*,
   *tools*, *code*, **reasoning**, **discipline**, and **epistemics**. *Reasoning* tests whether the
   model can actually **solve** a problem with one regex-checkable answer (multi-step arithmetic,
   letter-counting, a code-trace, an instruction transform), not merely follow a format — the
   dimension that keeps *quality* from saturating near 1.0 for any fluent model. *Discipline* tests
   whether the model honors prohibitions, above all **not reproducing the internal
   `[tier=...; source=...]` scaffolding it is shown** (the failure that forced the §10 output
   sanitizer); its probe replicates the production condition that triggers the leak (a tag-saturated
   recall block under the real soft instruction), sampled across runs since the leak is probabilistic.
   *Epistemics* tests whether the model actually **exploits** the tiered/provenance/gated context
   (§3) — defers to higher-tier facts, attributes to source, hedges on thin evidence — measured by
   the structured arm of the epistemic-competence experiment over a gauntlet: a **layered
   conflicting-tier** probe (defer to the high tier under noise), a **grounding** floor (recall a
   context-only nonce), and a **long-context needle**. The identity-bearing roles (`chat`,
   `reasoning`) gate on **both** *discipline* and *epistemics* (and *reasoning*): a model that leaks
   the scaffolding, ignores evidence tiers, OR can't solve a problem is never recommended to speak as
   the system — caught in qualification, not discovered in production.
2. **Coherence judgment** (judge + human anchor, only on survivors): the candidate answers a fixed
   **golden case** (prompt + fixed context + reference answer); a trusted model judges it against a
   rubric (faithful to context, cites the right memory, refuses to hallucinate). Sample outputs
   surface to a human for a thumbs up/down — the human is the calibration anchor.
3. **Continuous governance**: usage signal (correction rate, engagement) as a coherence proxy; the
   **golden set self-expands** — when a human corrects the running system, that correction becomes a
   new golden case. Re-qualify after a runtime/model update.

**Guardrails:** the judge can't catch an error it would also make → mitigated by the human anchor,
rubric+reference judging, and the council's adversarial structure. The qualifier ships a **canary
pair** (a known-good model that must score high, a garbled one that must score low); if the canary
inverts, the *qualifier* is broken → loud alarm, never a silent pass.

> **The full inference-engine spec** — discovery, recommended-model registry, first-run onboarding
> (wizard + headless), progressive/trusted-judge qualification, local-first routing, staleness, and
> the failure-mode guards — lives in [`docs/INFERENCE_ENGINE.md`](docs/INFERENCE_ENGINE.md). It
> extends this section and §5 into the model-agnostic engine the rest of Mimir bolts onto.

---

## 5. Architecture — the spine + the cognition

```
   turn(text, user) ──► ROUTER (assemble → route → reply)
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                   ▼
   build_context()   tier selection     side effects
  (epistemic assembly)     │            (bake, working mem)
        │                  │                   │
   ┌────┴─────┐    ┌────────┴────────┐   ┌──────┴───────┐
   │ STORAGE  │    │   MODEL         │   │  STORAGE     │
   │ GATEWAY  │    │   GATEWAY       │   │  GATEWAY     │
   │ (single  │    │ (provider POOL  │   │ (single      │
   │  writer) │    │  + priority +   │   │  writer)     │
   │  SQLite  │    │  retry)         │   └──────────────┘
   └──────────┘    └────────┬────────┘
        ▲                   │ provider adapters
        │         ┌─────────┴──────────────────┐
        │         │ LAN inference pool / API    │
        │         └────────────────────────────┘
        └──── ATTENTION GOVERNOR (background work yields to foreground)

   async cognition (off the hot path, gated by the governor):
     • SENTINEL — reflective review of last turn → note for next
     • INNER LIFE — idle background thinking → ideas
     • COUNCIL — adversarial deliberation over open questions
     • SLEEP — consolidate, hygiene, synthesize, self-model
```

**The three chokepoints:**
- **Model gateway** — every call goes through it: priority tiers, retry/backoff, transient-fail
  signaling (so background tasks defer instead of corrupting state on a busy backend). Behind it, a
  **provider pool**: multiple endpoints, health-checked, tier-routed, graceful degradation — plus
  single-endpoint and API adapters. Routing is **speed- and health-aware** (below).

**Speed-aware routing + ranked fallback (the live half of §4).** Qualification (§4) decides *which
models are acceptable for a role*; routing decides *which model on which node answers a given call*:
- **Live node speed, measured from real traffic.** Every real call is timed and folded into a
  per-`(node, model)` latency estimate (EWMA, in the same "seconds per ~256-token turn" unit the
  benchmark writes), so the pool learns each node's current speed **passively — no synthetic calls**.
  A rare **idle heartbeat** (default ~30 min, decoupled from the faster health refresh) tops up nodes
  that have gone quiet; real usage is the primary signal. Estimates **seed** from the catalogue's
  qualification snapshot (informed from turn one) and are **written back** so the placement view shows
  live, current speed — and a failed probe is recorded as *unmeasured*, never as fast.
- **Route to the healthiest, fastest node.** Within the healthy tier a call goes to the node with the
  lowest **expected wait** (`latency × current load`); the existing health gating (reachable /
  saturated / disabled / model-aware) is unchanged. With nothing measured yet this reduces to
  least-loaded — the prior behaviour.
- **Ranked fallback per role.** A role resolves not to one model but to an **ordered chain of
  acceptable models** (its qualified ranking, best first). Routing walks the chain: each model routes
  to its fastest healthy node; if every node for it is down, routing falls to the next acceptable
  model. So a **heterogeneous fleet** (Gemma only on node A, Qwen only on node B) still serves a role
  — Gemma@A, then Qwen@B. A **pinned** model is honoured exactly: a pin is never substituted.
- **Storage gateway** — one dedicated writer thread, priority queue, coalescing + batching. Reads
  stay direct (SQLite WAL allows many readers, one writer). Eliminates write-lock contention.
- **Attention governor** — a generic "foreground beats background" scheduler driven by software
  signals (chat-in-flight / consolidation-running / model-reload-pending / idle).

### 5a. The idle window — burst cognition + bidirectional RAG  **[partial → proposed]**

A finding from months of running this on small, distributed compute (the project's origin): **after
the model answers, the GPU goes idle while the user reads, thinks, and composes a reply.** In text
that's a few seconds; **on voice it's 30 seconds or more, routinely**. That window is not dead time
— it is *the* reclaimable resource that lets a small or distributed rig act like a much bigger brain.
Reference design (proven in the parent system, to be rebuilt public-clean):

**The burst worker — the idle window as a first-class, scheduled resource pool.** After each
response the engine fires a *burst window* and drains a queue of background tasks, with these
mechanics:

- **Two task classes.** *User-driven* tasks (the user asked for something — finishing a tool action,
  a document, research) run **continuously, no cap**, only checking "is a new query in flight?"
  between calls so latency stays low if the user interjects. *Autonomous* tasks (nobody asked —
  memory creation, error correction, prefetch, consolidation) are **slot-capped** per window.
- **Floating priority / pent-up demand.** An autonomous task's effective priority *accrues urgency
  the longer it hasn't run* (`effective = base − starved_seconds × rate`), so starved work floats to
  the top naturally instead of needing a hand-tuned schedule. Some tasks are fixed-priority (run
  every turn); some accrue slowly (run when they've waited long enough).
- **Interruptible — foreground always wins.** Between *every* burst call the worker checks for an
  incoming turn and **yields immediately, re-queuing the rest** (this is the attention governor +
  the pool's BACKGROUND/IDLE priority tier in action). Tasks are idempotent/resumable; the slot cap
  is the proactivity budget.
- **Idle takeover.** After a long quiet (the user is away or deep in thought), the cap lifts and the
  worker goes continuous — use all the cycles.
- **Surfaces feed forward.** A burst task may emit a *surface* — a short result injected into the
  *next* reply ("[background note: …]"), so off-path work re-enters the conversation.

**The multiplier insight:** inference too *slow* for the live path is **free in the idle window.**
A reply must stream now, so it can't afford deep retrieval, a thinking-mode pass, or multi-step
verification — but the burst worker can, because the user is already reading. This is how reclaimed
idle cycles buy big-GPU-quality grounding on small hardware.

**Bidirectional RAG — the model's output triggers its own memory.** Retrieval runs on the model's
**output**, not only the user's input: *if the system says something, that statement should be able
to trigger its own memory.* The highest-value jobs — verify the reply's claims against stored truth
(**error correction**), surface contradictions (feeding the uncertainty gate and sleep's
contradiction resolution), bake durable facts the model just committed to, bump salience of
re-touched memories, and **prefetch context for the likely next turn** — all run here.

**Latency doctrine (hard-won):** output-side RAG runs **after the model has spoken, in the burst
window — never as an inline two-pass.** A draft → retrieve → finalize pipeline *was tried and proved
too slow for chat.* Answering fast and grounding *after the fact* keeps latency flat. A **fast/slow
switch** governs depth: a text UI (read time) can afford more post-response work than a voice UI
(which needs it fastest) — so it is a config policy, not a fixed cost.

**Current state.** Mimir-0 has the bones: the attention governor, the pool's priority tiers + busy
deferral, and a *fixed* slice of post-response background cognition (sentinel, self-model, working
memory) **[partial]**. The general burst-window scheduler (two classes, pent-up priority, surfaces,
idle takeover) and output-side bidirectional RAG are **[proposed]** — and compose directly with the
inference engine, which is *built* to be distributed-and-idle-aware rather than single-shot.

> A concrete **build sketch** mapping this onto Mimir-0's parts (the scheduler, the data model, the
> bidirectional-RAG tasks, the phased plan B1–B4) is parked in
> [`docs/BURST_WORKER.md`](docs/BURST_WORKER.md) — not scheduled; do the inference-engine phases first.

---

## 6. v0 scope + acceptance test

v0 is **done** when this loop works and nothing more is needed to demo it:

> Boot empty (no seeded data). Have a short conversation. The system **bakes a memory** from it via
> the storage gateway. In a *later* turn, ask a follow-up whose answer depends on that memory; the
> reply **cites it**, retrieved through `build_context()` with correct evidence-tier and provenance,
> and the **sentinel** fires async and leaves a usable note for the next turn.

That exercises the entire spine: router → context → retrieval → model (gateway/pool) → reply →
sentinel(async) → bake(storage gateway) → recall. Inner council, sleep/consolidation, self-model,
procedural memory, and the entity graph are **all v0.1+**, layered only after the spine is alive.
The discipline of *not* building them first is what ships v0.

---

## 7. Build order — vertical slice first, then harden

Separate the **seam** from the **internals**:

- **Seam = law from turn 1.** Every write routes through the storage gateway; every model/embed call
  through the model gateway. The *interface* exists immediately and is never bypassed.
- **Internals = hardened after the loop is green.** Behind the seams, start minimal (a simple
  serialized writer, a single-endpoint call). Get the §6 skeleton breathing fast, *then* upgrade the
  internals (priority queue, retry/backoff, coalescing, provider pool).

Order: thin seams → vertical slice to the acceptance test → harden the chokepoints → layer the rest.
This delivers a demo early and avoids gold-plating the plumbing before cognition has said what it
needs.

---

## 8. Roadmap

- **v0 — the spine:** router, `build_context()`, retrieval + epistemics, memory bake/recall,
  the two gateways (seams first), sentinel. The acceptance loop above.
- **v0.1 — document ingestion (lead feature) — _txt/md landed_:** `ingest(path)` → extract →
  chunk → embed → a `document`-tier typed layer with file/section provenance, retrieved through
  `build_context()` like any other source (a document chunk is just a memory whose evidence tier
  is `document`, with a `source` column so re-ingest replaces rather than duplicates). Plain text
  + markdown ship in core (zero deps); PDF extraction ships behind the optional `[documents]`
  extra (`pypdf`), so the runtime contract holds. EPUB and *LLM compilation of documents into
  integrated, contradiction-resolved knowledge* remain later, optional layers — not the ingestion
  itself.
- **v0.1+ — cognition layers:** ~~working memory~~ _(landed: rolling cross-session salient
  context — a capped recency log of recent exchanges plus a periodically compressed summary,
  injected always-on just before the sentinel note)_, ~~self-model~~ _(landed: an evolving, generic
  self-model — the reasoning model authors a first-person identity grounded only in the store's
  own operational signals, refreshed off the hot path and injected first, always-on; bootstrapped
  from the first boot by a re-runnable **identity interview** that establishes eight universal
  anchors — name, operator, location, purpose, values, scope, boundaries, voice — interactively
  or from config)_, ~~procedural memory~~ _(landed: learned reasoning habits as trigger→procedure
  pairs, matched to a turn by cosine + structural overlap and injected as how-to guidance)_,
  ~~entity graph~~ _(landed: subject–relation–object triples
  extracted at bake time, deduped, retrieved by 1–2 hop traversal seeded on the query's entities
  and injected as a distinct connected-facts layer)_, ~~sleep/consolidation~~ _(landed: a
  deterministic maintenance pass — dedup, salience/confidence decay with the death-spiral
  guard, archival of low-salience provisionals, and conservative contradiction resolution over
  functional graph relations)_, ~~inner council~~ _(landed: adversarial deliberation — generic
  personas take positions in parallel, spread across auto-discovered models, and a synthesizer
  weighs them into a verdict stored as recallable understanding)_, ~~the seeding interview~~
  _(landed, Phase 1: a short get-to-know-you — what to call the assistant, who the operator is and
  what they do, their week, location, household, pets, interests — paired with the qualifying
  tournament. Captured model-free and persisted immediately as the operator's highest-provenance
  facts: `stated_by_primary_user` memories with `provenance="onboarding"`, one editable row per
  question living in one place, name/operator/location mirrored into the identity anchors. The
  LLM parse pass (one answer → several typed facts + triples, review-before-commit) is Phase 2)_,
  the qualification battery.
- **Adapters (separate extras/packages):** ~~a reference HTTP server~~ _(landed: a stdlib,
  zero-dependency reference web server + single-page UI — chat, the identity interview, document
  ingest; `python -m mimir.server`)_, streaming, an optional voice adapter, and example plugins
  (home-automation, etc.) that show the extension pattern without being core dependencies.

---

## 9. Extending Mimir 0

Three extension points, each with a tiny end-to-end example:
- **Tool** — give the model an action it can call.
- **Provider** — wrap any chat/embeddings backend behind the gateway.
- **Context source** — contribute a typed section to `build_context()`.

A 30-line working example earns more trust than an architecture diagram.

---

## 10. Failure modes & self-observation

The failure modes this design guards against hardest are not logical — they're **silent**.
Experience building large memory systems shows the dominant bug class is *quiet death*: a swallowed
exception, a truncated prompt section, a store that silently fell back, a route that drifted, a
background job that starved the foreground — each invisible for weeks because nothing failed
*loudly*. A silently-broken memory is worse than a crash: it manufactures false confidence.

So the core's first doctrine is **fail loud, self-check, stay observable.** The cognition core must
be the least-fragile, loudest-failing, self-testing part of the system.

**v0 mechanisms (cheap, built in from the start):**
- **No silent swallow** — no bare `except` in core without re-raise or an explicit, logged
  downgrade. A swallowed error is a banned pattern.
- **Schema versioning + migration runner** — a `schema_version`, a tiny migration runner (even with
  only v1), and a startup check that required tables/columns exist and match the code.
  Misconfiguration **fails loud with instructions** — it never silently falls back to an alternate
  store.
- **The acceptance test is also a runtime self-test** — §6's loop runs as an automated guard at
  startup and on a schedule (synthetic turn → must bake → must recall → sentinel must fire).
  "No writes / no recall / no sentinel over N turns" is a *fault*, not a quiet state. The self-test
  ships a canary so a broken self-test is itself loud.
- **Context accounting** — `build_context()` records per-section tokens requested vs admitted and
  whether truncation occurred; truncating a high-tier section is a warning, not silence. An
  introspection call exposes "what's in the prompt and how big," so "why did it forget X?" is
  debuggable without reading internals.
- **Budgeted section registry** — every registered context source declares a budget + priority; the
  core caps or disables a misbehaving source without starving core sections.
- **Internal scaffolding never leaks** — the provenance tags and epistemic flags that structure the
  prompt are stripped from the model's reply (streaming-safe), so a human never sees Mimir's internal
  annotations and the model can't re-learn the tag style by reading its own logged output. A small
  model that mimics the format is contained deterministically, not asked nicely.

**Principles (applied as each layer lands):**
- **Governor fail-safe** — if scheduling signals glitch, default to throttling background, never
  starving foreground. Background tasks are idempotent/resumable and never advance a bookmark on a
  skipped run.
- **Routing golden-set** — a small fixed battery of inputs that must always route to a given role;
  alert on drift. Thresholds live in versioned config, not inline constants.
- **Commitment tracking** — when the model says it will do something, a tool call is recorded or a
  structured pending-commitment is created; a promise is never silently lost.
- **Proactivity is an isolated plugin** — idle/proactive behavior runs as a plugin whose failure
  cannot touch the core turn→bake→recall loop, with a proactivity budget and an easy off-switch.

**Process** (`CONTRIBUTING`): each load-bearing claim in this document has a test asserting it
(executable spec); a change to core behavior updates this document in the same PR — prose drift is
a defect.

---

## 11. Status

Pre-alpha. **The v0 spine is alive and the §6 acceptance loop passes green** (boot → bake → recall
with provenance & evidence tier → async sentinel note), verified under an automated self-test and
against a live local model. Both gateways are hardened (priority queue, batching, coalescing,
retry/backoff, provider pool with health/failover). **v0.1 has begun**: document
ingestion (`ingest()` for text + markdown in core, PDF behind the `[documents]` extra) and an
evolving, generic **self-model** (identity authored from the store's own operational history,
refreshed off the hot path, injected always-on), working memory (rolling recency + compression),
an entity graph (subject–relation–object triples with 1–2 hop traversal), sleep/consolidation
(dedup, decay, archival, contradiction resolution), and an inner council (adversarial deliberation
across auto-discovered models), and procedural memory (learned trigger→procedure habits). The
model backend is now a **distributed fleet** (DESIGN §5): it auto-discovers Ollama nodes on the LAN
(zero setup on them — just `ollama serve`), catalogues their models, and routes each request to a
node that has the model, with active health checks and least-loaded selection — so the brain can
run on a tiny box and borrow GPUs over the network. The **qualification battery** (DESIGN §4) is
layered on top: a benchmark scores each model's speed and a capability "IQ test" (talk / tools /
code, deterministic) plus a coherence pass voted by a panel of other models, guarded by a canary
pair — filling the catalogue so model→role fitness is *tested, not asserted* — and the catalogue
drives **per-role recommendations** ("for chat, use X on node Y"). The whole DESIGN architecture
is now implemented end-to-end; it remains pre-alpha and unhardened, but the spine, every typed
knowledge layer, the async cognition, and the distributed/qualified model fleet are all live and
verified against a real multi-node LAN.
