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
of flattening everyone's knowledge into "you told me."

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
| `council` | persona pool | **auto-discovered** (see below) |

Each role entry carries its tuned params (context window, temperature, output budget). A per-model
constraint worth knowing: the context-window size must stay consistent across callers of the same
warm model, or you trigger an expensive reload — so it belongs in config, not code.

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
   nightly role. A model that fails here never costs a judge call.
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
  single-endpoint and API adapters.
- **Storage gateway** — one dedicated writer thread, priority queue, coalescing + batching. Reads
  stay direct (SQLite WAL allows many readers, one writer). Eliminates write-lock contention.
- **Attention governor** — a generic "foreground beats background" scheduler driven by software
  signals (chat-in-flight / consolidation-running / model-reload-pending / idle).

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
- **v0.1 — document ingestion (lead feature):** `ingest(path)` → extract → chunk → embed → a
  `document`-tier typed layer with file/page provenance, retrieved through `build_context()` like
  any other source (a document fact is just a memory whose evidence tier is `document`). Plain
  text + markdown in core (zero deps); PDF/EPUB extractors ship as an optional `[documents]` extra,
  so the runtime contract holds. *LLM compilation of documents into integrated, contradiction-
  resolved knowledge is a later, optional layer on top — not the ingestion itself.*
- **v0.1+ — cognition layers:** working memory, self-model, procedural memory, entity graph,
  sleep/consolidation, inner council, the qualification battery.
- **Adapters (separate extras/packages):** a reference HTTP server (+ streaming), an optional
  voice adapter, a demo UI, and example plugins (home-automation, etc.) that show the extension
  pattern without being core dependencies.

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

Pre-alpha / design phase. The architecture is specified; the spine is being built against §6's
acceptance test. Not yet usable. Contributions to anything beyond the v0 spine are frozen until the
acceptance loop passes green — the goal is a rock-solid memory→recall→reflect core before breadth.
