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
`stated_by_peer` > `inferred`. The tier becomes a gentle retrieval multiplier (at equal relevance,
better-sourced
facts win) and an explicit provenance tag in the prompt, so the model attributes correctly instead
of flattening everyone's knowledge into "you told me." That tag is an *internal* prompt convention:
it is deterministically stripped from the user-facing reply (see §10), so the scaffolding never
leaks into output.

**Who gets which tier is a server-side trust policy, not the caller's to declare.** An integration
caller picks the *speaker identity* (the `user` field) and its *kind* (`speaker_kind`:
`human`/`ai_peer`); the config decides how much that speaker is believed: `primary_user` →
`stated_by_primary_user`, `trusted_users` → `stated_by_trusted`, and any *other* named human (an
unknown caller, a guest) is attributed but written at `conversation` tier — never as fact.

**A peer AI is its own ontological category, not just an untrusted human.** A human is reporting
observation; a peer AI is emitting generated text that may be confabulated — or may be an *echo* of
something this system itself said, so that two agents agreeing manufactures a false sense of
corroboration ("agreement is an illusion of redundancy"). So a peer's input — declared per-turn
(`speaker_kind="ai_peer"`) or by config (`peer_agents`) — is written at `stated_by_peer` (0.95, below
human conversation), attributed and marked AI-sourced. **Kind wins over identity**: an agent can't
reach a human tier by also being named primary/trusted. So an exposed endpoint can't self-assert
trust, and two systems can converse without each treating the other's mistakes as gospel. (Zero-config
single-user keeps the convenience: with no policy set, the lone *human* speaker is the primary.)

### 3c. Confidence / salience decoupling (the foundational idea)
Two **separate** axes — conflating them is the bug this design exists to avoid:
- **confidence** = "is it TRUE?" — does *not* decay from disuse. Only low-tier, uncorroborated
  provisionals decay; authority-tier and corroborated facts never do.
- **salience** = "is it RELEVANT now?" — decays over time, bumps on access. Drives forgetting/archival.
  It decays **faster for the decaying tiers** (conversation/inferred — peer chatter and self-generated
  rumination) than for authority/document facts, so low-value provisional content goes dormant in weeks
  while a primary-user fact lingers for months — this is what makes the store *distil* rather than hoard.

**Access frequency measures relevance, not truth.** Archiving ≠ disbelieving — a resurfaced memory
is still trusted, and the archive step preserves confidence; it only drops a memory out of *active
recall*. Only **decaying-tier** memories that have faded below the salience floor are archived;
authority-tier and document facts are never archived for disuse. The failure this prevents: a true
fact buried below the injection floor because it wasn't accessed lately, decaying further because it
was never injected — a death spiral. Don't let "haven't used it lately" masquerade as "probably false."

### 3d. Uncertainty gate (deterministic, zero model cost)
After context assembly, count how many layers produced substantive content — recalled memory facts,
connected graph edges, wiki passages, **and cited library claims** (every independent grounding
layer must be counted; omitting one falsely starves the gate and makes the model deflect on material
it actually has). If a real question drew from ≤1 source, inject an explicit honesty flag: *say what
you don't know, name the gap, ask a clarifying question.* Pure pipeline introspection — no model
call. The mechanical antidote to confident hallucination.

### 3e. The assembly contract — `build_context()`
This is the heart. Given a turn + user, it produces an ordered, budgeted prompt: self-model →
identity/persona → **the current moment (time/season)** → typed knowledge sections (each capped, and
each fact **tagged with its age**) → goals → working memory → sentinel note (high-attention end slot)
→ uncertainty flag. It is the single point where "universal" must be kept strictly separate from any
deployment-specific context — the core ships the universal sections; everything else is a *registered
context source*.

**Temporal grounding.** The system has an always-on clock/calendar sense (`cognition/temporal.py`): a
compact "It is Thursday, January 15 2026, 2:30 PM. Season: winter (spring in 64 days)." line is
injected each turn, and every recalled fact carries a relative-age tag (`… ; 3 days ago`) so the
model reasons about recency instead of guessing. Timezone + hemisphere are `[locale]` config (default:
host zone, northern seasons) — universal, no place baked into core. Explicit time/date/season
questions are answered by a **deterministic intercept** (zero model cost) before any model call.

It also has a **temporal-awareness baseline**: a durable interaction log (one timestamp per turn)
lets it notice when the gap since you were last around is unusual *for your own rhythm* — "you
haven't been around in 14h (typically every ~6h)" or "the longest gap I've recorded." Pure statistics
over the log (median/p90/longest gap), zero model cost, and silent within normal rhythm — awareness,
not nagging. The same generic baseline machinery extends to entity/topic staleness later.

**Session history + restore.** A durable conversation log (the `conversation` table — one row per
exchange, pruned to a rolling window) is the lasting full turn history, distinct from the capped
EXCHANGE recency buffer (which working-memory compression clears). It restores the chat on UI load,
survives a process restart, and is **replayed to the model as real `user`/`assistant` messages** so a
turn has genuine continuity rather than summary-only context — the root fix for a model treating each
turn as a fresh start. A single-stream approximation of the home AI's session system.

**Temporal narratives** (`cognition/narratives.py`) give it a sense of *what happened* over time: a
hierarchical journal — **daily → weekly → monthly**, each tier compressed from the one below and
**lossy by design** (details fade, patterns persist, like human memory). Generated off the hot path in
the consolidation pass from generic sources (the running summary + recent exchanges + the facts
learned that period — no domain feeds), retained per-tier (10 / 5 / 13), and injected as a
`[Recent history:]` section (coarsest first) so a turn weeks later still has the shape of what came
before without dragging the raw transcript.

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
2. **measured-best** — the benchmarked, role-gated recommendation, ranked by a transparent **points
   total**: quality for *that* role (dominant) + speed (a strong, universal term — a slow model is bad
   for chat *and* background work) + a faint size prior to break near-ties toward capacity. Not
   quality alone, so a coin-flip tie on a saturated battery can't crown the wrong model; it
   future-proofs itself as new models are benchmarked;
3. **approved-family heuristic** — before any benchmark, a curated-family model near the role's
   ideal size (approved models win the first round);
4. **any reachable model** — last resort, so `auto` always yields something runnable.

The **embed role is auto-discovered too, but specially**: embedding models define the *vector space*,
so the choice must be stable — `auto` discovers an installed embedding model, **remembers** it
(persisted), and prefers the remembered one across restarts; if it goes unreachable the system stays
pinned to it and degrades to keyword recall (loud) rather than silently switching to an incompatible
model and corrupting recall. (Chat-style resolution explicitly excludes embedding models, and vice
versa, via a shared `is_embedding_model` check.) You provide whatever chat LLM(s) you like plus at
least one embedding model; pull the same embedding model on every node so recall survives a node loss.

A user who distrusts a model **disables** it (a bias veto) and resolution skips it everywhere. The
default is **local-only** — the LAN fleet is opt-in (`[backend] lan_backend`), never polled unless
asked. This serves the spread of users — one powerful machine, an edge node borrowing LAN GPUs, or
no Ollama at all — from a sensible zero-config default, with every level overridable.

### Council = auto-discovery, scales to the machine
The council uses **no fixed model list**. On start it asks the provider pool what's actually
installed, filters to eligible models, and assigns personas across them. **1 eligible model** →
single-model council, persona diversity via distinct system prompts. **N models** → N genuinely
different minds. Diversity is emergent from the hardware, not a config chore.

**The council debates in two rounds, not one shot.** Each voice first takes an opening position;
then, in a **rebuttal round**, every voice sees the others' openings and answers them — defending,
sharpening, or conceding. Both rounds fan across the fleet in parallel (the rebuttal round adds a
second pass, so wall-clock is ~2× a single round, not 5×), and both are persisted to the forum
(openings → rebuttals → verdict) so the debate reads in order. The synthesizer weighs the whole
exchange, so the verdict reflects arguments that *survived contact with their counter-arguments*,
not just five voices talking past each other.

**The verdict preserves dissent — it does not flatten the debate.** The synthesizer returns a
*structured* verdict, not a single agreeable paragraph: the conclusion, the **single strongest
objection that survived** the deliberation (with which voice raised it), and a **consensus** score
for how strongly the voices converged. The surviving objection rides into the stored memory
(`On '<q>': <conclusion>\nSurviving objection (<voice>): <dissent>`), so a later turn draws on the
*conclusion of its own disagreement* — counter-argument and all, not an unexamined gist. The
stored understanding's confidence is **derived from consensus** (a unanimous verdict is worth more
than a 3–2 split; a 50/50 split lands on the old flat default, and it never escapes the modest
INFERRED band). The objection persists as its own attributed forum post. Parsing is tolerant — a
model that ignores the format degrades to the whole reply as the conclusion, so a verdict is never
lost to a formatting slip (§10). _(Ported from the home-AI moderator's rule: the most valuable
output of an adversarial debate is the strongest objection that survives.)_

### The qualification battery — gate, rank, then watch
"Could model X do the job as well as Y?" — don't guess; measure deterministically, rank, then watch.

1. **Deterministic gate** (mechanical, zero judge cost): does output parse as valid JSON against the
   role schema? Required fields, sane types? **Consistency** across K runs (5/5 vs 3/5 is a real
   score). **Latency** against a per-role ceiling — too slow for `chat` may still qualify for a
   nightly role. **Latency** is each model's true decode throughput — from Ollama's own
   `eval_count`/`eval_duration` (pure generation, so a cold model-load or VRAM swap can't distort it),
   normalized to seconds per ~256-token turn — and it's measured for **every `(model, node)`
   pairing**, since a node can be far faster than the one a model's capability happened to be scored
   on. The gate scores six capability dimensions — *talk*,
   *tools*, *code*, **reasoning**, **discipline**, and **epistemics** (plus an informational *vision*
   check that doesn't affect the score). *Reasoning* tests whether the model can actually **solve** a
   problem with one regex-checkable answer — multi-step arithmetic, classic traps (bat-and-ball, the
   "all but" trap, a relational puzzle, a painted-cube count), letter-counting, a code-trace — and
   its cases are chosen **empirically**: run across a 3B→32B model spread on the fleet, keeping only
   those that *discriminate* strong models from weak, so *quality* can't saturate near 1.0 for any
   fluent model (the failure that made the picker recommend a small model over a clearly better big
   one). *Discipline* tests
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
2. **Rank by points** (over the gate's survivors): the recommendation is a transparent points total
   — quality for the role (dominant) + speed (a strong, *universal* term: a slow model is bad for
   chat *and* background) + a faint size prior to break near-ties — so a saturated quality score
   can't crown the wrong model on a coin-flip. The harder, empirically-chosen reasoning battery in
   filter 1 does the de-saturating job deterministically (no judge call).
3. **Continuous governance** *(forward-looking)*: usage signal (correction rate, engagement) as a
   coherence proxy; a **golden set that self-expands** — when a human corrects the running system,
   that correction becomes a new golden case. Re-qualify after a runtime/model update.

**Guardrail:** the acceptance loop itself ships a **canary** (a known-good run that must pass); if
the self-test breaks, that's loud, never a silent pass (§10).

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

### 5a. The idle window — burst cognition + bidirectional RAG  **[engine landed → extending]**

A finding from months of running this on small, distributed compute (the project's origin): **after
the model answers, the GPU goes idle while the user reads, thinks, and composes a reply.** In text
that's a few seconds; **on voice it's 30 seconds or more, routinely**. That window is not dead time
— it is *the* reclaimable resource that lets a small or distributed rig act like a much bigger brain.
Reference design (proven in the parent system, to be rebuilt public-clean):

> **Landed (the engine):** `cognition/burst.py` is the generic scheduler — pent-up-demand priority,
> the two task classes, the slot cap, interruptibility (an injected `is_busy` predicate), and
> surfaces. The brain routes all post-response cognition (sentinel, self-model, working memory,
> sleep/narratives) through it instead of N raw threads; `turn()` signals it, the next turn settles
> it. Output-triggered RAG and the live inner-life loop (the first step of idle-takeover continuous
> mode) have since landed below; the deep-idle two-voice dialogue is still to come.

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

**Current state.** The burst-window scheduler is **[landed]** (`cognition/burst.py`): two task
classes, pent-up priority, interruptible, surfaces. Post-response cognition (sentinel, self-model,
working memory) routes through it, as does **output-side bidirectional RAG [landed]** — after the
model replies, a burst task retrieves memory relevant to *its own reply* and surfaces it into the
next turn's prompt, so a thread the model itself opened gets grounded, not just the user's input (it
excludes the facts just baked from that reply, so it's not an echo; `Mimir._output_rag`). All of it
composes with the inference engine, which is *built* to be distributed-and-idle-aware rather than
single-shot.

**The live inner life — thinking in the long quiet. [landed]** The burst worker reclaims the *short*
window right after a reply; the inner-life loop (`cognition/inner_life.py`) reclaims the *long quiet*
between conversations — the first step of idle-takeover continuous mode. On a slow, user-tunable
cadence (default one thought every ~5 minutes) a daemon picks ONE universal stimulus — a recent
error, an un-deliberated conflict, the most salient memory, the working-memory thread — and composes
a brief first-person reflection with a cheap background model. The thought is stored as a
low-confidence, decaying memory (`provenance="inner life"`, `INFERRED`). A musing is a *reflection*,
not a knowledge fact, so it does not sit in the recall/knowledge block: at turn time inner-life
memories are split out of the knowledge candidates, and the one most relevant to the current turn —
if it clears a relevance bar — is surfaced as a single **framed, tentative background note** ("while
idle I'd been thinking…", weighed as the system's own idle thought, not as fact). So it **earns its
way in**, gated and framed, never force-injected. Two doctrines bound it: **chat priority** — it
routes *off* the chat model, yields the instant a turn starts (`should_think`), holds an idle floor
after each turn, and runs on a long cadence — and **edge cost** — one model call per cycle, paused
when the fleet is down, **off by default** until the operator opts in. The cadence and on/off live in
the UI (and `[inner_life]` config); a manual "think now" forces one cycle. **The inner life also feeds
the forum:** when the idle loop lands on a genuine, *fresh* conflict it occasionally convenes the full
**council** on it (a forum thread + verdict) instead of musing solo — gated to a daytime trickle (the
self-directed council enabled, a healthy fleet, an hourly cooldown) and sharing the sleep
deliberation's seen-set so the two never re-argue each other. So the council runs nightly in batch
*and* stirs during the day on whatever the system's own attention surfaces. Still **[proposed]**: the
deep-idle two-voice dialogue (propose→critique with memory grounding).

**The wall-clock sleep cycle — heavy maintenance needs a real window, not scraps. [landed]** The
burst worker's premise is that the model idles while the user reads the reply. Two things break it:
**streaming** keeps the model busy until the last token, and on a **slow machine** a single turn can
consume the whole post-response window. So the heavy, model-touching maintenance — consolidation
(dedup, decay, archive, contradiction hygiene), self-directed deliberation, temporal-narrative
roll-ups, a **self-knowledge bake** (its own README → recallable memory, so it knows what it is and
how it works), and an error-digest health pass — gets its **own wall-clock window** when nobody's
around (`cognition/sleep_cycle.py`, the analogue of sleep). A
daemon checks the clock against a user-set `[sleep]` window; inside it (and not already done today,
and not mid-turn) it runs the phases **in order, skipping any that won't fit the minutes left** —
graceful degradation on a slow box rather than starting work it can't finish. Each phase is
checkpointed per-day in a generic `kv` table, so a same-night restart **resumes** and the cycle
**never runs twice a day**, with **catch-up before noon** if the window was missed (a powered-off or
restarted host). It always yields to a live turn, and a manual "run sleep now" forces the full cycle
any time. This is the pattern lifted (clean) from the private home-AI's nightly cycle.

**Self-directed deliberation — the council argues the system's own conflicts. [landed]** A
`deliberate` phase (after consolidation) turns the inner council from a hand-invoked tool into
autonomous cognition (`cognition/deliberation.py`). Consolidation settles the clear-cut cases; what
remains are genuine *tensions* it deterministically surfaces: **graph tensions** (a subject with two+
objects under the same *non-functional* relation — functional ones like "lives in" are consolidation's
job) and **divergent near-duplicates** (memory pairs in a cosine band close enough to be the same
topic but not merged, whose text differs). A **hybrid curator** picks the few most worth arguing — an
LLM ranks them, with a deterministic weight order as the no-model fallback — and each is submitted to
the council; the verdict is stored as recallable understanding (`provenance="sleep deliberation"`).
Conflicts argued recently are skipped so it doesn't loop. This is the public-clean analogue of the
home-AI's nightly BBS/deliberation forum (its 16-persona forum + curator → our council + curator).

**The forum — deliberations made legible + governable. [landed]** Council runs (convened, asked, or
self-initiated in sleep) persist as **threads**: one post per persona (tagged with the node+model that
argued it — the fleet fan-out, visible), the verdict, and user comments (`forum_threads`/`forum_posts`,
schema v20). A `🏛 Forum` view toggles over the chat panel (like the memory graph) with full-admin
housekeeping — comment, close/reopen, delete a post or a whole thread — and an "Ask the council" box.
Comments are annotations, not inputs to the reasoning. This makes the system's adversarial reasoning
something the operator can *read and curate*, not a black box.

> A concrete **build sketch** mapping the *rest* of §5a onto Mimir-0's parts (the bidirectional-RAG
> tasks, idle takeover, the phased plan B1–B4) is parked in
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
  + markdown ship in core (zero deps); PDF + DOCX extraction ship behind the optional `[documents]`
  extra (`pypdf` + `python-docx`), so the runtime contract holds — a missing extra fails loud with the
  install instruction, never a silent skip. _Landed on top:_ a **`[documents]` drop folder** —
  the 📎 upload saves into it and the user can drop files in directly; an idle sleep phase ingests
  new/changed files (content-hashed) and writes a short per-document summary, a small browsable local
  "wiki" (`Mimir.upload_document`/`ingest_pending_documents`/`documents`; `/api/documents/*`). EPUB and
  fuller *LLM compilation of documents into integrated, contradiction-resolved knowledge* remain later
  layers — the per-doc summary is the first step of that, not the whole of it. The **Library layer**
  ("books I've read": gist in SQLite, detail in Markdown, progressive disclosure, with a Phase-2
  model-driven fetch tool) is specced and staged in [`docs/LIBRARY.md`](docs/LIBRARY.md) — planned,
  not built. (Distinct from the read-only Kiwix `[wiki]` reference source.)
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
- **v0.1+ — the extracted thinking layers** (the home AI's highest-leverage cognition, public-clean;
  see §3e/§5a): ~~temporal grounding~~ _(landed: clock/calendar line + per-fact age + no-LLM time
  intercept + interaction-gap awareness baselines)_, ~~temporal narratives~~ _(landed: hierarchical
  daily→weekly→monthly journal, lossy by design, written in the consolidation pass, injected as recent
  history)_, ~~the burst worker~~ _(landed: the idle-window scheduler that runs all post-response
  cognition — pent-up priority, two task classes, interruptible, surfaces; turn() signals, next turn
  settles)_, ~~session history~~ _(landed: a durable, restorable conversation log replayed to the model
  for real continuity, grouped into selectable sessions)_, ~~the visual memory graph~~ _(landed: the
  chat pane flips to a drifting force-directed galaxy of memory blobs + entities — importance to the
  centre, click to review/edit)_. Output-triggered (bidirectional) RAG + idle-takeover are the open §5a
  extensions.
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
- **The system sees its own failures (`diagnostics.py`)** — fail-loud is only half the loop; the
  other half is fail-*aware*. A bounded ring captures `WARNING`+ off the `mimir` logger, and two
  surfaces consume it: a **system-health section in the turn's context** (recent errors, windowed +
  capped) so the model knows when it's degraded and owns it ("my last sentinel pass failed") instead
  of carrying on oblivious, and a **`health` phase in the sleep cycle** that digests the period's
  errors so the nightly pass reviews what went wrong and the summary survives a restart.
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
ingestion (`ingest()` for text + markdown in core, PDF + DOCX behind the `[documents]` extra) and an
evolving, generic **self-model** (identity authored from the store's own operational history,
refreshed off the hot path, injected always-on), working memory (rolling recency + compression),
an entity graph (subject–relation–object triples with 1–2 hop traversal), sleep/consolidation
(dedup, decay, archival, contradiction resolution), and an inner council (adversarial deliberation
across auto-discovered models), and procedural memory (learned trigger→procedure habits). The
model backend is now a **distributed fleet** (DESIGN §5): it auto-discovers Ollama nodes on the LAN
(zero setup on them — just `ollama serve`), catalogues their models, and routes each request to a
node that has the model, with active health checks and least-loaded selection — so the brain can
run on a tiny box and borrow GPUs over the network. The **qualification battery** (DESIGN §4) is
layered on top: a benchmark scores each model on a deterministic capability "IQ test" (talk / tools /
code / reasoning / discipline / epistemics, with empirically-chosen reasoning cases that actually
discriminate) and speed-tests every `(model, node)` pairing — filling the catalogue so model→role
fitness is *tested, not asserted* — and a transparent **points** rank (quality + speed + size)
drives **per-role recommendations** ("for chat, use X on node Y"). The whole DESIGN architecture
is now implemented end-to-end; it remains pre-alpha and unhardened, but the spine, every typed
knowledge layer, the async cognition, and the distributed/qualified model fleet are all live and
verified against a real multi-node LAN.
