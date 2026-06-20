# Extensibility — a brain with ports for hands

**Status: design — Phase 1 + the Phase-2 motor-port foundation landing now; the rest proposed.**
Each subsection is tagged **[built]**, **[partial]**, or **[proposed]** so the doc stays honest.

Mimir-0 is a **brain, not an app**: it ships **no built-in hands** — no voice, avatar, Home Assistant,
sensors, or actuators. That doctrine does not change. What this document specifies is how *hands*
attach: instead of one narrow door (`POST /api/turn`, text→prose), the brain exposes **four typed
ports**, each a clean Protocol + a registry + a construction-time injection point. A **connector** is
anything a user writes against a port. The **core ships the ports and the protocols** (and trivial
reference connectors), and **never a real integration** — so it stays public-clean and
zero-dependency while making "build your own IO" a small job instead of a fork.

This honors the same law as the rest of the system (`DESIGN.md`): Python + SQLite + one chat + one
embeddings endpoint, **zero core runtime dependencies**, fail loud. A connector may pull in whatever
it likes; the core never does.

---

## The four ports

| Port | Direction | A connector provides | Protocol / hook |
|---|---|---|---|
| **① Sensory** | world → brain | a **context source** — a prompt section built from external state (a sensor reading, a calendar, an event) | `ContextSource` → `Section` |
| **② Motor** (the hands) | brain → world | a **tool** the model can invoke + a handler that does the thing | `Tool` + the guarded dispatcher |
| **③ Backend** | swap the "neurons" | a custom `Provider` (chat/embed) or `Embedder` | `Provider` / `Embedder` Protocols |
| **④ Reflex** | async / background | a background task that polls/pushes and re-enters via a **surface** | `BurstWorker.register(...)` |

All four attach the **same way**: implement a Protocol → register it (library call, or config) → it's
injected at construction. One mental model, four ports.

---

## ① Sensory port — external state into a turn **[built: Phase 1]**

The seam already exists and the assembler already consumes it; Phase 1 wires it through the turn.

A context source is a small object:
```python
@runtime_checkable
class ContextSource(Protocol):
    name: str
    tier: SectionTier          # HIGH | MEDIUM | LOW — where in the epistemic stack it sits
    budget_tokens: int
    def build(self, query: str, user: str | None) -> Section | None: ...
```
It returns a `Section` (`context/sections.py`) — a titled, tier-tagged, budget-bounded block — or
`None` to contribute nothing this turn. Register it:
```python
brain.register_context_source(WeatherSource(...))        # library
Mimir(config, context_sources=[WeatherSource(...)])      # or at construction
```
At each turn, every registered source's `build()` is called (guarded — a faulty source degrades, it
never breaks the turn, like all enrichment), and its `Section` is folded into the prompt with full
context accounting and the same budget/tier discipline as the built-in layers. **External events ride
the existing recall/section channel — no event bus, no new epistemics.** A sensor reading can also
simply be written to memory (it then recalls like any fact) or pushed via `POST /api/event`
(below) — three intake styles, all reusing machinery the core already has.

---

## ② Motor port — the hands **[partial: registry + guarded dispatcher + single-round loop built; multi-round ReAct proposed]**

The brain emits only prose today, so an external system can't learn *what it wants to do*. The motor
port adds a structured action channel: **registered tools the model can invoke**, run through **one
guarded dispatcher**.

**A tool:**
```python
@dataclass(slots=True)
class Tool:
    name: str                          # "set_light", "now", "send_message"
    description: str                   # shown to the model
    schema: dict                       # JSON-Schema for the args (validated before the handler runs)
    handler: Callable[[dict, ActionContext], str]   # returns a string result; NEVER raises
    state_changing: bool = False       # read-only vs actuating → gates confirmation + trust (below)
    keywords: tuple[str, ...] = ()     # cheap pre-selection so the model isn't handed every tool
    always: bool = False               # offered every turn regardless of keywords
```
Registered **replace-by-name** (hot-safe) via `brain.register_tool(tool)` / `Mimir(config,
tools=[...])`. A **keyword / always-on selector** caps how many tools reach the model per turn —
small local models degrade sharply when handed too many (a hard-won lesson from the parent system),
so this cap is load-bearing, not cosmetic.

**Invocation is tool-calling, not prose-scanning** (the chosen model): the model *deliberately*
invokes a tool by emitting a structured in-band call — `<TOOL name="x" args={…}>` — generalizing the
existing `<FETCH id=N>` marker, which is exactly this pattern for one hardcoded verb. The brain runs
each call through the dispatcher and re-invokes with the results in hand. **Built now:** a
**single-round** loop (offer relevant tools → the model may call → dispatch → re-invoke once →
`actions`). **Next:** a bounded **multi-round ReAct** loop (call-dedup, outcomes logged to procedural
memory). In-band markers (not the provider's native `tools` array) keep it working on small/local
models and the deterministic mock. Acting on what a reply merely *says* ("turning on the light") is
**out of core** by design — too sharp an edge. Tools reach the model only when registered *and*
keyword/always-selected for the turn, so a tool-free deployment pays nothing.

### The dispatcher — one guarded choke point **[foundation: Phase 2]**
*Every* action funnels through one executor; safety lives there, so it can't be bypassed:
- **Capability allow-list + arg-schema validation** — an unknown tool or a malformed call is refused,
  logged, never run.
- **Confirmation gate for `state_changing` tools** — read-only tools (a clock, a lookup) run free;
  an actuator can require an explicit approve step before it fires.
- **Trust-gated actuation — the differentiator.** Mimir already classifies every speaker
  (`speaker_kind`, evidence tiers, `peer_agents`). A **peer AI or a non-primary/low-tier speaker is
  structurally barred from `state_changing` tools** — the same policy that stops a peer laundering a
  fact into trusted memory stops it moving the hands. Read-only tools may stay open; actuation is
  gated to trusted human speakers by default. Trust is enforced in **code**, not the prompt.

**The structured-action output channel:** `TurnResult` and `/api/turn` gain an `actions` field —
`[{tool, args, result, status}]` — so an external system *sees* what the brain did or wanted. This is
the "what does the brain want to do" signal that's missing today.

---

## ③ Backend port — swap the neurons **[built: Phase 1]**

`Provider` (chat/embed) and `Embedder` are already clean Protocols; today only `Mimir(config,
provider=...)` injection exists and `build_provider`/`make_embedder` are closed `if/elif`. Phase 1
turns both into **open registries** so a third-party backend can be named in config *or* injected:
```python
register_provider("vllm", lambda spec: VllmProvider(spec))      # or
Mimir(config, provider=MyProvider(), embedder=MyEmbedder())
```
The core's mock/ollama/bootstrap/endpoint stay registered as the built-in defaults; nothing about the
gateway discipline changes (all calls still route through it).

---

## ④ Reflex port — background connectors **[built: Phase 1 exposes it]**

`BurstWorker` is already a generic, priority-scheduled, interruptible task pool with **surfaces** (a
result injected into the next reply). Phase 1 exposes it publicly:
```python
brain.register_burst_task("poll_sensor", make_task, base_priority=..., trigger=lambda ctx: ...)
```
so a connector can run async work (poll a device, push a notification) that re-enters the
conversation through a surface — the "the world interrupted" path, reusing the existing engine.

---

## Worked example built on the ports: the Notebook **[built]**

The **Notebook** (`cognition/notebook.py`, schema v23) is the first real cognition primitive built on
these ports, and it dogfoods two of them: lossless, name-addressable working memory the model curates
itself (markdown with `##` sections; SQLite via the gateway; decay-exempt — *memory is what it knows,
a notebook is what it's working on*). It attaches as a **`notebook` tool** on the motor port
(non-actuating — its own notes, so `state_changing=False`, safe always) and an **ambient catalog
section** on the sensory port (titles only, never bodies). Its signature mechanism is **read = RAG
re-trigger** (`read_with_memory`): a cold re-read runs the note back through recall so it reconnects
to current memory instead of being an orphaned clipping. `[notebook]`-gated; off-by-cap grooming
surfaces (never silently drops). This is the template a third-party connector follows.

## Worked example built on the ports: the Temporal Registry **[built]**

The **Temporal Registry** (`cognition/temporal_registry.py`, schema v24) is the second primitive on
these ports, and it fixes a specific failure of a narrative memory store: memory accumulates in
**mixed tense** ("planning to do X" … "X is underway" … "X is done"), all coexisting and ranked by
relevance/recency/tier — *not* by which one is currently true, so a status question can surface the
old, high-salience *planning* note and answer as if a finished thing is still upcoming. The registry
is the separate axis: a small, **authoritative, dated, status-tagged** ledger of milestones (STATE —
what is true *now*) beside the narrative store. It lives in its own table, so it is inherently exempt
from memory decay/archival.

It dogfoods three ports plus the sleep cycle:

- **Sensory** — the brain injects a high-attention `[Timeline]` section (current STATE) near the top
  of every prompt, so a "what's the status of …" is answered from what's true now.
- **Motor** — a **`record_milestone` tool** (non-actuating — it writes the brain's own ledger, not
  the world, so `state_changing=False`, safe always) lets the model log a state change the operator
  states ("we finished the migration today").
- **Self-model** — current-config milestones are pinned into the authored self-knowledge ("how am I
  set up"), so configuration is answered from authoritative state, not a stale note.
- **Reflex / sleep** — a deterministic **reconcile** pass during consolidation uses the ledger as the
  authority: it **demotes** a narrative memory that frames as still-upcoming something a milestone
  says is done, and **protects** a faded memory a current milestone confirms (lifting it above the
  archive floor). The guard is **distinctive tokens** — a proper-noun / number / rare word shared
  with the milestone — so it never clobbers an unrelated memory on a generic word like "system". No
  model call; pure functions, the brain owns the wiring. `[temporal_registry]`-gated.

## How connectors attach — three tiers

1. **Library injection** (the clean default, Phase 1) — `Mimir(config, provider=…, embedder=…,
   tools=[…], context_sources=[…])` plus public `register_*` methods. **[built]**
2. **Config module-paths** (Phase 3) — `[connectors] modules = ["mypkg.hands"]`; the server imports
   them at boot and each module's `register(brain)` runs. Closes the "config can't introduce code"
   gap. **Opt-in and trust-flagged** — importing arbitrary modules is a security decision, off by
   default, surfaced loudly. **[proposed]**
3. **Manifest / subprocess connectors** (Phase 4) — a language-agnostic, sandboxed, approval-gated
   "drop a folder in" contract (JSON-in / JSON-out), modeled on the parent system's skills. **[proposed]**

---

## The API surface — fleshed out **[proposed: Phase 3]**

- `POST /api/turn` gains optional `context` (pre-formed observations/sections) on input and an
  `actions` array on output (what the brain did/wants).
- `POST /api/event` — push an external observation into memory / a pending queue **without** a full
  turn; it surfaces on the next turn (the asynchronous "world → brain" channel).
- `GET /api/tools` — the registered capabilities, so a connector or UI can discover what hands exist.

---

## Interop: speak MCP **[proposed: Phase 4 — the high-leverage follow-up]**

The `Tool` model maps almost 1:1 onto **MCP (Model Context Protocol)**, the emerging standard for
wiring models to tools and data. An **MCP adapter** makes any MCP server a Mimir connector — so you
inherit the MCP ecosystem of hands instead of hand-building each, and Mimir's trust-gating wraps them.
The native registry is the mechanism; MCP is the ecosystem on top. (An MCP *client* is a connector,
optional, never a core dependency.)

---

## Doctrine compliance (non-negotiable)

- **Zero core dependencies** — ports + protocols only; a connector's deps are the connector's, never
  the core's. The mock provider + bootstrap embedder still boot the whole loop on Python alone.
- **Public-clean** — core ships the *slots*, never an integration. Reference connectors are trivial
  (a `now` clock, an echo) and exist to document the contract + drive tests.
- **Fail loud / never break the turn** — a faulty context source or reflex task degrades (logged);
  a tool handler returns an error *string*, never raises; an unknown/blocked action is refused loudly.
- **Trust in code** — actuation is gated by the same speaker-trust policy as memory writes; a peer AI
  cannot reach the hands by conversing, exactly as it cannot launder a fact.

---

## Phases

- **Phase 1 — wire the dormant seams.** Forward `ContextSource`/`extra_sections` through
  `turn()`/`turn_stream()`; open provider/embedder registries + `embedder=` injection; expose
  `register_burst_task`; make council personas overridable. → ports ①③④ real. **[building now]**
- **Phase 2 — the motor port.** `Tool` + `ToolRegistry` (register-replace-by-name, keyword/always-on
  selection) + the guarded dispatcher (schema-validate, **trust-gating**, handler-error-as-string) +
  `actions` in `TurnResult`/`/api/turn` + a single-round invocation in `turn()` generalizing the
  `<FETCH>` mechanism. **[built]** Still proposed: multi-round ReAct (call-dedup, outcome logging),
  a confirmation gate for state-changing tools, tool-calling in the *streaming* turn, and
  `register_tool` reference connectors.
- **Phase 3 — packaging + API.** Config module-paths, `POST /api/event`, `GET /api/tools`, `actions`
  + `context` on `/api/turn`. **[proposed]**
- **Phase 4 — ecosystem.** The MCP adapter + manifest/subprocess connectors. **[proposed]**
