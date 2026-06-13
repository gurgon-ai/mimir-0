# The Inference Engine — discovery, qualification, routing, onboarding

**Status: design — partially implemented.** This document specifies the model-agnostic inference
engine that the rest of Mimir bolts onto. It extends `DESIGN.md` §4 (model-agnostic by role) and §5
(the fleet) into a complete contract: how models are *discovered*, *measured*, *trusted*, *routed*,
and *kept current* — and how a new user goes from a clone to a running, qualified fleet without
reading the source. It is explicitly **beyond the scope of the private system Mimir-0 was distilled
from**, and deliberately so (see Motivation).

Each subsection marks **[built]**, **[partial]**, or **[proposed]** so the doc stays honest about
current state.

---

## 0. Relationship to the core spec

The core (`DESIGN.md`) holds regardless of this document: the §6 acceptance loop runs on a single
mock provider with zero models installed. This engine is the *production* path — what happens when a
real, possibly-distributed, possibly-heterogeneous set of models shows up. It must never weaken the
core's law: **Python + SQLite + one chat endpoint + one embeddings endpoint, zero core runtime
dependencies, fail loud.** If a step here needs more than that to function, it is the wrong step.

---

## 1. Motivation — why this is core, not decoration

Mimir's epistemic framework was *born on weak, distributed hardware*: two Raspberry Pi 5s and an
old gaming laptop. The layered, evidence-tiered, provenance-tagged RAG approach proved itself there
— it made *small* models behave far better than their size suggested. The powerful hardware came
**after** that realization, not before. The lesson that shaped this engine:

> If the framework is this good on poor hardware, then its value is *democratization* — it should
> let anyone get strong cognition out of whatever models and machines they have, by **measuring**
> models objectively and **routing** intelligently, rather than assuming a single big GPU.

So the inference engine exists to make the epistemic frame **hardware-agnostic and model-agnostic**:
- A user with one modest machine runs locally and the engine picks the best *local* model.
- A user with a Pi and a gaming PC on the LAN runs the brain on the Pi and borrows the GPU.
- A user with a pile of mixed machines pools them and gets multi-family adversarial reasoning for
  free.
- A user who installs a brand-new model next year gets it **objectively measured** and slotted in
  if it earns its place — the project **evergreens** instead of ossifying around today's models.

This is why qualification cannot be an opt-in afterthought: for most users (who do *not* have a
$10k AI box), the engine *is* the difference between good and useless cognition.

---

## 2. Principles (the contract)

1. **Model-agnostic.** No model is hardcoded or privileged in code. A curated *recommended* list is
   a documented, versioned default — never a lock-in. Any model the user installs can compete.
2. **Objective measurement.** A model earns a role by **measured** behaviour on a fixed battery
   (talk, tools, code, reasoning, discipline, epistemics, coherence), not by reputation or size.
3. **Local-first, distributed-optional.** Default is local-only; the LAN fleet is opt-in and, when
   enabled, *local is preferred* — remote nodes are for burst, overflow, or edge deployments.
4. **Evergreen.** New models are first-class. Stale measurements self-invalidate (model digest +
   battery version), so "install anything" stays *valid* over time.
5. **User override at every step.** Auto is the default, never the cage. A pin, a disable, a manual
   role assignment, or a skipped step always wins.
6. **Fail loud.** No silent fallback to an unqualified model; no silent staleness; misconfiguration
   and degradation are surfaced, not swallowed (`DESIGN.md` §10).
7. **Zero added dependencies, public-clean.** The wizard is the existing stdlib `http.server` UI;
   nothing here introduces a runtime dependency or copies private code.
8. **Headless parity.** Every wizard choice has a `mimir.toml` equivalent, so edge/server/CI deploys
   configure identically without a browser.

---

## 3. Architecture — the pipeline everything bolts to

```
        ┌── discovery ──┐   ┌── qualification ──┐   ┌── routing ──┐
 nodes →│ local + (LAN) │ → │ recommended-first │ → │ resolve per │ → turn()/council/bake/...
 models │ inventory     │   │ vet → vote → score│   │ role, local │
        └───────────────┘   └───────────────────┘   │ -first,     │
              ▲                      ▲               │ speed/qual  │
              │                      │               └─────────────┘
        node approval         seed registry (us)            │
        (LAN trust)           + known-as-judges        loud degrade
```

The engine exposes a stable internal surface — *"give me the model for role R"* and *"qualify what
you can reach"* — so cognition (turns, council, bake, sentinel, self-model) never touches discovery,
benchmarking, or scheduling. That separation is what lets everything else "bolt on."

---

## 3a. Data model & config contract (the concrete schemas)

Defining the shapes up front prevents schema creep. Three records, one config DSL, one data file.
Field names that already exist in the implementation are noted; the **[new]** tag marks what Phase
A–D add to the current `model_catalogue` / `model_prefs` tables (`DESIGN.md` §4).

**ModelDescriptor** — one per distinct model (the catalogue row, deduped across nodes):
```
id              "gemma4:e4b"          # the Ollama tag
family          "gemma"               # CANONICAL family (see family map below), not the raw tag
params_b        8.0
quantization    "Q4_K_M"
context_length  8192
nodes           ["http://192.168.2.50:11434", ...]   # every node that has it
digest          "sha256:1a2b…"        # [new] Ollama model digest — staleness key (§8)
recommended     true                  # [new] present in the registry (§4)
enabled         true                  # user veto (model_prefs.enabled)
scores          { <dimension>: ScoreRecord }   # talk/tools/code/reasoning/discipline/epistemics/coherence
quality         0.92                  # aggregate
return_time     0.86                  # fastest node, seconds
battery_version 3                     # [new] which battery produced `scores` (§8)
scored_at       <ts>
role_bans       [ {role:"chat", reason:"discipline 0.25 < 0.50"} ]   # [new] persisted, explainable (§10.9)
```

**ScoreRecord** — one per dimension, carries uncertainty (so a noisy point estimate isn't a hard
pass/fail; §6):
```
value     0.67     # mean over samples
samples   6
ci_low    0.41     # simple Wilson/normal bound — no heavy stats
ci_high   0.86
state     "pass" | "fail" | "needs_more_data"   # by where the CI sits vs the floor
```

**RouteDecision** — the audit log entry, appended as structured JSONL **and** human-readable (§11):
```
{ "ts": …, "role": "chat", "chosen": "gemma4:e4b", "node": "http://…",
  "reason": "measured-best" | "pin" | "recommended-heuristic" | "reachable-fallback" | "degraded-unqualified",
  "candidates": ["gemma4:e4b","qwen2.5:3b"], "battery_version": 3,
  "scores": {"discipline":0.83,"epistemics":1.0} }
```

**The recommended-models registry** — a *pure data file* (`recommended_models.toml`, versioned,
documented), never code:
```toml
[[model]]
family        = "gemma"
tag_patterns  = ["gemma4:e4b", "gemma4:e2b", "gemma3:12b"]
roles         = ["chat", "reasoning", "judge", "bake"]
expected      = { discipline = [0.8, 1.0], epistemics = [0.6, 1.0] }   # floors we have measured
judge_ok      = true        # trusted to vet unknown models (cold-start judges, §4)
min_params_b  = 5.0
notes         = "Strongest small-model epistemics in testing."
```

**The headless config DSL** (`mimir.toml`) — the single source of truth the wizard merely *writes*:
```toml
[backend]
mode             = "local"        # "local" | "lan"
discovery        = "on"           # "on" | "off"
refresh_interval_s = 60

[roles.chat]
model          = "auto"           # "auto" | "<tag>" (a pin)
allow_families = ["gemma", "qwen"]    # optional narrowing
deny_models    = ["gemma3:4b"]        # the bias veto, declaratively

[[lan.node]]
url      = "http://192.168.2.50:11434"
approved = true                   # NO context is routed to an unapproved node (§5.2)
```

**Family canonicalization** — a data map (not inline code), because vendors fork and rename:
```
"gemma*" → "gemma"   "qwen*" → "qwen"   "llama*"/"codellama" → "llama"   "phi*" → "phi" …
```

**Conflict resolution (mechanical).** A recommended model present on the machine is qualified
*first*, but its registry status grants **no grandfathering**: if its measured score on *this*
hardware falls below the registry's `expected` floor (or the role floor), it is **barred for that
role with a persisted `role_bans` reason** — measured beats recommended, always.

**Worked example — a Pi (brain) + an RTX box (Ollama), 3 models.** `mimir.toml`:
```toml
[backend]
mode = "lan"            # the Pi has no GPU; borrow the RTX box
discovery = "on"
[roles.chat]    model = "auto"
[roles.bake]    model = "auto"
[[lan.node]]
url = "http://192.168.2.50:11434"   # the RTX box
approved = true
```
Discovery finds `gemma4:e4b`, `qwen2.5:3b`, `nomic-embed-text` on the node. Qualification (recommended
gemma first → trusted judge → vets qwen) yields: `gemma4:e4b` passes chat/reasoning, `qwen2.5:3b`
passes (epistemics `needs_more_data` → one more sampling round → pass). Resolution: `chat →
gemma4:e4b` (`reason: measured-best`), `bake → qwen2.5:3b` (faster, talk-gated). One `RouteDecision`
per role is logged. The Pi holds memory; the RTX box does inference; nothing was hand-configured
beyond approving one node.

---

## 4. The recommended-models registry  **[built — Phase A]**

A **shipped, versioned, documented** list of models we have tested
(`src/mimir/cognition/recommended_models.toml`, loaded by `cognition/registry.py`), each with:
family, the roles it is fit for, expected score ranges (per dimension), a `judge_ok` flag, and a
minimum viable size. **[built]** Auto-routing now prefers a present recommended-for-the-role model
before any benchmark **[built]**; using `judge_ok` models as cold-start judges is wired in Phase C
**[proposed]**. Purpose:

- **Safe default before any benchmark.** On first run the engine prefers a recommended model that is
  actually present, so the *out-of-box* path can't silently land on a known-bad model (this closes
  the engine's worst failure mode — see §10.1). This is the concrete form of "test the local machine
  first for viable models, start with the ones we trust."
- **Bootstrap of trust for judging.** Recommended models that pass their expected scores become the
  **trusted judges** that vet *unknown* models — solving the cold-start problem of "who judges the
  judges" without assuming the user's fleet is any good.
- **Documentation.** The README/SETUP list these so a user knows what to pull for a great experience
  (e.g. the gemma4 family, qwen2.5/3.5, llama3.x, phi, mistral, command-r, deepseek, granite,
  internlm — families validated to clear the gates), while making clear it is a *starting point*,
  not a whitelist.

The registry is data, versioned alongside the battery (§8), and **maintained as a first-class doc**
(per the project's documentation convention). It never gates: an unknown model that *measures* well
is used; a recommended model that *measures* badly on this machine is not.

---

## 5. Onboarding — first run  **[proposed]**

On first boot with no usable configuration, the server starts and directs the user to the browser,
where a **setup page** runs once:

1. **How do you want to run?** Explain **local-only** vs **LAN pool** in plain language ("LAN pool =
   any machine on your network running `ollama serve` becomes a worker, zero setup on it"). Pick one
   or both. Default **local-only**.
2. **LAN node approval** (only if LAN chosen). Show the nodes discovered on the subnet and let the
   user **approve** which ones may receive context — because routing memory to an auto-discovered
   node is a data-exposure surface (§10.5). Nothing is used until approved.
3. **Automatic model discovery?** If yes, inventory what is reachable, then **qualify
   recommended-present models first** (usable in seconds), and offer to qualify the rest in the
   background. Show a time estimate; respect the machine's capacity (don't melt a Pi — §6).
4. **Review & override.** Present the proposed role→model assignments with *why* (scores, what was
   barred and why), and let the user pin, disable, or accept. Surface whether **multi-family
   adversarial reasoning** is available (≥2 families present) and what that buys them.
5. **Persist.** Write the choices to `mimir.toml` (and DB prefs) so boot is silent thereafter.

**Headless equivalent [proposed].** The same outcomes are reachable declaratively in `mimir.toml`
(`[backend]`, `[roles]`, approved nodes, discovery on/off), so a Pi or server configures without a
browser. The wizard *writes* exactly this file — there is one source of truth.

**Re-runnable [proposed].** A "re-run setup / re-qualify" action repeats the flow at any time (new
models, new hardware, a model update), preserving overrides unless changed.

---

## 6. Qualification  **[partial]**

The battery (`DESIGN.md` §4) scores **six deterministic dimensions** — **talk, tools, code,
reasoning, discipline, epistemics** — plus a judged **coherence** pass, outside-in ordered and
size/latency-bounded. **[built]**

What the non-obvious dimensions actually test **[built]**:

- **reasoning** — real problems with one regex-checkable answer (multi-step arithmetic,
  letter-counting, sequence completion, a code-trace, an instruction transform). This is what keeps
  `quality` from saturating near 1.0 for any fluent model: following a *format* is easy, *solving* is
  not. chat/reasoning/code gate on it.
- **discipline** — does **not** reproduce the internal `[tier=…; source=…]` scaffolding when shown it
  (sampled K×; a consistent leaker scores ~0 and is barred from the identity roles).
- **epistemics** — does it exploit Mimir's tiered/provenance/gated context (`DESIGN.md` §3)? The
  structured-arm score over a gauntlet: a **layered conflicting-tier** probe (a high-tier "blue" vs a
  low-tier "red" buried in filler → defer to the high tier *under noise* — the structured arm can, a
  flat blob can't), a **grounding** floor (recall a nonce that exists *only* in the context), and a
  **long-context needle** (a nonce planted mid-way through a ~2k-token document, past Ollama's 2048
  default). The chat-LLM qualifier; chat/reasoning gate on it.

**Representative latency [built].** `return_time` is timed from one *real-length* generation,
normalized to seconds per ~256-token turn — not the round-trip of a 3-token reply, which can't tell a
slow remote 12B from a snappy local 3B. **Context window [built]:** every benchmark call pins
`backend.benchmark_num_ctx` (default 24576 — the operational window, qualify at the size you deploy
at) so Ollama's 2048 default can't silently truncate the layered prompts, and the long-context probe
sizes its haystack to ~60% of it (testing the real window, not just clearing 2048). **Size bounds [built]:** `max_model_size_b` (ceiling) and `min_model_size_b` (floor —
on capable hardware, don't let a tiny model that scores "high enough" out-compete a bigger, genuinely
better one) bound the field; both are UI/config knobs.

**Capability and latency are orthogonal — and neither may contaminate the other [next, the design].**
This is the load-bearing rule for the distributed scheduler:

- **Capability is per-MODEL** (node-independent): establish it **once**, on whatever node can run it.
  A model that's slow on one weak edge is **never failed** — that fails the *(model, node) pairing*,
  not the model. So a capability call must **never be scored 0 because a node was slow**: if a node
  is too slow to test on, the model is **requeued to another node**, not marked incapable.
- **Latency is per-(MODEL, NODE)**: recorded so we don't re-probe a pairing, and used **only for
  routing/placement** (does *this* box meet the user's `max_latency_s`?), never to gate capability.
- **The objective is distributed compute, not "use every box."** On one-beast-plus-weak-edges
  hardware it is *correct* for everything to run on the beast — shipping a 4B to a Pi at 5s when the
  beast does two 26Bs in that time is a loss. An edge earns a role only by having the capability
  **and** meeting latency. The scheduler therefore: distributes to test fast, probes a node cheaply,
  and if it's too slow to bother → marks that (model, node) and **requeues the model to another
  node** (capability lands on the fastest node that can run it); per-node speed is recorded for
  routing. The latency cap is a **routing/finals** criterion, **not** a quality-round filter (an
  earlier "cap skips early everywhere" decision was reversed for exactly this reason).

Extensions this engine adds:

- **Recommended-first ordering [proposed].** Qualify trusted models before unknowns so the system is
  usable immediately and the trusted ones are available as judges.
- **Known-as-judges → multi-family vote [partial].** Coherence is judged by a panel; today it is
  guarded by a single canary pair **[built]**. The engine strengthens this: prefer *trusted* judges
  (from the registry), and weight by **family diversity** — a true cross-family vote is more robust
  than a single canary, but its confidence **scales with diversity** and must be reported honestly
  (§10.6), not presented as rigorous on a single-family fleet.
- **Sampling + confidence [proposed].** Concrete rule (no heavy stats): run *N* samples per
  dimension, keep the mean and a simple confidence bound. If the whole interval is above the floor →
  **pass**; entirely below → **fail**; if it straddles the floor → **needs_more_data** (run more
  samples, up to a cap, before deciding). Add hysteresis (a small margin) so a borderline model does
  not flap in and out of qualification between runs.
- **Concurrency + capacity awareness [proposed — full design in `BENCHMARK_SCHEDULER.md`].**
  Benchmark backend pools **concurrently** across nodes via a **work-stealing, one-worker-per-node**
  scheduler: each node tests one model at a time (VRAM), fast nodes claim more, and a model too slow
  to test on one node is **requeued to another** (never failed on speed). Bounded by a per-node
  concurrency cap (default 1), so qualification never starves real use or overwhelms a small box.
  Distributed *or* local-only, sized to the hardware the user actually has. Orchestration:
  - **Parallel across nodes, sequential within a node.** A node holds one model warm at a time
    (VRAM), so within a node it's warm → test → swap; but every node works at once. Each *distinct*
    model is dealt to one **home node** for scoring (quality is node-independent), spreading the
    work so nodes hit *different* models; the other nodes that have it are probed only for per-node
    *speed*. The orchestrator (on the head) dispatches async to the LAN nodes **and** runs its own
    models in the gaps — the head never blocks the edge, and results are collected as they land.
  - **Capability-aware assignment.** Among the nodes that have a model, deal **small models to the
    weaker edge nodes and big models to the strong head** — a Pi races through 3B models while the
    big GPU grinds the 26–30B ones, each busy with work it's suited to (and the weak node never
    chokes on a model it can barely hold). Dovetails with the outside-in (small/big) ordering.
  - **Triage, then score.** A fast first sweep checks each model is viable (responds within budget)
    and *warms* it; only survivors get the full battery — so a slow/broken model can't burn a slot.
  - **Honest live ETA.** Time the *test* separately from the *warmup* (warmup is overhead, not the
    score). After the first model, `remaining × running-average` gives an estimate, refined per node
    as models complete (overall ETA = the slowest node's). Order each node's list **outside-in
    (biggest, smallest, biggest, smallest…)** so the running average samples both extremes
    immediately and the ETA is stable and honest from the second model on — not the ballooning
    underestimate that smallest-first produces. Surface elapsed + estimated-remaining + per-node
    progress.

---

## 6a. The qualifying tournament  **[built]**

The web UI exposes qualification as a **staged, human-veto knock-out** (Fleet tab → "🏆 Run
qualifying tournament") — the interactive realization of triage→score, with the user in the loop
between rounds. It separates the two questions a fleet must answer, which behave differently across
machines:

- **Can the model do the job? (quality — node-independent.)** A model is as capable on any box, so it
  is scored **once**. **Round 0 · Qualifying** runs the cheap capability dimensions over the whole
  fleet (ephemeral — nothing saved); the user unticks who shouldn't advance; **Round 1 · Gauntlet**
  re-tests only the survivors through the full framework qualification (reasoning + the epistemic
  gauntlet), persisting scores.
- **Is it fast enough? (speed — node-dependent.)** The one genuinely per-machine question.
  **Round 2 · Finals** picks each role's champion *among the user's finalists only* (the veto beats
  the global best). **Round 3 · Vision** is reserved for the vision dimension.

The latency cap (`max_latency_s`) is an early **skip-gate** in every round (a model no node can run
under the cap never reaches the expensive gauntlet) and the **selection** criterion in the finals.
The objective, plainly: *the best-scoring model for this system that you're willing to wait for.*

Built on three staging primitives on `benchmark_fleet`: `only_models` (a round re-tests just the
survivors), `framework=False` (triage — cheap dimensions only), and `persist=False` (ephemeral — a
scouting round can't pollute the saved scores).

**Per-node toggle [built].** Each discovered edge node can be **toggled off** (a node-level veto,
mirroring the per-model one; `node_prefs` table, schema v13), excluding it from the pool's routing
(with a fail-safe if *every* node is vetoed, so chat never hard-blocks), from qualification, and from
recommendations — even if reachable. "All qualified machines" = the enabled nodes.

**The two axes — quality vs. speed [next].** A model needs both to be a candidate:
- **Eligible?** *Can it do the job* — quality + framework. **Node-independent → tested once.**
- **Fast enough?** *Can some enabled node run it under the cap* — **per-node.**

The hard speed cut is **early and cheap** (a model no node can run under the cap is skipped before the
expensive gauntlet — the latency cap as a skip-gate, applied every round). The **Finals is not an
elimination — it's placement/priority:** for the eligible survivors it measures real per-node
turn-speed → a model×node registry → and assigns each role its champion (best eligible model). Because
serving runs off the **pool** (least-loaded/reachable node that has the model, per call), "which box"
is a runtime decision the pool already makes; the finals just records the speeds and picks the model.
So the agreed restructure: the gauntlet stops per-node speed-probing (pure quality = faster), and the
finals becomes the dedicated per-node speed/placement round.

---

## 7. Routing  **[partial]**

Resolution hierarchy (`auto` roles): **pin > measured-best (role-gated) > recommended/approved-family
heuristic > any reachable model** — re-resolved on every rescan, with user disables vetoing at every
level. **[built]**

**The objective [built].** *The best-scoring model for this system that you're willing to wait for.*
Within the cap every role ranks on **pure quality** — a dominant model wins outright and speed only
breaks ties. So a 26B a second behind a 4B at a higher score wins. (This replaced an earlier
quality-minus-speed "balanced" formula for `chat`.)

**Latency is a USER-FACING concern only [next].** Most of Mimir's cognition is idle/between-turns
(council, the async sentinel, sleep, the burst worker reclaiming idle GPU) — for that work nobody is
waiting, so latency is irrelevant and **absolute capacity wins** (run the biggest, slowest, most
capable model). So `max_latency_s` applies **only to user-facing roles** (`chat`, tools-in-a-turn);
**idle roles** (`council`, `sentinel`, `reasoning`, `bake`, sleep, burst-worker) route to **best
quality, no cap**. A slow-but-brilliant model that fails the chat cap everywhere is **kept** and
routed to the idle roles where it's the best choice — never discarded for being slow. This is why the
qualification never fails capability on speed, and the cap is applied *last*, only where it matters.

**Identity roles** — `chat`, `reasoning`, `judge`, `sentinel` — are the roles that *speak or reason
as the system*. They have a hard rule: **never route to an unqualified model** (one without a
passing discipline + epistemics score) without a **noisy, user-acknowledged** exception. Stating
this in routing terms prevents silent regressions. `bake` and forward-looking `tools`/`code` are not
identity roles and gate only on their own capability.

Engine additions:
- **Local-first preference [partial].** When a LAN pool is enabled, a qualified *local* model is
  preferred for latency/privacy; remote nodes serve burst/overflow/edge. (Discovery defaults
  local-only **[built]**; explicit local-vs-remote preference within an enabled pool is **[proposed]**.)
- **Speed-aware live selection [proposed].** Among equally-qualified options, pick by live
  speed + least-loaded per call (the Phase-2 dynamic routing), not just a static per-role pick.
- **Loud degradation [proposed].** If no qualified model is available for an identity role (all
  disabled/offline), the engine **refuses to silently run an unqualified model** and surfaces it
  structurally: the API returns a clear error (e.g. HTTP 503 with a structured body naming the role
  and why), and the UI shows a banner — never a quiet downgrade.

---

## 8. Staleness & evergreening  **[proposed]**

Evergreening is only *valid* if measurements expire when they should:
- **Model digest.** Store Ollama's per-model digest; a changed digest under the same tag (a re-pull)
  invalidates that model's scores → re-qualify.
- **Battery version.** Stamp the battery; when *we* change probes or scorers (as has already
  happened), prior scores are not comparable → re-qualify on version bump.
- **Pending re-qual, not silent reuse.** When a model's scores are invalidated (digest or battery
  change), any role depending on it is marked **`pending re-qual`** and shown as such in the UI —
  the engine does **not** silently keep routing on stale scores.
- **Optional cadence.** A user may schedule periodic background re-qualification; off by default.

---

## 9. Multi-family adversarial reasoning  **[built, to be surfaced]**

The inner council already spreads personas across whatever models are installed (`DESIGN.md` §4) —
genuinely different model families give genuinely different minds, which strengthens both
deliberation and judging. The engine's job is to **document and present** this in onboarding: tell
the user that running a *variety* of families unlocks stronger adversarial reasoning and more
trustworthy qualification, and show whether their current fleet has it. "Family" is resolved through
the canonical **family map** (§3a) — kept as data, not inline code — so vendor forks and renames
don't quietly collapse two distinct families into one (which would fake diversity).

---

## 10. Failure modes & guards (the "get it right" list)

1. **Out-of-box lands on a bad model** *(the worst one)* → the recommended-first default (§4/§5) and a
   refusal to route identity roles to un-measured models close this; qualification becomes part of
   onboarding, not an opt-in users skip.
2. **Brittle / low-sample scorers near a hard floor** → sampling + confidence + "needs more data" +
   hysteresis (§6); the multi-family vote as a second signal.
3. **Stale scores after a model update** → digest + battery-version invalidation (§8).
4. **Silent degradation to an unqualified model** → loud surfacing + refusal (§7).
5. **Data exposure to LAN nodes** → explicit node approval before any context is routed (§5.2).
6. **False confidence on a single-family fleet** → confidence scales with diversity and is reported
   honestly; lean on trusted seed judges (§6, §9).
7. **Circular judging (the fleet judges itself)** → trusted recommended models as judges bootstrap
   trust from outside the fleet (§4).
8. **Resource exhaustion from benchmarking** → concurrency bounded by node capacity (§6).
9. **Opaque decisions** → record and surface *why* each role got each model, and why others were
   barred (§5.4, §11).

---

## 11. UX cues & observability  **[proposed]**

- The Model Pool tab prompts action when models are unqualified ("3 models not yet measured —
  qualify now?"), rather than only showing badges.
- Bars are explained inline ("gemma3:4b barred from chat: discipline 0.25 < 0.5").
- Role changes are announced ("chat: gemma3:12b → gemma4:e4b after re-qualification").
- **Decision log** persists the routing rationale as structured **JSONL** (machine-readable, so a
  user can diff behaviour across versions or hardware) *and* renders human-readably in the UI.
- **Dry-run mode** — the engine explains what it *would* route for each role, and why, **without
  calling any model**. Invaluable for debugging a fleet before committing real inference.

---

## 12. Non-negotiables (must hold for every part)

Zero core runtime dependencies · stdlib-only web wizard · public-clean (no private code) · fail-loud
· headless parity with the wizard · the core §6 loop still boots with no models installed.

---

## 13. Build phases

- **A — Recommended-models registry** (documented + versioned) and route from it by default.
  Smallest change, immediately closes failure mode §10.1. ("Document recommended models; test local
  first; start with known-good.")
- **B — First-run setup wizard** (local/LAN, node approval, discovery opt-in), persisted to config,
  with the **headless declarative equivalent**. *Scope guard:* the wizard does **only** mode
  selection, LAN node approval, discovery toggle, and accept/reject of the auto-generated role map —
  detailed tuning and score inspection are deferred to a later "advanced" view, so it can't balloon.
- **C — Progressive/automatic qualification** (recommended-first → vet-the-rest, concurrent,
  capacity-aware, local-prioritized). *Order within the phase:* ship **trusted judges + a minimal
  multi-family vote first** (most of the benefit), then add the full sampling/confidence machinery.
- **D — Staleness (digest + battery version), loud degradation, decision audit/explainability,
  speed-aware live routing.**

---

## 14. Acceptance — how we know it's right

1. A fresh user with only Ollama + one recommended model installed reaches a *qualified, correctly
   routed* chat without reading the source or running a manual command.
2. A user who installs a brand-new model gets it objectively measured and used **iff** it earns a
   role — with the reason visible.
3. The same outcomes are reachable headlessly via `mimir.toml` (Pi/edge).
4. No path routes an identity role to an unqualified model without a loud notice.
5. On a multi-family fleet, qualification uses cross-family judging; on a single-family fleet, it
   says so and leans on trusted seed judges — never presenting false rigor.
6. The core §6 acceptance loop still boots and passes with zero models installed.
