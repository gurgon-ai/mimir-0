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
   (talk, tools, code, discipline, epistemics, coherence), not by reputation or size.
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

## 4. The recommended-models registry  **[proposed]**

A **shipped, versioned, documented** list of models we have tested, each with: family, the roles it
is fit for, expected score ranges (per dimension), and a minimum viable size. Purpose:

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

The battery (`DESIGN.md` §4) scores five deterministic dimensions — **talk, tools, code,
discipline, epistemics** — plus a judged **coherence** pass, smallest-first, capacity-capped. **[built]**

Extensions this engine adds:

- **Recommended-first ordering [proposed].** Qualify trusted models before unknowns so the system is
  usable immediately and the trusted ones are available as judges.
- **Known-as-judges → multi-family vote [partial].** Coherence is judged by a panel; today it is
  guarded by a single canary pair **[built]**. The engine strengthens this: prefer *trusted* judges
  (from the registry), and weight by **family diversity** — a true cross-family vote is more robust
  than a single canary, but its confidence **scales with diversity** and must be reported honestly
  (§10.6), not presented as rigorous on a single-family fleet.
- **Sampling + confidence [proposed].** Dimensions are sampled and the score carries variance. Near
  the floor, prefer a **"needs more data"** state over a hard pass/fail on a noisy point estimate;
  add hysteresis so a borderline model does not flap in and out of qualification between runs.
- **Concurrency + capacity awareness [proposed].** Benchmark backend pools **concurrently** across
  nodes, but bounded by each node's capacity so qualification never starves real use or overwhelms a
  small box. Distributed *or* local-only, sized to the hardware the user actually has.

---

## 7. Routing  **[partial]**

Resolution hierarchy (`auto` roles): **pin > measured-best (role-gated) > recommended/approved-family
heuristic > any reachable model** — re-resolved on every rescan, with user disables vetoing at every
level. **[built]**

Engine additions:
- **Local-first preference [partial].** When a LAN pool is enabled, a qualified *local* model is
  preferred for latency/privacy; remote nodes serve burst/overflow/edge. (Discovery defaults
  local-only **[built]**; explicit local-vs-remote preference within an enabled pool is **[proposed]**.)
- **Speed-aware live selection [proposed].** Among equally-qualified options, pick by live
  speed + least-loaded per call (the Phase-2 dynamic routing), not just a static per-role pick.
- **Loud degradation [proposed].** If no qualified model is available for a role (all disabled/offline),
  the engine surfaces it prominently and refuses to silently run an unqualified model.

---

## 8. Staleness & evergreening  **[proposed]**

Evergreening is only *valid* if measurements expire when they should:
- **Model digest.** Store Ollama's per-model digest; a changed digest under the same tag (a re-pull)
  invalidates that model's scores → re-qualify.
- **Battery version.** Stamp the battery; when *we* change probes or scorers (as has already
  happened), prior scores are not comparable → re-qualify on version bump.
- **Optional cadence.** A user may schedule periodic background re-qualification; off by default.

---

## 9. Multi-family adversarial reasoning  **[built, to be surfaced]**

The inner council already spreads personas across whatever models are installed (`DESIGN.md` §4) —
genuinely different model families give genuinely different minds, which strengthens both
deliberation and judging. The engine's job is to **document and present** this in onboarding: tell
the user that running a *variety* of families unlocks stronger adversarial reasoning and more
trustworthy qualification, and show whether their current fleet has it.

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
- A decision log persists the routing rationale for audit.

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
  with the **headless declarative equivalent**.
- **C — Progressive/automatic qualification** (recommended-first → vet-the-rest, concurrent,
  capacity-aware, local-prioritized, trusted-judge multi-family vote, sampling/confidence).
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
