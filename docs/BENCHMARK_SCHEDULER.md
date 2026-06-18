# The benchmark scheduler — concurrent, distributed qualification

**Status: design [proposed].** The build that makes fleet qualification parallel. Grounded in what
we learned shipping the sequential version (`cognition/benchmark.py`) and the live testing that
shaped it. Companion to `INFERENCE_ENGINE.md` §6/§6a; this is the detailed scheduler spec.

> **Note (superseded in part):** the **coherence / peer-review judging** discussed below (§4-ish, the
> `judges_trustworthy` canary, the panel-voted coherence pass) was **removed** — it scored every
> model the same middling yellow and discriminated nothing. Its de-saturation job is now done
> deterministically by an **empirically-chosen harder reasoning battery** (cases probed across the
> fleet, kept only if they separate strong models from weak), and ranking is a transparent **points**
> total (quality + speed + size). Concurrency is built; the benchmark also **speed-tests every
> `(model, node)` pairing** as an automatic final phase. Read the judging sections as history.

---

## 0. Framing — what this is and isn't

Qualifying the fleet is an **initialization event**: it runs once at setup (re-runnable later as
"re-qualify"), and on first run it is **paced by the onboarding interview** filling the wait (see
`mimir_foundational_interview.md`). So the priorities, in order:

1. **Correct** — never fail a model's capability because of one node's speed; never record a
   half-finished battery as a score; never silently drop a model.
2. **Minimal duplication** — capability is established **once**; a (model, node) latency is measured
   **once**; nothing is re-probed.
3. **Fast** — wall-clock matters for UX, but it is *not* critical. We can afford a proper scheduler.

This ordering is why we choose a correct work-stealing design over a quick-but-lossy one.

## 1. The two orthogonal questions (the invariant that drives everything)

- **Capability — per MODEL, node-independent.** A model is as capable on any box. Test it **once**,
  on whatever node can run it. A model slow on a weak edge is **never failed** — that fails the
  *(model, node) pairing*, not the model.
- **Latency — per (MODEL, NODE).** Recorded so we never re-probe a pairing; used **only** for
  routing/placement (does *this* box meet `max_latency_s`?), **never** to gate capability.

The objective is **distributed compute, not "use every box."** An edge earns a role only by having
the capability **and** meeting latency. On one-beast-plus-weak-edges hardware it is *correct* for
everything to land on the beast — the scheduler must converge there naturally, not fight it.

## 2. Hard constraints

- **One model per node at a time (VRAM).** A node holds one model warm; testing two at once thrashes
  it. The per-node worker *is* the serialization point. (A `per_node_concurrency` > 1 is a later
  refinement for nodes with spare VRAM — default 1.)
- **All writes through the storage gateway** (already single-writer/thread-safe) — so persisting
  scores/speeds from many workers is safe without extra locking on the DB.
- **Calls fail fast, no retries** (the hang fixes): per-call `__timeout_s__`, `max_retries=0`, and
  single-token warmups stay. A wedged call dies in seconds and that (model, node) attempt is abandoned.
- **The pre-gate measures a representative turn** (already built): warm + a ~64-token generation
  normalized to seconds/turn — the units the latency cap is in.

## 3. Data model

Per run, in memory (the durable scores/speeds go to `model_catalogue` via the gateway):

```
candidates[model]  = [enabled http nodes that have the model]      # who could test it
tried[model]       = set(nodes already attempted and too slow)     # never retry a pairing
done[model]        = ModelBenchmark | NO_VIABLE_NODE | None         # capability outcome
pending            = set(models needing capability, not yet exhausted)
in_flight          = int  (models currently being probed/tested)   # for termination
speed[(model,node)]= seconds/turn                                  # recorded as measured
```

`NO_VIABLE_NODE` is **not** a capability failure — it means "no reachable node could test this within
the test budget." Surfaced loudly (the user can add a capable node, raise the budget, or accept it).

## 4. The algorithm — work-stealing, one worker per node

One thread per enabled http node. The worker *is* the node's VRAM lock, so no separate per-node mutex
is needed — a node only ever does one thing at a time.

```
worker(node N):
  loop:
    with lock:
      M = claim_for(N)                 # a pending model on N that N hasn't tried (policy below)
      while M is None:
        if not pending and in_flight == 0:
          notify_all(); return         # nothing left and nothing can be requeued → done
        wait(cond, timeout=1s)         # another worker may requeue something N can take
        M = claim_for(N)
      pending.discard(M); in_flight += 1     # claimed → in flight

    speed = probe(N, M)                # warm + representative latency (seconds/turn); bounded
    record speed[(M,N)]               # per-node viability, for routing/finals

    if speed <= TEST_BUDGET:
      try:
        bench = run_battery(M, pinned to N)    # capability; fail-fast calls
        with lock:
          done[M] = bench; persist(bench, speed on N); in_flight -= 1; notify_all()
        continue
      except battery_error:                    # node crashed / transport died mid-battery
        pass                                    # fall through to requeue (don't record a partial)

    # too slow to TEST here, or the attempt failed → requeue to another node, never fail the model
    with lock:
      tried[M].add(N); in_flight -= 1
      if candidates[M] - tried[M]:
        pending.add(M)                          # a node that hasn't tried it may still qualify it
      else:
        done[M] = NO_VIABLE_NODE
      notify_all()
```

**Why this terminates.** Each (model, node) is attempted at most once (the `tried` set), so total
work is finite. A worker exits only when `pending` is empty **and** `in_flight == 0` — i.e. nothing
to claim and nothing that could be requeued. Every state change (`done`, requeue) is followed by
`notify_all()`, so a waiting worker re-checks; the `timeout=1s` wait is a liveness backstop, not the
mechanism.

**Why it's distributed-compute-correct.** Fast nodes finish batteries sooner → claim more; a weak
edge that probes a big model finds it over `TEST_BUDGET`, requeues it, and the beast picks it up. On
a one-beast rig the beast ends up doing ~everything and the edges contribute a few cheap probes —
exactly right. On a balanced fleet the work spreads and runs in parallel.

## 5. The claim policy (capability-aware)

`claim_for(N)` chooses among the models `N` could attempt. The policy shapes throughput; correctness
holds for any choice.

- **Default: largest-untried-first that fits.** A node claims the **largest** pending model it hasn't
  tried and that fits its capacity (when capacity is known — VRAM/RAM from discovery, or learned).
  Big models gravitate to strong nodes; a weak node skips models it can't hold and grabs small ones.
- **Avoid wasted probes:** if a node's capacity is known and a model clearly won't fit, don't claim
  it (let a bigger node, or `NO_VIABLE_NODE`, handle it). Until capacity is known, the `TEST_BUDGET`
  probe is the cheap backstop — one ~64-token generation, then requeue.
- **Outside-in within a node** (biggest, smallest, biggest…) keeps the per-node ETA honest, as today.

## 6. Per-node concurrency [refinement]

Default one model per node. A node with spare VRAM can run `k` models at once; expose
`per_node_concurrency` (config or a quick probe of how many fit). Then a node runs up to `k` worker
slots. Keep the default at 1 — it's the safe, universal floor, and the init-event framing means we
don't *need* to squeeze each node.

## 7. Integration with the tournament rounds

- **Round 0 Qualifying & Round 1 Gauntlet** run **through this scheduler** — quality only, distributed,
  never cutting on speed. (Triage = cheap dimensions; Gauntlet = full framework on survivors — same
  scheduler, different battery depth.)
- **Round 2 Finals = the speed/placement round.** For the eligible survivors, fill in the per-node
  speed matrix: we already have `speed[(M,N)]` for the nodes we tested on; the finals measures the
  *remaining* enabled nodes (skipping pairings already recorded — minimal duplication).

**Latency is a USER-FACING concern only — apply it last, and only where someone is waiting.** Most of
Mimir's work is idle/between-turns (council, the sentinel's async pass, sleep, the burst worker
reclaiming idle GPU); for that, latency is irrelevant and **absolute capacity wins** — you'd run the
biggest, slowest, most capable model. So roles split by latency-sensitivity:

- **User-facing roles (`chat`, tools-in-a-turn):** apply `max_latency_s`. A (model, node) is
  **viable** iff `speed ≤ cap`; the role is routable iff it has ≥1 viable node; champion = best
  quality among routable, placed on its fastest viable node. The pool does live node selection from there.
- **Idle roles (council, sentinel, reasoning, bake, sleep, burst-worker):** **no cap** — champion =
  best quality, full stop; placed on its fastest *capable* node (speed only breaks ties).

A slow-but-brilliant model that fails the chat cap everywhere is therefore **kept** and routed to the
idle roles where it's the best choice — it just drops off the user-facing shortlist. The scheduler's
"never fail capability on speed" invariant (§1) is what makes this possible: capacity is preserved
for every model, and the cap is a final, user-facing-only filter.

### Distribution's payoff is parallelism, not "best model faster" — the placement matrix [built]

The beast almost always wins user-facing (best model, fastest). An edge earns its keep a *different*
way: by running **background cognition concurrently** — council, sentinel, the nightly backlog — so
the beast stays free for the user. For that work latency barely matters (idle/capacity-bound), so the
bar for an edge is **"can it run a qualified-enough model and return before the next idle tick,"** not
"is it fast." A runner-up like gemma4:e4b returning in 4s on the macbook is useless for chat but a
perfect **background worker**.

So the finals produce a **placement matrix**, not a single champion:

- **The column we were missing:** the scheduler tests each model's capability *once* (on whatever node
  ran it — usually the beast). The finals must then probe each qualified model's speed on the **other
  nodes it's installed on**, so we know *where each model can actually run*.
- **User-facing roles** → best model on its fastest viable node (cap applies). Usually the beast.
- **Idle/background roles** → **offload to an edge** when a capable-enough model runs there within a
  *generous* background tolerance — freeing the beast. The edges become the background-cognition fleet
  (the council/sentinel/nightly work runs *off* the interactive box, in parallel).
- **Provisioning gap, surfaced loudly:** if a capable runner-up is installed **only on the beast**,
  the finals say so — *"qualified, but on no edge; pull it to an edge to use it as a background
  worker."* This is the bridge to auto-distributing models across the LAN fleet (the discover →
  qualify → distribute vision): qualification reveals *which* runner-ups are worth pushing to edges.

**Cast a wide net — the output is a graded, queryable fleet-capability map, not a single champion.**
The non-user work (async sentinel, adversarial council, nightly review) doesn't want *the* best model;
it wants a **diverse roster of good-enough ones** spread across the LAN — the parent runs a 16-persona
council across 7 model families precisely because diversity beats a single brain for adversarial
reasoning. So qualification grades **every "yellow-and-above" model on every node it can run on**, and
role assignment is a **fuzzy query over the map, not a fixed pick**:

- **User-facing (chat):** strict — top quality, under the cap, on a fast node.
- **Background / council / sentinel:** generous — *any* yellow-and-above model that **runs** on an
  idle edge qualifies; deliberately favor a **spread of families** for adversarial value.
- Downstream subsystems query the map ("3 diverse council members that fit on idle edges," "what can
  the macbook host for background reasoning") instead of consulting a hand-maintained list.

This graded map — pool × node × per-dimension score × speed — is the thing the hand-tuned parent
lacks. It's *learned*, and it self-updates whenever the fleet or the model lineup changes (the
evergreen property): a new model is graded and placed the day it's installed, no human in the loop.

**The second lineup is qualified with the user-facing limits OFF.** `max_model_size_b` and
`max_latency_s` are *user-facing* constraints — they exist so chat stays fast on the user's hardware.
But the background/adversarial/specialized lineup is **capacity-bound, not latency-bound**, so those
limits must not exclude its candidates: the big coder a user caps out of chat, a 122B MoE, the
slow-but-brilliant models are *exactly* the council/code/background picks. (Observed: a user's real
coding model was size-capped out, so `code` defaulted to the chat champion — correct for the chat
pool, wrong as a final answer.) So qualification grades a **wider pool** than chat will ever route to;
the user-facing caps are applied only when assigning the *user-facing* roles. A model is never
excluded from the second lineup for being too big or too slow — only from chat.

**[built]** The second lineup ships in two parts: **selection** — `council_roster()` ranks within
each model family then round-robins *across* families (diversity beats ranking for adversarial work),
capacity-bound and never latency-gated; and **grading** — `benchmark_council_pool()` grades the
above-cap models (the 30–36B coders, the 122B MoE) with the caps off, **in place / no rescan** so the
main pool's scores survive (the `complete_speed_matrix` discipline). Surfaced as the 🏟️ Council view
(3/5/7 seats, families represented, bench) + 🏋️ Qualify big models. The per-node **placement matrix**
(`placement_matrix()` + 📊 view) is the display side: every model on every node it runs on, with that
node's speed and the node's winner. Still open: explicit background/council *roles* in `ROLE_NEEDS`
(loose, non-discipline-gated) so role assignment can query the roster directly.

### Coherence is a post-qualification peer-review pass, not a qualification gate [next]

Coherence is the one **judge-based** dimension — a panel of *other* models rates a candidate's answer
for faithfulness. Running it *inside* the qualification battery (as today) has a chicken-and-egg flaw:
a trustworthy judge panel requires **qualified** models, but during qualification none exist yet, so
the judges are "first-3-available" (weak ones included) → conservative, noisy, run-to-run-unstable
scores that drag the ranking around for no signal. (Observed live: capable 24–27B models all landing
~0.65 coherence — that's the panel, not the candidates. And the rubric rewards *terseness*: the only
model to score green was a 2B, because it answered with the bare facts while the bigger, more helpful
models got dinged for "invented details" — i.e. for elaborating. Coherence-as-judged penalizes
exactly the helpfulness you want.)

The fix mirrors the parent's nightly **peer-review** phase: **defer coherence out of the deterministic
qualification entirely.** The deterministic dimensions decide who qualifies; then a **post-
qualification peer-review pass** runs coherence on the survivors only, judged by the **top qualified
models** (a real trusted panel — never a model judging itself). Three wins: faster (survivors only),
better-calibrated (qualified judges), and the qualification ranking stops carrying judge noise. This
is capacity-bound, latency-irrelevant work — a natural fit for the finals or for idle/nightly time.

**The scoring redesign — a criteria rubric, not a vibe number [decided 2026-06-13].** Relocating the
pass isn't enough; the *scoring* is the deeper bug. Observed live: **every** model lands yellow
(~0.65). That's not the models — it's three compounding measurement faults: (1) **LLM central
tendency** — asked to "rate faithfulness 0.0–1.0," judges cluster at 0.6–0.8 almost regardless of
input; (2) **mean-of-N compresses** further toward the center; (3) the **probe is trivial** (one
simple fact every model gets right), so there's no real spread and the only variance left *is* judge
noise. The tell that it's the judge: the `judges_trustworthy` canary **passes** (good ranked above
garbled) while real answers all compress to ~0.65 — judges separate extremes but flatten the
realistic middle. So the redesign:

- **Stop asking for a number — ask for discrete checks.** The judge returns a small structured
  verdict (JSON), e.g. `{"used_required_facts": bool, "invented_detail": bool, "contradicted": bool}`.
  Score = fraction of criteria passed. Discrete groundedness checks escape vibe-compression, and
  parsing the JSON sidesteps the `_parse_score` first-number bug ("on a scale of 0 to 1 …" → `0`).
- **A probe with room to fail.** A short context with **≥2 facts that must be used**, a **trap**
  (a plausible-but-absent detail a sloppy model invents), and something that **must be grounded**.
  Now answers genuinely differ, so the dimension can discriminate.
- **Fixes the terseness bias for free.** Criteria grade *groundedness* (used the facts, invented
  nothing, contradicted nothing) — a model that elaborates with *grounded* detail isn't penalized;
  only invented/contradictory detail counts against it. (Today's "free of invented details" rubric
  dinged the helpful 24–27B models and crowned a terse 2B.)
- **Qualified judges, self excluded** (per the peer-review relocation above).

One structured criteria verdict thus fixes central tendency, the parse bug, and the terseness bias in
a single move — and gives a *checkable* sub-score instead of a number pulled from the judge's vibe.

## 8. Failure modes (fail-loud)

- **Node dies mid-battery** → the attempt raises → that (model, node) is marked tried and the model
  requeued to other nodes. The dead node's worker keeps failing fast and contributes nothing more.
- **Model qualifies on no node** (`NO_VIABLE_NODE`) → recorded and surfaced in the result/coverage
  readout (not hidden): "N models couldn't be tested on any reachable node within the budget."
- **Whole fleet too slow** → every model `NO_VIABLE_NODE` → a loud result, never a silent empty pass
  (the canary/self-test discipline, DESIGN §10).
- **A worker stuck on a genuinely-wedged socket** → the per-call timeout + no-retry bound it; the
  1s condition-wait backstop keeps other workers from blocking on it.

## 9. Progress & ETA under concurrency

Report `completed / total` (capability outcomes, not attempts) and an elapsed-based ETA
(`elapsed / completed × remaining`). Stream each completion via `on_result(bench, node)` as today, so
the live board fills in as nodes finish — and the per-model elapsed timer already distinguishes
"grinding" from "stuck."

## 10. Concurrency caps & politeness

- Bound total in-flight by the number of enabled nodes (one model each) — the worker count.
- Honor the per-node concurrency cap (default 1) so we never overwhelm a small box.
- Leave the user's real inference untouched: qualification is `BACKGROUND` priority and these are
  *direct, pinned* calls, so they don't fight production routing on the pool.

## 11. Open decisions

- **Capacity discovery:** read VRAM/RAM per node (Ollama doesn't expose it directly) vs. *learn* it
  (a model that OOMs/refuses on a node → mark that node can't hold ≥ that size). Leaning learn-it:
  simpler, and it self-corrects. A failed load is already a signal.
- **Claim policy tuning:** largest-first vs capability-matched. Start largest-untried-that-fits;
  revisit if a real fleet shows wasted probing.
- **Re-qualify deltas:** on re-run, skip models whose digest + battery_version are unchanged
  (`INFERENCE_ENGINE.md` §8 staleness) so re-qualification only tests what's new — the biggest real
  speedup, and it composes with this scheduler.

## 12. Build order

1. **Pin a battery to a node** — `benchmark_model(node=…)`: a model's whole battery runs on one warm,
   direct provider (no pool thrash). Coherence judges still use the gateway pool.
2. **The scheduler** (§4) wrapping the quality rounds — worker-per-node, claim/probe/test/requeue,
   termination. Tests: termination on a mock multi-node fleet; a too-slow node requeues to a fast
   one; capability never scored from a timed-out attempt; one-beast converges on the beast.
3. **Finals speed round** (§7) — fill the per-node matrix (skipping recorded pairings), apply the cap,
   place roles.
4. **Refinements** — per-node concurrency, capacity learning, re-qualify deltas.

Keep the mock/single-node path **sequential and order-deterministic** (one worker) so the existing
executable specs hold; concurrency engages only on a real multi-node http fleet.
