# The benchmark scheduler — concurrent, distributed qualification

**Status: design [proposed].** The build that makes fleet qualification parallel. Grounded in what
we learned shipping the sequential version (`cognition/benchmark.py`) and the live testing that
shaped it. Companion to `INFERENCE_ENGINE.md` §6/§6a; this is the detailed scheduler spec.

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
  *remaining* enabled nodes (skipping pairings already recorded — minimal duplication). Then apply
  `max_latency_s`: a (model, node) is **viable** iff `speed ≤ cap`; a model is routable iff it has
  ≥1 viable node; each role's champion = best quality among routable, placed on its fastest viable
  node. The pool does live node selection per call from there.

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
