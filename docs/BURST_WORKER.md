# Burst Worker & Bidirectional RAG — build sketch

**Status: design sketch — NOT scheduled.** Parked until the inference-engine phases land. This is a
"don't lose it" map of how the idle-window doctrine (DESIGN §5a) would be built on Mimir-0's actual
parts — enough for a future session to start from, not a full spec. Public-clean: patterns only, no
private/domain code.

## Goal (one line)

Turn the post-response idle window into a scheduled, interruptible work pool that does memory
creation, error correction, and **output-side (bidirectional) RAG** on otherwise-dead GPU cycles —
so a small/distributed rig grounds like a big one, with no added live latency.

## What we reuse (the bones already exist)

- **Model gateway priority tiers** (`model/priority.py`: `BACKGROUND`/`IDLE`) + transient-fail
  deferral + saturation breaker — burst calls route at `IDLE`, so the pool already defers them when a
  real turn is in flight.
- **Storage gateway** (single-writer) — burst writes (baked facts, notes) go through it safely.
- **The brain's existing background spawning** (`_start_background`, `_spawn_sentinel`,
  `_maybe_refresh_*`) — generalize this fixed set into the scheduler.
- **`build_context()`** — already has an end-slot pattern (the sentinel note); surfaced burst results
  slot in the same way.

## Data model (Mimir-0 terms)

```python
@dataclass
class ResponseContext:        # snapshot of the just-finished turn, handed to task factories
    user: str | None; user_text: str; reply: str; turn_index: int; ...

@dataclass
class BurstTask:
    name: str
    run: Callable[[], BurstResult]
    base_priority: float          # lower = sooner
    accrual_rate: float = 0.0     # priority improves per idle second (pent-up demand)
    user_requested: bool = False  # user-driven (continuous) vs autonomous (slot-capped)
    queued_at / last_ran_at / max_age_s
    @property
    def effective_priority(self): return base_priority - starved_seconds * accrual_rate

@dataclass
class BurstResult:
    done: bool = True; requeue: bool = False
    surface: str = ""             # short text injected into the NEXT reply
```

## The scheduler (`cognition/burst.py`, proposed)

- A single daemon worker thread + a priority queue, started lazily.
- **`register(task_factory, trigger=None)`** — tasks register once; after each turn the brain calls
  **`signal_response_sent(ctx)`**, which evaluates triggers, enqueues, and wakes the worker.
- **Two classes:** *user-driven* run continuously (no cap) until done/interrupted; *autonomous* get
  N slots/window, sorted by `effective_priority`.
- **Interruptible:** the brain sets a lightweight **turn-in-flight flag** on entry to
  `turn()`/`turn_stream()`. The worker checks it **between every task**; on a new turn it yields and
  requeues the rest. (This is the doctrine's "foreground always wins" — and it *replaces* today's
  `_join_background()` blocking: a fast follow-up turn no longer waits on background work.)
- **Idle takeover:** after a long quiet, lift the slot cap and run continuous.
- **Surfaces:** `get_pending_surfaces()` drains results; `build_context()` injects them as a
  "background note" section on the next turn.

## Bidirectional RAG as burst tasks

Output-side RAG is just the highest-value autonomous tasks, run **post-response** (never inline
two-pass — that was tried and is too slow for chat, DESIGN §5a):

- **verify/correct** — retrieve memory about claims in `reply`; if it contradicts a higher-tier
  fact, log/flag it (feeds the uncertainty gate + sleep's contradiction resolution).
- **commit** — if the model said it would remember/do something, bake/act on it (error correction
  for "agreed but didn't follow through").
- **prefetch** — retrieve for the likely next turn from `reply`, warming the cache.
- **bump salience** of re-touched memories.

A **fast/slow switch** (config: `[burst] depth = "fast" | "full"`) tunes how much runs — a text UI
affords more than voice. Default text-first.

## Build phases (small, each shippable)

- **B1 — Scheduler core:** `cognition/burst.py` (queue, two classes, floating priority, interrupt
  flag, idle takeover) + migrate the existing sentinel/self-model/WM onto it (replaces
  `_join_background` blocking). Pure refactor + the interrupt win; no new cognition.
- **B2 — Surfaces:** `build_context()` background-note section + the drain path.
- **B3 — Bidirectional RAG tasks:** verify/correct, commit, prefetch; the fast/slow switch.
- **B4 — Tooling/observability:** burst status introspection, decision/burst log (JSONL).

## Non-negotiables

Zero new deps · every burst call at `IDLE` priority through the gateway · **interruptible between
tasks, foreground always wins** · tasks idempotent/resumable, never advance a bookmark on a
pre-empted run · slot cap = the proactivity budget · output-RAG is post-response only · the core §6
loop still passes with the burst worker absent.

## Current state

Mimir-0 has the priority tiers, the single-writer gateway, and a *fixed* post-response background set
(sentinel/self-model/working-memory) that currently **blocks** the next turn. Everything above is
**[proposed]**. The single biggest first win is **B1** — it generalizes what exists *and* removes the
existing next-turn latency stall.
