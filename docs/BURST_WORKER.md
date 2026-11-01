# The burst worker — post-response cognition in the idle window

`cognition/burst.py`; DESIGN §5a.

## The idea

After a reply, the model sits idle while the user reads and types: a few seconds in text, often
half a minute on voice. The burst worker runs everything that belongs between one reply and the next
turn in that window, as one prioritised queue that yields when the next turn arrives, instead of a set
of independent threads competing for the model.

## What runs in it (the brain's default registration, `Mimir._register_burst_tasks`)

| task | class | what it does |
|---|---|---|
| `sentinel` | user-requested (every turn, runs first) | reflects on the turn and leaves a note the next prompt carries |
| `actions_followup` | user-requested | if a tool call this turn was refused or crashed, surfaces it as an unkept promise so the model does it another way or says so; only status `ok` counts as done |
| `output_rag` | autonomous (default on) | retrieves memory relevant to the model's *own reply* and grounds the next turn with it; flags a reply that contradicts a higher-tier fact |
| `self_model` | autonomous, on its cadence | re-synthesises the identity from operational history |
| `working_memory` | autonomous, on its cadence | folds the oldest exchanges into a summary rebuilt from archived sources |
| `sleep` | autonomous, only if `[sleep] every > 0` | the turn-cadence consolidation path; the wall-clock window (`cognition/sleep_cycle.py`) is the default |

Anything registered through the background-reflex port (`Mimir.register_burst_task`) joins the same
queue. Every task is registered in the capability ledger and touched when it runs or fails.

## Mechanics (`BurstWorker`)

- `register(name, factory, base_priority, accrual_rate, user_requested, trigger, max_age_s)`: a
  recurring task; `factory(ctx)` returns the work or `None` to skip this turn.
- `signal(ctx)`: the brain calls this after each reply with a `ResponseContext` (user text, reply,
  user, turn index, the turn's tool calls). Triggers are evaluated, work is queued, the worker wakes.
- **Two classes.** User-requested tasks run every window, uncapped. Autonomous tasks get
  `autonomous_slots` per window (default 5), ordered by effective priority, where a starved task's
  priority rises by `accrual_rate` per idle second (pent-up demand).
- **Interruptible between tasks.** The worker checks `is_busy()` (the brain's turn-in-flight flag)
  before each task and yields; the rest is requeued.
- **Settling.** The next turn calls `wait_idle()` (bounded, 30 s) and only then sets its in-flight
  flag, so queued work drains first and the prompt includes the sentinel note and self-model it
  produced. The order matters: the worker yields while the flag is set, so setting it before waiting
  makes the turn wait on work that can't run, until the timeout. An in-flight model call is not
  cancelled; a fast follow-up can wait on one.
- **Surfaces.** A task's `BurstResult.surface` is drained by the next turn's context build and
  injected as a background note.
- **Priority at the gateway.** Burst calls go through the model gateway at the role's priority
  (the `reasoning` role, `USER_ADJACENT` by default), not `IDLE`; the gateway's saturation admission
  and the brain's turn flag are what keep them off the live path.
- `on_ran(name, ok)`: a listener the brain wires to the capability ledger.

## Not built

Idle takeover (lifting the slot cap after a long quiet), a prefetch task for the likely next turn,
a fast/slow depth switch, and a JSONL decision log. Each is small and each is welcome.
