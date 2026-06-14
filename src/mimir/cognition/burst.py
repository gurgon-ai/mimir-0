"""The burst worker — the idle window as a first-class scheduled resource pool (DESIGN §5a).

After each turn the model sits idle while the user reads and composes a reply. This turns that
window into a scheduled pool of background cognition, with three properties from the home AI:

- **Pent-up-demand priority** — a task's effective priority improves the longer it has gone unrun
  (``effective = base − starved_seconds × accrual_rate``), so starved work floats up on its own
  instead of needing a hand-tuned schedule. Lower number = runs sooner.
- **Two task classes** — USER-DRIVEN tasks (the user asked for something) run continuously, only
  yielding to a new query; AUTONOMOUS tasks (nobody asked) are slot-capped per window and ordered by
  effective priority.
- **Interruptible** — foreground always wins. Between tasks the worker checks an injected
  ``is_busy`` predicate and defers the rest of the window if a turn is in flight.

A task may emit a **surface** — a short note drained into the next reply's prompt ("[Background
note: …]") — so off-path work re-enters the conversation. The worker is generic: it schedules
callables; *what* they do lives in the tasks the brain registers (sentinel, self-model, working
memory, sleep). All state is per-instance and the clock is injected, so tests drive ``signal()`` +
``drain_once()`` deterministically without threads; the brain runs the threaded loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("mimir.burst")

# A task factory returns the work to do for this context (a zero-arg callable), or None to skip.
TaskFn = Callable[[], "BurstResult"]
TaskFactory = Callable[["ResponseContext"], TaskFn | None]
Trigger = Callable[["ResponseContext"], bool]


@dataclass(slots=True)
class ResponseContext:
    """A snapshot of the just-finished turn, handed to task factories."""

    user_text: str
    reply: str
    user: str | None = None
    turn_index: int = 0


@dataclass(slots=True)
class BurstResult:
    """What a burst task returns. ``surface`` (if any) is injected into the next reply's prompt."""

    done: bool = True       # True → drop from the queue
    requeue: bool = False   # True (and not done) → run again next window
    surface: str = ""       # a note to carry into the next turn


@dataclass(slots=True)
class _Task:
    name: str
    fn: TaskFn
    base_priority: float
    accrual_rate: float
    user_requested: bool
    queued_at: float
    last_ran_at: float = 0.0
    max_age_s: float = 600.0


@dataclass(slots=True)
class _Recurring:
    name: str
    factory: TaskFactory
    base_priority: float
    accrual_rate: float
    user_requested: bool
    trigger: Trigger | None
    max_age_s: float


class BurstWorker:
    """Schedules background tasks into the post-response window (DESIGN §5a)."""

    def __init__(
        self, *, autonomous_slots: int = 5, is_busy: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._slots = max(1, autonomous_slots)
        self._is_busy = is_busy or (lambda: False)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._recurrings: dict[str, _Recurring] = {}
        self._queue: list[_Task] = []
        self._surfaces: list[str] = []
        self._last_ran: dict[str, float] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()
        self._stats = {"windows": 0, "tasks_run": 0, "surfaces": 0}

    # -- registration -----------------------------------------------------------------

    def register(
        self, name: str, factory: TaskFactory, *, base_priority: float = 50.0,
        accrual_rate: float = 0.1, user_requested: bool = False,
        trigger: Trigger | None = None, max_age_s: float = 600.0,
    ) -> None:
        """Register a recurring task, evaluated after every response (or when ``trigger`` passes).

        ``factory(ctx)`` returns the work (a zero-arg callable) or ``None`` to skip this turn. Lower
        ``base_priority`` runs sooner; ``accrual_rate`` is how fast a starved task's priority rises.
        ``user_requested`` puts it in the continuous (uncapped) class.
        """
        with self._lock:
            self._recurrings[name] = _Recurring(
                name=name, factory=factory, base_priority=base_priority,
                accrual_rate=accrual_rate, user_requested=user_requested,
                trigger=trigger, max_age_s=max_age_s,
            )

    # -- the signal after each response -----------------------------------------------

    def signal(self, ctx: ResponseContext) -> None:
        """Enqueue the tasks whose triggers fire for this turn, then wake the worker."""
        now = self._clock()
        with self._lock:
            recurrings = list(self._recurrings.values())
        for r in recurrings:
            try:
                if r.trigger is not None and not r.trigger(ctx):
                    continue
                fn = r.factory(ctx)
            except Exception as exc:  # a factory must never break the turn that signalled it (§10)
                log.warning("burst: factory %r failed: %s", r.name, exc)
                continue
            if fn is None:
                continue
            with self._lock:
                self._queue.append(_Task(
                    name=r.name, fn=fn, base_priority=r.base_priority,
                    accrual_rate=r.accrual_rate, user_requested=r.user_requested,
                    queued_at=now, last_ran_at=self._last_ran.get(r.name, 0.0),
                    max_age_s=r.max_age_s,
                ))
        if self.pending():
            self._idle.clear()  # mark busy so wait_idle() blocks until this work is drained
        self._wake.set()

    # -- draining ---------------------------------------------------------------------

    def _effective_priority(self, task: _Task, now: float) -> float:
        starved = now - max(task.last_ran_at, task.queued_at)
        return task.base_priority - starved * task.accrual_rate

    def drain_once(self) -> list[str]:
        """Run one burst window and return the surfaces produced. Synchronous + deterministic.

        User-driven tasks run first, continuously (re-running a requeued one until done), only
        yielding to ``is_busy``. Autonomous tasks then run up to the slot cap, lowest effective
        priority first, also yielding to ``is_busy`` and requeuing the rest.
        """
        self._idle.clear()
        try:
            now = self._clock()
            with self._lock:
                live = [t for t in self._queue if now - t.queued_at <= t.max_age_s]
                self._queue.clear()
            user_tasks = [t for t in live if t.user_requested]
            auto_tasks = sorted(
                (t for t in live if not t.user_requested),
                key=lambda t: self._effective_priority(t, now),
            )
            ran: list[str] = []
            requeue: list[_Task] = []

            # Phase 1 — user-driven: continuous, no slot cap, yield only to a new query.
            for task in user_tasks:
                first = True
                while True:
                    if self._is_busy() and not first:
                        requeue.append(task)
                        break
                    first = False
                    result = self._run(task, ran)
                    if result is None or result.done or not result.requeue:
                        break

            # Phase 2 — autonomous: slot-capped, priority-ordered, interruptible.
            used = 0
            for i, task in enumerate(auto_tasks):
                if self._is_busy():
                    requeue.extend(auto_tasks[i:])
                    break
                if used >= self._slots:
                    requeue.append(task)
                    continue
                result = self._run(task, ran)
                used += 1
                if result is not None and result.requeue and not result.done:
                    task.last_ran_at = self._clock()
                    requeue.append(task)

            with self._lock:
                self._queue.extend(requeue)
                self._stats["windows"] += 1
                self._stats["tasks_run"] += len(ran)
                surfaces, self._surfaces = self._surfaces, []
            if ran:
                log.info("burst: window ran %d task(s): %s", len(ran), ", ".join(ran))
            return surfaces
        finally:
            self._idle.set()

    def _run(self, task: _Task, ran: list[str]) -> BurstResult | None:
        try:
            result = task.fn() or BurstResult()
        except Exception as exc:  # one task's failure never aborts the window (§10)
            log.warning("burst: task %r failed: %s", task.name, exc)
            return None
        ran.append(task.name)
        with self._lock:
            self._last_ran[task.name] = self._clock()
            if result.surface:
                self._surfaces.append(result.surface)
                self._stats["surfaces"] += 1
        return result

    # -- surfaces (drained into the next prompt) --------------------------------------

    def drain_surfaces(self) -> list[str]:
        """Pop and return the surfaces produced by completed tasks (for the next turn's prompt)."""
        with self._lock:
            out, self._surfaces = self._surfaces, []
        return out

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    # -- the threaded loop (the brain runs this; tests use drain_once directly) --------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._loop, name="mimir-burst", daemon=True)
        thread.start()
        self._thread = thread  # publish only once started — stop() never joins an unstarted thread

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while self.pending() and not self._stop.is_set():
                if self._is_busy():
                    break  # foreground wins — wait for the next signal
                before = self.pending()
                self.drain_once()
                if self.pending() >= before:
                    break  # made no progress (all requeued/busy) — don't spin

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue is drained and no window is running — settles background before the
        next turn. Returns True if it reached idle, False on timeout (e.g. work kept deferring)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._idle.wait(timeout=0.05) and self.pending() == 0:
                return True
        return self.pending() == 0

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
