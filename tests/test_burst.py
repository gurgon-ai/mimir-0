"""Executable spec for the burst worker (DESIGN §5a): priority, two classes, interrupt, surfaces.

Deterministic — an injected clock drives pent-up demand, and tests call signal()/drain_once()
directly so there are no threads to race.
"""

from __future__ import annotations

from mimir.cognition.burst import BurstResult, BurstWorker, ResponseContext

_CTX = ResponseContext(user_text="hi", reply="hello", user="operator", turn_index=1)


def _clock(t: list[float]):
    return lambda: t[0]


def test_runs_a_registered_task_after_a_response() -> None:
    w = BurstWorker()
    ran: list[str] = []
    w.register("note", lambda ctx: (lambda: (ran.append(ctx.user_text), BurstResult())[1]))
    w.signal(_CTX)
    w.drain_once()
    assert ran == ["hi"]


def test_factory_returning_none_skips_the_task() -> None:
    w = BurstWorker()
    calls: list[str] = []
    w.register("maybe", lambda ctx: None)  # never produces work
    w.register("always", lambda ctx: (lambda: (calls.append("ran"), BurstResult())[1]))
    w.signal(_CTX)
    w.drain_once()
    assert calls == ["ran"]


def test_autonomous_slot_cap_defers_excess_lowest_priority_last() -> None:
    t = [0.0]
    w = BurstWorker(autonomous_slots=2, clock=_clock(t))
    ran: list[str] = []
    for name, pri in (("a", 10.0), ("b", 20.0), ("c", 30.0)):
        w.register(name, (lambda n: lambda ctx: lambda: (ran.append(n), BurstResult())[1])(name),
                   base_priority=pri, accrual_rate=0.0)
    w.signal(_CTX)
    w.drain_once()
    assert ran == ["a", "b"]          # two best-priority ran
    assert w.pending() == 1            # "c" deferred to the next window
    w.drain_once()
    assert ran == ["a", "b", "c"]


def test_pent_up_demand_floats_a_starved_task_up() -> None:
    t = [0.0]
    w = BurstWorker(autonomous_slots=1, clock=_clock(t))
    order: list[str] = []
    # "fresh" has the better base priority; "starved" accrues urgency fast.
    w.register("fresh", lambda ctx: lambda: (order.append("fresh"), BurstResult())[1],
               base_priority=10.0, accrual_rate=0.0)
    w.register("starved", lambda ctx: lambda: (order.append("starved"), BurstResult())[1],
               base_priority=50.0, accrual_rate=1.0)
    # Starved queued at t=0, fresh at t=100: starved's effective pri = 50 - 100 = -50, beats 10.
    w.signal(ResponseContext(user_text="x", reply="y"))  # queues both at t=0
    t[0] = 100.0
    w.drain_once()
    assert order[0] == "starved"  # the long-waiting task wins despite a worse base priority


def test_foreground_interrupts_autonomous_work() -> None:
    busy = {"v": False}
    w = BurstWorker(autonomous_slots=10, is_busy=lambda: busy["v"])
    ran: list[str] = []
    for name in ("a", "b"):
        w.register(name, (lambda n: lambda ctx: lambda: (ran.append(n), BurstResult())[1])(name),
                   base_priority=10.0, accrual_rate=0.0)
    w.signal(_CTX)
    busy["v"] = True       # a new query arrives before the window runs
    w.drain_once()
    assert ran == []                # autonomous work yields entirely
    assert w.pending() == 2         # ...and is requeued for later
    busy["v"] = False
    w.drain_once()
    assert set(ran) == {"a", "b"}


def test_user_driven_runs_continuously_until_done() -> None:
    w = BurstWorker()
    counter = {"n": 0}

    def factory(ctx):
        def run():
            counter["n"] += 1
            return BurstResult(done=counter["n"] >= 3, requeue=counter["n"] < 3)
        return run

    w.register("doc", factory, user_requested=True)
    w.signal(_CTX)
    w.drain_once()
    assert counter["n"] == 3  # looped to completion in one window (no slot cap on user-driven)


def test_surfaces_are_collected_and_drained() -> None:
    w = BurstWorker()
    w.register("noticer", lambda ctx: lambda: BurstResult(surface="I noticed something."))
    w.signal(_CTX)
    surfaced = w.drain_once()
    assert surfaced == ["I noticed something."]
    # drain_once already cleared them; a follow-up drain_surfaces is empty.
    assert w.drain_surfaces() == []


def test_threaded_worker_drains_on_signal_and_settles() -> None:
    # The brain runs the worker as a thread: signal() wakes it, wait_idle() blocks until drained.
    w = BurstWorker()
    ran: list[int] = []
    w.register("t", lambda ctx: lambda: (ran.append(1), BurstResult())[1])
    w.start()
    try:
        w.signal(_CTX)
        assert w.wait_idle(timeout=5.0)
        assert ran == [1] and w.pending() == 0
    finally:
        w.stop()


def test_a_failing_task_does_not_break_the_window() -> None:
    w = BurstWorker()
    ran: list[str] = []
    w.register("boom", lambda ctx: lambda: (_ for _ in ()).throw(RuntimeError("nope")),
               base_priority=1.0)
    w.register("ok", lambda ctx: lambda: (ran.append("ok"), BurstResult())[1], base_priority=2.0)
    w.signal(_CTX)
    w.drain_once()
    assert ran == ["ok"]  # the healthy task still ran
