"""Executable spec for live node-speed tracking + speed-aware routing (DESIGN §5).

Real traffic is the primary signal (passive measurement on every call); a rare idle probe tops up
quiet nodes; routing prefers the node that will answer soonest (latency × load).
"""

from __future__ import annotations

from mimir.model.latency import (
    LATENCY_NORM_TOKENS,
    LatencyStat,
    normalize_latency,
)
from mimir.model.pool import ProviderPool
from mimir.model.priority import Priority
from mimir.model.provider import Message

# -- the pure core: normalization + EWMA ----------------------------------------------


def test_normalize_is_verbosity_independent() -> None:
    # Same per-token throughput, different reply lengths → near-identical normalized s/turn, so a
    # chatty and a terse model on the same node aren't mis-ranked by how much they happened to say.
    long_reply = "x" * 4000  # ~1000 tokens
    short_reply = "x" * 400  # ~100 tokens
    a = normalize_latency(10.0, long_reply)   # 10s / 1000 tok
    b = normalize_latency(1.0, short_reply)   # 1s / 100 tok
    assert abs(a - b) < 0.01
    assert a == round(10.0 / 1000 * LATENCY_NORM_TOKENS, 3)


def test_normalize_floors_tiny_replies() -> None:
    # A 3-token "ok" can't divide-by-tiny into a nonsense-fast number; the floor caps the divisor.
    assert normalize_latency(2.0, "ok") == round(2.0 / 32 * LATENCY_NORM_TOKENS, 3)


def test_ewma_first_real_sample_replaces_seed_then_blends() -> None:
    stat = LatencyStat()
    stat.seed(9.0)                                   # a frozen benchmark number
    assert stat.value == 9.0 and stat.samples == 0   # seeds don't count as observations
    stat.observe(1.0, alpha=0.5, now=1.0)            # first REAL sample replaces the seed outright
    assert stat.value == 1.0 and stat.samples == 1
    stat.observe(3.0, alpha=0.5, now=2.0)            # then it's an EWMA: 0.5*3 + 0.5*1
    assert stat.value == 2.0 and stat.samples == 2


def test_seed_never_overwrites_a_real_sample() -> None:
    stat = LatencyStat()
    stat.observe(2.0, alpha=0.3, now=1.0)
    stat.seed(9.0)                # lived experience already exists → the seed is ignored
    assert stat.value == 2.0


# -- the pool: passive measurement + speed-aware routing ------------------------------


class _TimedProvider:
    """A provider whose every chat advances a shared fake clock by ``dur`` seconds."""

    def __init__(self, name: str, dur: float, clock: dict[str, float], reply: str = "y" * 400):
        self.name = name
        self.dur = dur
        self.clock = clock
        self.reply = reply
        self.calls = 0

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        self.calls += 1
        self.clock["t"] += self.dur
        return self.reply

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def list_models(self) -> list[str]:
        return ["m"]


def _noop_sleep(_: float) -> None:
    return None


def test_passive_measurement_records_latency_on_real_calls() -> None:
    clock = {"t": 0.0}
    p = _TimedProvider("A", dur=2.0, clock=clock)
    pool = ProviderPool([("A", p)], clock=lambda: clock["t"], sleep=_noop_sleep)
    pool.chat("m", [], {}, priority=Priority.USER_ADJACENT)
    snap = pool.latency_snapshot()
    assert ("A", "m") in snap
    assert snap[("A", "m")]["samples"] == 1
    assert snap[("A", "m")]["return_time"] == normalize_latency(2.0, p.reply)


def test_routing_prefers_the_faster_node() -> None:
    clock = {"t": 0.0}
    a = _TimedProvider("A", dur=9.0, clock=clock)
    b = _TimedProvider("B", dur=1.0, clock=clock)
    pool = ProviderPool([("A", a), ("B", b)], clock=lambda: clock["t"], sleep=_noop_sleep)
    # Seed both from "qualification": A slow, B fast. The next call must route to B without us
    # having to discover it the hard way (seeding makes routing informed from the first turn).
    pool.seed_latency({("A", "m"): 9.0, ("B", "m"): 1.0})
    pool.chat("m", [], {}, priority=Priority.USER_ADJACENT)
    assert b.calls == 1 and a.calls == 0  # fastest healthy node wins


def test_idle_nodes_excludes_busy_and_disabled() -> None:
    clock = {"t": 0.0}
    a = _TimedProvider("A", dur=1.0, clock=clock)
    b = _TimedProvider("B", dur=1.0, clock=clock)
    pool = ProviderPool([("A", a), ("B", b)], clock=lambda: clock["t"], sleep=_noop_sleep)
    pool.refresh()  # mark both reachable + inventory
    assert set(pool.idle_nodes()) == {"A", "B"}
    pool.set_disabled_nodes({"A"})
    assert pool.idle_nodes() == ["B"]  # a vetoed node is never probed


def test_probe_latency_records_without_real_traffic() -> None:
    clock = {"t": 0.0}
    a = _TimedProvider("A", dur=3.0, clock=clock)
    pool = ProviderPool([("A", a)], clock=lambda: clock["t"], sleep=_noop_sleep)
    pool.refresh()
    out = pool.probe_latency("A", "m", [{"role": "user", "content": "hi"}], {})
    assert out == normalize_latency(3.0, a.reply)
    assert pool.latency_snapshot()[("A", "m")]["samples"] == 1
