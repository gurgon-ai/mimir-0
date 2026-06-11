"""Executable spec for the provider pool: retry/backoff, transient signaling, health, failover."""

from __future__ import annotations

import pytest

from mimir.errors import ProviderError
from mimir.model.pool import ProviderPool
from mimir.model.priority import Priority
from mimir.model.provider import Message


class FakeProvider:
    """Scripted provider: fails the first ``fail_times`` calls, then returns ``reply``."""

    def __init__(
        self, name: str, *, fail_times: int = 0, transient: bool = True, reply: str = "ok"
    ) -> None:
        self.name = name
        self.fail_times = fail_times
        self.transient = transient
        self.reply = reply
        self.calls = 0

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(f"{self.name} fail #{self.calls}", transient=self.transient)
        return self.reply

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def _noop_sleep(_: float) -> None:
    return None


def test_retries_transient_then_succeeds() -> None:
    p = FakeProvider("A", fail_times=2, transient=True, reply="done")
    pool = ProviderPool([("A", p)], max_retries=2, sleep=_noop_sleep)
    assert pool.chat("m", [], {}, priority=Priority.USER_ADJACENT) == "done"
    assert p.calls == 3  # 2 transient fails + 1 success
    assert pool.get_stats()["retries"] == 2


def test_non_transient_fails_fast_without_retry() -> None:
    p = FakeProvider("A", fail_times=1, transient=False)
    pool = ProviderPool([("A", p)], max_retries=3, sleep=_noop_sleep)
    with pytest.raises(ProviderError) as ei:
        pool.chat("m", [], {}, priority=Priority.USER_ADJACENT)
    assert not ei.value.transient
    assert p.calls == 1  # no retries on a permanent error


def test_failover_to_healthy_endpoint() -> None:
    a = FakeProvider("A", fail_times=10, transient=True)  # always fails
    b = FakeProvider("B", reply="from-b")
    pool = ProviderPool([("A", a), ("B", b)], max_retries=1, sleep=_noop_sleep)
    assert pool.chat("m", [], {}, priority=Priority.USER_ADJACENT) == "from-b"
    assert b.calls == 1
    assert pool.get_stats()["failovers"] >= 1


def test_all_transient_raises_transient() -> None:
    a = FakeProvider("A", fail_times=10, transient=True)
    pool = ProviderPool([("A", a)], max_retries=1, sleep=_noop_sleep)
    with pytest.raises(ProviderError) as ei:
        pool.chat("m", [], {}, priority=Priority.CHAT_CRITICAL)
    assert ei.value.transient


def test_saturation_breaker_defers_background_but_lets_chat_through() -> None:
    clock = {"t": 0.0}
    a = FakeProvider("A", fail_times=10**9, transient=True)  # never recovers
    pool = ProviderPool(
        [("A", a)],
        max_retries=0,
        sat_threshold=3,
        sat_window_s=100.0,
        sat_cooldown_s=50.0,
        clock=lambda: clock["t"],
        sleep=_noop_sleep,
    )

    # Three user-adjacent failures trip the saturation breaker.
    for _ in range(3):
        with pytest.raises(ProviderError):
            pool.chat("m", [], {}, priority=Priority.USER_ADJACENT)
    assert "A" in pool.get_stats()["saturated"]
    assert a.calls == 3

    # Background work now defers immediately without hammering the saturated endpoint.
    with pytest.raises(ProviderError) as ei:
        pool.chat("m", [], {}, priority=Priority.BACKGROUND)
    assert ei.value.transient
    assert a.calls == 3  # not called again

    # Chat-critical still attempts the saturated endpoint as a last resort.
    with pytest.raises(ProviderError):
        pool.chat("m", [], {}, priority=Priority.CHAT_CRITICAL)
    assert a.calls == 4


def test_success_clears_saturation_history() -> None:
    a = FakeProvider("A", fail_times=2, transient=True, reply="ok")
    pool = ProviderPool([("A", a)], max_retries=2, sat_threshold=3, sleep=_noop_sleep)
    assert pool.chat("m", [], {}, priority=Priority.USER_ADJACENT) == "ok"
    # The two transient failures were retried within one successful call; history cleared.
    assert pool.get_stats()["saturated"] == {}


def test_embed_routes_through_pool() -> None:
    p = FakeProvider("A")
    pool = ProviderPool([("A", p)], sleep=_noop_sleep)
    out = pool.embed("m", ["x", "y"], priority=Priority.USER_ADJACENT)
    assert out == [[1.0, 0.0], [1.0, 0.0]]


class StreamFake:
    """A provider that streams tokens."""

    def __init__(self, name: str, tokens: list[str]) -> None:
        self.name = name
        self.tokens = tokens

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        return "".join(self.tokens)

    def chat_stream(self, model: str, messages: list[Message], params: dict[str, object]):
        yield from self.tokens

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_chat_stream_yields_tokens() -> None:
    p = StreamFake("S", ["a", "b", "c"])
    pool = ProviderPool([("S", p)], sleep=_noop_sleep)
    assert list(pool.chat_stream("m", [], {}, priority=Priority.CHAT_CRITICAL)) == ["a", "b", "c"]


def test_chat_stream_falls_back_to_oneshot() -> None:
    # FakeProvider has no chat_stream → the pool yields its single-shot reply once.
    p = FakeProvider("A", reply="hello world")
    pool = ProviderPool([("A", p)], sleep=_noop_sleep)
    assert list(pool.chat_stream("m", [], {}, priority=Priority.CHAT_CRITICAL)) == ["hello world"]


def test_chat_stream_fails_over_before_first_token() -> None:
    a = FakeProvider("A", fail_times=10, transient=True)  # raises on the first token
    b = StreamFake("B", ["from-", "b"])
    pool = ProviderPool([("A", a), ("B", b)], max_retries=0, sleep=_noop_sleep)
    out = "".join(pool.chat_stream("m", [], {}, priority=Priority.USER_ADJACENT))
    assert out == "from-b"
