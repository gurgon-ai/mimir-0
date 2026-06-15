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


def test_max_retries_override_disables_retry() -> None:
    # Default: a transient failure is retried and succeeds.
    a = FakeProvider("A", fail_times=1, transient=True)
    assert ProviderPool([("A", a)], max_retries=2, sleep=_noop_sleep).chat(
        "m", [], {}, priority=Priority.BACKGROUND) == "ok"
    assert a.calls == 2

    # Override to 0 (what the benchmark passes): one shot, no retry — the failure propagates so a
    # slow/wedged model fails fast instead of being retried into a multi-minute stall.
    b = FakeProvider("B", fail_times=1, transient=True)
    pool = ProviderPool([("B", b)], max_retries=2, sleep=_noop_sleep)
    with pytest.raises(ProviderError):
        pool.chat("m", [], {}, priority=Priority.BACKGROUND, max_retries=0)
    assert b.calls == 1


def test_split_timeout_pulls_the_reserved_key() -> None:
    from mimir.model.providers.ollama import _split_timeout
    t, rest = _split_timeout({"num_ctx": 8192, "__timeout_s__": 30})
    assert t == 30.0 and rest == {"num_ctx": 8192}  # the key is stripped from the Ollama options
    t2, rest2 = _split_timeout({"num_ctx": 8192})
    assert t2 is None and rest2 == {"num_ctx": 8192}  # absent → provider's own timeout


def test_disabled_node_is_skipped_with_a_fail_safe() -> None:
    a = FakeProvider("A", reply="from-a")
    b = FakeProvider("B", reply="from-b")
    pool = ProviderPool([("A", a), ("B", b)], max_retries=0, sleep=_noop_sleep)
    # Veto node A → routing goes to B only.
    pool.set_disabled_nodes({"A"})
    assert pool.chat("m", [], {}, priority=Priority.CHAT_CRITICAL) == "from-b"
    assert a.calls == 0
    # Veto BOTH → fail-safe: chat must still run rather than hard-block (DESIGN §10), so A is used.
    pool.set_disabled_nodes({"A", "B"})
    assert pool.chat("m", [], {}, priority=Priority.CHAT_CRITICAL) in ("from-a", "from-b")


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


class FleetProvider(FakeProvider):
    """A provider that advertises an inventory (so the pool learns its models) and tags replies."""

    def __init__(self, name: str, models: list[str]) -> None:
        super().__init__(name, reply=f"reply-from-{name}")
        self._models = models

    def list_models(self) -> list[str]:
        return self._models


def test_council_placements_one_per_node_distinct_models() -> None:
    a = FleetProvider("A", ["gemma:12b", "nomic-embed-text"])
    b = FleetProvider("B", ["gemma:12b", "qwen:14b"])
    c = FleetProvider("C", ["qwen:14b"])
    pool = ProviderPool([("A", a), ("B", b), ("C", c)], sleep=_noop_sleep)
    pool.refresh()
    placements = pool.council_placements()
    nodes = [n for n, _m in placements]
    assert nodes == ["A", "B", "C"]                  # one slot per reachable node
    assert all("embed" not in m for _n, m in placements)  # embedding models excluded
    # greedy distinctness: A→gemma, B→qwen (gemma taken), C→qwen (only option)
    assert dict(placements)["A"] == "gemma:12b"
    assert dict(placements)["B"] == "qwen:14b"


def test_chat_on_pins_to_node() -> None:
    a = FleetProvider("A", ["m"])
    b = FleetProvider("B", ["m"])
    pool = ProviderPool([("A", a), ("B", b)], sleep=_noop_sleep)
    pool.refresh()
    assert pool.chat_on("B", "m", [], {}, priority=Priority.BACKGROUND) == "reply-from-B"
    assert b.calls == 1 and a.calls == 0  # pinned to B, A untouched


def test_chat_on_falls_back_when_node_missing() -> None:
    a = FleetProvider("A", ["m"])
    pool = ProviderPool([("A", a)], sleep=_noop_sleep)
    pool.refresh()
    # node "Z" doesn't exist → fall back to ordinary routing (lands on A), persona not lost
    assert pool.chat_on("Z", "m", [], {}, priority=Priority.BACKGROUND) == "reply-from-A"


def test_role_node_pin_routes_to_that_node() -> None:
    from mimir.config import RoleSpec
    from mimir.model.gateway import ModelGateway

    a = FleetProvider("A", ["m"])
    b = FleetProvider("B", ["m"])
    pool = ProviderPool([("A", a), ("B", b)], sleep=_noop_sleep)
    pool.refresh()
    gw = ModelGateway(pool, {"chat": RoleSpec(model="m")})

    gw.set_role_model("chat", "m", node="B")          # pin chat onto node B
    assert gw.role_nodes() == {"chat": "B"}
    assert gw.chat("chat", []) == "reply-from-B"
    assert b.calls >= 1 and a.calls == 0              # ran on B, never touched A

    gw.set_role_model("chat", "m", node=None)         # clear the pin → routes freely again
    assert gw.role_nodes() == {}
