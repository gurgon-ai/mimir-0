"""Executable spec for the distributed fleet: discovery, model-aware routing, catalogue."""

from __future__ import annotations

from mimir.cognition.fleet import fleet_report, scan_fleet
from mimir.config import BackendConfig, Config, ProviderSpec, RoleSpec
from mimir.embed.base import EmbeddingMode
from mimir.model.discovery import discover_node_urls, normalize_url
from mimir.model.gateway import ModelGateway
from mimir.model.pool import ProviderPool
from mimir.model.priority import Priority
from mimir.model.provider import Message, ModelInfo
from mimir.storage.gateway import StorageGateway


class FleetFake:
    """A node holding a fixed set of models; records every chat it served."""

    def __init__(self, name: str, models: list[str]) -> None:
        self.name = name
        self._models = models
        self.calls: list[str] = []

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        self.calls.append(model)
        return f"{self.name} answering with {model}"

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def model_details(self) -> list[ModelInfo]:
        return [ModelInfo(name=m, family="fam", params_b=float(len(m))) for m in self._models]


# -- discovery ------------------------------------------------------------------------


def test_normalize_url_adds_scheme_and_port() -> None:
    assert normalize_url("192.168.2.5") == "http://192.168.2.5:11434"
    assert normalize_url("192.168.2.5:11434") == "http://192.168.2.5:11434"
    assert normalize_url("http://host:1234") == "http://host:1234"


def test_discovery_merges_local_configured_and_scanned() -> None:
    backend = BackendConfig(
        lan_backend=True, subnet="10.0.0.0/30", nodes=["10.0.0.99"], scan_concurrency=4
    )
    reachable = {"http://10.0.0.1:11434", "http://10.0.0.2:11434"}
    urls = discover_node_urls(backend, probe=lambda u: u in reachable)
    assert "http://127.0.0.1:11434" in urls  # local always
    assert "http://10.0.0.99:11434" in urls  # configured always (not probed)
    assert "http://10.0.0.1:11434" in urls  # scanned + reachable
    assert "http://10.0.0.3:11434" not in urls  # scanned but unreachable → dropped


# -- model-aware routing --------------------------------------------------------------


def test_routes_to_the_node_that_has_the_model() -> None:
    a = FleetFake("A", ["llama", "qwen"])
    b = FleetFake("B", ["gemma-only"])
    pool = ProviderPool([("A", a), ("B", b)])
    pool.refresh()  # populate inventories
    out = pool.chat("gemma-only", [], {}, priority=Priority.CHAT_CRITICAL)
    assert out.startswith("B")  # only B has it
    assert a.calls == [] and b.calls == ["gemma-only"]


def test_unknown_model_falls_back_optimistically() -> None:
    a = FleetFake("A", ["llama"])
    pool = ProviderPool([("A", a)])
    pool.refresh()
    # no node advertises 'mystery' → still attempts (optimistic), doesn't hard-fail to find it
    out = pool.chat("mystery", [], {}, priority=Priority.CHAT_CRITICAL)
    assert out.startswith("A")


# -- ranked fallback routing (heterogeneous fleet, DESIGN §4/§5) ----------------------


class PickyFake:
    """A node holding fixed models; optionally fails every chat (a 'down' node) — for failover."""

    def __init__(self, name: str, models: list[str], *, fail: bool = False) -> None:
        self.name = name
        self._models = models
        self.fail = fail
        self.calls: list[str] = []

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        if self.fail:
            from mimir.errors import ProviderError
            raise ProviderError(f"{self.name} down", transient=True)
        self.calls.append(model)
        return f"{self.name} answering with {model}"

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def model_details(self) -> list[ModelInfo]:
        return [ModelInfo(name=m, family="fam", params_b=1.0) for m in self._models]


def test_role_falls_back_to_the_next_model_across_nodes() -> None:
    # The heterogeneous case: node A has only gemma (and is down), node B has only qwen. The chat
    # role's chain is [gemma, qwen]; gemma's only node fails → routing falls to qwen on node B.
    a = PickyFake("A", ["gemma"], fail=True)
    b = PickyFake("B", ["qwen"])
    pool = ProviderPool([("A", a), ("B", b)], max_retries=0, sleep=lambda _: None)
    pool.refresh()
    gw = ModelGateway(pool, {"chat": RoleSpec("gemma")})
    gw.set_role_fallbacks("chat", ["gemma", "qwen"])
    out = gw.chat("chat", [], priority=Priority.CHAT_CRITICAL)
    assert "qwen" in out and b.calls == ["qwen"]  # served by the fallback model on the other node


def test_chain_is_pruned_to_reachable_models() -> None:
    # A chain may name a model no live node has (qualified elsewhere/earlier). The prune keeps only
    # models the cached inventory can run, so the first *reachable* preference wins immediately.
    b = PickyFake("B", ["qwen"])
    pool = ProviderPool([("B", b)], max_retries=0, sleep=lambda _: None)
    pool.refresh()
    gw = ModelGateway(pool, {"chat": RoleSpec("qwen")})
    gw.set_role_fallbacks("chat", ["gemma", "qwen"])  # gemma isn't installed anywhere
    out = gw.chat("chat", [], priority=Priority.CHAT_CRITICAL)
    assert b.calls == ["qwen"]  # gemma pruned, qwen served — no wasted attempt on the absent model
    assert out.endswith("qwen")


def test_pinned_role_is_never_substituted() -> None:
    # No fallback chain set → the role routes to exactly its model. A failure raises rather than
    # silently swapping in another model (a pin is the operator's explicit choice; DESIGN §4).
    import pytest

    from mimir.errors import ProviderError
    a = PickyFake("A", ["gemma"], fail=True)
    pool = ProviderPool([("A", a)], max_retries=0, sleep=lambda _: None)
    pool.refresh()
    gw = ModelGateway(pool, {"chat": RoleSpec("gemma")})  # pinned, no chain
    with pytest.raises(ProviderError):
        gw.chat("chat", [], priority=Priority.CHAT_CRITICAL)


# -- catalogue ------------------------------------------------------------------------


def test_scan_builds_catalogue(db_path: str) -> None:
    a = FleetFake("nodeA", ["llama", "qwen"])
    b = FleetFake("nodeB", ["gemma"])
    gateway = ModelGateway(ProviderPool([("nodeA", a), ("nodeB", b)]), {})
    with StorageGateway(db_path) as storage:
        result = scan_fleet(gateway, storage)
        assert result.nodes == 2
        assert result.models == 3
        report = fleet_report(storage)
        assert report["nodes"] == 2
        assert {m["model"] for m in report["by_node"]["nodeA"]} == {"llama", "qwen"}


def test_fleet_config_builds_pool(db_path: str, monkeypatch) -> None:
    """A [backend] config with no reachable nodes still builds (localhost endpoint, marked down)."""
    import mimir.brain as brain_mod

    # avoid real network: discovery returns just localhost (the prober marks it down if absent)
    monkeypatch.setattr(brain_mod, "discover_node_urls", lambda backend: ["http://127.0.0.1:11434"])
    cfg = Config(
        storage_path=db_path,
        roles={"chat": RoleSpec("m"), "bake": RoleSpec("m"), "reasoning": RoleSpec("m")},
        provider=ProviderSpec(type="ollama"),
        backend=BackendConfig(lan_backend=False, refresh_interval_s=0),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )
    m = brain_mod.Mimir(cfg)
    try:
        stats = m._model.get_stats()
        assert "http://127.0.0.1:11434" in stats["endpoints"]
    finally:
        m.close()
