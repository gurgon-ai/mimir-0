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
