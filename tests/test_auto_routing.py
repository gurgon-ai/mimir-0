"""Executable spec for `auto` model routing (DESIGN §4): resolution hierarchy, enable/disable.

pin > measured-best > approved-family heuristic > any reachable model, with disabled models
vetoed at every level. "As automatic as possible, but configurable."
"""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.fleet import fleet_model_pool, resolve_auto_model
from mimir.config import AUTO_MODEL, Config, ProviderSpec, RoleSpec, _parse_roles
from mimir.embed.base import EmbeddingMode
from mimir.model.gateway import ModelGateway
from mimir.model.providers.mock import MockProvider
from mimir.storage.gateway import StorageGateway
from mimir.storage.models import CatalogueEntry
from mimir.storage.repo import (
    disabled_models,
    replace_catalogue,
    set_model_enabled,
    update_catalogue_scores,
)


def _cat(model: str, family: str, params_b: float, node: str = "http://n1:11434") -> CatalogueEntry:
    return CatalogueEntry(node=node, model=model, family=family, params_b=params_b, scanned_at=1.0)


# -- config: auto sentinel ------------------------------------------------------------


def test_role_without_model_defaults_to_auto() -> None:
    roles = _parse_roles({"chat": {"temperature": 0.5}, "bake": {"model": "x"}})
    assert roles["chat"].model == AUTO_MODEL
    assert roles["chat"].params == {"temperature": 0.5}
    assert roles["bake"].model == "x"  # an explicit pin is preserved


# -- enable / disable persistence -----------------------------------------------------


def test_enable_disable_round_trips(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        assert disabled_models(sg) == set()
        set_model_enabled(sg, "gemma:7b", False)
        assert disabled_models(sg) == {"gemma:7b"}
        set_model_enabled(sg, "gemma:7b", True)  # re-enable (upsert)
        assert disabled_models(sg) == set()
    finally:
        sg.close()


# -- resolution hierarchy -------------------------------------------------------------


def test_approved_family_wins_the_first_round(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        # No benchmark scores yet → heuristic. Same size, but only one is an approved family.
        replace_catalogue(
            sg, [_cat("llama3.1:8b", "llama", 8.0), _cat("oddball:8b", "oddfam", 8.0)]
        )
        got = resolve_auto_model(sg, "chat", available={"llama3.1:8b", "oddball:8b"})
        assert got == "llama3.1:8b"
    finally:
        sg.close()


def test_recommended_model_wins_heuristic_over_approved(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        # No benchmark yet → heuristic. Both are the approved `gemma`/`qwen` families, but only
        # qwen2.5:3b is registry-recommended for chat (gemma3:4b is deliberately excluded). The
        # recommended one wins even though gemma3:4b is closer to the ideal chat size — this is the
        # out-of-box guard that stops `auto` landing on the known-weak model.
        replace_catalogue(sg, [_cat("gemma3:4b", "gemma", 4.0), _cat("qwen2.5:3b", "qwen", 3.0)])
        got = resolve_auto_model(sg, "chat", available={"gemma3:4b", "qwen2.5:3b"})
        assert got == "qwen2.5:3b"
    finally:
        sg.close()


def test_only_weak_model_present_still_resolves(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        # If the ONLY reachable model isn't recommended, auto still yields something runnable
        # (pre-benchmark we can't know it's bad; the wizard/docs steer users to pull a good one).
        replace_catalogue(sg, [_cat("gemma3:4b", "gemma", 4.0)])
        assert resolve_auto_model(sg, "chat", available={"gemma3:4b"}) == "gemma3:4b"
    finally:
        sg.close()


def test_measured_best_overrides_the_heuristic(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [_cat("qwen2.5:14b", "qwen", 14.0), _cat("gemma:7b", "gemma", 7.0)])
        # gemma benchmarks well and is disciplined; the bigger qwen scores worse.
        update_catalogue_scores(
            sg, "gemma:7b", return_time=0.5, quality=0.9,
            talk=1.0, tools=0.9, code=0.9, coherence=None, discipline=1.0, epistemics=1.0,
            reasoning=1.0,
        )
        update_catalogue_scores(
            sg, "qwen2.5:14b", return_time=2.0, quality=0.6,
            talk=0.6, tools=0.5, code=0.5, coherence=None, discipline=0.5, epistemics=0.5,
            reasoning=0.5,
        )
        # Measured-best wins over the larger heuristic pick.
        assert resolve_auto_model(sg, "chat", available={"qwen2.5:14b", "gemma:7b"}) == "gemma:7b"
    finally:
        sg.close()


def test_disabled_model_is_vetoed(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [_cat("llama3.1:8b", "llama", 8.0), _cat("gemma:7b", "gemma", 7.0)])
        # Unvetoed, chat picks llama (8B is closer to the ideal than 7B).
        avail = {"llama3.1:8b", "gemma:7b"}
        assert resolve_auto_model(sg, "chat", available=avail) == "llama3.1:8b"
        set_model_enabled(sg, "llama3.1:8b", False)
        m = resolve_auto_model(sg, "chat", available=avail, disabled=disabled_models(sg))
        assert m == "gemma:7b"  # the user's veto pushes it to the next-best
    finally:
        sg.close()


def test_falls_back_to_any_reachable_with_empty_catalogue(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        assert resolve_auto_model(sg, "chat", available=set()) is None
        assert resolve_auto_model(sg, "chat", available={"somemodel:7b"}) == "somemodel:7b"
    finally:
        sg.close()


# -- model pool view ------------------------------------------------------------------


def test_model_pool_flags_passed_disabled_and_excludes_embeds(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [
            _cat("gemma:7b", "gemma", 7.0),
            _cat("llama3.1:8b", "llama", 8.0),
            _cat("nomic-embed-text", "nomic", 0.1),
        ])
        update_catalogue_scores(
            sg, "gemma:7b", return_time=0.5, quality=0.9,
            talk=1.0, tools=1.0, code=1.0, coherence=None, discipline=1.0,
        )
        set_model_enabled(sg, "llama3.1:8b", False)
        pool = fleet_model_pool(
            sg, disabled=disabled_models(sg),
            active_roles={"chat": "gemma:7b"}, auto_roles={"chat"},
        )
        models = {m["model"]: m for m in pool["models"]}
        assert "nomic-embed-text" not in models  # embedding models are not routable chat models
        assert models["gemma:7b"]["passed"] is True and models["gemma:7b"]["enabled"] is True
        assert models["gemma:7b"]["roles"] == ["chat"]
        assert models["llama3.1:8b"]["enabled"] is False
        assert models["llama3.1:8b"]["passed"] is False  # never benchmarked
    finally:
        sg.close()


# -- gateway stop-gap + brain wiring --------------------------------------------------


def test_gateway_stopgaps_auto_to_a_reachable_model() -> None:
    gw = ModelGateway(MockProvider(), {"chat": RoleSpec(AUTO_MODEL)})
    spec = gw._role("chat")
    assert spec.model in {"mock-a", "mock-b", "mock-c"}  # not the literal "auto"


def test_brain_resolves_auto_roles_at_boot(db_path: str) -> None:
    cfg = Config(
        storage_path=db_path,
        roles={
            "chat": RoleSpec(AUTO_MODEL),
            "bake": RoleSpec("mock-b"),  # an explicit pin
            "reasoning": RoleSpec(AUTO_MODEL),
        },
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )
    m = Mimir(cfg)
    try:
        roles = m._model.roles_view()
        assert roles["chat"].model in {"mock-a", "mock-b", "mock-c"}
        assert roles["reasoning"].model in {"mock-a", "mock-b", "mock-c"}
        assert roles["bake"].model == "mock-b"  # pin untouched
    finally:
        m.close()
