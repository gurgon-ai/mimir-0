"""Executable spec for `auto` model routing (DESIGN §4): resolution hierarchy, enable/disable.

pin > measured-best > approved-family heuristic > any reachable model, with disabled models
vetoed at every level. "As automatic as possible, but configurable."
"""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition.fleet import (
    fleet_model_pool,
    recommend_roles,
    resolve_auto_model,
    roster_for,
)
from mimir.config import AUTO_MODEL, Config, ProviderSpec, RoleSpec, _parse_roles
from mimir.embed.base import EmbeddingMode
from mimir.model.gateway import ModelGateway
from mimir.model.providers.mock import MockProvider
from mimir.storage.gateway import StorageGateway
from mimir.storage.models import CatalogueEntry
from mimir.storage.repo import (
    disabled_models,
    disabled_nodes,
    replace_catalogue,
    set_model_enabled,
    set_node_enabled,
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


def test_disabled_node_drops_its_models_from_recommendations(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        # Same model on two nodes; a LAN-only model on the fast node. Score both.
        replace_catalogue(sg, [
            _cat("gemma:7b", "gemma", 7.0, node="http://local:11434"),
            _cat("gemma:7b", "gemma", 7.0, node="http://edge:11434"),
            _cat("qwen2.5:14b", "qwen", 14.0, node="http://edge:11434"),  # only on the edge
        ])
        for m in ("gemma:7b", "qwen2.5:14b"):
            update_catalogue_scores(
                sg, m, return_time=1.0, quality=0.9, talk=1.0, tools=0.9, code=0.9,
                coherence=None, discipline=1.0, epistemics=1.0, reasoning=1.0,
            )
        # With both nodes enabled, the edge-only qwen is a candidate for chat.
        recs = recommend_roles(sg, disabled_nodes=disabled_nodes(sg))
        assert any(r and r["model"] == "qwen2.5:14b" for r in recs.values())

        # Veto the edge node: qwen (edge-only) vanishes from every recommendation; gemma survives
        # (it's also on the local node).
        set_node_enabled(sg, "http://edge:11434", False)
        recs2 = recommend_roles(sg, disabled_nodes=disabled_nodes(sg))
        chosen = {r["model"] for r in recs2.values() if r}
        assert "qwen2.5:14b" not in chosen
        assert recs2["chat"] is not None and recs2["chat"]["node"] == "http://local:11434"
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


# -- the second lineup: loose roles + the harness staffing query (DESIGN §5a) ---------


def _score(sg: StorageGateway, model: str, *, quality: float, discipline: float = 1.0,
           reasoning: float = 1.0) -> None:
    update_catalogue_scores(
        sg, model, return_time=1.0, quality=quality, talk=1.0, tools=1.0, code=1.0,
        coherence=None, discipline=discipline, epistemics=1.0, reasoning=reasoning,
    )


def test_background_and_council_are_not_discipline_gated(db_path: str) -> None:
    # The second lineup is reasoning-gated, NOT discipline-gated: a capable model that "leaks" the
    # identity (low discipline) is barred from chat/reasoning but staffs background work + a council
    # seat. The bridge that lets the harness use the big/undisciplined models the voice can't.
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [_cat("creative:13b", "qwen", 13.0)])
        _score(sg, "creative:13b", quality=0.9, discipline=0.2)  # reasoning 1.0, discipline 0.2
        recs = recommend_roles(sg)
        assert recs["chat"] is None and recs["reasoning"] is None      # discipline 0.20 < 0.50
        assert recs["background"]["model"] == "creative:13b"           # reasoning floor only
        assert recs["council"]["model"] == "creative:13b"
        # The harness query staffs it too — board eligibility and the roster can't disagree (§10).
        assert [p["model"] for p in roster_for(sg, "background")] == ["creative:13b"]
        assert "creative:13b" in [m["model"] for m in roster_for(sg, "council", n=5)]
    finally:
        sg.close()


def test_roster_for_returns_up_to_n_best_first(db_path: str) -> None:
    # "Give me N for role R": single-best roles return up to n role-eligible models, quality first.
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [_cat("a:8b", "qwen", 8.0), _cat("b:8b", "gemma", 8.0),
                               _cat("c:8b", "mistral", 8.0)])
        _score(sg, "a:8b", quality=0.95)
        _score(sg, "b:8b", quality=0.90)
        _score(sg, "c:8b", quality=0.85)
        assert [p["model"] for p in roster_for(sg, "background", n=2)] == ["a:8b", "b:8b"]
    finally:
        sg.close()


def test_roster_for_council_is_a_diverse_pool_not_top_n(db_path: str) -> None:
    # Pool roles route to the diversity picker: families before depth (contrast single-best roles).
    sg = StorageGateway(db_path)
    try:
        replace_catalogue(sg, [_cat("q1", "qwen", 8.0), _cat("q2", "qwen", 8.0),
                               _cat("g1", "gemma", 8.0)])
        _score(sg, "q1", quality=0.95)
        _score(sg, "q2", quality=0.93)
        _score(sg, "g1", quality=0.80)
        assert [p["family"] for p in roster_for(sg, "council", n=2)] == ["qwen", "gemma"]
    finally:
        sg.close()


def test_roster_for_rejects_an_unknown_role(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        with pytest.raises(ValueError):
            roster_for(sg, "nonsense")
    finally:
        sg.close()


def test_brain_harness_can_staff_itself(db_path: str) -> None:
    # The facade the harness calls: background_model() + council_members(), honouring vetoes.
    cfg = Config(
        storage_path=db_path,
        roles={
            "chat": RoleSpec("mock-a"),
            "bake": RoleSpec("mock-a"),
            "reasoning": RoleSpec("mock-a"),
        },
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )
    m = Mimir(cfg)
    try:
        replace_catalogue(m._storage, [_cat("a:8b", "qwen", 8.0), _cat("b:8b", "gemma", 8.0)])
        _score(m._storage, "a:8b", quality=0.95)
        _score(m._storage, "b:8b", quality=0.90)
        assert m.background_model() == "a:8b"
        assert set(m.council_members(n=2)) == {"a:8b", "b:8b"}  # diverse: both families seated
        m.set_model_enabled("a:8b", False)                       # a user veto removes it everywhere
        assert m.background_model() == "b:8b"
        assert m.council_members(n=2) == ["b:8b"]
    finally:
        m.close()


def test_auto_role_gets_a_ranked_fallback_chain(db_path: str) -> None:
    # An auto role resolves to an ORDERED chain of acceptable models (best first), not just one — so
    # a heterogeneous fleet still serves the role across nodes. The chain is the qualified ranking,
    # pruned to reachable models (DESIGN §4/§5).
    cfg = Config(
        storage_path=db_path,
        roles={
            "chat": RoleSpec(AUTO_MODEL),
            "bake": RoleSpec("mock-a"),
            "reasoning": RoleSpec("mock-a"),
        },
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )
    m = Mimir(cfg)
    try:
        # Catalogue two chat-eligible models (named to match what the mock advertises, so they're
        # "reachable"); mock-a outscores mock-b.
        replace_catalogue(m._storage, [_cat("mock-a", "fam", 8.0), _cat("mock-b", "fam", 8.0)])
        _score(m._storage, "mock-a", quality=0.95)
        _score(m._storage, "mock-b", quality=0.90)
        m._resolve_auto_roles()
        assert m._model.fallbacks_view()["chat"] == ["mock-a", "mock-b"]  # ranked, best first
    finally:
        m.close()


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


def test_manual_role_pin_persists_across_restart(mock_config: Config) -> None:
    # A manual model selection must survive a restart (it's saved to kv, re-applied on boot).
    with Mimir(mock_config) as m:
        m.set_role("chat", "hand-picked-model")
        assert m._model.roles_view()["chat"].model == "hand-picked-model"
    with Mimir(mock_config) as m2:                  # same DB → pin restored
        assert m2._model.roles_view()["chat"].model == "hand-picked-model"
        assert "chat" not in m2._auto_roles         # a pin leaves the auto set
