"""Extensibility ports (docs/EXTENSIBILITY.md): connectors attach without forking core. Phase 1 —
the sensory port (context sources), the backend provider registry, and overridable personas."""

from __future__ import annotations

from mimir.brain import Mimir, build_provider, register_provider
from mimir.config import Config, ProviderSpec
from mimir.context.sections import Section, SectionTier
from mimir.model.providers.mock import MockProvider


class _ClockSource:
    """A trivial reference context source — the kind of thing a 'hand' connector provides."""

    name = "clock"
    tier = SectionTier.MEDIUM
    budget_tokens = 64

    def __init__(self, stamp: str) -> None:
        self._stamp = stamp

    def build(self, query: str, user: str | None) -> Section | None:
        return Section(name=self.name, title="The current time:", body=self._stamp, tier=self.tier)


def test_context_source_folds_into_the_prompt(brain: Mimir) -> None:
    brain.register_context_source(_ClockSource("2026-06-20 09:00"))
    r = brain.turn("hello", user="alex")
    assert "2026-06-20 09:00" in r.context.prompt  # the external section reached the prompt
    assert any(s["name"] == "clock" for s in r.context.introspect()["sections"])


def test_context_source_injected_at_construction(mock_config: Config) -> None:
    with Mimir(mock_config, context_sources=[_ClockSource("high noon")]) as m:
        assert "high noon" in m.turn("hi", user="alex").context.prompt


def test_register_replaces_a_same_named_source(brain: Mimir) -> None:
    brain.register_context_source(_ClockSource("ZULU-stamp"))
    brain.register_context_source(_ClockSource("YANKEE-stamp"))  # same name → replaces
    prompt = brain.turn("hello", user="alex").context.prompt
    assert "YANKEE-stamp" in prompt and "ZULU-stamp" not in prompt


def test_a_faulty_context_source_degrades_not_breaks(brain: Mimir) -> None:
    class _Boom:
        name, tier, budget_tokens = "boom", SectionTier.LOW, 16

        def build(self, query: str, user: str | None) -> Section | None:
            raise RuntimeError("connector blew up")

    brain.register_context_source(_Boom())
    assert brain.turn("hello", user="alex").reply  # the turn still completes (§10)


def test_provider_registry_is_open(brain: Mimir) -> None:
    register_provider("myfake", lambda spec: MockProvider())
    assert isinstance(build_provider(ProviderSpec(type="myfake")), MockProvider)


def test_council_personas_are_overridable(mock_config: Config) -> None:
    custom = [("oracle", "answer plainly"), ("gadfly", "disagree on principle")]
    with Mimir(mock_config, council_personas=custom) as m:
        result = m.deliberate("breadth or depth?")
        assert {p.persona for p in result.positions} == {"oracle", "gadfly"}
