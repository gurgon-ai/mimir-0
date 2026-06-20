"""Extensibility ports (docs/EXTENSIBILITY.md): connectors attach without forking core. Phase 1 —
the sensory port (context sources), the backend provider registry, and overridable personas."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir, build_provider, register_provider
from mimir.cognition.tools import (
    ActionContext,
    Tool,
    ToolCall,
    ToolRegistry,
    dispatch,
    parse_tool_calls,
)
from mimir.config import Config, ProviderSpec
from mimir.context.sections import Section, SectionTier
from mimir.model.providers.mock import MockProvider


def _echo() -> Tool:
    return Tool(name="echo", description="echo the text back",
                handler=lambda a, c: f"echo:{a.get('text', '')}",
                schema={"text": {"required": True}}, keywords=("echo",), always=True)


def _light() -> Tool:
    return Tool(name="set_light", description="turn a light on/off",
                handler=lambda a, c: f"light {a.get('state')}", state_changing=True,
                schema={"state": {"required": True}}, always=True)


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


# -- ② motor port: the guarded dispatcher --------------------------------------------------------

def _reg(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def test_dispatch_runs_a_read_only_tool() -> None:
    call = dispatch(_reg(_echo()), ToolCall("echo", {"text": "hi"}), ActionContext(trusted=False))
    assert call.status == "ok" and call.result == "echo:hi"


def test_dispatch_blocks_state_changing_for_untrusted_speaker() -> None:
    call = dispatch(_reg(_light()), ToolCall("set_light", {"state": "on"}),
                    ActionContext(speaker_kind="ai_peer", trusted=False))
    assert call.status == "blocked"  # a peer/guest cannot move the hands


def test_dispatch_allows_state_changing_for_trusted_speaker() -> None:
    call = dispatch(_reg(_light()), ToolCall("set_light", {"state": "on"}),
                    ActionContext(trusted=True))
    assert call.status == "ok" and call.result == "light on"


def test_dispatch_unknown_and_invalid_and_raising() -> None:
    reg = _reg(_echo(), Tool(name="boom", description="x", handler=lambda a, c: 1 / 0))
    assert dispatch(reg, ToolCall("nope", {}), ActionContext()).status == "unknown"
    assert dispatch(reg, ToolCall("echo", {}), ActionContext()).status == "error"  # missing arg
    blown = dispatch(reg, ToolCall("boom", {}), ActionContext())
    assert blown.status == "error" and "failed" in blown.result  # handler raise → error string


def test_parse_tool_calls() -> None:
    calls = parse_tool_calls('sure: <TOOL name="echo" args={"text": "x"}> done')
    assert len(calls) == 1 and calls[0].tool == "echo" and calls[0].args == {"text": "x"}
    assert parse_tool_calls("no calls here") == []


def test_tool_round_trip_through_a_turn(brain: Mimir, monkeypatch: pytest.MonkeyPatch) -> None:
    brain.register_tool(_echo())

    def fake_chat(role: str, messages: list, **k: object) -> str:
        sys = messages[0]["content"]
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if role == "chat" and "Tools you may call" in sys and not has_assistant:
            return '<TOOL name="echo" args={"text": "ping"}>'   # the model invokes the tool
        return "Final answer, tool used."                        # the re-invoke (+ any other call)

    monkeypatch.setattr(brain._model, "chat", fake_chat)
    r = brain.turn("please echo ping", user="alex")
    assert r.reply == "Final answer, tool used."
    assert len(r.actions) == 1
    assert r.actions[0].tool == "echo" and r.actions[0].status == "ok"
    assert r.actions[0].result == "echo:ping"
