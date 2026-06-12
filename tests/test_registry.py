"""Executable spec for the recommended-models registry (INFERENCE_ENGINE.md §4)."""

from __future__ import annotations

from mimir.cognition.registry import (
    is_recommended,
    is_trusted_judge,
    recommended_models,
    registry_version,
)


def test_registry_loads_as_data() -> None:
    assert registry_version() >= 1
    entries = recommended_models()
    assert entries and {e.family for e in entries} >= {"gemma", "qwen", "llama"}


def test_known_weak_gemma_excluded_strong_included() -> None:
    # The headline guard: gemma3:4b (leaks tags, ignores tiers) is NOT a recommended chat model;
    # the strong gemmas are. This is what stops `auto` landing on it out of the box.
    assert is_recommended("gemma4:e4b", "chat")
    assert is_recommended("gemma3:12b", "chat")
    assert not is_recommended("gemma3:4b", "chat")
    assert not is_recommended("gemma3:1b", "chat")


def test_role_specific_recommendation() -> None:
    # llama is fine for chat but excluded from tools (weak tool-call JSON).
    assert is_recommended("llama3.2:3b", "chat")
    assert not is_recommended("llama3.2:3b", "tools")
    # An unknown model isn't recommended for anything.
    assert not is_recommended("mysterymodel:7b")


def test_trusted_judges() -> None:
    assert is_trusted_judge("gemma4:e4b") and is_trusted_judge("qwen2.5:3b")
    assert not is_trusted_judge("granite3.1-moe:3b")  # judge_ok = false
    assert not is_trusted_judge("gemma3:4b")  # not recommended at all
