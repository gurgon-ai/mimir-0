"""Executable spec for identity anchors and the initialization interview."""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition.identity import ANCHOR_KEYS, render_anchors
from mimir.config import Config
from mimir.interview import run_interview
from mimir.storage.gateway import StorageGateway
from mimir.storage.repo import get_identity_anchors, set_identity_anchor


def test_anchor_upsert_updates_not_duplicates(db_path: str) -> None:
    with StorageGateway(db_path) as gw:
        set_identity_anchor(gw, "name", "First")
        set_identity_anchor(gw, "name", "Second")
        assert get_identity_anchors(gw) == {"name": "Second"}


def test_establish_and_pending(brain: Mimir) -> None:
    assert {k for k, _ in brain.pending_identity_questions()} == set(ANCHOR_KEYS)
    brain.establish_identity({"name": "Mimir", "location": "a home server"})
    assert brain.identity_anchors() == {"name": "Mimir", "location": "a home server"}
    assert {k for k, _ in brain.pending_identity_questions()} == set(ANCHOR_KEYS) - {
        "name",
        "location",
    }


def test_blank_and_unknown_answers_ignored(brain: Mimir) -> None:
    brain.establish_identity({"name": "   ", "bogus_key": "x"})
    assert brain.identity_anchors() == {}


def test_render_anchors_first_person() -> None:
    assert render_anchors({}) is None
    text = render_anchors({"name": "Mimir", "purpose": "to remember"})
    assert text is not None
    assert "My name is Mimir." in text
    assert "My purpose is to remember." in text


def test_anchors_injected_into_self_model_section(brain: Mimir) -> None:
    brain.establish_identity({"name": "Mimir", "purpose": "to remember"})
    r = brain.turn("hello", user="greg")
    sm = next((s for s in r.context.sections if s.name == "self_model"), None)
    assert sm is not None
    assert "My name is Mimir." in sm.body
    assert "My purpose is to remember." in sm.body


def test_config_anchors_established_at_boot(mock_config: Config) -> None:
    mock_config.identity_anchors = {"name": "Helios", "operator": "the lab"}
    with Mimir(mock_config) as m:
        assert m.identity_anchors() == {"name": "Helios", "operator": "the lab"}
        r = m.turn("hi")
        sm = next(s for s in r.context.sections if s.name == "self_model")
        assert "My name is Helios." in sm.body


def test_run_interview_collects_answers(brain: Mimir, monkeypatch: pytest.MonkeyPatch) -> None:
    # One answer per anchor, in interview order.
    answers = iter([f"answer_{k}" for k in ANCHOR_KEYS])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    result = run_interview(brain)
    assert result == {k: f"answer_{k}" for k in ANCHOR_KEYS}


def test_run_interview_noop_when_already_established(brain: Mimir) -> None:
    brain.establish_identity({k: f"v_{k}" for k in ANCHOR_KEYS})
    # All anchors set → no questions; run_interview must not call input().
    result = run_interview(brain)
    assert result == {k: f"v_{k}" for k in ANCHOR_KEYS}


def test_run_interview_revise_updates_and_keeps(
    brain: Mimir, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain.establish_identity({k: f"old_{k}" for k in ANCHOR_KEYS})

    # In revise mode every anchor is re-asked; blank keeps the current value. Change only `name`.
    def fake_input(prompt: str = "") -> str:
        return "New Name" if prompt.strip().startswith("What is your name?") else ""

    monkeypatch.setattr("builtins.input", fake_input)
    result = run_interview(brain, revise=True)
    assert result["name"] == "New Name"
    assert result["purpose"] == "old_purpose"  # untouched anchors retained
