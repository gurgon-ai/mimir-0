"""Executable spec for session history + restore (DESIGN §3a): a durable conversation log that
survives a restart, replays to the model as real messages, and restores the UI on load."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.config import Config
from mimir.storage.gateway import StorageGateway
from mimir.storage.repo import recent_conversation, record_conversation_turn


def test_conversation_log_round_trips_and_prunes(db_path: str) -> None:
    sg = StorageGateway(db_path)
    try:
        for i in range(5):
            record_conversation_turn(sg, user="alex", user_text=f"q{i}", reply=f"a{i}", keep=3)
        turns = recent_conversation(sg, user="alex")
        assert [t["user_text"] for t in turns] == ["q2", "q3", "q4"]  # oldest→newest, pruned to 3
        assert turns[-1]["reply"] == "a4"
    finally:
        sg.close()


def test_history_persists_across_a_restart(mock_config: Config) -> None:
    # The "restore" guarantee: a new Mimir on the same DB still has the conversation.
    m1 = Mimir(mock_config)
    m1.turn("the north gate sticks in the cold", user="operator")
    m1.wait_for_sentinel()
    m1.close()
    m2 = Mimir(mock_config)
    try:
        h = m2.history(user="operator")
        assert any("north gate sticks" in t["user_text"] for t in h)
    finally:
        m2.close()


def test_recent_turns_replay_to_the_model_as_messages(brain: Mimir) -> None:
    brain.turn("first thing I said", user="operator")
    msgs = brain._history_messages("operator", brain._resolve_session())
    assert len(msgs) >= 2
    assert msgs[0] == {"role": "user", "content": "first thing I said"}
    assert msgs[1]["role"] == "assistant"  # the reply, as a real assistant message


def test_new_session_starts_a_clean_context(brain: Mimir) -> None:
    brain.turn("the gate is broken", user="operator")
    brain.start_new_session()
    # A fresh conversation replays nothing from the previous one.
    assert brain._history_messages("operator", brain._resolve_session()) == []
    sessions = brain.sessions(user="operator")
    assert sessions and "the gate is broken" in (sessions[-1]["summary"] or "")


def test_intercept_turns_are_logged_too(brain: Mimir) -> None:
    brain.turn("what day is it?", user="operator")  # the deterministic time intercept
    assert any("what day is it" in t["user_text"] for t in brain.history(user="operator"))
