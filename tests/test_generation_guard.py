"""Executable spec for the store-boundary guard and the lessons it carries (2026-10-31).

Nothing the model failed to produce may be stored, folded on, or filed as an argument. Working
memory
is rebuilt from sources, never from its own previous summary. The operational num_ctx reaches every
chat path. An empty body with done_reason == "length" under thinking is retried without thinking.
Embeddings are batched and shape-checked. The time intercept counts by position. A handler's refusal
is reported as blocked, not ok.
"""

from __future__ import annotations

import pytest

from mimir.brain import Mimir
from mimir.cognition import sentinel as sentinel_mod
from mimir.cognition.generation import (
    EmptyGeneration,
    is_valid_generation,
    rejection_reason,
    require_generation,
)
from mimir.cognition.temporal import answer_time_query
from mimir.cognition.working_memory import (
    SOURCE_WINDOW,
    recent_exchanges,
    record_exchange,
    synthesize_working_memory,
)
from mimir.storage.models import MemoryKind
from mimir.storage.repo import count_memories, recent_by_kind

# -- the guard itself --------------------------------------------------------------------


def test_guard_rejects_empty_and_placeholders_and_accepts_content() -> None:
    assert rejection_reason("") == "empty"
    assert rejection_reason("   \n") == "empty"
    assert rejection_reason(None) == "no text"
    assert rejection_reason("[unavailable: timeout]").startswith("placeholder")
    assert rejection_reason("Error: provider down").startswith("placeholder")
    assert rejection_reason("ok").startswith("too short")
    assert is_valid_generation("The garlic goes in around October, per the operator.")
    with pytest.raises(EmptyGeneration):
        require_generation("", what="test note")
    assert require_generation("  a usable sentence  ", what="x") == "a usable sentence"


# -- working memory: rejected fold saves nothing and archives nothing ------------------------


def test_an_empty_fold_saves_nothing_and_keeps_every_exchange(brain: Mimir, monkeypatch) -> None:
    for i in range(10):
        record_exchange(brain._storage, user="g", user_text=f"point {i}", reply="noted")
    monkeypatch.setattr(brain._model, "chat", lambda *a, **k: "")
    assert (
        synthesize_working_memory(brain._model, brain._storage, fold_threshold=10, keep_recent=4)
        is None
    )
    assert count_memories(brain._storage, kind=MemoryKind.EXCHANGE) == 10  # nothing folded
    assert count_memories(brain._storage, kind=MemoryKind.WORKING_MEMORY) == 0  # nothing saved
    assert count_memories(brain._storage, kind=MemoryKind.EXCHANGE, include_archived=True) == 10


def test_a_fold_archives_its_sources_and_never_reads_its_own_summary(
    brain: Mimir, monkeypatch
) -> None:
    briefs: list[str] = []

    def fake_chat(role: str, messages: list, **k: object) -> str:
        briefs.append(messages[-1]["content"])
        return f"Summary #{len(briefs)}: the user made several points."

    monkeypatch.setattr(brain._model, "chat", fake_chat)

    for i in range(10):
        record_exchange(brain._storage, user="g", user_text=f"point {i}", reply="noted")
    first = synthesize_working_memory(
        brain._model, brain._storage, fold_threshold=10, keep_recent=4
    )
    assert first is not None
    assert count_memories(brain._storage, kind=MemoryKind.EXCHANGE) == 4  # raw
    assert (
        count_memories(brain._storage, kind=MemoryKind.EXCHANGE, include_archived=True) == 10
    )  # kept
    archived = [
        e for e in recent_exchanges(brain._storage, 100, include_archived=True) if e.archived
    ]
    assert len(archived) == 6 and all("point" in e.text for e in archived)

    # second fold: the brief carries the archived sources and the new ones, NOT the prior summary
    for i in range(10, 16):
        record_exchange(brain._storage, user="g", user_text=f"point {i}", reply="noted")
    second = synthesize_working_memory(
        brain._model, brain._storage, fold_threshold=10, keep_recent=4
    )
    assert second is not None
    assert "Summary #1" not in briefs[-1]  # no self-feeding
    assert "Previous working memory" not in briefs[-1]
    assert "point 0" in briefs[-1] and "point 11" in briefs[-1]  # rebuilt from sources, old and new
    total_archived = count_memories(
        brain._storage, kind=MemoryKind.EXCHANGE, include_archived=True
    ) - count_memories(brain._storage, kind=MemoryKind.EXCHANGE)
    assert total_archived <= SOURCE_WINDOW


def test_recent_by_kind_hides_archived_by_default(brain: Mimir) -> None:
    for i in range(3):
        record_exchange(brain._storage, user="g", user_text=f"e{i}", reply="r")
    from mimir.storage.repo import archive_memories

    rows = recent_by_kind(brain._storage, MemoryKind.EXCHANGE, limit=10)
    archive_memories(brain._storage, [rows[0].id])
    assert len(recent_by_kind(brain._storage, MemoryKind.EXCHANGE, limit=10)) == 2
    assert (
        len(recent_by_kind(brain._storage, MemoryKind.EXCHANGE, limit=10, include_archived=True))
        == 3
    )


# -- sentinel and council: a failed voice is not a note and not an argument -------------------


def test_an_empty_sentinel_reflection_is_an_error_not_a_note(brain: Mimir, monkeypatch) -> None:
    monkeypatch.setattr(brain._model, "chat", lambda *a, **k: "   ")
    with pytest.raises(EmptyGeneration):
        sentinel_mod.run_sentinel(
            brain._model, brain._storage, user="g", turn_text="hi", reply="hello"
        )
    assert count_memories(brain._storage, kind=MemoryKind.SENTINEL_NOTE) == 0


def test_council_drops_an_unavailable_voice_and_refuses_an_empty_verdict(
    brain: Mimir, monkeypatch
) -> None:
    from mimir.cognition import council as C
    from mimir.cognition.council import Position

    spoke = Position(persona="A", model="m", text="A real position with some substance to it.")
    dead = Position(persona="B", model="m", text="[unavailable: timeout]")
    assert C._others([spoke, dead], 0) == []  # the dead voice is not on the floor
    assert C._others([spoke, dead], 1) == [("A", spoke.text)]
    seen: list[str] = []

    def fake_chat(role: str, messages: list, **k: object) -> str:
        seen.append(messages[-1]["content"])
        return ""  # synthesis returns nothing

    monkeypatch.setattr(brain._model, "chat", fake_chat)
    with pytest.raises(EmptyGeneration):
        C._synthesize(brain._model, "q?", [spoke, dead], [])
    assert "[unavailable" not in seen[-1] and "A real position" in seen[-1]
    with pytest.raises(EmptyGeneration):
        C._synthesize(brain._model, "q?", [dead], [])  # nobody spoke: nothing to synthesise


# -- gateway: the operational window reaches the bypass paths ---------------------------------


def test_operational_num_ctx_reaches_chat_with_model(brain: Mimir, monkeypatch) -> None:
    seen: dict = {}

    def fake_pool_chat(model, messages, params, priority=None, max_retries=None):
        seen.update(params or {})
        return "ok text here"

    monkeypatch.setattr(brain._model._pool, "chat", fake_pool_chat)
    brain._model.set_operational_num_ctx(12345)
    brain._model.chat_with_model("mock", [{"role": "user", "content": "x"}])
    assert seen.get("num_ctx") == 12345
    brain._model.chat_with_model(
        "mock", [{"role": "user", "content": "x"}], params={"num_ctx": 777}
    )
    assert seen.get("num_ctx") == 777  # an explicit per-call pin still wins


# -- provider: think + empty + length → one retry without thinking; embed batching -------------


def test_empty_body_under_thinking_is_retried_without_thinking(monkeypatch) -> None:
    from mimir.model.providers.ollama import OllamaProvider

    prov = OllamaProvider("http://x")
    posts: list[dict] = []

    def fake_post(path, payload, timeout=None):
        posts.append(payload)
        if payload.get("think"):
            return {"message": {"content": ""}, "done_reason": "length", "eval_count": 2400}
        return {"message": {"content": "an answer"}, "done_reason": "stop", "eval_count": 12}

    monkeypatch.setattr(prov, "_post", fake_post)
    text, n, _ = prov.chat_timed("m", [{"role": "user", "content": "q"}], {"think": True})
    assert text == "an answer" and len(posts) == 2 and posts[1]["think"] is False
    posts.clear()
    text, _, _ = prov.chat_timed("m", [{"role": "user", "content": "q"}], {"think": False})
    assert len(posts) == 1  # no retry when thinking was off


def test_embeddings_are_batched_and_shape_checked(monkeypatch) -> None:
    from mimir.errors import ProviderError
    from mimir.model.providers.ollama import EMBED_BATCH, OllamaProvider

    prov = OllamaProvider("http://x")
    sizes: list[int] = []

    def fake_post(path, payload, timeout=None):
        sizes.append(len(payload["input"]))
        return {"embeddings": [[0.1, 0.2] for _ in payload["input"]]}

    monkeypatch.setattr(prov, "_post", fake_post)
    out = prov.embed("e", [f"t{i}" for i in range(EMBED_BATCH * 2 + 5)])
    assert len(out) == EMBED_BATCH * 2 + 5 and sizes == [EMBED_BATCH, EMBED_BATCH, 5]

    def bad_post(path, payload, timeout=None):
        return {"embeddings": [None for _ in payload["input"]]}  # the silent-None failure

    monkeypatch.setattr(prov, "_post", bad_post)
    with pytest.raises(ProviderError):
        prov.embed("e", ["a", "b"])


# -- time intercept: position, not content ---------------------------------------------------


def test_time_intercept_counts_by_position() -> None:
    import datetime as dt

    now = dt.datetime(2026, 10, 31, 9, 0)
    assert answer_time_query("what day is it", now) is not None
    assert answer_time_query("hey mimir, what time is it?", now) is not None
    assert (
        answer_time_query("how many days to finish the report", now) is None
    )  # 'days to' mid-turn
    assert answer_time_query("remind me what day the vet comes", now) is None
    assert answer_time_query("what day should I prune the apples this year then", now) is None


# -- tools: a refusal is blocked, not ok; a peer may read the notebook but not rewrite it -----


def test_a_peer_cannot_rewrite_the_notebook_and_the_refusal_is_blocked(brain: Mimir) -> None:
    from mimir.cognition.tools import ActionContext, ToolCall, dispatch

    reg = brain._tools
    call = dispatch(
        reg,
        ToolCall(tool="notebook", args={"op": "write", "title": "t", "body": "x y z"}),
        ActionContext(user="peer", speaker_kind="ai_peer", trusted=False),
    )
    assert call.status == "blocked" and call.result.startswith("refused:")
    human = dispatch(
        reg,
        ToolCall(tool="notebook", args={"op": "write", "title": "t", "body": "x y z"}),
        ActionContext(user="alex", speaker_kind="human", trusted=False),
    )
    assert human.status == "ok"
    wipe = dispatch(
        reg,
        ToolCall(tool="notebook", args={"op": "write", "title": "t", "body": ""}),
        ActionContext(user="alex", speaker_kind="human", trusted=True),
    )
    assert wipe.status == "blocked"  # an empty write would wipe it


def test_milestone_tool_accepts_both_spellings_of_current_config(brain: Mimir) -> None:
    from mimir.cognition.tools import ActionContext, ToolCall, dispatch
    from mimir.storage.repo import list_milestones

    call = dispatch(
        brain._tools,
        ToolCall(
            tool="record_milestone",
            args={
                "title": "Solar",
                "statement": "The array is live.",
                "status": "done",
                "current_config": True,
            },
        ),
        ActionContext(user="alex", speaker_kind="human", trusted=True),
    )
    assert call.status == "ok"
    rows = list_milestones(brain._storage)
    assert any(
        getattr(r, "is_current_config", None)
        or (isinstance(r, dict) and r.get("is_current_config"))
        for r in rows
    )
