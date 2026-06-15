"""The council forum (DESIGN §5a): deliberations persist as threads; the user can comment + keep
house (close/reopen, delete posts, delete threads). Exercised through the brain (mock-backed)."""

from __future__ import annotations

from mimir.brain import Mimir


def test_deliberation_creates_a_thread(brain: Mimir) -> None:
    result = brain.deliberate("Should we plant garlic in October?")
    assert result.thread_id is not None
    threads = brain.forum_threads()
    assert len(threads) == 1
    t = threads[0]
    assert t["question"].startswith("Should we plant garlic")
    assert t["status"] == "open" and t["posts"] >= 2  # persona positions + the verdict

    full = brain.forum_thread(result.thread_id)
    kinds = {p["kind"] for p in full["posts"]}
    assert "position" in kinds and "verdict" in kinds


def test_comment_and_housekeeping(brain: Mimir) -> None:
    tid = brain.deliberate("Tea or coffee?").thread_id

    brain.forum_comment(tid, "I think both have their place.")
    posts = brain.forum_thread(tid)["posts"]
    comment = next(p for p in posts if p["kind"] == "comment")
    assert comment["content"].startswith("I think both")

    # close / reopen
    brain.forum_set_status(tid, "closed")
    assert brain.forum_thread(tid)["status"] == "closed"
    brain.forum_set_status(tid, "open")
    assert brain.forum_thread(tid)["status"] == "open"

    # delete a single post
    before = len(brain.forum_thread(tid)["posts"])
    brain.forum_delete_post(comment["id"])
    assert len(brain.forum_thread(tid)["posts"]) == before - 1

    # delete the whole thread
    brain.forum_delete_thread(tid)
    assert brain.forum_thread(tid) is None
    assert brain.forum_threads() == []
