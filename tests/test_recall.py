"""Hybrid recall (DESIGN §3a/§3d): the degraded keyword-only path must not false-ground a query —
a single shared token on a multi-word query would silence the uncertainty gate on an unrelated q."""

from __future__ import annotations

from mimir.retrieval.hybrid import retrieve
from mimir.storage.models import EvidenceTier, Memory


def _mem(text: str) -> Memory:
    return Memory(text=text, evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER)


def test_degraded_recall_needs_two_token_overlap_on_a_long_query() -> None:
    mems = [_mem("the north gate latch was replaced in March")]
    # query_vec=None → keyword-only (null embedder). Shares only "gate" → too thin to count.
    assert retrieve("when does the front gate open for visitors", None, mems, top_k=5) == []
    # shares "gate" + "latch" → a real hit
    assert len(retrieve("is the gate latch holding", None, mems, top_k=5)) == 1


def test_degraded_recall_allows_a_single_token_short_query() -> None:
    mems = [_mem("My favorite color is teal")]
    # a 1-token query where the one match IS the point — not blocked
    assert len(retrieve("teal", None, mems, top_k=5)) == 1
