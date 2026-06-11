"""Executable spec for the entity graph: triples, dedup, 1–2 hop traversal (DESIGN §3a)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.graph import retrieve_connected, store_triples
from mimir.context.build import build_context
from mimir.embed.base import EmbeddingMode
from mimir.storage.repo import count_triples, traverse_from_entities


def test_store_dedups_case_insensitively(brain: Mimir) -> None:
    assert store_triples(brain._storage, [["Greg", "lives in", "Colorado"]], user="greg") == 1
    # same triple, different case → deduped (0 new)
    assert store_triples(brain._storage, [["greg", "Lives In", "colorado"]], user="greg") == 0
    assert count_triples(brain._storage) == 1


def test_store_skips_malformed(brain: Mimir) -> None:
    n = store_triples(brain._storage, [["a", "b"], ["x", "", "z"], ["s", "r", "o"]], user=None)
    assert n == 1  # only the well-formed, non-blank triple


def test_traverse_one_hop_both_directions(brain: Mimir) -> None:
    store_triples(
        brain._storage,
        [["Greg", "lives in", "Colorado"], ["Greg", "likes", "tea"]],
        user="greg",
    )
    assert len(traverse_from_entities(brain._storage, ["greg"])) == 2  # subject side
    assert len(traverse_from_entities(brain._storage, ["colorado"])) == 1  # object side


def test_retrieve_connected_seeds_from_query(brain: Mimir) -> None:
    store_triples(brain._storage, [["My favorite color", "is", "teal"]], user="greg")
    hit = retrieve_connected(brain._storage, "what is my favorite color?")
    assert len(hit) == 1 and hit[0].object == "teal"
    # an unrelated query names no known entity
    assert retrieve_connected(brain._storage, "what is the capital of France?") == []


def test_retrieve_connected_two_hops(brain: Mimir) -> None:
    store_triples(
        brain._storage,
        [["Alice", "knows", "Bob"], ["Bob", "works at", "Acme"]],
        user=None,
    )
    one = retrieve_connected(brain._storage, "tell me about Alice", hops=1)
    assert len(one) == 1  # Alice → Bob only
    two = retrieve_connected(brain._storage, "tell me about Alice", hops=2)
    assert len(two) == 2  # plus Bob → Acme


def test_build_context_graph_section_counts_as_sources() -> None:
    bundle = build_context(
        query="q",
        user=None,
        identity="id",
        retrieved=[],
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
        graph_facts=["A — rel → B", "C — rel → D"],
    )
    section = next(s for s in bundle.sections if s.name == "entity_graph")
    assert "A — rel → B" in section.body
    assert bundle.source_count == 2  # connected edges are grounding sources


def test_bake_extracts_and_stores_triples(brain: Mimir) -> None:
    brain.turn("Greg is from Colorado.", user="greg")
    assert count_triples(brain._storage) >= 1


def test_brain_injects_connected_facts(brain: Mimir) -> None:
    brain.turn("My favorite color is teal.", user="greg")
    brain.wait_for_sentinel()
    r = brain.turn("What is my favorite color?", user="greg")
    section = next((s for s in r.context.sections if s.name == "entity_graph"), None)
    assert section is not None
    assert "teal" in section.body.lower()
