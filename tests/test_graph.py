"""Executable spec for the entity graph: triples, dedup, 1–2 hop traversal (DESIGN §3a)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.graph import retrieve_connected, store_triples
from mimir.context.build import build_context
from mimir.embed.base import EmbeddingMode
from mimir.storage.repo import count_triples, traverse_from_entities


def test_store_dedups_case_insensitively(brain: Mimir) -> None:
    assert store_triples(brain._storage, [["Alex", "lives in", "Colorado"]], user="alex") == 1
    # same triple, different case → deduped (0 new)
    assert store_triples(brain._storage, [["alex", "Lives In", "colorado"]], user="alex") == 0
    assert count_triples(brain._storage) == 1


def test_store_skips_malformed(brain: Mimir) -> None:
    n = store_triples(brain._storage, [["a", "b"], ["x", "", "z"], ["s", "r", "o"]], user=None)
    assert n == 1  # only the well-formed, non-blank triple


def test_traverse_one_hop_both_directions(brain: Mimir) -> None:
    store_triples(
        brain._storage,
        [["Alex", "lives in", "Colorado"], ["Alex", "likes", "tea"]],
        user="alex",
    )
    assert len(traverse_from_entities(brain._storage, ["alex"])) == 2  # subject side
    assert len(traverse_from_entities(brain._storage, ["colorado"])) == 1  # object side


def test_retrieve_connected_seeds_from_query(brain: Mimir) -> None:
    store_triples(brain._storage, [["My favorite color", "is", "teal"]], user="alex")
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
    brain.turn("Alex is from Colorado.", user="alex")
    assert count_triples(brain._storage) >= 1


def test_brain_injects_connected_facts(brain: Mimir) -> None:
    brain.turn("My favorite color is teal.", user="alex")
    brain.wait_for_sentinel()
    r = brain.turn("What is my favorite color?", user="alex")
    section = next((s for s in r.context.sections if s.name == "entity_graph"), None)
    assert section is not None
    assert "teal" in section.body.lower()


def test_graph_map_blobs_entities_and_links(brain: Mimir) -> None:
    # A memory mentioning "barn" + a relation barn—near→gate → memory blob, two entities, and both
    # link kinds (relation + the memory's "mentions").
    from mimir.cognition.graph import build_graph_map
    from mimir.storage.models import EvidenceTier, Memory, Triple
    from mimir.storage.repo import save_memory, save_triple

    save_memory(brain._storage, Memory(
        text="the barn is freshly painted", evidence_tier=EvidenceTier.CONVERSATION, salience=2.0))
    save_triple(brain._storage, Triple(subject="barn", relation="near", object="gate"))

    m = build_graph_map(brain._storage)
    ids = {n["id"] for n in m["nodes"]}
    assert any(n["type"] == "memory" for n in m["nodes"])
    assert "e:barn" in ids and "e:gate" in ids
    rel = [link for link in m["links"] if link["label"] == "near"]
    assert rel and {rel[0]["source"], rel[0]["target"]} == {"e:barn", "e:gate"}
    assert any(link["target"] == "e:barn" and link["label"] == "mentions" for link in m["links"])
    mem = next(n for n in m["nodes"] if n["type"] == "memory")
    assert {"mid", "text", "tier", "salience"} <= set(mem)  # editable fields for the inspector


def test_edit_and_forget_memory(brain: Mimir) -> None:
    from mimir.storage.models import EvidenceTier, Memory
    from mimir.storage.repo import get_memory, save_memory

    mid = save_memory(brain._storage, Memory(
        text="initial", evidence_tier=EvidenceTier.CONVERSATION, salience=1.0))
    brain.edit_memory(mid, text="edited text", salience=3.5)
    m = get_memory(brain._storage, mid)
    assert m.text == "edited text" and m.salience == 3.5
    brain.forget_memory(mid)
    assert get_memory(brain._storage, mid) is None
