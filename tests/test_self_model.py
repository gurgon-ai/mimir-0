"""Executable spec for the evolving, generic self-model (v0.1+, DESIGN §3a)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.cognition.self_model import build_brief, gather_signals, synthesize_self_model
from mimir.config import Config
from mimir.context.build import build_context
from mimir.embed.base import EmbeddingMode
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import latest_self_model, save_memory


def test_signals_are_generic_store_stats(brain: Mimir) -> None:
    save_memory(
        brain._storage,
        Memory(
            text="alex likes tea",
            user="alex",
            evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,
        ),
    )
    save_memory(
        brain._storage,
        Memory(text="a doc chunk", evidence_tier=EvidenceTier.DOCUMENT, source="/tmp/x.md"),
    )
    sig = gather_signals(brain._storage)
    assert sig.total_memories == 2
    assert sig.documents == 1
    assert sig.distinct_users == 1
    assert sig.tier_counts.get("document") == 1
    brief = build_brief(sig)
    assert "2 memories" in brief and "1 from documents" in brief


def test_synthesis_stores_grounded_self_model(brain: Mimir) -> None:
    save_memory(brain._storage, Memory(text="a fact", user="greg"))
    mem = synthesize_self_model(brain._model, brain._storage)
    assert mem.kind is MemoryKind.SELF_MODEL
    assert mem.text.strip()
    assert latest_self_model(brain._storage) is not None
    # The mock grounds its description in the signals brief (which counts memories).
    assert "memor" in mem.text.lower()


def test_build_context_puts_self_model_first() -> None:
    bundle = build_context(
        query="hi",
        user=None,
        identity="You are the seed persona.",
        retrieved=[],
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
        self_knowledge="I have come to hold a handful of facts about one user.",
    )
    names = [s.name for s in bundle.sections]
    assert names[0] == "self_model"
    assert names[1] == "identity"
    # self-model is rendered above the seed persona in the prompt
    assert bundle.prompt.index("come to hold") < bundle.prompt.index("seed persona")


def test_brain_seeds_self_model_on_first_turn_and_injects_it(brain: Mimir) -> None:
    brain.turn("My favorite color is teal.", user="greg")
    brain.wait_for_sentinel()  # joins background incl. the self-model refresh
    assert latest_self_model(brain._storage) is not None

    r2 = brain.turn("What is my favorite color?", user="greg")
    assert r2.context.sections[0].name == "self_model"
    assert "memory system" in r2.context.prompt  # the mock's self-description phrasing


def test_explicit_refresh_returns_current_self_model(brain: Mimir) -> None:
    mem = brain.refresh_self_model()
    assert mem.kind is MemoryKind.SELF_MODEL
    current = latest_self_model(brain._storage)
    assert current is not None and current.text == mem.text


def test_self_model_can_be_disabled(mock_config: Config) -> None:
    mock_config.self_model_refresh_every = 0
    with Mimir(mock_config) as m:
        m.turn("hello there", user="x")
        m.wait_for_sentinel()
        assert latest_self_model(m._storage) is None  # disabled → no synthesis


def test_self_model_excluded_from_recall(brain: Mimir) -> None:
    """The self-model occupies its own slot; it must never compete in the knowledge section."""
    brain.refresh_self_model()
    r = brain.turn("anything at all", user="z")
    # self-model rows are kind=SELF_MODEL, never retrieved into the knowledge section
    assert all(
        s.name != "knowledge" or "self-model synthesis" not in s.body for s in r.context.sections
    )
