"""Executable spec for build_context(): epistemics, the uncertainty gate, accounting."""

from __future__ import annotations

from mimir.context.build import build_context
from mimir.embed.base import EmbeddingMode
from mimir.prompts import RECALL_CLOSE, RECALL_OPEN
from mimir.retrieval.hybrid import ScoredMemory
from mimir.storage.models import EvidenceTier, Memory


def _scored(text: str, tier: EvidenceTier, provenance: str, score: float, mid: int) -> ScoredMemory:
    mem = Memory(text=text, evidence_tier=tier, provenance=provenance, id=mid)
    return ScoredMemory(memory=mem, score=score, keyword=score, vector=0.0)


def test_provenance_is_rendered_not_flattened() -> None:
    retrieved = [
        _scored(
            "favorite color is teal",
            EvidenceTier.STATED_BY_PRIMARY_USER,
            "stated by alex",
            0.9,
            1,
        ),
    ]
    bundle = build_context(
        query="what is my favorite color?",
        user="alex",
        identity="You are Mimir.",
        retrieved=retrieved,
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    assert RECALL_OPEN in bundle.prompt and RECALL_CLOSE in bundle.prompt
    assert "tier=stated_by_primary_user" in bundle.prompt
    assert "source=stated by alex" in bundle.prompt
    assert bundle.retrieved_ids == [1]


def test_uncertainty_gate_fires_on_thin_evidence() -> None:
    one = [_scored("teal", EvidenceTier.CONVERSATION, "x", 0.5, 1)]
    bundle = build_context(
        query="what is my favorite color?",
        user=None,
        identity="id",
        retrieved=one,
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    assert bundle.uncertainty_triggered
    assert bundle.source_count == 1
    assert "epistemic check" in bundle.prompt


def test_uncertainty_gate_silent_with_corroboration() -> None:
    two = [
        _scored("teal", EvidenceTier.CONVERSATION, "x", 0.5, 1),
        _scored("teal indeed", EvidenceTier.CONVERSATION, "y", 0.4, 2),
    ]
    bundle = build_context(
        query="what is my favorite color?",
        user=None,
        identity="id",
        retrieved=two,
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    assert not bundle.uncertainty_triggered
    assert bundle.source_count == 2


def test_uncertainty_gate_ignores_non_questions() -> None:
    bundle = build_context(
        query="i like hiking",
        user=None,
        identity="id",
        retrieved=[],
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    assert not bundle.uncertainty_triggered


def test_sentinel_note_occupies_end_slot() -> None:
    note = Memory(text="follow up about the trip")
    bundle = build_context(
        query="hello",
        user=None,
        identity="id",
        retrieved=[],
        sentinel_note=note,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    assert "follow up about the trip" in bundle.prompt
    # the note section is the last non-uncertainty section (high-attention end slot)
    assert bundle.sections[-1].name == "sentinel_note"


def test_token_accounting_and_introspection() -> None:
    retrieved = [_scored("a fact", EvidenceTier.CONVERSATION, "x", 0.5, 1)]
    bundle = build_context(
        query="hello there",
        user=None,
        identity="You are Mimir.",
        retrieved=retrieved,
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=4096,
    )
    info = bundle.introspect()
    assert info["embed_mode"] == "bootstrap"
    assert "NOT semantic" in info["embed_mode_banner"]
    assert info["admitted_tokens"] > 0
    assert any(s["name"] == "knowledge" for s in info["sections"])


def test_knowledge_truncates_under_tight_budget() -> None:
    retrieved = [
        _scored(
            f"fact number {i} with some length",
            EvidenceTier.CONVERSATION,
            "x",
            0.9 - i * 0.01,
            i,
        )
        for i in range(20)
    ]
    bundle = build_context(
        query="tell me",
        user=None,
        identity="id",
        retrieved=retrieved,
        sentinel_note=None,
        embed_mode=EmbeddingMode.BOOTSTRAP,
        budget_tokens=80,  # deliberately tiny
    )
    knowledge = next(s for s in bundle.sections if s.name == "knowledge")
    assert knowledge.truncated
    assert len(bundle.retrieved_ids) < 20
    assert any("truncated" in w for w in bundle.warnings)
