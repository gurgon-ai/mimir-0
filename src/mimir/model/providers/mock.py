"""A deterministic, dependency-free provider for tests, CI, and the acceptance self-test.

It makes **no network calls** and needs no model server, so the §6 acceptance loop can run
green anywhere — no Ollama, no GPU, no account. It is deterministic on purpose: identical
inputs always yield identical outputs, so the loop is a reliable self-test (DESIGN §10).

It is honest about being a mock. It does not pretend to reason — it reflects the assembled
context back in a structured, predictable way, which is exactly enough to prove the spine:
that a baked memory is retrieved, attributed, and surfaced in a later turn.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from ...embed.locality import LocalityHashEmbedder
from ...prompts import (
    BAKE_MARKER,
    COUNCIL_PERSONA_MARKER,
    COUNCIL_SYNTH_MARKER,
    RECALL_CLOSE,
    RECALL_OPEN,
    SELF_MODEL_MARKER,
    SENTINEL_MARKER,
    WORKING_MEMORY_MARKER,
)
from ..provider import Message, ModelInfo

_EMBED_DIM = 64


class MockProvider:
    """Deterministic stand-in for a real chat+embeddings backend."""

    def __init__(self) -> None:
        # A locality-hashing embedder gives deterministic vectors with real lexical
        # overlap, so endpoint-mode tests retrieve correctly through the mock.
        self._embedder = LocalityHashEmbedder(dim=_EMBED_DIM)

    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        if BAKE_MARKER in system:
            return self._bake(user)
        if COUNCIL_SYNTH_MARKER in system:
            return self._council_synth(user)
        if COUNCIL_PERSONA_MARKER in system:
            return self._council_persona(system, user)
        if WORKING_MEMORY_MARKER in system:
            return self._working_memory(user)
        if SELF_MODEL_MARKER in system:
            return self._self_model(user)
        if SENTINEL_MARKER in system:
            return self._reflect(user)
        return self._reply(system, user)

    def chat_stream(
        self, model: str, messages: list[Message], params: dict[str, object]
    ) -> Iterator[str]:
        """Stream the same deterministic reply, word by word (so the UI streaming path works)."""
        reply = self.chat(model, messages, params)
        for i, word in enumerate(reply.split(" ")):
            yield word if i == 0 else " " + word

    def list_models(self) -> list[str]:
        """A few distinct names so the council's multi-model assignment path is exercised."""
        return ["mock-a", "mock-b", "mock-c"]

    def model_details(self) -> list[ModelInfo]:
        """Deterministic catalogue metadata: distinct families and weights for fleet tests."""
        return [
            ModelInfo(name="mock-a", family="alpha", params_b=3.0, quantization="Q4"),
            ModelInfo(name="mock-b", family="beta", params_b=8.0, quantization="Q4"),
            ModelInfo(name="mock-c", family="gamma", params_b=27.0, quantization="Q5"),
        ]

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed(t) for t in texts]

    # -- task behaviors ---------------------------------------------------------------

    @staticmethod
    def _bake(user_text: str) -> str:
        """Extract a durable fact + a simple ``X is Y`` triple from a declarative statement."""
        stripped = user_text.strip()
        if not stripped or stripped.endswith("?"):
            return json.dumps({"facts": [], "triples": []})
        return json.dumps({"facts": [stripped], "triples": _mock_triples(stripped)})

    @staticmethod
    def _council_persona(system: str, question: str) -> str:
        """A deterministic, persona-specific position (name parsed from the marker)."""
        name = "voice"
        marker = f"[{COUNCIL_PERSONA_MARKER} "
        if marker in system:
            name = system.split(marker, 1)[1].split("]", 1)[0].strip()
        snippet = question.strip()[:48]
        return f"As the {name}, on '{snippet}': here is my distinct position."

    @staticmethod
    def _council_synth(question: str) -> str:
        return "Council verdict: weighing the perspectives, a balanced conclusion emerges."

    @staticmethod
    def _working_memory(brief: str) -> str:
        """Deterministic rolling summary referencing how much context it folded."""
        exchanges = brief.count("you:")
        return f"Working summary: tracking {exchanges} recent exchange(s) of context."

    @staticmethod
    def _self_model(brief: str) -> str:
        """Author a deterministic self-description grounded in the brief's first line."""
        first = brief.strip().splitlines()[0] if brief.strip() else "I hold no memories yet."
        return f"I am a memory system shaped by use. {first}"

    @staticmethod
    def _reflect(turn_text: str) -> str:
        """Leave a short, deterministic note for the next turn."""
        snippet = turn_text.strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return f"Note to self: the user just said '{snippet}'. Follow up if it becomes relevant."

    @staticmethod
    def _reply(system: str, user_text: str) -> str:
        """Compose a reply that surfaces whatever the recall block injected."""
        recalled = _extract_recall(system)
        if recalled:
            return "Based on what you've told me — " + " ".join(recalled)
        return "I don't have anything on record about that yet. Could you tell me more?"


def _mock_triples(text: str) -> list[list[str]]:
    """A deterministic, naive triple from an ``X is Y`` statement (enough to drive the graph)."""
    cleaned = text.strip().rstrip(".!?")
    lowered = cleaned.lower()
    if " is " in lowered:
        idx = lowered.index(" is ")
        subject, obj = cleaned[:idx].strip(), cleaned[idx + 4 :].strip()
        if subject and obj:
            return [[subject, "is", obj]]
    return []


def _extract_recall(system: str) -> list[str]:
    """Pull the cleaned memory texts out of the <RECALL>…</RECALL> block, if present."""
    if RECALL_OPEN not in system or RECALL_CLOSE not in system:
        return []
    inner = system.split(RECALL_OPEN, 1)[1].split(RECALL_CLOSE, 1)[0]
    texts: list[str] = []
    for line in inner.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        # build_context renders "- <text> [tier=…; source=…]"; keep the text portion.
        text = body.split(" [tier=", 1)[0].strip()
        if text:
            texts.append(text)
    return texts
