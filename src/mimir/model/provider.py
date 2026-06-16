"""The provider seam — the adapter contract behind the model gateway.

A provider wraps exactly two capabilities the runtime contract allows: one chat endpoint
and one embeddings endpoint (DESIGN §2). Everything model-shaped in Mimir flows through a
provider, and every provider is reached only through the ``ModelGateway`` — never called
directly by cognition.

A provider receives the concrete ``model`` name and tuned ``params`` the gateway resolved
from config for the role; it does not know about roles. This keeps role→model mapping in
config, where the law says it belongs (DESIGN §4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# A chat message: {"role": "system"|"user"|"assistant", "content": "..."}.
Message = dict[str, str]

_PARAMS_RE = re.compile(r"([\d.]+)\s*[bB]")

# Substrings that mark a model as embeddings-only (no chat endpoint). Ollama reports this via
# `capabilities`, but the live routing path only has model *names* — and several embedding models
# don't contain "embed" (all-minilm, bge, gte, mxbai, …), so a bare "embed" check misses them and
# the router/council tries to chat with them (HTTP 400). Practical, not exhaustive.
_EMBEDDING_MARKERS = ("embed", "minilm", "bge", "gte", "mxbai", "arctic-embed", "nomic")


def is_embedding_model(name: str) -> bool:
    """Heuristic: is this model name an embeddings-only model (no chat)? Used to keep embedding
    models out of chat routing, the council roster, and `auto` chat-role resolution."""
    lowered = (name or "").lower()
    return any(marker in lowered for marker in _EMBEDDING_MARKERS)


@dataclass(slots=True)
class ModelInfo:
    """Catalogue metadata for one installed model (from a provider's model listing)."""

    name: str
    family: str = ""
    params_b: float = 0.0  # parameter count in billions (the 'weight'), 0 if unknown
    quantization: str = ""
    context_length: int = 0
    size_bytes: int = 0
    capabilities: list[str] = field(default_factory=list)


def parse_params_b(parameter_size: str) -> float:
    """Parse Ollama's ``parameter_size`` (e.g. '11.9B', '7B') into billions of params."""
    match = _PARAMS_RE.search(parameter_size or "")
    return float(match.group(1)) if match else 0.0


@runtime_checkable
class Provider(Protocol):
    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        """Return the assistant's reply text for a chat completion."""
        ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...
