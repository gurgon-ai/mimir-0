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

from typing import Protocol, runtime_checkable

# A chat message: {"role": "system"|"user"|"assistant", "content": "..."}.
Message = dict[str, str]


@runtime_checkable
class Provider(Protocol):
    def chat(self, model: str, messages: list[Message], params: dict[str, object]) -> str:
        """Return the assistant's reply text for a chat completion."""
        ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...
