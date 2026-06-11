"""``Mimir`` — the facade. Import it, hand it a config, call ``.turn()`` (DESIGN §1).

This is where the spine is wired into the §6 loop:

    turn(text, user)
      → embed the query → recall via the storage gateway → build_context() (assemble)
      → model (chat) → reply
      → record access · bake new facts (storage gateway) · fire the sentinel (async)

Everything routes through the two gateways. The sentinel runs off the hot path and its failure
cannot break the loop. The next turn joins the prior sentinel before assembling, so a note is
always ready when it matters — without making the reply wait on reflection.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .cognition.bake import bake
from .cognition.sentinel import run_sentinel
from .config import Config, ProviderSpec, load_config
from .context.build import ContextBundle, build_context
from .embed.base import Embedder, EmbeddingMode
from .embed.endpoint import EndpointEmbedder, NullEmbedder
from .embed.locality import LocalityHashEmbedder
from .errors import ConfigError
from .model.gateway import ModelGateway
from .model.provider import Provider
from .model.providers.mock import MockProvider
from .model.providers.ollama import OllamaProvider
from .retrieval.hybrid import retrieve
from .storage.gateway import StorageGateway
from .storage.models import Memory, MemoryKind
from .storage.repo import latest_sentinel_note, list_memories, record_access

log = logging.getLogger("mimir")

# How many memories the knowledge section may draw on per turn (pre-budget). Hardening
# (adaptive top-k, SQL-side prefiltering) is a later session; v0 keeps it a simple constant.
DEFAULT_TOP_K = 6


@dataclass(slots=True)
class TurnResult:
    """What a turn produced: the reply, the assembled context (introspectable), and bakes."""

    reply: str
    context: ContextBundle
    baked: list[Memory]


def build_provider(spec: ProviderSpec) -> Provider:
    """Construct the provider named in config. Fails loud on an unknown type."""
    if spec.type == "mock":
        return MockProvider()
    if spec.type == "ollama":
        return OllamaProvider(host=str(spec.options.get("host", "http://localhost:11434")))
    raise ConfigError(
        f"unknown provider type {spec.type!r}; supported: 'ollama', 'mock'. See docs/SETUP.md."
    )


def make_embedder(config: Config, model: ModelGateway) -> Embedder:
    """Pick the embedder for the configured mode (DESIGN kickoff decision; see docs/SETUP.md)."""
    if config.embed_mode is EmbeddingMode.BOOTSTRAP:
        return LocalityHashEmbedder(dim=config.embed_dim)
    if config.embed_mode is EmbeddingMode.ENDPOINT:
        return EndpointEmbedder(model)
    return NullEmbedder()


class Mimir:
    """The cognition core. One instance owns one store and one provider."""

    def __init__(self, config: Config, *, provider: Provider | None = None) -> None:
        config.validate()
        self.config = config
        self._storage = StorageGateway(config.storage_path)
        self._model = ModelGateway(provider or build_provider(config.provider), config.roles)
        self._embedder: Embedder = make_embedder(config, self._model)
        # Fail-loud visibility: announce the active embedding mode so no one mistakes the
        # cheap bootstrap path for poor memory (DESIGN §10; kickoff decision).
        log.info("Mimir online | embeddings: %s", self._embedder.mode.banner())

        self._pending_sentinel: threading.Thread | None = None
        self._last_sentinel_error: BaseException | None = None

    @classmethod
    def from_config(cls, path: str) -> Mimir:
        """Construct from a ``mimir.toml`` path."""
        return cls(load_config(path))

    # -- the turn ---------------------------------------------------------------------

    def turn(self, text: str, user: str | None = None) -> TurnResult:
        # Make sure the previous turn's sentinel note has landed before we assemble.
        self._join_sentinel()

        # 1. Recall: embed the query, pull candidates, rank them.
        query_vec = self._embedder.embed(text)
        candidates = list_memories(self._storage, user=user, kind=MemoryKind.MEMORY)
        retrieved = retrieve(text, query_vec, candidates, top_k=DEFAULT_TOP_K)
        note = latest_sentinel_note(self._storage, user)

        # 2. Assemble the epistemic prompt.
        bundle = build_context(
            query=text,
            user=user,
            identity=self.config.identity,
            retrieved=retrieved,
            sentinel_note=note,
            embed_mode=self._embedder.mode,
            budget_tokens=self.config.context_budget_tokens,
        )

        # 3. Generate the reply through the model gateway.
        reply = self._model.chat(
            "chat",
            [
                {"role": "system", "content": bundle.prompt},
                {"role": "user", "content": text},
            ],
        )

        # 4. Side effects through the storage gateway: relevance bookkeeping + bake.
        record_access(self._storage, bundle.retrieved_ids)
        baked = bake(
            self._model,
            self._storage,
            self._embedder,
            turn_text=text,
            user=user,
            primary_user=self.config.primary_user,
        )

        # 5. Fire the sentinel off the hot path.
        self._spawn_sentinel(user=user, turn_text=text, reply=reply)

        return TurnResult(reply=reply, context=bundle, baked=baked)

    # -- sentinel plumbing ------------------------------------------------------------

    def _spawn_sentinel(self, *, user: str | None, turn_text: str, reply: str) -> None:
        def _run() -> None:
            try:
                run_sentinel(
                    self._model, self._storage, user=user, turn_text=turn_text, reply=reply
                )
            except BaseException as exc:  # logged downgrade — never touches the turn
                self._last_sentinel_error = exc
                log.error(
                    "sentinel failed (off the hot path; this turn is unaffected): %s",
                    exc,
                    exc_info=True,
                )

        self._last_sentinel_error = None
        thread = threading.Thread(target=_run, name="mimir-sentinel", daemon=True)
        thread.start()
        self._pending_sentinel = thread

    def _join_sentinel(self) -> None:
        if self._pending_sentinel is not None:
            self._pending_sentinel.join(timeout=30)
            self._pending_sentinel = None

    def wait_for_sentinel(self) -> None:
        """Block until the most recent turn's sentinel note has been written (or failed)."""
        self._join_sentinel()

    @property
    def last_sentinel_error(self) -> BaseException | None:
        """The exception from the last sentinel run, if it failed; else ``None``."""
        return self._last_sentinel_error

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:
        self._join_sentinel()
        self._storage.close()

    def __enter__(self) -> Mimir:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
