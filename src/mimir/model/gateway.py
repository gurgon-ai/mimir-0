"""The model gateway — the second chokepoint (DESIGN §5).

**The law:** every chat and every embedding call goes through here. The gateway resolves a
*role* (``chat``, ``bake``, ``reasoning``, ``embed``) to a concrete model + tuned params from
config and a default priority, then hands off to the provider pool behind it.

The pool (``pool.py``) provides the hardened internals: retry/backoff, transient-fail signaling,
a saturation breaker, health tracking, and failover across endpoints. The gateway itself stays a
thin role→model resolver, so cognition never touches scheduling concerns. Callers are unchanged:
``chat(role, messages)`` / ``embed(role, texts)`` still work; ``priority`` is an optional override.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from ..config import AUTO_MODEL, RoleSpec
from ..errors import ModelGatewayError
from .pool import ProviderPool
from .priority import DEFAULT_ROLE_PRIORITY, Priority
from .provider import Message, ModelInfo, Provider

log = logging.getLogger("mimir.model")


class ModelGateway:
    def __init__(
        self,
        provider: Provider | list[Provider] | ProviderPool,
        roles: dict[str, RoleSpec],
    ) -> None:
        self._roles = roles
        if isinstance(provider, ProviderPool):
            self._pool = provider
        elif isinstance(provider, list):
            endpoints = [
                (getattr(p, "name", f"endpoint-{i}"), p) for i, p in enumerate(provider)
            ]
            self._pool = ProviderPool(endpoints)
        else:
            self._pool = ProviderPool([(getattr(provider, "name", "endpoint-0"), provider)])

    def set_role_model(self, role: str, model: str) -> None:
        """Re-point a role at a different model, keeping its tuned params (for auto-apply)."""
        existing = self._roles.get(role)
        params = existing.params if existing is not None else {}
        self._roles[role] = RoleSpec(model=model, params=params)

    def roles_view(self) -> dict[str, RoleSpec]:
        """A read-only snapshot of the current role→spec mapping (for introspection / the UI)."""
        return dict(self._roles)

    def _role(self, role: str) -> RoleSpec:
        spec = self._roles.get(role)
        if spec is None:
            raise ModelGatewayError(
                f"no model configured for role {role!r}; known roles: {sorted(self._roles)}"
            )
        if spec.model == AUTO_MODEL:
            # The brain resolves `auto` to a concrete model once inventory lands (DESIGN §4); until
            # then, stop-gap to any reachable model so a turn never fails on an unresolved role.
            want_embed = role == "embed"
            picks = [
                m for m in self._pool.available_models() if ("embed" in m.lower()) == want_embed
            ]
            if not picks:
                raise ModelGatewayError(
                    f"role {role!r} is set to 'auto' but no suitable model is reachable yet"
                )
            return RoleSpec(model=picks[0], params=spec.params)
        return spec

    def _priority(self, role: str, override: Priority | None) -> Priority:
        if override is not None:
            return override
        return DEFAULT_ROLE_PRIORITY.get(role, Priority.USER_ADJACENT)

    def chat(
        self, role: str, messages: list[Message], *, priority: Priority | None = None
    ) -> str:
        """Route a chat completion for ``role`` through the pool (retry/failover handled there)."""
        spec = self._role(role)
        return self._pool.chat(
            spec.model, messages, spec.params, priority=self._priority(role, priority)
        )

    def chat_stream(
        self, role: str, messages: list[Message], *, priority: Priority | None = None
    ) -> Iterator[str]:
        """Stream a chat completion for ``role`` through the pool (token-by-token)."""
        spec = self._role(role)
        return self._pool.chat_stream(
            spec.model, messages, spec.params, priority=self._priority(role, priority)
        )

    def embed(
        self, role: str, texts: list[str], *, priority: Priority | None = None
    ) -> list[list[float]]:
        """Route an embeddings call for ``role`` through the pool."""
        spec = self._role(role)
        return self._pool.embed(
            spec.model, texts, priority=self._priority(role, priority)
        )

    # -- inner council support --------------------------------------------------------

    def available_models(self) -> list[str]:
        """Models installed across the provider pool — the council's auto-discovery (DESIGN §4)."""
        return self._pool.available_models()

    def default_council_model(self) -> str:
        """Fallback model when discovery finds nothing: the council, reasoning, or chat role."""
        for role in ("council", "reasoning", "chat"):
            spec = self._roles.get(role)
            if spec is not None:
                return spec.model
        raise ModelGatewayError("no model configured for the council")

    def _council_params(self) -> dict[str, object]:
        for role in ("council", "reasoning"):
            spec = self._roles.get(role)
            if spec is not None:
                return spec.params
        return {}

    def chat_with_model(
        self, model: str, messages: list[Message], *,
        priority: Priority = Priority.BACKGROUND, params: dict[str, object] | None = None,
    ) -> str:
        """Chat against a specific discovered model (bypassing role→model resolution).

        Used by the council to spread personas across models. Params come from the council/reasoning
        role config, so tuning still lives in config (DESIGN §4). ``params`` merges over those — the
        benchmark uses it to pin a consistent ``num_ctx`` so long prompts aren't truncated to the
        Ollama default.
        """
        merged = {**self._council_params(), **(params or {})}
        return self._pool.chat(model, messages, merged, priority=priority)

    def get_stats(self) -> dict[str, object]:
        return self._pool.get_stats()

    # -- fleet lifecycle (delegates to the pool) --------------------------------------

    def refresh_inventory(self) -> None:
        self._pool.refresh()

    def start_prober(self, interval_s: float) -> None:
        self._pool.start_prober(interval_s)

    def stop_prober(self) -> None:
        self._pool.stop_prober()

    def inventory_details(self) -> list[tuple[str, str, list[ModelInfo]]]:
        return self._pool.inventory_details()
