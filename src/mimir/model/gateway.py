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
from ..errors import ModelGatewayError, ProviderError
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
        # Ordered acceptable models per role (best first) — the role's fallback chain (DESIGN §5).
        # Set by the brain from the qualified ranking; a chat routes down it, so a heterogeneous
        # fleet (Gemma on node A, Qwen on node B) still serves the role by falling Gemma → Qwen.
        # Empty for a pinned role: a pin is honoured exactly, never substituted.
        self._fallbacks: dict[str, list[str]] = {}
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

    def set_role_fallbacks(self, role: str, models: list[str]) -> None:
        """Set a role's ordered acceptable-model chain (best first) — routing walks it on failure.

        De-duplicated, order preserved. An empty list clears the chain (the role then routes to its
        single resolved/pinned model). The brain sets this from the qualified per-role ranking so a
        heterogeneous fleet still serves the role across nodes (DESIGN §4/§5).
        """
        seen: dict[str, None] = {}
        for m in models:
            seen.setdefault(m, None)
        if seen:
            self._fallbacks[role] = list(seen)
        else:
            self._fallbacks.pop(role, None)

    def roles_view(self) -> dict[str, RoleSpec]:
        """A read-only snapshot of the current role→spec mapping (for introspection / the UI)."""
        return dict(self._roles)

    def fallbacks_view(self) -> dict[str, list[str]]:
        """A read-only snapshot of each role's fallback chain (for introspection / the UI)."""
        return {role: list(chain) for role, chain in self._fallbacks.items()}

    def _ordered_models(self, role: str) -> tuple[list[str], dict[str, object]]:
        """The role's acceptable models, best first, plus its tuned params. The chain is pruned to
        models the cached inventory says can run now (so a fallback never targets a vanished model);
        if the prune empties it (or inventory isn't known yet), the full chain is tried as-is. With
        no chain set, falls back to the single resolved/pinned model (the ``auto`` stop-gap works).
        """
        params = self._roles[role].params if role in self._roles else {}
        chain = self._fallbacks.get(role)
        if chain:
            known = self._pool.known_models()
            present = [m for m in chain if m in known] if known else []
            return (present or chain), params
        return [self._role(role).model], params

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
        """Route a chat completion for ``role``, walking its fallback chain (DESIGN §4/§5).

        Each model routes through the pool (which picks the fastest healthy node for it and fails
        over across that model's nodes). If a model is exhausted with a *transient* failure (every
        node for it is down/saturated), routing falls to the next acceptable model — so a fleet
        where no single model is everywhere still serves the role. A permanent error fails fast.
        """
        models, params = self._ordered_models(role)
        prio = self._priority(role, priority)
        last: ProviderError | None = None
        for model in models:
            try:
                return self._pool.chat(model, messages, params, priority=prio)
            except ProviderError as exc:
                last = exc
                if not exc.transient:
                    raise  # bad request / parse error — the next model won't fare better
        raise last or ModelGatewayError(f"no acceptable model for role {role!r}")

    def chat_stream(
        self, role: str, messages: list[Message], *, priority: Priority | None = None
    ) -> Iterator[str]:
        """Stream a chat completion for ``role``, walking its fallback chain (token-by-token).

        Fallover to the next model happens only *before the first token* — once tokens have streamed
        we are committed (restarting would duplicate output), so a failure after that propagates.
        """
        models, params = self._ordered_models(role)
        prio = self._priority(role, priority)
        last: ProviderError | None = None
        for model in models:
            started = False
            try:
                for token in self._pool.chat_stream(model, messages, params, priority=prio):
                    started = True
                    yield token
                return
            except ProviderError as exc:
                last = exc
                if started or not exc.transient:
                    raise  # mid-stream, or permanent — can't safely fall to another model
        raise last or ModelGatewayError(f"no acceptable model for role {role!r}")

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
        max_retries: int | None = None,
    ) -> str:
        """Chat against a specific discovered model (bypassing role→model resolution).

        Used by the council to spread personas across models. Params come from the council/reasoning
        role config, so tuning still lives in config (DESIGN §4). ``params`` merges over those — the
        benchmark uses it to pin a consistent ``num_ctx`` (and a tight per-call timeout) so long
        prompts aren't truncated and a slow model fails fast. ``max_retries`` overrides the pool
        default — the benchmark passes 0 so one slow call isn't retried into a multi-minute stall.
        """
        merged = {**self._council_params(), **(params or {})}
        return self._pool.chat(model, messages, merged, priority=priority, max_retries=max_retries)

    def council_placements(self) -> list[tuple[str, str]]:
        """One ``(node, model)`` per reachable node — for fanning the council across the fleet."""
        return self._pool.council_placements()

    def chat_on_node(
        self, node: str, model: str, messages: list[Message], *,
        priority: Priority = Priority.BACKGROUND, params: dict[str, object] | None = None,
        max_retries: int | None = None,
    ) -> str:
        """Run a council persona on a specific node (DESIGN §5). Params come from council/reasoning
        config; falls back to routing if that node is unavailable, so no persona is lost."""
        merged = {**self._council_params(), **(params or {})}
        return self._pool.chat_on(
            node, model, messages, merged, priority=priority, max_retries=max_retries,
        )

    def get_stats(self) -> dict[str, object]:
        return self._pool.get_stats()

    # -- fleet lifecycle (delegates to the pool) --------------------------------------

    def set_disabled_nodes(self, names: set[str]) -> None:
        """Veto fleet nodes by name — routing skips them even if reachable (DESIGN §5)."""
        self._pool.set_disabled_nodes(names)

    def refresh_inventory(self) -> None:
        self._pool.refresh()

    def start_prober(self, interval_s: float) -> None:
        self._pool.start_prober(interval_s)

    def stop_prober(self) -> None:
        self._pool.stop_prober()

    def inventory_details(self) -> list[tuple[str, str, list[ModelInfo]]]:
        return self._pool.inventory_details()

    # -- live latency / speed-aware routing (delegates to the pool, DESIGN §5) ---------

    def seed_latency(self, seeds: dict[tuple[str, str], float]) -> None:
        """Prime per-(node, model) latency from the catalogue's qualification snapshot."""
        self._pool.seed_latency(seeds)

    def latency_snapshot(self) -> dict[tuple[str, str], dict[str, object]]:
        """Live per-(node, model) latency from real traffic (for write-back/introspection)."""
        return self._pool.latency_snapshot()

    def idle_nodes(self) -> list[str]:
        """Reachable, non-vetoed nodes with nothing in flight — targets for the idle heartbeat."""
        return self._pool.idle_nodes()

    def probe_latency(
        self, node: str, model: str, messages: list[Message], params: dict[str, object]
    ) -> float | None:
        """Probe one node+model once and record its latency (the rare idle heartbeat)."""
        return self._pool.probe_latency(node, model, messages, params)
