"""The provider pool — retry/backoff, transient signaling, health, failover (DESIGN §5).

Behind the model gateway sits a pool of one or more provider endpoints. The pool implements the
resilience patterns proven in the parent system's Ollama broker, reimplemented clean and
provider-agnostic:

- **Retry with exponential backoff** on *transient* failures (a busy/unreachable backend),
  classified via ``ProviderError.transient``. Non-transient failures (bad request, parse error)
  fail fast — retrying or failing over won't help.
- **Saturation breaker** — if an endpoint throws N transient failures within a window, it is
  marked *saturated* for a cooldown. While saturated it is skipped for background work, attempted
  with a single try for user-adjacent work, and still attempted (last resort) for chat.
- **Failover** — on a transient failure the pool moves to the next healthy endpoint; only when
  every candidate is exhausted does it raise a transient error so the caller can defer.
- **Graceful degradation** — background tiers fail fast (defer) when nothing healthy is left,
  rather than hammering a struggling backend and starving the foreground.

The pool is deterministic-testable: the clock and sleep are injectable, so saturation/backoff
behavior can be exercised without real time passing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TypeVar

from ..errors import ProviderError
from .latency import LatencyStat, normalize_latency
from .priority import Priority
from .provider import Message, ModelInfo, Provider

log = logging.getLogger("mimir.model.pool")

R = TypeVar("R")


def _list_model_infos(provider: Provider) -> list[ModelInfo]:
    """Model metadata for a provider — rich ``model_details`` if it has it, else names only."""
    details = getattr(provider, "model_details", None)
    if details is not None:
        return list(details())
    lister = getattr(provider, "list_models", None)
    if lister is not None:
        return [ModelInfo(name=n) for n in lister()]
    return []


def _provider_stream(
    provider: Provider, model: str, messages: list[Message], params: dict[str, object]
) -> Iterator[str]:
    """Stream from a provider, falling back to a single-shot ``chat`` if it can't stream."""
    stream = getattr(provider, "chat_stream", None)
    if stream is not None:
        yield from stream(model, messages, params)
    else:
        yield provider.chat(model, messages, params)


@dataclass
class _Endpoint:
    name: str
    provider: Provider
    failures: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    saturated_until: float = 0.0
    models: set[str] = field(default_factory=set)  # known inventory (empty = unknown, route freely)
    reachable: bool = True  # active-health signal (set by refresh)
    inflight: int = 0  # current in-flight calls, for least-loaded selection
    latency: dict[str, LatencyStat] = field(default_factory=dict)  # live s/turn per model here


class ProviderPool:
    """Routes model calls across endpoints with retry, health, and failover."""

    def __init__(
        self,
        endpoints: list[tuple[str, Provider]],
        *,
        max_retries: int = 2,
        backoff_base_s: float = 1.0,
        sat_window_s: float = 30.0,
        sat_threshold: int = 5,
        sat_cooldown_s: float = 30.0,
        latency_alpha: float = 0.3,
        default_latency_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoints:
            raise ValueError("ProviderPool needs at least one endpoint")
        self._endpoints = [_Endpoint(name=n, provider=p) for n, p in endpoints]
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._sat_window_s = sat_window_s
        self._sat_threshold = sat_threshold
        self._sat_cooldown_s = sat_cooldown_s
        # Speed-aware routing: weight on the newest sample in the per-(node, model) EWMA, and the
        # assumed cost of a node we haven't measured yet (so an unknown node is still tried — and so
        # sampled — rather than starved or blindly trusted). See model/latency.py.
        self._latency_alpha = latency_alpha
        self._default_latency_s = default_latency_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._disabled: set[str] = set()   # node names the user vetoed (excluded from routing)
        self._stats = {"calls": 0, "ok": 0, "retries": 0, "failovers": 0, "transient_fails": 0}
        self._stop_prober = threading.Event()
        self._prober: threading.Thread | None = None

    # -- public API -------------------------------------------------------------------

    def chat(
        self, model: str, messages: list[Message], params: dict[str, object], *, priority: Priority,
        max_retries: int | None = None,
    ) -> str:
        return self._route(
            priority, lambda p: p.chat(model, messages, params), "chat", model,
            max_retries=max_retries,
        )

    def embed(
        self, model: str, texts: list[str], *, priority: Priority
    ) -> list[list[float]]:
        return self._route(priority, lambda p: p.embed(model, texts), "embed", model)

    def chat_on(
        self, node: str, model: str, messages: list[Message], params: dict[str, object], *,
        priority: Priority = Priority.BACKGROUND, max_retries: int | None = None,
        fallback: bool = True,
    ) -> str:
        """Run a chat on a *specific* node — for fanning the council across the whole fleet (§5).

        Pins the call to ``node`` so concurrent persona calls light up every machine instead of
        piling onto the best node for a model. Records load + latency like normal routing. If the
        node is gone/disabled or fails and ``fallback`` is set, it degrades to ordinary routing so a
        persona is never lost to one flaky box; with ``fallback=False`` the failure propagates.
        """
        with self._lock:
            self._stats["calls"] += 1
            ep = next(
                (e for e in self._endpoints if e.name == node and e.name not in self._disabled),
                None,
            )
        if ep is None:
            if fallback:
                return self.chat(
                    model, messages, params, priority=priority, max_retries=max_retries
                )
            raise ProviderError(f"node {node!r} unavailable for chat", transient=True)
        base_retries = self._max_retries if max_retries is None else max_retries
        with self._lock:
            ep.inflight += 1
        try:
            result, elapsed = self._attempt(
                ep, lambda p: p.chat(model, messages, params), base_retries
            )
        except ProviderError:
            self._record_failure(ep)
            if fallback:
                return self.chat(
                    model, messages, params, priority=priority, max_retries=max_retries
                )
            raise
        else:
            if isinstance(result, str):
                self._observe_latency(ep, model, elapsed, result)
            self._record_success(ep)
            with self._lock:
                self._stats["ok"] += 1
            return result
        finally:
            with self._lock:
                ep.inflight -= 1

    def council_placements(self) -> list[tuple[str, str]]:
        """One ``(node, model)`` per reachable, non-disabled node — the council's fleet spread.

        Each node contributes one slot, on a non-embedding model it actually has; models are chosen
        greedily to be **distinct across nodes** where inventory allows (more minds, not just more
        copies). Empty when no node inventory is known (e.g. a single local provider) — the council
        then falls back to model-routing. The order follows discovery; the caller round-robins it.
        """
        with self._lock:
            eps = [
                e for e in self._endpoints
                if e.reachable and e.name not in self._disabled and e.models
            ]
        used: set[str] = set()
        placements: list[tuple[str, str]] = []
        for endpoint in eps:
            options = sorted(m for m in endpoint.models if "embed" not in m.lower())
            if not options:
                continue
            pick = next((m for m in options if m not in used), options[0])
            used.add(pick)
            placements.append((endpoint.name, pick))
        return placements

    def chat_stream(
        self, model: str, messages: list[Message], params: dict[str, object], *, priority: Priority
    ) -> Iterator[str]:
        """Stream a chat completion from a healthy endpoint.

        Failover happens only *before the first token* (peeking it). Once a stream has started
        we are committed to that endpoint — a mid-stream failure propagates rather than silently
        restarting on another backend (which would duplicate output). No mid-stream retry.
        """
        with self._lock:
            self._stats["calls"] += 1
        candidates = self._candidates(priority, model)
        if not candidates:
            with self._lock:
                self._stats["transient_fails"] += 1
            raise ProviderError(
                "no available provider endpoint for chat stream (all saturated); deferring",
                transient=True,
            )

        last_exc: ProviderError | None = None
        for index, (endpoint, _clamp) in enumerate(candidates):
            gen = _provider_stream(endpoint.provider, model, messages, params)
            started = self._clock()
            try:
                first = next(gen)
            except StopIteration:
                self._record_success(endpoint)
                return
            except ProviderError as exc:
                self._record_failure(endpoint)
                last_exc = exc
                if not exc.transient:
                    raise
                continue  # failed before any token — safe to try the next endpoint
            except Exception as exc:
                self._record_failure(endpoint)
                raise ProviderError(f"{endpoint.name}: {type(exc).__name__}: {exc}") from exc

            if index > 0:
                with self._lock:
                    self._stats["failovers"] += 1
            self._record_success(endpoint)
            with self._lock:
                self._stats["ok"] += 1
            chunks = [first]
            yield first
            for token in gen:  # remaining tokens; a mid-stream error propagates as-is
                chunks.append(token)
                yield token
            # Passive measurement of the streamed turn (wall-clock of the full generation; the
            # consumer drains it promptly on the hot path). Same per-token normalization as chat.
            self._observe_latency(endpoint, model, self._clock() - started, "".join(chunks))
            return

        with self._lock:
            self._stats["transient_fails"] += 1
        raise ProviderError(
            "all provider endpoint(s) failed for chat stream; deferring", transient=True
        ) from last_exc

    def available_models(self) -> list[str]:
        """Distinct models installed across the endpoints (for council auto-discovery, DESIGN §4).

        Endpoints that can't list (no ``list_models`` or it errors) contribute nothing — discovery
        is best-effort and never fatal.
        """
        seen: list[str] = []
        for endpoint in self._endpoints:
            lister = getattr(endpoint.provider, "list_models", None)
            if lister is None:
                continue
            try:
                models = lister()
            except Exception as exc:  # discovery is best-effort
                log.warning("model: could not list models on %s: %s", endpoint.name, exc)
                continue
            for model in models:
                if model not in seen:
                    seen.append(model)
        return seen

    def set_disabled_nodes(self, names: set[str]) -> None:
        """Replace the vetoed node names; routing skips them (fail-safe if every node is vetoed)."""
        with self._lock:
            self._disabled = set(names)

    def known_models(self) -> set[str]:
        """Models the *cached* inventory says are installed on a reachable node (no network call).

        Refreshed by the prober's ``refresh``. Used to prune a role's fallback chain to models that
        can actually run right now — cheaply, on every turn — unlike ``available_models`` which
        lists live. Empty until the first inventory lands (caller should then not prune).
        """
        with self._lock:
            out: set[str] = set()
            for e in self._endpoints:
                if e.reachable:
                    out |= e.models
            return out

    def get_stats(self) -> dict[str, object]:
        with self._lock:
            now = self._clock()
            return {
                **self._stats,
                "endpoints": [e.name for e in self._endpoints],
                "nodes_up": sum(1 for e in self._endpoints if e.reachable),
                "saturated": {
                    e.name: round(e.saturated_until - now, 1)
                    for e in self._endpoints
                    if e.saturated_until > now
                },
                # Live speed: each node's fastest currently-known model (s/turn) — the routing
                # signal, surfaced for the UI/placement view. None-valued stats are omitted.
                "latency": {
                    e.name: min(
                        (s.value for s in e.latency.values() if s.value is not None),
                        default=None,
                    )
                    for e in self._endpoints
                    if any(s.value is not None for s in e.latency.values())
                },
            }

    # -- active health / inventory (the fleet) ----------------------------------------

    def refresh(self) -> None:
        """Re-inventory every endpoint's models + reachability (active health, DESIGN §5).

        Best-effort: an unreachable node is marked down (and dropped from model-aware routing)
        rather than raising. Run once at boot and periodically by the prober.
        """
        for endpoint in self._endpoints:
            try:
                names = {m.name for m in _list_model_infos(endpoint.provider)}
            except Exception as exc:  # node down / can't list
                log.warning("model: endpoint %s unreachable on refresh: %s", endpoint.name, exc)
                with self._lock:
                    endpoint.reachable = False
                    endpoint.models = set()
                continue
            with self._lock:
                endpoint.reachable = True
                endpoint.models = names

    def inventory_details(self) -> list[tuple[str, str, list[ModelInfo]]]:
        """(endpoint_name, endpoint_label, models) for each endpoint — for the fleet catalogue."""
        out: list[tuple[str, str, list[ModelInfo]]] = []
        for endpoint in self._endpoints:
            try:
                infos = _list_model_infos(endpoint.provider)
            except Exception as exc:
                log.warning("model: could not inventory %s: %s", endpoint.name, exc)
                infos = []
            out.append((endpoint.name, endpoint.name, infos))
        return out

    def start_prober(self, interval_s: float) -> None:
        """Start a background thread that calls ``refresh`` every ``interval_s`` (0 = no prober)."""
        if interval_s <= 0 or self._prober is not None:
            return

        def _loop() -> None:
            while not self._stop_prober.wait(interval_s):
                try:
                    self.refresh()
                except Exception as exc:  # the prober must never die
                    log.warning("model: prober refresh failed: %s", exc)

        self._prober = threading.Thread(target=_loop, name="mimir-model-prober", daemon=True)
        self._prober.start()

    def stop_prober(self) -> None:
        self._stop_prober.set()
        if self._prober is not None:
            self._prober.join(timeout=5)
            self._prober = None

    # -- routing ----------------------------------------------------------------------

    def _route(
        self, priority: Priority, fn: Callable[[Provider], R], kind: str,
        model: str | None = None, *, max_retries: int | None = None,
    ) -> R:
        with self._lock:
            self._stats["calls"] += 1
        candidates = self._candidates(priority, model)
        if not candidates:
            with self._lock:
                self._stats["transient_fails"] += 1
            raise ProviderError(
                f"no available provider endpoint for {kind}"
                f"{f' (model {model})' if model else ''} (all saturated/missing); deferring",
                transient=True,
            )

        base_retries = self._max_retries if max_retries is None else max_retries
        last_exc: ProviderError | None = None
        for index, (endpoint, clamp) in enumerate(candidates):
            if index > 0:
                with self._lock:
                    self._stats["failovers"] += 1
            attempt_retries = 0 if clamp else base_retries
            with self._lock:
                endpoint.inflight += 1
            try:
                result, elapsed = self._attempt(endpoint, fn, attempt_retries)
            except ProviderError as exc:
                self._record_failure(endpoint)
                last_exc = exc
                if not exc.transient:
                    raise  # permanent failure — failover won't help
                continue  # transient — try the next endpoint
            else:
                # Passive measurement (DESIGN §5): learn this node's speed from the real call we
                # just made — no synthetic probe. Only chat (str) output is token-normalizable.
                if kind == "chat" and model is not None and isinstance(result, str):
                    self._observe_latency(endpoint, model, elapsed, result)
                self._record_success(endpoint)
                with self._lock:
                    self._stats["ok"] += 1
                return result
            finally:
                with self._lock:
                    endpoint.inflight -= 1

        with self._lock:
            self._stats["transient_fails"] += 1
        raise ProviderError(
            f"all {len(candidates)} provider endpoint(s) failed for {kind}; deferring",
            transient=True,
        ) from last_exc

    def _attempt(
        self, endpoint: _Endpoint, fn: Callable[[Provider], R], max_retries: int
    ) -> tuple[R, float]:
        """One endpoint, with retry/backoff on transient failures.

        Returns ``(result, elapsed_s)`` where ``elapsed_s`` times only the *successful* provider
        call — backoff sleeps and failed attempts are excluded, so the latency sample reflects the
        node's real throughput, not how flaky it was getting there.
        """
        for attempt in range(max_retries + 1):
            started = self._clock()
            try:
                result = fn(endpoint.provider)
            except ProviderError as exc:
                if exc.transient and attempt < max_retries:
                    wait = self._backoff_base_s * (2**attempt)  # 1s, 2s, 4s, ...
                    with self._lock:
                        self._stats["retries"] += 1
                    log.warning(
                        "model: transient fail on %s (attempt %d/%d); retrying in %.1fs: %s",
                        endpoint.name,
                        attempt + 1,
                        max_retries + 1,
                        wait,
                        exc,
                    )
                    self._sleep(wait)
                    continue
                raise
            except Exception as exc:  # non-ProviderError → name it, treat as permanent
                raise ProviderError(
                    f"{endpoint.name}: {type(exc).__name__}: {exc}"
                ) from exc
            else:
                return result, self._clock() - started
        raise ProviderError(f"{endpoint.name}: retry loop exhausted")  # defensive; unreachable

    # -- admission / health -----------------------------------------------------------

    def _candidates(
        self, priority: Priority, model: str | None = None
    ) -> list[tuple[_Endpoint, bool]]:
        """Endpoints to try, in order, each flagged whether to clamp retries to a single try.

        Model-aware (DESIGN §5): only nodes that *have* the model are considered (endpoints with
        unknown inventory are included optimistically; if none is known to have it, we fall back to
        all and let the call fail naturally). Within a tier, least-loaded first so bursts spread.

        - CHAT_CRITICAL: every endpoint, full retries (chat must run; saturated as last resort).
        - USER_ADJACENT: healthy first (full retries), then saturated with a single try.
        - BACKGROUND/IDLE: healthy only — if none, defer (empty list → transient fail).
        """
        now = self._clock()
        with self._lock:
            eps = [e for e in self._endpoints if e.reachable and e.name not in self._disabled]
            if not eps:  # all unreachable/disabled → fail safe, don't hard-block (DESIGN §10)
                eps = [e for e in self._endpoints if e.name not in self._disabled] or \
                    list(self._endpoints)
            if model is not None:
                known = [e for e in eps if e.models]
                if known:
                    havers = [e for e in eps if not e.models or model in e.models]
                    if havers:  # else: nobody known to have it → keep all (optimistic)
                        eps = havers
            # Within a tier, order by EXPECTED WAIT — measured latency × current load — so a call
            # routes to the node that will answer it soonest (DESIGN §5). With no latency known yet
            # (all at the default seed) this reduces to least-loaded-first, the prior behaviour.
            healthy = sorted(
                (e for e in eps if e.saturated_until <= now),
                key=lambda e: self._expected_wait(e, model),
            )
            saturated = sorted(
                (e for e in eps if e.saturated_until > now),
                key=lambda e: self._expected_wait(e, model),
            )

        if priority == Priority.CHAT_CRITICAL:
            return [(e, False) for e in healthy] + [(e, False) for e in saturated]
        if priority == Priority.USER_ADJACENT:
            return [(e, False) for e in healthy] + [(e, True) for e in saturated]
        return [(e, False) for e in healthy]

    def _record_failure(self, endpoint: _Endpoint) -> None:
        now = self._clock()
        with self._lock:
            dq = endpoint.failures
            while dq and now - dq[0] > self._sat_window_s:
                dq.popleft()
            dq.append(now)
            if len(dq) >= self._sat_threshold:
                endpoint.saturated_until = now + self._sat_cooldown_s
                dq.clear()
                log.warning(
                    "model: endpoint %r SATURATED (%d+ transient fails in %.0fs); "
                    "background work deferred, chat still attempted for %.0fs",
                    endpoint.name,
                    self._sat_threshold,
                    self._sat_window_s,
                    self._sat_cooldown_s,
                )

    def _record_success(self, endpoint: _Endpoint) -> None:
        with self._lock:
            endpoint.failures.clear()
            endpoint.saturated_until = 0.0

    # -- live latency (speed-aware routing, DESIGN §5) --------------------------------

    def _expected_wait(self, endpoint: _Endpoint, model: str | None) -> float:
        """This node's estimated seconds-to-answer for ``model`` right now: measured (or seeded, or
        default) per-turn latency scaled by current load. Caller holds ``self._lock``."""
        base = self._default_latency_s
        if model is not None:
            stat = endpoint.latency.get(model)
            if stat is not None and stat.value is not None:
                base = stat.value
        return base * (endpoint.inflight + 1)

    def _observe_latency(
        self, endpoint: _Endpoint, model: str, elapsed_s: float, output: str
    ) -> None:
        """Fold a real (or probe) generation's wall-time into this node's per-model estimate."""
        sample = normalize_latency(elapsed_s, output)
        with self._lock:
            endpoint.latency.setdefault(model, LatencyStat()).observe(
                sample, alpha=self._latency_alpha, now=self._clock()
            )

    def seed_latency(self, seeds: dict[tuple[str, str], float]) -> None:
        """Prime per-``(node, model)`` estimates from the catalogue's ``return_time`` snapshot so
        routing starts informed, not cold (DESIGN §5). A seed only applies while no real sample has
        landed for that pair — lived experience always wins over the frozen benchmark."""
        with self._lock:
            by_name = {e.name: e for e in self._endpoints}
            for (node, model), value in seeds.items():
                ep = by_name.get(node)
                if ep is not None and value is not None:
                    ep.latency.setdefault(model, LatencyStat()).seed(value)

    def latency_snapshot(self) -> dict[tuple[str, str], dict[str, object]]:
        """Live per-``(node, model)`` latency — for introspection and write-back to the placement
        matrix: ``return_time`` (s/turn), ``samples`` (real obs), ``age_s`` (since last)."""
        now = self._clock()
        with self._lock:
            return {
                (e.name, model): {
                    "return_time": stat.value,
                    "samples": stat.samples,
                    "age_s": round(now - stat.last_ts, 1) if stat.last_ts else None,
                }
                for e in self._endpoints
                for model, stat in e.latency.items()
                if stat.value is not None
            }

    def idle_nodes(self) -> list[str]:
        """Reachable, non-vetoed endpoints with nothing in flight — safe targets for the rare idle
        latency heartbeat (probing a busy node would both add load and skew its measurement)."""
        with self._lock:
            return [
                e.name for e in self._endpoints
                if e.reachable and e.inflight == 0 and e.name not in self._disabled
            ]

    def probe_latency(
        self, node: str, model: str, messages: list[Message], params: dict[str, object]
    ) -> float | None:
        """Send ONE probe generation to a specific ``node``+``model`` and record its latency — the
        idle heartbeat (DESIGN §5; kept rare — real traffic is the primary signal). Returns the
        normalized s/turn, or ``None`` if the node is gone/busy/errored (a probe must never raise
        into the caller, and a failed probe is not recorded as fast)."""
        with self._lock:
            ep = next((e for e in self._endpoints if e.name == node), None)
            skip = ep is None or not ep.reachable or ep.inflight > 0 or node in self._disabled
        if ep is None or skip:
            return None
        started = self._clock()
        try:
            out = ep.provider.chat(model, messages, params)
        except Exception as exc:  # a probe failure must never propagate; just leave the estimate be
            log.warning("model: idle latency probe failed on %s/%s: %s", node, model, exc)
            return None
        elapsed = self._clock() - started
        self._observe_latency(ep, model, elapsed, out)
        return normalize_latency(elapsed, out)
