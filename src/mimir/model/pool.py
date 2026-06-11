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
from .priority import Priority
from .provider import Message, Provider

log = logging.getLogger("mimir.model.pool")

R = TypeVar("R")


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
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._stats = {"calls": 0, "ok": 0, "retries": 0, "failovers": 0, "transient_fails": 0}

    # -- public API -------------------------------------------------------------------

    def chat(
        self, model: str, messages: list[Message], params: dict[str, object], *, priority: Priority
    ) -> str:
        return self._route(priority, lambda p: p.chat(model, messages, params), "chat")

    def embed(
        self, model: str, texts: list[str], *, priority: Priority
    ) -> list[list[float]]:
        return self._route(priority, lambda p: p.embed(model, texts), "embed")

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
        candidates = self._candidates(priority)
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
            yield first
            yield from gen  # remaining tokens; a mid-stream error propagates as-is
            return

        with self._lock:
            self._stats["transient_fails"] += 1
        raise ProviderError(
            "all provider endpoint(s) failed for chat stream; deferring", transient=True
        ) from last_exc

    def get_stats(self) -> dict[str, object]:
        with self._lock:
            now = self._clock()
            return {
                **self._stats,
                "endpoints": [e.name for e in self._endpoints],
                "saturated": {
                    e.name: round(e.saturated_until - now, 1)
                    for e in self._endpoints
                    if e.saturated_until > now
                },
            }

    # -- routing ----------------------------------------------------------------------

    def _route(self, priority: Priority, fn: Callable[[Provider], R], kind: str) -> R:
        with self._lock:
            self._stats["calls"] += 1
        candidates = self._candidates(priority)
        if not candidates:
            with self._lock:
                self._stats["transient_fails"] += 1
            raise ProviderError(
                f"no available provider endpoint for {kind} (all saturated); deferring",
                transient=True,
            )

        last_exc: ProviderError | None = None
        for index, (endpoint, clamp) in enumerate(candidates):
            if index > 0:
                with self._lock:
                    self._stats["failovers"] += 1
            max_retries = 0 if clamp else self._max_retries
            try:
                result = self._attempt(endpoint, fn, max_retries)
            except ProviderError as exc:
                self._record_failure(endpoint)
                last_exc = exc
                if not exc.transient:
                    raise  # permanent failure — failover won't help
                continue  # transient — try the next endpoint
            else:
                self._record_success(endpoint)
                with self._lock:
                    self._stats["ok"] += 1
                return result

        with self._lock:
            self._stats["transient_fails"] += 1
        raise ProviderError(
            f"all {len(candidates)} provider endpoint(s) failed for {kind}; deferring",
            transient=True,
        ) from last_exc

    def _attempt(self, endpoint: _Endpoint, fn: Callable[[Provider], R], max_retries: int) -> R:
        """One endpoint, with retry/backoff on transient failures."""
        for attempt in range(max_retries + 1):
            try:
                return fn(endpoint.provider)
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
        raise ProviderError(f"{endpoint.name}: retry loop exhausted")  # defensive; unreachable

    # -- admission / health -----------------------------------------------------------

    def _candidates(self, priority: Priority) -> list[tuple[_Endpoint, bool]]:
        """Endpoints to try, in order, each flagged whether to clamp retries to a single try.

        - CHAT_CRITICAL: every endpoint, full retries (chat must run; saturated as last resort).
        - USER_ADJACENT: healthy first (full retries), then saturated with a single try.
        - BACKGROUND/IDLE: healthy only — if none, defer (empty list → transient fail).
        """
        now = self._clock()
        with self._lock:
            healthy = [e for e in self._endpoints if e.saturated_until <= now]
            saturated = [e for e in self._endpoints if e.saturated_until > now]

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
