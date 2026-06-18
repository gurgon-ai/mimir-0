"""The reference local-model provider: Ollama over HTTP, using only stdlib ``urllib``.

This keeps the runtime contract intact — no third-party HTTP client in core. It targets a
local Ollama server (``http://localhost:11434`` by default). See ``docs/SETUP.md`` for how to
install Ollama and pull recommended models.

Connection-level failures are raised as ``ProviderError(transient=True)`` so background
cognition can back off against a busy or down backend instead of corrupting state (DESIGN §5).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from ...errors import ProviderError
from ..provider import Message, ModelInfo, parse_params_b

log = logging.getLogger("mimir.model.ollama")

_DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT_S = 120


class OllamaProvider:
    def __init__(self, host: str = _DEFAULT_HOST, *, timeout: float = _TIMEOUT_S) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout

    def chat(self, model: str, messages: list[Message], params: dict[str, Any]) -> str:
        return self.chat_timed(model, messages, params)[0]

    def chat_timed(
        self, model: str, messages: list[Message], params: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Like :meth:`chat`, but also returns Ollama's ``eval_count`` (tokens generated) and
        ``eval_duration`` (ns of *pure generation*, excluding model-load / VRAM-swap). Callers time
        latency from these instead of wall-clock so a cold load can't masquerade as a slow model — a
        fast MoE (gemma4:26b, 4B active, ~210 tok/s) caught mid swap was recording a fake ~38s/turn
        and losing speed-weighted roles (DESIGN §4). Returns ``(content, eval_count,
        eval_duration_ns)``; counts are 0 when Ollama omits them, so the caller can fall back to a
        wall-clock estimate."""
        timeout, params = _split_timeout(params)
        think, opts = _split_think(params)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": _to_options(opts),
        }
        data = self._post("/api/chat", payload, timeout=timeout)
        try:
            content = str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected /api/chat response shape from Ollama: {data!r}"
            ) from exc
        return content, int(data.get("eval_count") or 0), int(data.get("eval_duration") or 0)

    def _tags(self) -> list[dict[str, Any]]:
        req = urllib.request.Request(f"{self._host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            raise ProviderError(
                f"could not reach Ollama at {self._host}: {exc}", transient=True
            ) from exc
        models = data.get("models", [])
        return [m for m in models if isinstance(m, dict) and "name" in m]

    def list_models(self) -> list[str]:
        """Discover installed model names via Ollama's /api/tags (council auto-discovery)."""
        return [str(m["name"]) for m in self._tags()]

    def has_vision(self, model: str) -> bool:
        """Does this model carry a vision projector? Read from ``/api/show`` (its ``capabilities`` +
        the ``model_info`` clip/vision tensor keys), which — unlike the ``/api/tags`` capabilities
        the catalogue stores — is consistent across Ollama versions. Used only to decide whether a
        model is *worth* empirically probing for vision per node (whether it actually SEES on a node
        is a separate, version-dependent probe). False on any error (no claim → don't probe)."""
        try:
            data = self._post("/api/show", {"model": model}, timeout=self._timeout)
        except (ProviderError, OSError, ValueError) as exc:
            log.warning("ollama: /api/show failed for %s (vision unknown): %s", model, exc)
            return False
        caps = [str(c).lower() for c in (data.get("capabilities") or [])]
        if "vision" in caps:
            return True
        info = data.get("model_info") or {}
        return any("vision" in str(k).lower() or "clip" in str(k).lower() for k in info)

    def model_details(self) -> list[ModelInfo]:
        """Rich model metadata for the fleet catalogue (family, weight, quant, capabilities)."""
        out: list[ModelInfo] = []
        for m in self._tags():
            details = m.get("details") or {}
            out.append(
                ModelInfo(
                    name=str(m["name"]),
                    family=str(details.get("family", "")),
                    params_b=parse_params_b(str(details.get("parameter_size", ""))),
                    quantization=str(details.get("quantization_level", "")),
                    context_length=int(details.get("context_length", 0) or 0),
                    size_bytes=int(m.get("size", 0) or 0),
                    capabilities=[str(c) for c in (m.get("capabilities") or [])],
                )
            )
        return out

    def chat_stream(
        self, model: str, messages: list[Message], params: dict[str, Any]
    ) -> Iterator[str]:
        """Stream a chat completion: Ollama's ``stream=true`` newline-delimited JSON deltas."""
        _timeout, params = _split_timeout(params)   # streaming uses the provider timeout; strip it
        think, opts = _split_think(params)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": _to_options(opts),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            transient = exc.code >= 500 or exc.code in (404, 422)
            raise ProviderError(
                f"Ollama returned HTTP {exc.code} for /api/chat (stream)", transient=transient
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"could not reach Ollama at {self._host} ({exc.reason}). Is it running?",
                transient=True,
            ) from exc
        with resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a malformed keep-alive line; the stream continues
                content = (obj.get("message") or {}).get("content")
                if content:
                    yield content
                if obj.get("done"):
                    break

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": texts}
        data = self._post("/api/embed", payload)
        try:
            vectors = data["embeddings"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected /api/embed response shape from Ollama: {data!r}"
            ) from exc
        return [[float(x) for x in vec] for vec in vectors]

    # -- transport --------------------------------------------------------------------

    def _post(
        self, path: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        url = f"{self._host}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        deadline = timeout or self._timeout
        try:
            with urllib.request.urlopen(req, timeout=deadline) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # The server answered with an error status. 5xx and 404/422 (Ollama returns these
            # briefly while loading/unloading a model) are transient → worth a retry. Other
            # 4xx are request-level and permanent → fail fast.
            detail = exc.read().decode("utf-8", "replace")
            # …but a 404 "model not found, try pulling it first" is PERMANENT for this node — the
            # model isn't installed, so retrying just burns time and spams the log. Fail fast.
            model_missing = exc.code == 404 and "not found" in detail.lower()
            transient = not model_missing and (exc.code >= 500 or exc.code in (404, 422))
            raise ProviderError(
                f"Ollama returned HTTP {exc.code} for {path}: {detail}",
                transient=transient,
            ) from exc
        except urllib.error.URLError as exc:
            # Could not reach the server at all — transient; let callers defer/retry.
            raise ProviderError(
                f"could not reach Ollama at {self._host} ({exc.reason}). Is it running? "
                f"See docs/SETUP.md.",
                transient=True,
            ) from exc
        except TimeoutError as exc:
            # Socket timeout — the node took the full deadline and still didn't answer. Transient
            # (worth failing over), but flagged ``timeout`` so the pool stops retrying it and cools
            # it down instead of burning another full deadline on a node that isn't answering.
            raise ProviderError(
                f"Ollama request to {path} timed out after {deadline}s",
                transient=True, timeout=True,
            ) from exc
        try:
            parsed: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Ollama returned non-JSON for {path}: {raw!r}") from exc
        return parsed


_TIMEOUT_KEY = "__timeout_s__"


def _split_timeout(params: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """Pull a per-call timeout out of the params (a reserved key, not an Ollama option). Lets the
    benchmark bound a single call far tighter than the provider's production ceiling, so a slow
    model fails fast instead of stalling the run; ``None`` falls back to the provider's timeout."""
    opts = dict(params)
    raw = opts.pop(_TIMEOUT_KEY, None)
    return (float(raw) if raw else None), opts


def _split_think(params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Pull ``think`` out of the tuned params — it is a TOP-LEVEL Ollama field, not an option, so
    a ``think`` left in ``options`` is silently ignored.

    Defaults to **off**: thinking-mode noticeably slows generation and rarely improves output in
    practice; it earns its cost only in specific spots (e.g. some models on tool selection). Set
    ``think = true`` (or a level string) on a role to opt in where it helps. ``think=false`` is
    accepted by non-thinking models too, so sending it everywhere is safe.
    """
    opts = dict(params)
    think = opts.pop("think", False)
    return think, opts


def _to_options(params: dict[str, Any]) -> dict[str, Any]:
    """Translate Mimir's tuned params into Ollama's ``options`` block.

    ``max_tokens`` is mapped to Ollama's ``num_predict``; everything else passes through
    (``temperature``, ``num_ctx``, ``top_p``, …). ``num_ctx`` must stay consistent across
    callers of the same warm model, which is why it lives in config (DESIGN §4).
    """
    options = dict(params)
    if "max_tokens" in options:
        options["num_predict"] = options.pop("max_tokens")
    return options
