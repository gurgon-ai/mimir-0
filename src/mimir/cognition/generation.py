"""generation.py — the guard at every store boundary that persists model output.

A rejected, empty or placeholder generation must never be stored as if it were content. Each of
these happens in practice: a gateway refuses a call and hands back a fallback value, which an
ingest stores as an empty document; a reasoning model spends its whole budget thinking and returns
an empty body, which is saved as the next turn's note; a working-memory fold gets an empty summary
back, saves it, and deletes the exchanges it was meant to summarise. The check lives here, once,
and every writer calls it.

``is_valid_generation`` is deliberately conservative and rejects nothing a model would plausibly
mean: the empty string, whitespace, the council's own ``[unavailable: …]`` placeholder, and the
handful of error-shaped prefixes a provider or dispatcher can return as text.
"""

from __future__ import annotations

MIN_CHARS = 8

# Text that is a report of failure, not content. Lower-cased prefix match after stripping.
_PLACEHOLDER_PREFIXES = (
    "[unavailable",
    "[error",
    "[failed",
    "[model unavailable",
    "[no response",
    "error:",
    "provider error",
    "<error",
    "unavailable:",
)


class EmptyGeneration(RuntimeError):
    """The model returned nothing usable; the caller must not store, delete or fold on it."""


def rejection_reason(text: object, *, min_chars: int = MIN_CHARS) -> str | None:
    """Why ``text`` must not be stored, or ``None`` if it may be."""
    if text is None:
        return "no text"
    s = str(text).strip()
    if not s:
        return "empty"
    if len(s) < min_chars:
        return f"too short ({len(s)} chars)"
    low = s.lower()
    for p in _PLACEHOLDER_PREFIXES:
        if low.startswith(p):
            return f"placeholder ({p!r})"
    return None


def is_valid_generation(text: object, *, min_chars: int = MIN_CHARS) -> bool:
    return rejection_reason(text, min_chars=min_chars) is None


def require_generation(text: object, *, what: str, min_chars: int = MIN_CHARS) -> str:
    """Return the stripped text, or raise :class:`EmptyGeneration` naming what was being made."""
    reason = rejection_reason(text, min_chars=min_chars)
    if reason is not None:
        raise EmptyGeneration(f"{what}: model returned nothing usable ({reason}); not stored")
    return str(text).strip()
