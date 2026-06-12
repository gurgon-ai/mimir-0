"""Strip internal epistemic annotations from model output before it reaches a human.

The prompt renders recalled facts with provenance tags like ``[tier=...; source=...]`` and
may inject an ``[epistemic check]`` flag. These are an *internal* prompt convention — they
must never appear in a user-facing reply. Capable models follow the instruction not to echo
them; small models (observed: ``gemma3:4b``) absorb the *style* and spray invented tags onto
their own sentences. Instruction alone is therefore not enough, so we strip them deterministically.

Two entry points:

- :func:`strip_epistemic_tags` — clean a complete string (non-streaming replies, stored text).
- :class:`StreamTagStripper` — a stateful filter for token-by-token streaming, where a tag may
  be split across deltas. It holds back text from an open ``[`` until it can decide whether the
  bracket is a tag (drop it) or ordinary content (emit it).
"""

from __future__ import annotations

import re

# A tag is ``[tier=...]``, ``[source=...]`` or the literal ``[epistemic check]``. We also eat a
# single run of leading whitespace so dropping a trailing tag doesn't leave a double space.
_TAG_RE = re.compile(
    r"[ \t]*\[(?:(?:tier|source)\s*=[^\]]*|epistemic check)\]",
    re.IGNORECASE,
)

# An open ``[`` longer than this without closing can't be one of our short tags — stop holding it.
_MAX_TAG_LEN = 160


def strip_epistemic_tags(text: str) -> str:
    """Remove ``[tier=...]`` / ``[source=...]`` / ``[epistemic check]`` annotations from ``text``.

    Collapses the whitespace a removed tag leaves behind so the prose stays clean. Idempotent.
    """
    cleaned = _TAG_RE.sub("", text)
    # Tidy artifacts: spaces before punctuation, doubled spaces, trailing spaces per line.
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned


class StreamTagStripper:
    """Streaming-safe tag stripper: ``feed`` deltas in, get emittable (clean) text out.

    A tag can straddle two deltas, so once we see an unclosed ``[`` we hold everything from it
    until either the bracket closes (then we know to drop or keep it) or the held run grows past
    :data:`_MAX_TAG_LEN` (then it isn't one of our tags, so we release it). Call :meth:`flush`
    at end-of-stream to release anything still held.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        """Consume a streamed chunk; return text safe to emit now (may be empty).

        Trailing whitespace and any open ``[`` are held back, because a tag (which the regex
        eats together with the whitespace in front of it) may continue in the next delta.
        """
        self._buf += delta
        out: list[str] = []
        while True:
            open_idx = self._buf.find("[")
            if open_idx == -1:
                # No bracket. Emit everything except a trailing whitespace run, which could turn
                # out to sit in front of a tag arriving in the next delta.
                emit = self._buf.rstrip(" \t")
                out.append(emit)
                self._buf = self._buf[len(emit) :]
                break
            # Text before the bracket is safe, minus the whitespace run adjacent to the "[".
            before = self._buf[:open_idx]
            keep = before.rstrip(" \t")
            ws = before[len(keep) :]
            if keep:
                out.append(keep)
            rest = self._buf[open_idx:]  # starts at "["
            close_idx = rest.find("]")
            if close_idx == -1:
                # Bracket still open — hold "ws + [...". Release it only if it grows too long to
                # be one of our tags (then the "[" was ordinary content).
                held = ws + rest
                if len(held) > _MAX_TAG_LEN:
                    out.append(ws + "[")
                    self._buf = rest[1:]
                    continue
                self._buf = held
                break
            candidate = ws + rest[: close_idx + 1]
            if _TAG_RE.fullmatch(candidate):
                # A complete tag (with any leading whitespace) — drop it, keep scanning after it.
                self._buf = rest[close_idx + 1 :]
                continue
            # A closed bracket that isn't our tag (e.g. "[1]"): release "ws + [" and rescan.
            out.append(ws + "[")
            self._buf = rest[1:]
        return "".join(out)

    def flush(self) -> str:
        """Release any held text at end-of-stream (an unclosed ``[`` was ordinary content)."""
        tail = strip_epistemic_tags(self._buf)
        self._buf = ""
        return tail
