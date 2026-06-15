"""Error capture for self-observability (DESIGN §10).

The doctrine is **fail loud**: every downgrade in core is a logged ``log.error``/``log.warning``.
That covers *emitting* failures; this adds the *introspection* side — a bounded ring capturing
``WARNING``+ off the ``mimir`` logger, so the system can report its own recent failures: to the
operator (the UI), and into the prompt so the model knows when it's degraded ("my last sentinel pass
failed", "a fleet node went down") instead of carrying on oblivious.

Pure stdlib ``logging``. A handler's ``emit`` must never raise, so capture is wrapped defensively.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
_FORMATTER = logging.Formatter()  # formatException lives on Formatter, not Handler


@dataclass(slots=True)
class ErrorRecord:
    ts: float
    level: str
    logger: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {"ts": self.ts, "level": self.level, "logger": self.logger, "message": self.message}


class RingErrorHandler(logging.Handler):
    """Keeps the most recent ``capacity`` log records (``WARNING``+) in an in-memory ring."""

    def __init__(self, capacity: int = 200) -> None:
        super().__init__(level=logging.WARNING)
        self._ring: deque[ErrorRecord] = deque(maxlen=capacity)
        self._counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                formatted = _FORMATTER.formatException(record.exc_info).strip().splitlines()
                if formatted:
                    message = f"{message} — {formatted[-1]}"  # the exception's final line
            # deque.append + dict writes are atomic under the GIL; no extra lock needed for this.
            self._ring.append(ErrorRecord(record.created, record.levelname, record.name, message))
            self._counts[record.levelname] = self._counts.get(record.levelname, 0) + 1
        except Exception:  # a handler must never raise — it would break the call that logged
            pass

    def recent(self, *, limit: int = 10, min_level: str = "WARNING") -> list[ErrorRecord]:
        """The most recent records at or above ``min_level``, oldest-first, capped to ``limit``."""
        floor = _LEVEL_ORDER.get(min_level, 2)
        items = [r for r in list(self._ring) if _LEVEL_ORDER.get(r.level, 2) >= floor]
        return items[-limit:]

    def within(self, seconds: float, now: float, *, limit: int = 10) -> list[ErrorRecord]:
        """Records from the last ``seconds`` (the live 'most recent errors' window), capped."""
        cutoff = now - seconds
        items = [r for r in list(self._ring) if r.ts >= cutoff]
        return items[-limit:]

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


_handler: RingErrorHandler | None = None


def install_error_capture(capacity: int = 200) -> RingErrorHandler:
    """Attach (once) the ring handler to the ``mimir`` logger and return it. Idempotent."""
    global _handler
    if _handler is None:
        _handler = RingErrorHandler(capacity)
        logging.getLogger("mimir").addHandler(_handler)
    return _handler


def render_errors(records: list[ErrorRecord]) -> str:
    """A compact, model-readable rendering of recent errors for the system-health section."""
    lines = []
    for r in records:
        where = r.logger[6:] if r.logger.startswith("mimir.") else r.logger  # strip the 'mimir.'
        lines.append(f"- [{r.level.lower()}] {where}: {r.message}")
    return "\n".join(lines)
