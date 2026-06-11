"""Section primitives for ``build_context()`` and the context-source registration seam.

A prompt is assembled from typed ``Section`` objects, each with its own budget and its own
accounting. The universal sections (identity, knowledge, sentinel note, uncertainty flag) ship
in core; deployment-specific context arrives as a registered ``ContextSource`` — the single
point where "universal" is kept strictly separate from anything app-specific (DESIGN §3e).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable


def estimate_tokens(text: str) -> int:
    """A cheap, dependency-free token estimate.

    Deliberately approximate (~4 chars/token). Context accounting needs a consistent ruler
    to flag truncation and report sizes (DESIGN §10), not exact provider tokenization.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class SectionTier(IntEnum):
    """How load-bearing a section is, for truncation policy.

    Truncating a HIGH section is a warning (a high-tier fact got dropped); truncating a LOW
    section is routine. Ordering also gives a stable priority when budget is tight.
    """

    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass(slots=True)
class Section:
    """One rendered slice of the prompt, with its accounting attached."""

    name: str
    title: str
    body: str
    tier: SectionTier = SectionTier.MEDIUM
    substantive: bool = False  # did it produce real content that counts toward source_count?
    requested_tokens: int = 0
    admitted_tokens: int = 0
    truncated: bool = False
    meta: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """The text that goes into the prompt for this section (title + body)."""
        return f"{self.title}\n{self.body}" if self.body else self.title


@runtime_checkable
class ContextSource(Protocol):
    """A registered contributor of a typed section to ``build_context()`` (DESIGN §3e, §10).

    Each source declares a budget + priority so the core can cap or disable a misbehaving
    source without starving core sections. v0 ships the seam; the universal sections do not
    go through it, but future typed layers (documents, working memory, …) will.
    """

    name: str
    budget_tokens: int
    tier: SectionTier

    def build(self, query: str, user: str | None) -> Section | None:
        """Produce this source's section for the turn, or ``None`` if it has nothing."""
        ...


_QUESTION_WORDS = {
    "what", "who", "whom", "whose", "when", "where", "why", "how", "which",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "am",
}
_FIRST_WORD_RE = re.compile(r"[a-z]+")


def is_question(text: str) -> bool:
    """Heuristic: does this turn ask something? Drives the uncertainty gate (DESIGN §3d)."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    match = _FIRST_WORD_RE.match(stripped.lower())
    return bool(match and match.group() in _QUESTION_WORDS)
