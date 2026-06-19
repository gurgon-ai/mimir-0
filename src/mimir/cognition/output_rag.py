"""Output-triggered RAG helpers (DESIGN §5a): grounding *and self-correction* on the model's OWN
reply. After a turn, the burst worker retrieves memory relevant to what the model just said; if
that reply contradicts a fact that outranks its own generation, the system flags it to itself for
the next turn — catching its own drift from what it knows ("edge-awareness is the human part").

The retrieval orchestration lives in the brain (it needs the gateways + model); the pure pieces —
which beliefs outrank a generated reply, parsing the self-check verdict, and composing the
surfaces — live here so they're testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.hybrid import ScoredMemory
from ..storage.models import EvidenceTier, Memory

# Tiers that carry more authority than the system's own freshly generated reply. A reply that
# contradicts one of these is a genuine self-inconsistency worth flagging; a clash with a peer or
# inferred memory is not (the reply is, itself, that calibre of source). DESIGN §3b.
AUTHORITY_TIERS = frozenset({
    EvidenceTier.STATED_BY_PRIMARY_USER,
    EvidenceTier.STATED_BY_TRUSTED,
    EvidenceTier.DOCUMENT,
    EvidenceTier.MULTI_SOURCE,
})

_CHECK_RE = re.compile(r"^\s*(CONTRADICTS|NOTE)\s*:\s*", re.IGNORECASE | re.MULTILINE)
_NONE_VALUES = frozenset({"", "none", "n/a", "nil", "no", "0"})


@dataclass(slots=True)
class OutputCheck:
    """A detected self-contradiction: 1-based index of the contradicted belief + a short note."""

    index: int
    note: str = ""


def authority_beliefs(scored: list[ScoredMemory]) -> list[Memory]:
    """The retrieved memories that outrank a generated reply — the ones worth a self-check."""
    return [s.memory for s in scored if s.memory.evidence_tier in AUTHORITY_TIERS]


def parse_output_check(raw: str, count: int) -> OutputCheck | None:
    """Parse the self-check verdict into a contradiction, or ``None`` for no conflict.

    Tolerant by design: anything that isn't a valid in-range ``CONTRADICTS`` index — a missing
    label, ``none``, an out-of-range number, garbage — reads as "no conflict", so a formatting slip
    never fabricates a correction (a false self-correction is worse than a missed one)."""
    text = (raw or "").strip()
    matches = list(_CHECK_RE.finditer(text))
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[match.group(1).lower()] = text[match.end() : end].strip()
    verdict = fields.get("contradicts", "")
    if verdict.strip().lower() in _NONE_VALUES:
        return None
    num = re.search(r"\d+", verdict)
    if not num:
        return None
    index = int(num.group(0))
    if index < 1 or index > count:
        return None
    note = fields.get("note", "").strip()
    if note.lower() in _NONE_VALUES:
        note = ""
    return OutputCheck(index=index, note=note)


def correction_surface(belief: Memory, note: str) -> str:
    """The tentative self-correction surfaced into the next turn (a framed caution, not a fact)."""
    base = f'Self-check: what you just said may conflict with something you hold — "{belief.text}"'
    return f"{base} ({note})" if note else base


def relevance_surface(scored: list[ScoredMemory]) -> str:
    """The grounding note surfaced when the reply opened a thread your memory speaks to."""
    lines = "\n".join(f"- {s.memory.text}" for s in scored)
    return f"Possibly relevant from memory, on what you last said:\n{lines}"
