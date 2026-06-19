"""Deep-idle dialogue — inner-life Slice 3 (DESIGN §5a).

When the quiet runs *long*, the system holds a short two-voice dialogue with itself instead of a
single solo musing (Slice 1): a **reflective** voice proposes its honest current thinking, a
**skeptical** voice presses, the reflective voice grounds or concedes — then an insight is drawn.

The load-bearing mechanism (ported generic from the home AI's inner-dialogue) is **information
asymmetry**: the reflective voice is given the recent context; the skeptic is NOT, so it cannot take
a claim about "what I just said/did" on trust and must demand it be grounded in stored memory. That
forces the reflective voice to either cite what it actually holds or admit it's reconstructing —
exactly what a one-shot musing skips. A second mechanism, *convergence-as-validation*, lives in the
brain: an insight that re-derives independently earns confidence rather than piling up.

Pure + dependency-light like ``inner_life.py`` / ``sleep_cycle.py``: the brain supplies the model
call and the recent context; these functions are deterministic so tests drive them directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..prompts import (
    DEEP_IDLE_EXTRACT_SYSTEM,
    DEEP_IDLE_REFLECT_SYSTEM,
    DEEP_IDLE_SKEPTIC_SYSTEM,
)

# Voice labels (also the forum-style transcript tags).
REFLECT = "reflective"
SKEPTIC = "skeptic"

INSIGHT_TYPES = frozenset({"self_knowledge", "conflict", "gap", "debatable"})
_DEFAULT_TYPE = "self_knowledge"
_DEFAULT_CONFIDENCE = 0.4

_LABEL_RE = re.compile(r"^\s*(INSIGHT|TYPE|CONFIDENCE)\s*:\s*", re.IGNORECASE | re.MULTILINE)

Chat = Callable[[list[dict[str, str]]], str]


@dataclass(slots=True)
class DialogueTurn:
    voice: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"voice": self.voice, "text": self.text}


@dataclass(slots=True)
class DeepInsight:
    """The distilled takeaway: one insight, a coarse ``type`` (for light routing), and how grounded
    the dialogue made it feel (clamped to the single-pass ceiling by the brain)."""

    text: str
    kind: str = _DEFAULT_TYPE
    confidence: float = _DEFAULT_CONFIDENCE


def _render(turns: list[DialogueTurn]) -> str:
    return "\n".join(f"[{t.voice}] {t.text}" for t in turns)


def run_dialogue(
    chat: Chat, matter: str, recent_context: str, *, max_turns: int = 4
) -> list[DialogueTurn]:
    """Drive the asymmetric dialogue and return the transcript. ``matter`` is the stimulus; the
    reflective voice ALSO sees ``recent_context`` (the skeptic never does — that is the asymmetry).
    Caps at ``max_turns`` voices; stops early if a voice returns nothing. The injected ``chat`` is
    the brain's off-chat background router."""
    turns: list[DialogueTurn] = []
    context = (f"\n\nWhat's recently been going on:\n{recent_context}"
               if recent_context.strip() else "")

    opening = chat([
        {"role": "system", "content": DEEP_IDLE_REFLECT_SYSTEM},
        {"role": "user",
         "content": f"The matter: {matter}{context}\n\nOffer your honest thinking."},
    ]).strip()
    if not opening:
        return turns
    turns.append(DialogueTurn(REFLECT, opening))

    # Then alternate skeptic-challenge → reflective-defend until the turn cap.
    while len(turns) < max(2, max_turns):
        challenge = chat([
            {"role": "system", "content": DEEP_IDLE_SKEPTIC_SYSTEM},
            {"role": "user", "content":
                f"The matter: {matter}\n\nThe dialogue so far:\n{_render(turns)}\n\n"
                "Challenge the latest reflection."},
        ]).strip()
        if not challenge:
            break
        turns.append(DialogueTurn(SKEPTIC, challenge))
        if len(turns) >= max(2, max_turns):
            break
        defense = chat([
            {"role": "system", "content": DEEP_IDLE_REFLECT_SYSTEM},
            {"role": "user", "content":
                f"The matter: {matter}{context}\n\nThe dialogue so far:\n{_render(turns)}\n\n"
                "Respond to the challenge — ground your claim in memory, concede, or refine."},
        ]).strip()
        if not defense:
            break
        turns.append(DialogueTurn(REFLECT, defense))
    return turns


def extract_insight(chat: Chat, matter: str, turns: list[DialogueTurn]) -> DeepInsight | None:
    """Distil the dialogue into one structured insight, or ``None`` if it reached nothing usable."""
    if not turns:
        return None
    raw = chat([
        {"role": "system", "content": DEEP_IDLE_EXTRACT_SYSTEM},
        {"role": "user", "content": f"The matter: {matter}\n\nThe dialogue:\n{_render(turns)}"},
    ])
    return parse_insight(raw)


def parse_insight(raw: str) -> DeepInsight | None:
    """Parse the labelled extractor output. Tolerant: no ``INSIGHT`` label → treat the whole text as
    the insight; an unknown type → the default; an unparseable confidence → the default."""
    text = (raw or "").strip()
    if not text:
        return None
    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return DeepInsight(text=text)
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[match.group(1).lower()] = text[match.end() : end].strip()
    insight = fields.get("insight", "").strip()
    if not insight:
        return None
    kind = fields.get("type", "").strip().lower()
    if kind not in INSIGHT_TYPES:
        kind = _DEFAULT_TYPE
    conf = _parse_conf(fields.get("confidence", ""))
    return DeepInsight(text=insight, kind=kind, confidence=conf)


def _parse_conf(value: str) -> float:
    match = re.search(r"\d*\.?\d+", value)
    if not match:
        return _DEFAULT_CONFIDENCE
    try:
        return max(0.0, min(1.0, float(match.group(0))))
    except ValueError:
        return _DEFAULT_CONFIDENCE
