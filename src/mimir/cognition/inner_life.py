"""The live inner life: a low-frequency idle loop that thinks between conversations (DESIGN §5a).

The burst worker (``cognition/burst.py``) reclaims the *short* window right after a reply; this
reclaims the *long quiet* — the minutes and hours when nobody is talking. On a slow, user-tunable
cadence (default one thought every few minutes) it picks ONE universal stimulus — a recent error,
an un-deliberated conflict, the most salient memory, the working-memory thread — and composes a
brief reflection with a cheap background model. The thought is stored as a low-confidence, decaying
memory (provenance ``"inner life"``); it "earns its way" back into conversation only through
ordinary recall, never force-injected.

Two hard constraints shape every line (the §5a doctrine):
- **Chat priority** — it must never slow a live turn. It routes OFF the chat model, yields the
  instant a turn starts, runs on a long cadence, and stays quiet for a floor after each turn.
- **Edge cost** — it must be cheap on modest/distributed compute. One model call per cycle, paused
  when the fleet is down, OFF by default until the operator opts in.

Pure + dependency-light like ``sleep_cycle.py``: the brain runs the daemon and supplies the model
call; these functions are deterministic so tests drive them directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..embed.base import Embedder
from ..prompts import INNER_LIFE_SYSTEM
from ..storage.gateway import StorageGateway
from ..storage.models import MemoryKind
from ..storage.repo import list_memories
from .deliberation import NON_BELIEF_TIERS, surface_conflicts

log = logging.getLogger("mimir.inner_life")

# Kind order for tie-breaking when picking: a live error or tension earns attention before idle
# musing on an old memory; the working-memory thread is the last resort.
_KIND_PRIORITY = ("error", "conflict", "exchange", "memory", "working_memory")


@dataclass(slots=True)
class Stimulus:
    """One thing to dwell on this cycle: its ``kind`` (for rotation), the ``prompt`` handed to the
    voice, and a stable-ish ``key`` for light variety/dedup."""

    kind: str
    prompt: str
    key: str


def should_think(
    *,
    enabled: bool,
    turn_active: bool,
    degraded: bool,
    now: float,
    last_turn_at: float,
    last_thought_at: float,
    cadence_s: float,
    idle_floor_s: float,
) -> tuple[bool, str]:
    """Pure gate: may the loop think right now? Returns ``(ok, reason)``. Chat priority and edge
    cost live here — a turn in flight, a down fleet, too-soon-after-a-turn, or within-cadence all
    say no. ``last_turn_at``/``last_thought_at`` of 0 mean 'never' (no floor to clear)."""
    if not enabled:
        return False, "disabled"
    if turn_active:
        return False, "turn in flight"
    if degraded:
        return False, "fleet degraded"
    if last_turn_at and now - last_turn_at < idle_floor_s:
        return False, "too soon after a turn"
    if last_thought_at and now - last_thought_at < cadence_s:
        return False, "within cadence"
    return True, "ok"


def _clip(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def gather_stimuli(
    storage: StorageGateway,
    *,
    embedder: Embedder | None = None,
    recent_errors: list[str] | None = None,
    working_memory_text: str = "",
    last_exchange: tuple[str, str] | None = None,
) -> list[Stimulus]:
    """Build candidate stimuli from universal signals only (generic-core principle). Cheap and
    read-only: a recent error, an un-deliberated conflict, the most salient memory, the
    working-memory thread, the last exchange. The brain supplies the error/WM/exchange inputs (they
    aren't pure storage reads)."""
    out: list[Stimulus] = []

    for err in (recent_errors or [])[:1]:  # one error is plenty to muse on
        out.append(Stimulus(
            kind="error",
            prompt=(f"I recently hit an error while working: {_clip(err)}. "
                    "What might it mean for how I'm running, and is it worth flagging?"),
            key=f"error:{_clip(err, 60)}",
        ))

    conflicts = surface_conflicts(storage, embedder=embedder)
    if conflicts:
        c = conflicts[0]
        out.append(Stimulus(kind="conflict", prompt=c.question, key=c.key))

    # Dwell on *stated beliefs* only — not reference docs (DOCUMENT) and not the system's own output
    # (prior musings and council verdicts are INFERRED), which would just make it loop on itself.
    real = sorted(
        (m for m in list_memories(storage, user=None, kind=MemoryKind.MEMORY)
         if not m.archived and m.evidence_tier not in NON_BELIEF_TIERS),
        key=lambda m: m.salience, reverse=True,
    )
    if real:
        m = real[0]
        out.append(Stimulus(
            kind="memory",
            prompt=f"Something I hold: {_clip(m.text)}. What does it connect to, or open up?",
            key=f"memory:{m.id}",
        ))

    if working_memory_text.strip():
        out.append(Stimulus(
            kind="working_memory",
            prompt=(f"Where my recent thinking left off: {_clip(working_memory_text)}. "
                    "What thread there is worth picking back up?"),
            key="working_memory",
        ))

    if last_exchange:
        user_text, reply = last_exchange
        out.append(Stimulus(
            kind="exchange",
            prompt=(f"Earlier it was said: {_clip(user_text)!r} and I answered {_clip(reply)!r}. "
                    "On reflection, what did I miss or want to revisit?"),
            key="exchange",
        ))
    return out


def pick_stimulus(stimuli: list[Stimulus], *, avoid_kind: str | None = None) -> Stimulus | None:
    """Choose one stimulus by kind priority, preferring a kind different from last time (so it
    doesn't fixate on one thread). Deterministic — no RNG, so tests stay stable."""
    if not stimuli:
        return None
    ordered = sorted(
        stimuli,
        key=lambda s: _KIND_PRIORITY.index(s.kind) if s.kind in _KIND_PRIORITY
        else len(_KIND_PRIORITY),
    )
    for s in ordered:
        if s.kind != avoid_kind:
            return s
    return ordered[0]


def compose_thought(chat: Callable[[list[dict[str, str]]], str], stimulus: Stimulus) -> str:
    """Compose one reflection with the injected ``chat`` callable (the brain routes it to a cheap
    background model, off the chat model). Returns the trimmed thought (may be empty)."""
    messages = [
        {"role": "system", "content": INNER_LIFE_SYSTEM},
        {"role": "user", "content": stimulus.prompt},
    ]
    return (chat(messages) or "").strip()
