"""The sentinel: an async reflective pass that leaves a note for the next turn (DESIGN §2, §5).

After a turn, the sentinel (``reasoning`` role) reviews what just happened and writes a short
note to the system's future self. That note is stored as a ``SENTINEL_NOTE`` memory and lands
in the high-attention end slot of the *next* turn's prompt.

It runs **off the hot path** and its failure must never touch the core turn→bake→recall loop
(DESIGN §10). The brain runs it in the background; this module just does the reflection+write and
raises on failure so the brain's wrapper can log it loudly without crashing the turn.
"""

from __future__ import annotations

import logging

from ..model.gateway import ModelGateway
from ..prompts import SENTINEL_SYSTEM
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import save_memory

log = logging.getLogger("mimir.sentinel")


def run_sentinel(
    model: ModelGateway,
    storage: StorageGateway,
    *,
    user: str | None,
    turn_text: str,
    reply: str,
) -> Memory:
    """Reflect on the just-completed turn and persist a note for the next one.

    Returns the stored note. Raises on a model/storage failure — the caller (the brain's
    background runner) is responsible for logging that loudly and keeping it off the hot path.
    """
    review = f"User said: {turn_text}\nYou replied: {reply}"
    note_text = model.chat(
        "reasoning",
        [
            {"role": "system", "content": SENTINEL_SYSTEM},
            {"role": "user", "content": review},
        ],
    ).strip()

    note = Memory(
        text=note_text,
        kind=MemoryKind.SENTINEL_NOTE,
        evidence_tier=EvidenceTier.INFERRED,  # a reflection, not a stated fact
        confidence=0.5,
        salience=1.0,
        embedding=None,  # notes are fetched by recency, not similarity
        provenance="sentinel reflection",
        user=user,
    )
    save_memory(storage, note)
    log.info("sentinel: left a note for the next turn")
    return note
