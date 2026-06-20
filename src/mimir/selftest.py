"""The §6 acceptance loop, runnable as a self-test (DESIGN §6, §10).

"No bake / no recall / no sentinel" is a *fault*, not a quiet state. This module runs the
whole loop against the deterministic mock provider and asserts each stage happened. The reference
server runs it at **startup** (`mimir.server.main`, skippable with ``--no-selftest``) so a broken
cognition core refuses to boot; it is also exposed as ``python -m mimir.selftest`` for CI + humans.

It ships a **canary**: a negative control query that must NOT recall the planted fact. If the
canary inverts (everything matches everything), retrieval is degenerate and the self-test would
pass for the wrong reasons — so we fail loud on the canary itself, never silently.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .brain import Mimir
from .config import Config, ProviderSpec, RoleSpec
from .embed.base import EmbeddingMode
from .errors import SelfTestError
from .retrieval.hybrid import ScoredMemory, retrieve
from .storage.models import EvidenceTier, MemoryKind
from .storage.repo import count_memories, latest_sentinel_note, list_memories

log = logging.getLogger("mimir.selftest")

_FACT = "My favorite color is teal."
_QUESTION = "What is my favorite color?"
_CANARY_QUESTION = "What is the capital of France?"  # unrelated; must NOT recall the fact
_USER = "selftest"


@dataclass(slots=True)
class SelfTestReport:
    baked: bool
    recalled: bool
    correct_tier: bool
    sentinel_fired: bool
    canary_held: bool

    @property
    def ok(self) -> bool:
        return all(
            (self.baked, self.recalled, self.correct_tier, self.sentinel_fired, self.canary_held)
        )


def _mock_config(storage_path: str) -> Config:
    role = RoleSpec(model="mock")
    return Config(
        storage_path=storage_path,
        roles={"chat": role, "bake": role, "reasoning": role},
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
    )


def run_self_test() -> SelfTestReport:
    """Run the loop end-to-end and return a report. Raises ``SelfTestError`` on any failure."""
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "selftest.db")
        with Mimir(_mock_config(db)) as m:
            # Turn 1: state a fact. Expect it to bake.
            r1 = m.turn(_FACT, user=_USER)
            m.wait_for_sentinel()
            baked = count_memories(m._storage, kind=MemoryKind.MEMORY) >= 1 and bool(r1.baked)

            # Turn 2: ask. Expect recall, attributed at the primary-user tier.
            r2 = m.turn(_QUESTION, user=_USER)
            recalled = ("teal" in r2.reply.lower()) and r2.context.source_count >= 1
            correct_tier = any(
                s.memory.evidence_tier is EvidenceTier.STATED_BY_PRIMARY_USER
                for s in _retrieved(m, _QUESTION)
            )

            # Sentinel must have left a usable note for the next turn.
            m.wait_for_sentinel()
            note = latest_sentinel_note(m._storage, _USER)
            sentinel_fired = note is not None and bool(note.text.strip())

            # Canary: an unrelated question must NOT surface the planted fact.
            r3 = m.turn(_CANARY_QUESTION, user=_USER)
            canary_held = "teal" not in r3.reply.lower() and r3.context.source_count == 0

    report = SelfTestReport(
        baked=baked,
        recalled=recalled,
        correct_tier=correct_tier,
        sentinel_fired=sentinel_fired,
        canary_held=canary_held,
    )
    if not report.ok:
        raise SelfTestError(
            "acceptance loop self-test failed: "
            f"baked={report.baked} recalled={report.recalled} "
            f"correct_tier={report.correct_tier} sentinel_fired={report.sentinel_fired} "
            f"canary_held={report.canary_held}. The cognition core is not healthy."
        )
    return report


def _retrieved(m: Mimir, query: str) -> list[ScoredMemory]:
    """Re-run just the recall step for assertions, without a full turn's side effects."""
    vec = m._embedder.embed(query)
    candidates = list_memories(m._storage, user=_USER, kind=MemoryKind.MEMORY)
    return retrieve(query, vec, candidates, top_k=6)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = run_self_test()
    except SelfTestError as exc:
        log.error("%s", exc)
        return 1
    log.info(
        "self-test PASSED — baked, recalled (attributed), sentinel fired, canary held: %s",
        report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
