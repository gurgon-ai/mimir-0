"""Re-tier maintenance: drop a wrongly-trusted speaker's baked memories to a lower tier (e.g. a peer
AI ingested as primary-user before [identity] primary_user was set)."""

from __future__ import annotations

from mimir.brain import Mimir
from mimir.storage.models import EvidenceTier, Memory, MemoryKind
from mimir.storage.repo import list_memories, save_memory


def test_retier_speaker_lowers_only_that_speaker(brain: Mimir) -> None:
    for txt in ("peer claim A", "peer claim B"):
        save_memory(brain._storage, Memory(
            text=txt, kind=MemoryKind.MEMORY,
            evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER, provenance="stated by mimir-home"))
    save_memory(brain._storage, Memory(
        text="operator fact", kind=MemoryKind.MEMORY,
        evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER, provenance="stated by operator"))

    moved = brain.retier_speaker("mimir-home")          # default → conversation
    assert moved == 2

    by_prov = {m.provenance: m.evidence_tier for m in
               list_memories(brain._storage, user=None, kind=MemoryKind.MEMORY)}
    assert by_prov["stated by mimir-home"] is EvidenceTier.CONVERSATION   # demoted
    assert by_prov["stated by operator"] is EvidenceTier.STATED_BY_PRIMARY_USER  # untouched
