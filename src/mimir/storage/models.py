"""The domain model for stored knowledge.

A ``Memory`` is the atom of Mimir's store. It carries the two decoupled axes the whole
design turns on — **confidence** (is it true?) and **salience** (is it relevant now?) —
plus an **evidence tier** (how was it sourced?) and a **provenance** tag (who said it).
See DESIGN §3.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import Enum


class EvidenceTier(Enum):
    """How a memory was sourced — assigned at write time (DESIGN §3b).

    The tier is a *gentle* retrieval multiplier: at equal relevance, better-sourced
    facts win. It is also rendered as an explicit provenance tag in the prompt, so the
    model attributes correctly instead of flattening everyone into "you told me."

    The ``multiplier`` is deliberately gentle — it breaks ties, it does not bulldoze
    relevance. Truth ≠ relevance (DESIGN §3c).
    """

    STATED_BY_PRIMARY_USER = ("stated_by_primary_user", 1.30)
    STATED_BY_TRUSTED = ("stated_by_trusted", 1.20)
    DOCUMENT = ("document", 1.10)
    MULTI_SOURCE = ("multi_source", 1.10)
    CONVERSATION = ("conversation", 1.00)
    INFERRED = ("inferred", 0.90)

    def __init__(self, key: str, multiplier: float) -> None:
        self.key = key
        self.multiplier = multiplier

    @classmethod
    def from_key(cls, key: str) -> EvidenceTier:
        for tier in cls:
            if tier.key == key:
                return tier
        raise ValueError(f"unknown evidence tier: {key!r}")

    @property
    def decays(self) -> bool:
        """Whether *confidence* may decay from disuse for this tier.

        Only low-tier, uncorroborated provisionals decay; authority-tier and
        corroborated facts never do (DESIGN §3c). Salience always decays regardless —
        that is a separate axis and lives on the row, not here.
        """
        return self in (EvidenceTier.INFERRED, EvidenceTier.CONVERSATION)


class MemoryKind(Enum):
    """What *role* a stored row plays in assembly.

    Recallable knowledge is ``MEMORY``; the sentinel's note to the next turn is
    ``SENTINEL_NOTE`` (a distinct high-attention slot in ``build_context()``, DESIGN §3e);
    the synthesized always-on identity is ``SELF_MODEL`` (DESIGN §3a). Only ``MEMORY`` rows
    are recalled — notes and the self-model occupy their own dedicated prompt slots and never
    compete for the knowledge section. Future kinds register here without a schema change.
    """

    MEMORY = "memory"
    SENTINEL_NOTE = "sentinel_note"
    SELF_MODEL = "self_model"
    EXCHANGE = "exchange"  # a raw recent turn, for working-memory recency (capped, then folded)
    WORKING_MEMORY = "working_memory"  # the rolling compressed summary of recent context


@dataclass(slots=True)
class Memory:
    """One stored belief or note.

    ``id`` is ``None`` until the storage gateway has written it. ``embedding`` is the
    vector for similarity retrieval, or ``None`` in keyword-only (degraded) mode.
    """

    text: str
    kind: MemoryKind = MemoryKind.MEMORY
    evidence_tier: EvidenceTier = EvidenceTier.CONVERSATION
    confidence: float = 0.7
    salience: float = 1.0
    embedding: list[float] | None = None
    provenance: str = "conversation"
    user: str | None = None
    # For document chunks (evidence_tier=DOCUMENT): the originating file/uri, so re-ingest
    # can replace a document's chunks cleanly. NULL for conversation memories and notes.
    source: str | None = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    # Archived by consolidation: excluded from active recall, kept in the store. Archiving is
    # not disbelieving — a resurfaced archived memory is still trusted (DESIGN §3c).
    archived: bool = False
    id: int | None = None
    # Free-form extension bag, serialized to JSON. Kept tiny in v0 (DESIGN: resist breadth).
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Triple:
    """One subject–relation–object edge in the entity graph (DESIGN §3a).

    ``subject`` and ``object`` are entity nodes; ``relation`` is the labeled edge between them.
    Stored case-insensitively deduped, indexed both ways for 1–2 hop traversal.
    """

    subject: str
    relation: str
    object: str
    user: str | None = None
    provenance: str = "conversation"
    confidence: float = 0.8
    created_at: float = 0.0
    id: int | None = None

    def render(self) -> str:
        """A compact, readable edge for the prompt: ``subject — relation → object``."""
        return f"{self.subject} — {self.relation} → {self.object}"


def embedding_to_blob(vec: list[float] | None) -> bytes | None:
    """Pack an embedding into a compact little-endian float32 BLOB (stdlib only)."""
    if vec is None:
        return None
    return array("f", vec).tobytes()


def blob_to_embedding(blob: bytes | None) -> list[float] | None:
    """Unpack a float32 BLOB back into a list of floats."""
    if blob is None:
        return None
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)
