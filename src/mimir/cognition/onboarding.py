"""The seeding interview — the operator's first, highest-provenance facts (DESIGN §9; see
``docs/mimir_foundational_interview.md``).

A short get-to-know-you, run alongside the qualifying tournament, that orients the system from the
first turn: what to call the assistant, who the operator is and what they do, their week, where this
is, who else is here, pets, interests. These answers are the **bedrock** — everything the system
later recalls is anchored to them — so they are stored at the top evidence tier
(``stated_by_primary_user``, 1.30×) with ``provenance="onboarding"``.

They **live in one place** (the ``provenance="onboarding"`` rows) and are **editable at any time**:
one memory row per question, keyed by ``meta["onboarding_key"]``, so re-answering updates in place
and the whole interview is re-runnable. Two answers also mirror into identity **anchors** (assistant
name, location), because anchors inject verbatim into the always-on self-model every turn.

**Capture is model-free and crash-safe** (the doc's §2 law): each answer persists the instant it's
given — no chat model required, so it works while the fleet is still being qualified. (A smarter LLM
parse pass — splitting one answer into several typed facts + graph triples, with review before
commit — is a later phase; the deterministic capture here already makes each answer a real,
top-tier, editable orienting fact.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..embed.base import Embedder
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import (
    delete_memory,
    list_memories,
    save_memory,
    set_identity_anchor,
)

ONBOARDING_PROVENANCE = "onboarding"
_ONBOARDING_SALIENCE = 1.5  # load-bearing orientation, not incidental — surfaces ahead of chatter
_META_KEY = "onboarding_key"


@dataclass(frozen=True, slots=True)
class OnboardingQuestion:
    """One interview question and where its answer goes.

    ``fact`` is the first-person statement the answer becomes as a memory (so recall reads natural,
    not as a raw form field). ``anchor`` mirrors the answer into an identity anchor when set (the
    self-model injects anchors verbatim), in addition to storing the onboarding memory.
    """

    key: str
    question: str
    fact: str  # `{answer}` placeholder — the stored memory text
    anchor: str | None = None  # identity-anchor key to also set, or None


# The question set (the user's get-to-know-you). Order is the interview order. Deliberately short —
# the smallest set of durable facts that orient behaviour from turn one (foundational-interview §3).
ONBOARDING_QUESTIONS: list[OnboardingQuestion] = [
    OnboardingQuestion(
        "assistant_name", "First — what would you like to call me?",
        "The operator likes to call me {answer}.", anchor="name",
    ),
    OnboardingQuestion(
        "operator", "And who are you — what should I call you?",
        "My primary operator is {answer}.", anchor="operator",
    ),
    OnboardingQuestion(
        "work", "What do you do — your work or main focus?",
        "The operator's work / main focus: {answer}.",
    ),
    OnboardingQuestion(
        "schedule",
        "What does a normal week look like — routines, and when are you usually away or asleep?",
        "The operator's typical week, and their away/asleep window: {answer}.",
    ),
    OnboardingQuestion(
        "location", "Where is this — your setting, or where I'm running?",
        "I am situated in {answer}.", anchor="location",
    ),
    OnboardingQuestion(
        "household", "Who else is around here I should know?",
        "The operator shares this place with: {answer}.",
    ),
    OnboardingQuestion(
        "pets", "Any pets or other regulars in the day-to-day?",
        "Pets / regulars in the household: {answer}.",
    ),
    OnboardingQuestion(
        "interests", "What are you into — interests worth my knowing so I'm useful, not generic?",
        "The operator's interests: {answer}.",
    ),
]
_BY_KEY: dict[str, OnboardingQuestion] = {q.key: q for q in ONBOARDING_QUESTIONS}


def _onboarding_rows(storage: StorageGateway) -> dict[str, Memory]:
    """The current onboarding memories, keyed by their question key (one row per question)."""
    rows: dict[str, Memory] = {}
    for mem in list_memories(storage, kind=MemoryKind.MEMORY):
        if mem.provenance == ONBOARDING_PROVENANCE:
            key = mem.meta.get(_META_KEY)
            if key:
                rows[key] = mem
    return rows


def record_answer(
    storage: StorageGateway, embedder: Embedder, *, key: str, answer: str,
    primary_user: str | None = None,
) -> Memory | None:
    """Store (or update) one interview answer as a top-tier onboarding fact. Upsert by ``key``.

    A blank answer **clears** that fact (so editing to empty removes it). Mirrors name/operator/
    location into the matching identity anchor too. Returns the stored memory, or ``None`` if it was
    cleared or the key is unknown (unknown keys are ignored, never a silent miswrite — DESIGN §10).
    """
    q = _BY_KEY.get(key)
    if q is None:
        return None
    answer = answer.strip()
    existing = _onboarding_rows(storage).get(key)
    if existing is not None and existing.id is not None:
        delete_memory(storage, existing.id)  # upsert: one row per question, re-answering replaces
    if not answer:
        return None  # cleared
    text = q.fact.format(answer=answer)
    mem = Memory(
        text=text,
        kind=MemoryKind.MEMORY,
        evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER,  # the operator stating it = top evidence
        confidence=0.95,
        salience=_ONBOARDING_SALIENCE,
        provenance=ONBOARDING_PROVENANCE,
        user=primary_user,
        embedding=embedder.embed(text),
        meta={_META_KEY: key, "question": q.question, "answer": answer},
    )
    save_memory(storage, mem)
    if q.anchor is not None:
        set_identity_anchor(storage, q.anchor, answer)  # mirror into the always-on self-model
    return mem


def onboarding_profile(storage: StorageGateway) -> list[dict[str, Any]]:
    """The interview as the editable 'one place': every question, its current answer (or ``None``),
    and where the answer lands — for the Profile panel and the interview strip."""
    rows = _onboarding_rows(storage)
    out: list[dict[str, Any]] = []
    for q in ONBOARDING_QUESTIONS:
        mem = rows.get(q.key)
        out.append({
            "key": q.key,
            "question": q.question,
            "answer": mem.meta.get("answer") if mem is not None else None,
            "fact": mem.text if mem is not None else None,
            "anchor": q.anchor,
        })
    return out


def pending_onboarding(storage: StorageGateway) -> list[dict[str, str]]:
    """The questions not yet answered — what the interview still needs (drives the strip + the
    'run me' setup prompt). Empty when the interview is complete."""
    answered = set(_onboarding_rows(storage))
    return [
        {"key": q.key, "question": q.question}
        for q in ONBOARDING_QUESTIONS
        if q.key not in answered
    ]
