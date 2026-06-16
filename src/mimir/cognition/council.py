"""The inner council: adversarial multi-perspective deliberation (DESIGN §0.4, §4, §5).

Given an open question, a roster of generic adversarial personas each take a position — spread
across whatever models are installed (auto-discovered from the provider pool) — and a synthesizer
weighs the positions into a balanced verdict. The verdict is stored as recallable understanding,
so the system can later draw on the *conclusion of its own disagreement*.

Model assignment is emergent from the hardware (DESIGN §4): one eligible model → a single-model
council whose diversity comes from distinct persona prompts; N models → N genuinely different
minds. Positions are gathered in parallel; a failed voice degrades to a noted gap, never a crash.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..embed.base import Embedder
from ..model.gateway import ModelGateway
from ..model.provider import is_embedding_model
from ..prompts import (
    COUNCIL_PERSONAS,
    COUNCIL_SYNTH_SYSTEM,
    council_persona_system,
)
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import add_forum_post, create_forum_thread, save_memory

log = logging.getLogger("mimir.council")

_MAX_PARALLEL = 16  # cap on concurrent persona calls — wide enough to light up a whole fleet


@dataclass(slots=True)
class Position:
    persona: str
    model: str
    text: str
    node: str = ""  # which fleet node argued this (blank = routed, not pinned)


@dataclass(slots=True)
class CouncilResult:
    question: str
    positions: list[Position]
    verdict: str
    memory_id: int | None
    thread_id: int | None = None  # the persisted forum thread for this deliberation


def _eligible_models(model: ModelGateway) -> list[str]:
    """Discovered models eligible to host a persona — embedding models filtered out."""
    discovered = [m for m in model.available_models() if not is_embedding_model(m)]
    if discovered:
        return discovered
    return [model.default_council_model()]  # nothing discovered → fall back to a configured model


def _ask_persona(
    model: ModelGateway, name: str, stance: str, node: str, on_model: str, question: str
) -> str:
    messages = [
        {"role": "system", "content": council_persona_system(name, stance)},
        {"role": "user", "content": question},
    ]
    if node:  # pin to a specific fleet node so the council fans across the whole fleet (DESIGN §5)
        return model.chat_on_node(node, on_model, messages)
    return model.chat_with_model(on_model, messages)


def _assignments(
    models: list[str], placements: list[tuple[str, str]]
) -> list[tuple[str, str, str, str]]:
    """(persona, stance, node, model) for each persona. With a known fleet, spread round-robin
    across nodes (one machine per slot until they wrap) so every node works at once; otherwise
    round-robin across distinct models and let routing place each call (node left blank)."""
    out: list[tuple[str, str, str, str]] = []
    for i, (name, stance) in enumerate(COUNCIL_PERSONAS):
        if placements:
            node, on_model = placements[i % len(placements)]
        else:
            node, on_model = "", models[i % len(models)]
        out.append((name, stance, node, on_model))
    return out


def _gather_positions(
    model: ModelGateway, question: str, models: list[str], placements: list[tuple[str, str]]
) -> list[Position]:
    """Each persona takes a position, in parallel, fanned across the fleet's nodes when known."""
    assignments = _assignments(models, placements)
    positions = [Position(n, m, "", node=node) for n, _s, node, m in assignments]
    width = len(placements) if placements else len(models)
    workers = max(1, min(_MAX_PARALLEL, len(assignments), width))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ask_persona, model, name, stance, node, on_model, question): idx
            for idx, (name, stance, node, on_model) in enumerate(assignments)
        }
        for future in futures:
            idx = futures[future]
            try:
                positions[idx].text = future.result().strip()
            except Exception as exc:  # one voice failing must not sink the council
                log.warning("council: persona %r unavailable: %s", positions[idx].persona, exc)
                positions[idx].text = f"[unavailable: {exc}]"
    return positions


def _synthesize(model: ModelGateway, question: str, positions: list[Position]) -> str:
    rendered = "\n".join(f"[{p.persona}] {p.text}" for p in positions)
    brief = f"Question: {question}\n\nCouncil positions:\n{rendered}"
    return model.chat(
        "reasoning",
        [
            {"role": "system", "content": COUNCIL_SYNTH_SYSTEM},
            {"role": "user", "content": brief},
        ],
    ).strip()


def deliberate(
    model: ModelGateway,
    storage: StorageGateway,
    embedder: Embedder,
    *,
    question: str,
    user: str | None = None,
    provenance: str = "inner council",
) -> CouncilResult:
    """Run a council deliberation and persist the verdict as recallable understanding.

    ``provenance`` tags the stored verdict — defaults to user-convened ``"inner council"``; the
    sleep cycle passes ``"sleep deliberation"`` for self-initiated arguments (DESIGN §5a)."""
    models = _eligible_models(model)
    placements = model.council_placements()
    if placements:
        log.info("council: deliberating across %d node(s): %s", len(placements),
                 [f"{n}:{m}" for n, m in placements])
    else:
        log.info("council: deliberating across %d model(s): %s", len(models), models)
    positions = _gather_positions(model, question, models, placements)
    verdict = _synthesize(model, question, positions)

    mem = Memory(
        text=f"On '{question}': {verdict}",
        kind=MemoryKind.MEMORY,  # a recallable synthesis (a 'understanding'-style memory)
        evidence_tier=EvidenceTier.INFERRED,
        confidence=0.6,
        salience=1.0,
        embedding=embedder.embed(verdict),
        provenance=provenance,
        user=user,
    )
    save_memory(storage, mem)
    thread_id = _persist_thread(storage, question, positions, verdict, provenance)
    return CouncilResult(
        question=question, positions=positions, verdict=verdict, memory_id=mem.id,
        thread_id=thread_id,
    )


def _persist_thread(
    storage: StorageGateway, question: str, positions: list[Position], verdict: str, source: str,
) -> int | None:
    """Persist the deliberation as a browsable forum thread (positions + verdict). Best-effort —
    a forum write must never sink the deliberation, whose verdict is already saved (§10)."""
    try:
        thread_id = create_forum_thread(storage, question=question, source=source, verdict=verdict)
        for pos in positions:
            add_forum_post(
                storage, thread_id=thread_id, author=pos.persona, kind="position",
                content=pos.text, model=pos.model, node=pos.node,
            )
        add_forum_post(storage, thread_id=thread_id, author="synthesis", kind="verdict",
                       content=verdict)
        return thread_id
    except Exception as exc:
        log.warning("council: could not persist forum thread: %s", exc)
        return None
