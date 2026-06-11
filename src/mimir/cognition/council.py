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
from ..prompts import (
    COUNCIL_PERSONAS,
    COUNCIL_SYNTH_SYSTEM,
    council_persona_system,
)
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import save_memory

log = logging.getLogger("mimir.council")

_MAX_PARALLEL = 5


@dataclass(slots=True)
class Position:
    persona: str
    model: str
    text: str


@dataclass(slots=True)
class CouncilResult:
    question: str
    positions: list[Position]
    verdict: str
    memory_id: int | None


def _eligible_models(model: ModelGateway) -> list[str]:
    """Discovered models eligible to host a persona — embedding models filtered out."""
    discovered = [m for m in model.available_models() if "embed" not in m.lower()]
    if discovered:
        return discovered
    return [model.default_council_model()]  # nothing discovered → fall back to a configured model


def _ask_persona(model: ModelGateway, name: str, stance: str, on_model: str, question: str) -> str:
    return model.chat_with_model(
        on_model,
        [
            {"role": "system", "content": council_persona_system(name, stance)},
            {"role": "user", "content": question},
        ],
    )


def _gather_positions(model: ModelGateway, question: str, models: list[str]) -> list[Position]:
    """Each persona takes a position, assigned round-robin across the models, in parallel."""
    assignments = [
        (name, stance, models[i % len(models)])
        for i, (name, stance) in enumerate(COUNCIL_PERSONAS)
    ]
    positions: list[Position] = [Position(n, m, "") for n, _s, m in assignments]
    with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL, len(assignments))) as pool:
        futures = {
            pool.submit(_ask_persona, model, name, stance, on_model, question): idx
            for idx, (name, stance, on_model) in enumerate(assignments)
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
) -> CouncilResult:
    """Run a council deliberation and persist the verdict as recallable understanding."""
    models = _eligible_models(model)
    log.info("council: deliberating across %d model(s): %s", len(models), models)
    positions = _gather_positions(model, question, models)
    verdict = _synthesize(model, question, positions)

    mem = Memory(
        text=f"On '{question}': {verdict}",
        kind=MemoryKind.MEMORY,  # a recallable synthesis (a 'understanding'-style memory)
        evidence_tier=EvidenceTier.INFERRED,
        confidence=0.6,
        salience=1.0,
        embedding=embedder.embed(verdict),
        provenance="inner council",
        user=user,
    )
    save_memory(storage, mem)
    return CouncilResult(question=question, positions=positions, verdict=verdict, memory_id=mem.id)
