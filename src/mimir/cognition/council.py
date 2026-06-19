"""The inner council: adversarial multi-perspective deliberation (DESIGN §0.4, §4, §5).

Given an open question, a roster of generic adversarial personas debate in two rounds — an opening
position, then a rebuttal where each voice answers the others — spread across whatever models are
installed (auto-discovered from the provider pool). A synthesizer weighs the whole debate into a
structured verdict (conclusion + the surviving objection + a consensus score), stored as recallable
understanding so the system can later draw on the *conclusion of its own disagreement*.

Model assignment is emergent from the hardware (DESIGN §4): one eligible model → a single-model
council whose diversity comes from distinct persona prompts; N models → N genuinely different
minds. Each round is gathered in parallel; a failed voice degrades to a noted gap, never a crash.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..embed.base import Embedder
from ..model.gateway import ModelGateway
from ..model.provider import is_embedding_model
from ..prompts import (
    COUNCIL_PERSONAS,
    COUNCIL_SYNTH_SYSTEM,
    council_persona_system,
    council_rebuttal_user,
)
from ..storage.gateway import StorageGateway
from ..storage.models import EvidenceTier, Memory, MemoryKind
from ..storage.repo import add_forum_post, create_forum_thread, save_memory

log = logging.getLogger("mimir.council")

_MAX_PARALLEL = 16  # cap on concurrent persona calls — wide enough to light up a whole fleet

# A verdict's confidence rides on how strongly the council converged: a split deliberation is
# worth less as recallable understanding than a unanimous one. The band stays modest — this is
# still INFERRED — mapping consensus 0.0→floor, 1.0→floor+span (a 50/50 split lands on the old
# flat 0.6, so behavior is continuous).
_CONF_FLOOR = 0.45
_CONF_SPAN = 0.30
_DEFAULT_CONSENSUS = 0.5  # neutral prior when the synthesizer gives no usable number

# Match the labelled fields the synthesizer is asked to emit. DISSENT_BY precedes DISSENT in the
# alternation so the longer label wins at a shared position (leftmost-longest by ordering).
_LABEL_RE = re.compile(
    r"^\s*(VERDICT|DISSENT_BY|DISSENT|CONSENSUS)\s*:\s*", re.IGNORECASE | re.MULTILINE
)
_NONE_VALUES = frozenset({"", "none", "n/a", "no dissent", "nil"})


@dataclass(slots=True)
class Position:
    persona: str
    model: str
    text: str
    node: str = ""  # which fleet node argued this (blank = routed, not pinned)


@dataclass(slots=True)
class Verdict:
    """The synthesized outcome: the conclusion, the strongest objection that survived the debate
    (with who raised it), and how strongly the voices converged."""

    summary: str
    dissent: str = ""
    dissent_by: str = ""
    consensus: float = _DEFAULT_CONSENSUS


@dataclass(slots=True)
class CouncilResult:
    question: str
    positions: list[Position]
    verdict: str  # the synthesized conclusion (the verdict's summary)
    memory_id: int | None
    thread_id: int | None = None  # the persisted forum thread for this deliberation
    dissent: str = ""  # the strongest objection that survived (empty if the voices agreed)
    dissent_by: str = ""
    consensus: float = _DEFAULT_CONSENSUS
    rebuttals: list[Position] = field(default_factory=list)  # round two — each voice answers


def _eligible_models(model: ModelGateway) -> list[str]:
    """Discovered models eligible to host a persona — embedding models filtered out."""
    discovered = [m for m in model.available_models() if not is_embedding_model(m)]
    if discovered:
        return discovered
    return [model.default_council_model()]  # nothing discovered → fall back to a configured model


def _ask_persona(
    model: ModelGateway, name: str, stance: str, node: str, on_model: str, user_content: str
) -> str:
    messages = [
        {"role": "system", "content": council_persona_system(name, stance)},
        {"role": "user", "content": user_content},
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


def _gather(
    model: ModelGateway,
    assignments: list[tuple[str, str, str, str]],
    width: int,
    content_for: Callable[[int], str],
    *,
    round_label: str = "position",
) -> list[Position]:
    """Run one debate round in parallel — every persona answers ``content_for(idx)`` on its
    assigned node/model, fanned across the fleet. A failed voice degrades to a noted gap."""
    positions = [Position(n, m, "", node=node) for n, _s, node, m in assignments]
    workers = max(1, min(_MAX_PARALLEL, len(assignments), width))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ask_persona, model, name, stance, node, on_model, content_for(idx)): idx
            for idx, (name, stance, node, on_model) in enumerate(assignments)
        }
        for future in futures:
            idx = futures[future]
            try:
                positions[idx].text = future.result().strip()
            except Exception as exc:  # one voice failing must not sink the council
                log.warning("council: %s by %r unavailable: %s",
                            round_label, positions[idx].persona, exc)
                positions[idx].text = f"[unavailable: {exc}]"
    return positions


def _others(positions: list[Position], idx: int) -> list[tuple[str, str]]:
    """The floor a persona must answer in round two: every voice's opening but its own."""
    return [(p.persona, p.text) for j, p in enumerate(positions) if j != idx]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _confidence_from_consensus(consensus: float) -> float:
    """A verdict the council agreed on is worth more than one it split on — but stays INFERRED."""
    return round(_CONF_FLOOR + _CONF_SPAN * _clamp01(consensus), 2)


def _clean_dissent(value: str) -> str:
    """Normalize an absent/"none" objection to empty (the model is told to write 'none')."""
    cleaned = value.strip().strip("[]").rstrip(".")
    return "" if cleaned.lower() in _NONE_VALUES else value.strip()


def _parse_float(value: str, default: float) -> float:
    match = re.search(r"\d*\.?\d+", value)
    if not match:
        return default
    try:
        return _clamp01(float(match.group(0)))
    except ValueError:
        return default


def _parse_verdict(raw: str) -> Verdict:
    """Parse the synthesizer's labelled output. Tolerant by design: a model that ignores the
    format (no VERDICT label) degrades gracefully to the whole text as the conclusion, so a
    verdict is never lost to a formatting slip (§10)."""
    text = raw.strip()
    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return Verdict(summary=text)
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[match.group(1).lower()] = text[match.end() : end].strip()
    summary = fields.get("verdict", "").strip()
    if not summary:
        return Verdict(summary=text)
    dissent = _clean_dissent(fields.get("dissent", ""))
    dissent_by = _clean_dissent(fields.get("dissent_by", "")) if dissent else ""
    consensus = _parse_float(fields.get("consensus", ""), _DEFAULT_CONSENSUS)
    return Verdict(summary=summary, dissent=dissent, dissent_by=dissent_by, consensus=consensus)


def _synthesize(
    model: ModelGateway, question: str, openings: list[Position], rebuttals: list[Position]
) -> Verdict:
    opening_block = "\n".join(f"[{p.persona}] {p.text}" for p in openings)
    rebuttal_block = "\n".join(f"[{p.persona}] {p.text}" for p in rebuttals)
    brief = (
        f"Question: {question}\n\nOpening positions:\n{opening_block}\n\n"
        f"Rebuttals:\n{rebuttal_block}"
    )
    raw = model.chat(
        "reasoning",
        [
            {"role": "system", "content": COUNCIL_SYNTH_SYSTEM},
            {"role": "user", "content": brief},
        ],
    )
    return _parse_verdict(raw)


def _verdict_memory_text(question: str, verdict: Verdict) -> str:
    """The recallable form: the conclusion, with the surviving objection riding along so a later
    turn draws on the *conclusion of its own disagreement* — dissent and all, not a flat gist."""
    body = f"On '{question}': {verdict.summary}"
    if verdict.dissent:
        who = f" ({verdict.dissent_by})" if verdict.dissent_by else ""
        body += f"\nSurviving objection{who}: {verdict.dissent}"
    return body


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
    assignments = _assignments(models, placements)
    width = len(placements) if placements else len(models)
    openings = _gather(model, assignments, width, lambda _i: question)
    rebuttals = _gather(
        model, assignments, width,
        lambda i: council_rebuttal_user(question, openings[i].text, _others(openings, i)),
        round_label="rebuttal",
    )
    verdict = _synthesize(model, question, openings, rebuttals)

    mem = Memory(
        text=_verdict_memory_text(question, verdict),
        kind=MemoryKind.MEMORY,  # a recallable synthesis (a 'understanding'-style memory)
        evidence_tier=EvidenceTier.INFERRED,
        confidence=_confidence_from_consensus(verdict.consensus),
        salience=1.0,
        embedding=embedder.embed(verdict.summary),
        provenance=provenance,
        user=user,
    )
    save_memory(storage, mem)
    thread_id = _persist_thread(storage, question, openings, rebuttals, verdict, provenance)
    return CouncilResult(
        question=question, positions=openings, verdict=verdict.summary, memory_id=mem.id,
        thread_id=thread_id, dissent=verdict.dissent, dissent_by=verdict.dissent_by,
        consensus=verdict.consensus, rebuttals=rebuttals,
    )


def _persist_thread(
    storage: StorageGateway, question: str, openings: list[Position], rebuttals: list[Position],
    verdict: Verdict, source: str,
) -> int | None:
    """Persist the deliberation as a browsable forum thread (openings → rebuttals → the surviving
    objection → verdict). Best-effort — a forum write must never sink the deliberation, whose
    verdict is already saved (§10)."""
    try:
        thread_id = create_forum_thread(
            storage, question=question, source=source, verdict=verdict.summary
        )
        for pos in openings:
            add_forum_post(
                storage, thread_id=thread_id, author=pos.persona, kind="position",
                content=pos.text, model=pos.model, node=pos.node,
            )
        for reb in rebuttals:
            add_forum_post(
                storage, thread_id=thread_id, author=reb.persona, kind="rebuttal",
                content=reb.text, model=reb.model, node=reb.node,
            )
        if verdict.dissent:  # the objection that survived — kept as its own post, attributed
            add_forum_post(
                storage, thread_id=thread_id, author=verdict.dissent_by or "synthesis",
                kind="dissent", content=verdict.dissent,
            )
        add_forum_post(storage, thread_id=thread_id, author="synthesis", kind="verdict",
                       content=verdict.summary)
        return thread_id
    except Exception as exc:
        log.warning("council: could not persist forum thread: %s", exc)
        return None
