"""``build_context()`` — the epistemic assembly. This is the heart (DESIGN §3e).

Given a turn, the universal sections, and the retrieval results, it produces an ordered,
budgeted prompt with explicit epistemics:

    identity → typed knowledge (attributed, tier-ordered) → [registered sources]
      → sentinel note (high-attention end slot) → uncertainty flag

Three doctrines are built in, not bolted on:

- **Provenance, not flattening** — each recalled fact is rendered with its evidence tier and
  source, so the model attributes correctly (DESIGN §3b).
- **The uncertainty gate** — deterministic, zero model cost: a real question grounded in ≤1
  source gets an explicit honesty flag (DESIGN §3d).
- **Context accounting** — per-section tokens requested vs admitted, truncation warnings, and an
  ``introspect()`` call that answers "what's in the prompt and how big" plus the active embed
  mode (DESIGN §10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..cognition.temporal import relative_age
from ..embed.base import EmbeddingMode
from ..prompts import CONVERSATION_STYLE, RECALL_CLOSE, RECALL_OPEN, uncertainty_flag
from ..retrieval.hybrid import ScoredMemory
from ..storage.models import Memory
from .sections import Section, SectionTier, estimate_tokens, is_question

log = logging.getLogger("mimir.context")

# A small fixed allowance so the uncertainty flag (if it fires) always fits the budget.
_UNCERTAINTY_RESERVE = 64


@dataclass(slots=True)
class ContextBundle:
    """The assembled prompt plus everything needed to introspect and act on it."""

    prompt: str
    sections: list[Section]
    source_count: int
    uncertainty_triggered: bool
    embed_mode: EmbeddingMode
    budget_tokens: int
    requested_tokens: int
    admitted_tokens: int
    retrieved_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def introspect(self) -> dict[str, Any]:
        """Answer 'what's in the prompt and how big' — debuggable without reading internals."""
        return {
            "embed_mode": self.embed_mode.value,
            "embed_mode_banner": self.embed_mode.banner(),
            "budget_tokens": self.budget_tokens,
            "requested_tokens": self.requested_tokens,
            "admitted_tokens": self.admitted_tokens,
            "source_count": self.source_count,
            "uncertainty_triggered": self.uncertainty_triggered,
            "warnings": list(self.warnings),
            "sections": [
                {
                    "name": s.name,
                    "tier": s.tier.name,
                    "requested_tokens": s.requested_tokens,
                    "admitted_tokens": s.admitted_tokens,
                    "truncated": s.truncated,
                    "substantive": s.substantive,
                }
                for s in self.sections
            ],
        }


def _memory_line(mem: Memory, now_ts: float | None = None) -> str:
    """Render one recalled fact, attributed to its tier, source, and **age** (never flattened).

    Internal whitespace/newlines are collapsed so each memory stays a single line — the
    ``<RECALL>`` block is one-memory-per-line, and multi-line content (e.g. a document chunk)
    must not break that contract. With ``now_ts`` the fact carries a relative-age tag ("3 days ago")
    so the model can reason about recency instead of guessing (DESIGN §3e).
    """
    text = " ".join(mem.text.split())
    tags = f"tier={mem.evidence_tier.key}; source={mem.provenance}"
    if now_ts is not None and mem.created_at:
        tags += f"; {relative_age(mem.created_at, now_ts)}"
    return f"- {text} [{tags}]"


def _build_knowledge_section(
    retrieved: list[ScoredMemory], budget_tokens: int, now_ts: float | None = None
) -> tuple[Section | None, list[int]]:
    """Assemble the attributed knowledge section within its token budget.

    Memories arrive best-first. We admit lines until the budget is spent; anything dropped
    marks the section truncated. Returns the section and the ids actually admitted (for access
    bookkeeping). When recall is **empty** the section is still rendered, stating plainly that there
    is no memory — so the model says "I don't have any memory of this" rather than confabulating
    (DESIGN §3d). Never silently absent.
    """
    if not retrieved:
        title = "What you know that's relevant:"
        body = (f"{RECALL_OPEN}\nNo stored memory is relevant to this. If the user is asking about "
                f"something you'd need to remember, say you have no memory of it — do not guess.\n"
                f"{RECALL_CLOSE}")
        section = Section(
            name="knowledge", title=title, body=body, tier=SectionTier.HIGH, substantive=False,
            requested_tokens=estimate_tokens(f"{title}\n{body}"),
            admitted_tokens=estimate_tokens(f"{title}\n{body}"),
        )
        return section, []

    title = (
        "What you know that's relevant — each fact is attributed. Use these naturally in your "
        "reply and attribute in plain words when it matters; do NOT copy the bracketed "
        "[tier=...; source=...] tags into your response:"
    )
    all_lines = [_memory_line(s.memory, now_ts) for s in retrieved]
    # Requested = the section as if everything fit (for honest accounting).
    full_body = f"{RECALL_OPEN}\n" + "\n".join(all_lines) + f"\n{RECALL_CLOSE}"
    requested = estimate_tokens(f"{title}\n{full_body}")

    # Fixed framing cost (title + the two RECALL markers) that must be paid before any line.
    framing = estimate_tokens(f"{title}\n{RECALL_OPEN}\n{RECALL_CLOSE}")
    admitted_lines: list[str] = []
    admitted_ids: list[int] = []
    used = framing
    for scored, line in zip(retrieved, all_lines, strict=True):
        cost = estimate_tokens(line) + 1  # +1 for the joining newline
        if admitted_lines and used + cost > budget_tokens:
            break
        admitted_lines.append(line)
        if scored.memory.id is not None:
            admitted_ids.append(scored.memory.id)
        used += cost

    body = f"{RECALL_OPEN}\n" + "\n".join(admitted_lines) + f"\n{RECALL_CLOSE}"
    section = Section(
        name="knowledge",
        title=title,
        body=body,
        tier=SectionTier.HIGH,
        substantive=bool(admitted_lines),
        requested_tokens=requested,
        admitted_tokens=estimate_tokens(f"{title}\n{body}"),
        truncated=len(admitted_lines) < len(all_lines),
    )
    return section, admitted_ids


def build_context(
    *,
    query: str,
    user: str | None,
    identity: str,
    retrieved: list[ScoredMemory],
    sentinel_note: Memory | None,
    embed_mode: EmbeddingMode,
    budget_tokens: int,
    self_knowledge: str | None = None,
    working_memory: str | None = None,
    graph_facts: list[str] | None = None,
    procedures: list[str] | None = None,
    time_context: str | None = None,
    temporal_awareness: str | None = None,
    recent_history: str | None = None,
    background_notes: str | None = None,
    wiki_context: str | None = None,
    library: str | None = None,
    library_count: int = 0,
    system_health: str | None = None,
    now_ts: float | None = None,
    extra_sections: list[Section] | None = None,
) -> ContextBundle:
    """Assemble the epistemic prompt for one turn. Pure: no I/O, no model calls.

    The caller (the brain) does retrieval and embedding first, then hands the results here.
    Keeping assembly pure makes it unit-testable and keeps the epistemic logic in one place.
    """
    sections: list[Section] = []
    warnings: list[str] = []

    # 1. Self-model — the synthesized identity, first and always-on (DESIGN §3a, §3e). Distinct
    #    from the seed persona below: this is what the system has come to be through use.
    self_model_section: Section | None = None
    if self_knowledge:
        self_model_section = Section(
            name="self_model",
            title=(
                "Who you are — your established identity and self-knowledge. Speak and act as "
                "this; never adopt another name or invert who serves whom:"
            ),
            body=self_knowledge,
            tier=SectionTier.HIGH,
            requested_tokens=estimate_tokens(self_knowledge),
            admitted_tokens=estimate_tokens(self_knowledge),
        )
        sections.append(self_model_section)

    # 2. Identity / persona — the authored seed, always-on, high tier (DESIGN §3a, §3e).
    identity_section = Section(
        name="identity",
        title="Who you are:",
        body=identity,
        tier=SectionTier.HIGH,
        requested_tokens=estimate_tokens(identity),
        admitted_tokens=estimate_tokens(identity),
    )
    sections.append(identity_section)

    # 2-bis. Conversational style — always-on, framework-level (whatever the identity says): the
    #        blunt "don't greet every turn" rule, since each turn is sent without prior chat msgs.
    style_section = Section(
        name="style",
        title="How to converse:",
        body=CONVERSATION_STYLE,
        tier=SectionTier.HIGH,
        requested_tokens=estimate_tokens(CONVERSATION_STYLE),
        admitted_tokens=estimate_tokens(CONVERSATION_STYLE),
    )
    sections.append(style_section)

    # 2a. Temporal grounding — the clock/calendar line, always-on, so the model can reason about
    #     recency and dates instead of guessing (DESIGN §3e). Small, high-tier, never truncated.
    time_section: Section | None = None
    if time_context:
        time_section = Section(
            name="time",
            title="The current moment:",
            body=time_context,
            tier=SectionTier.HIGH,
            requested_tokens=estimate_tokens(time_context),
            admitted_tokens=estimate_tokens(time_context),
        )
        sections.append(time_section)

    # Reserve budget for the always-present pieces, then give the rest to knowledge.
    self_model_tokens = self_model_section.admitted_tokens if self_model_section else 0
    style_tokens = style_section.admitted_tokens
    time_tokens = time_section.admitted_tokens if time_section else 0
    graph_body = "\n".join(f"- {f}" for f in (graph_facts or []))
    graph_tokens = estimate_tokens(graph_body) + 8 if graph_facts else 0
    procedures_body = "\n".join(f"- {p}" for p in (procedures or []))
    procedures_tokens = estimate_tokens(procedures_body) + 8 if procedures else 0
    working_memory_tokens = estimate_tokens(working_memory) + 8 if working_memory else 0
    awareness_tokens = estimate_tokens(temporal_awareness) + 8 if temporal_awareness else 0
    history_tokens = estimate_tokens(recent_history) + 8 if recent_history else 0
    notes_tokens = estimate_tokens(background_notes) + 8 if background_notes else 0
    wiki_tokens = estimate_tokens(wiki_context) + 8 if wiki_context else 0
    library_tokens = estimate_tokens(library) + 8 if library else 0
    sentinel_tokens = (
        estimate_tokens(sentinel_note.text) + 8 if sentinel_note is not None else 0
    )
    extra_reserved = sum(s.admitted_tokens for s in (extra_sections or []))
    knowledge_budget = max(
        0,
        budget_tokens
        - self_model_tokens
        - identity_section.admitted_tokens
        - style_tokens
        - time_tokens
        - graph_tokens
        - procedures_tokens
        - working_memory_tokens
        - awareness_tokens
        - history_tokens
        - notes_tokens
        - wiki_tokens
        - library_tokens
        - sentinel_tokens
        - extra_reserved
        - _UNCERTAINTY_RESERVE,
    )

    # 2. Typed knowledge (the memory layer in v0).
    knowledge_section, retrieved_ids = _build_knowledge_section(
        retrieved, knowledge_budget, now_ts
    )
    if knowledge_section is not None:
        sections.append(knowledge_section)
        if knowledge_section.truncated:
            msg = "knowledge section truncated: a high-tier fact may have been dropped"
            warnings.append(msg)
            log.warning(msg)

    # 2b. Reference — live lookups from the offline encyclopedia (Kiwix/ZIM), attributed and clearly
    #     external, so the model can cite "per Wikipedia" apart from its own memory (DESIGN §9).
    wiki_count = 0
    if wiki_context:
        wiki_count = wiki_context.count("\n\n") + 1
        sections.append(
            Section(
                name="reference",
                title="Reference — from your offline encyclopedia (attribute as Wikipedia):",
                body=wiki_context,
                tier=SectionTier.MEDIUM,
                substantive=True,
                requested_tokens=estimate_tokens(wiki_context),
                admitted_tokens=estimate_tokens(wiki_context),
            )
        )

    # 2c. Library — cited claims distilled from the system's own long-form reading ("books I've
    #     read"), each tagged with its source title + locator. Adjacent to memory, not replacing it;
    #     the full source/composite is fetched on demand. See docs/LIBRARY.md.
    if library:
        sections.append(
            Section(
                name="library",
                title="From your library (your own reading — cited; a source can be loaded):",
                body=library,
                tier=SectionTier.MEDIUM,
                substantive=True,
                requested_tokens=estimate_tokens(library),
                admitted_tokens=estimate_tokens(library),
            )
        )

    # 3. Entity graph — connected facts, a second typed knowledge layer (DESIGN §3a).
    if graph_facts:
        sections.append(
            Section(
                name="entity_graph",
                title="What's connected to this (entity graph):",
                body=graph_body,
                tier=SectionTier.HIGH,
                substantive=True,
                requested_tokens=estimate_tokens(graph_body),
                admitted_tokens=estimate_tokens(graph_body),
            )
        )

    # 3b. Procedural memory — learned how-to guidance (methods, not facts; not a grounding
    #     source, so it does not count toward source_count) (DESIGN §3a).
    if procedures:
        sections.append(
            Section(
                name="procedures",
                title="How you've learned to handle this kind of situation:",
                body=procedures_body,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(procedures_body),
                admitted_tokens=estimate_tokens(procedures_body),
            )
        )

    # 4. Registered context sources (the seam; empty in v0 core).
    for extra in extra_sections or []:
        sections.append(extra)

    # source_count = how much substantive typed knowledge fed this turn (DESIGN §3d): admitted
    # memory facts, connected graph edges, wiki passages, and cited library claims — independent
    # grounding layers. The library is real, attributed evidence; omitting it falsely tripped the
    # uncertainty gate on questions answered from one's own reading, making the model deflect.
    source_count = len(retrieved_ids) + len(graph_facts or []) + wiki_count + max(0, library_count)

    # 3c. Recent history — the temporal-narrative arc (month → week → lately), longer-horizon
    #     context before working memory's recency (DESIGN §3a/§3e). Lossy summaries, not facts.
    if recent_history:
        sections.append(
            Section(
                name="recent_history",
                title="Recent history (your own journal):",
                body=recent_history,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(recent_history),
                admitted_tokens=estimate_tokens(recent_history),
            )
        )

    # 4. Working memory — rolling salient context, just before the end slot (DESIGN §3e).
    if working_memory:
        sections.append(
            Section(
                name="working_memory",
                title="Recent context you're carrying (working memory):",
                body=working_memory,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(working_memory),
                admitted_tokens=estimate_tokens(working_memory),
            )
        )

    # 4-bis. System health — recent errors the system logged (DESIGN §10). Self-observability: the
    #        model should know when it's degraded ("my last sentinel pass failed") rather than carry
    #        on oblivious. Only present when something actually went wrong recently.
    if system_health:
        sections.append(
            Section(
                name="system_health",
                title="System self-check — recent errors to be aware of (own them honestly):",
                body=system_health,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(system_health),
                admitted_tokens=estimate_tokens(system_health),
            )
        )

    # 4a. Background notes — what the burst worker surfaced from its own follow-up thinking since
    #     the last turn (DESIGN §5a). Off-path work re-entering the conversation; medium tier.
    if background_notes:
        sections.append(
            Section(
                name="background_notes",
                title="From your own follow-up thinking since we last spoke:",
                body=background_notes,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(background_notes),
                admitted_tokens=estimate_tokens(background_notes),
            )
        )

    # 4b. Temporal awareness — a deterministic baseline note ("you've been away longer than usual"),
    #     zero model cost, so the system feels time passing (DESIGN §3e). Soft signal, medium tier.
    if temporal_awareness:
        sections.append(
            Section(
                name="temporal_awareness",
                title="Temporal awareness:",
                body=temporal_awareness,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(temporal_awareness),
                admitted_tokens=estimate_tokens(temporal_awareness),
            )
        )

    # 5. Sentinel note — the high-attention END slot (DESIGN §3e).
    if sentinel_note is not None:
        sections.append(
            Section(
                name="sentinel_note",
                title="Note from your last reflection (carry this forward):",
                body=sentinel_note.text,
                tier=SectionTier.MEDIUM,
                requested_tokens=estimate_tokens(sentinel_note.text),
                admitted_tokens=estimate_tokens(sentinel_note.text),
            )
        )

    # 6. Uncertainty gate — deterministic, zero model cost (DESIGN §3d).
    uncertainty_triggered = is_question(query) and source_count <= 1
    if uncertainty_triggered:
        flag = uncertainty_flag(source_count)
        sections.append(
            Section(
                name="uncertainty",
                title="",
                body=flag,
                tier=SectionTier.HIGH,
                requested_tokens=estimate_tokens(flag),
                admitted_tokens=estimate_tokens(flag),
            )
        )

    prompt = "\n\n".join(s.render() for s in sections)
    requested_total = sum(s.requested_tokens for s in sections)
    admitted_total = sum(s.admitted_tokens for s in sections)

    if admitted_total > budget_tokens:
        msg = (
            f"assembled prompt ({admitted_total} tok) exceeds budget ({budget_tokens} tok); "
            f"identity/sentinel are kept even when over budget"
        )
        warnings.append(msg)
        log.warning(msg)

    return ContextBundle(
        prompt=prompt,
        sections=sections,
        source_count=source_count,
        uncertainty_triggered=uncertainty_triggered,
        embed_mode=embed_mode,
        budget_tokens=budget_tokens,
        requested_tokens=requested_total,
        admitted_tokens=admitted_total,
        retrieved_ids=retrieved_ids,
        warnings=warnings,
    )
