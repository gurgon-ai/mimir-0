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

from ..embed.base import EmbeddingMode
from ..prompts import RECALL_CLOSE, RECALL_OPEN, uncertainty_flag
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


def _memory_line(mem: Memory) -> str:
    """Render one recalled fact, attributed to its tier and source (never flattened).

    Internal whitespace/newlines are collapsed so each memory stays a single line — the
    ``<RECALL>`` block is one-memory-per-line, and multi-line content (e.g. a document chunk)
    must not break that contract.
    """
    text = " ".join(mem.text.split())
    return f"- {text} [tier={mem.evidence_tier.key}; source={mem.provenance}]"


def _build_knowledge_section(
    retrieved: list[ScoredMemory], budget_tokens: int
) -> tuple[Section | None, list[int]]:
    """Assemble the attributed knowledge section within its token budget.

    Memories arrive best-first. We admit lines until the budget is spent; anything dropped
    marks the section truncated. Returns the section (or ``None`` if nothing relevant) and
    the ids actually admitted (for access bookkeeping).
    """
    if not retrieved:
        return None, []

    title = "What you know that's relevant (each fact is attributed — honor the source):"
    all_lines = [_memory_line(s.memory) for s in retrieved]
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
            title="What you've come to understand about yourself (from your own history):",
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

    # Reserve budget for the always-present pieces, then give the rest to knowledge.
    self_model_tokens = self_model_section.admitted_tokens if self_model_section else 0
    working_memory_tokens = estimate_tokens(working_memory) + 8 if working_memory else 0
    sentinel_tokens = (
        estimate_tokens(sentinel_note.text) + 8 if sentinel_note is not None else 0
    )
    extra_reserved = sum(s.admitted_tokens for s in (extra_sections or []))
    knowledge_budget = max(
        0,
        budget_tokens
        - self_model_tokens
        - identity_section.admitted_tokens
        - working_memory_tokens
        - sentinel_tokens
        - extra_reserved
        - _UNCERTAINTY_RESERVE,
    )

    # 2. Typed knowledge (the memory layer in v0).
    knowledge_section, retrieved_ids = _build_knowledge_section(retrieved, knowledge_budget)
    if knowledge_section is not None:
        sections.append(knowledge_section)
        if knowledge_section.truncated:
            msg = "knowledge section truncated: a high-tier fact may have been dropped"
            warnings.append(msg)
            log.warning(msg)

    # 3. Registered context sources (the seam; empty in v0 core).
    for extra in extra_sections or []:
        sections.append(extra)

    # source_count = how many substantive typed knowledge layers fed this turn (DESIGN §3d).
    # v0 has one such layer (memory); each admitted fact counts as a source within it.
    source_count = len(retrieved_ids)

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
