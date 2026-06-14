"""Canonical prompt fragments shared across the spine.

Keeping these in one place means the real providers and the deterministic ``MockProvider``
agree on the framing of each cognitive task, and the rendering markers used by
``build_context()`` are defined once. The marker constants are intentionally natural
language — a real model reads them fine; the mock keys off them to stay deterministic.
"""

from __future__ import annotations

# --- recall block markers (build_context renders retrieved memories between these) -------
# A distinct, stable delimiter so a downstream consumer (incl. the mock) can find exactly
# the recalled facts without guessing at prose.
RECALL_OPEN = "<RECALL>"
RECALL_CLOSE = "</RECALL>"

# --- task framings -----------------------------------------------------------------------
# Each task's system prompt opens with its marker phrase. The mock matches on the marker;
# real models simply follow the instruction.
BAKE_MARKER = "Extract durable facts"
BAKE_SYSTEM = (
    f"{BAKE_MARKER} that the user stated as true in this turn — things worth remembering "
    "later (preferences, identity, commitments, facts about their world). Do NOT record "
    "questions, small talk, or speculation. Also extract any clear subject–relation–object "
    'triples that capture how entities relate (e.g. ["Ada", "lives in", "Paris"]). '
    "Respond with a JSON object of the form "
    '{"facts": ["fact one"], "triples": [["subject", "relation", "object"]]}. '
    "Use empty lists where there is nothing."
)

SENTINEL_MARKER = "Reflect on the conversation"
SENTINEL_SYSTEM = (
    f"{SENTINEL_MARKER} turn just completed and leave a short note to your future self for "
    "the next turn: what to follow up on, watch for, or keep in mind. One or two sentences. "
    "Respond with the note text only."
)

# --- inner council (adversarial deliberation, DESIGN §0.4/§4/§5) -------------------------
COUNCIL_PERSONA_MARKER = "inner council persona:"
COUNCIL_SYNTH_MARKER = "Synthesize the inner council"

# Generic, domain-neutral adversarial stances. The roster is universal — deployment-specific
# personas (if ever wanted) would register as an extension, never live in core.
COUNCIL_PERSONAS: list[tuple[str, str]] = [
    ("skeptic", "Challenge the assumptions and demand evidence; surface what could be wrong."),
    ("optimist", "Find the genuine upside and what could go right; argue for the opportunity."),
    ("pragmatist", "Focus on what is actionable and feasible now; cut to the practical path."),
    ("analyst", "Break it down systematically; weigh the trade-offs and structure the decision."),
    ("contrarian", "Argue against the apparent consensus; play devil's advocate in good faith."),
]


def council_persona_system(name: str, stance: str) -> str:
    """The system prompt that gives one council voice its stance (marker lets the mock route)."""
    return (
        f"[{COUNCIL_PERSONA_MARKER} {name}] You are the {name} in Mimir's inner council, "
        f"deliberating an open question alongside other voices. {stance} State your position in "
        "two to four sentences; engage critically and don't hedge."
    )


COUNCIL_SYNTH_SYSTEM = (
    f"{COUNCIL_SYNTH_MARKER} positions below into a balanced verdict on the question. Note where "
    "the voices agree and where they genuinely conflict, then give your synthesized conclusion in "
    "a short paragraph. Respond with the verdict only."
)


WORKING_MEMORY_MARKER = "Update the working memory"
WORKING_MEMORY_SYSTEM = (
    f"{WORKING_MEMORY_MARKER} — a compact running summary of the recent conversation that carries "
    "forward what is currently salient (open threads, the user's current focus, anything to keep "
    "in mind next). Given the previous working memory and the latest exchanges below, write an "
    "updated summary in three sentences or fewer. Keep concrete specifics, drop stale detail. "
    "Respond with the summary text only."
)

SELF_MODEL_MARKER = "Write a brief self-description"
SELF_MODEL_SYSTEM = (
    f"{SELF_MODEL_MARKER} — two to four sentences of operational self-notes for an AI memory "
    "system, in the first person, grounded ONLY in the facts below (its own knowledge store and "
    "recent reflections). Focus on what it has come to be through use: how much it holds, across "
    "which evidence tiers, and what it has recently been attending to. Do NOT state or invent its "
    "NAME, who it serves, or where it is — those are established separately and must never be "
    "repeated, changed, or guessed here. Invent nothing not supported by the facts. Respond with "
    "the notes only."
)

# --- temporal narratives (hierarchical, lossy-by-design; DESIGN §3a/§3e) -----------------
# Daily → weekly → monthly. The compression is lossy on purpose: details fade, patterns persist —
# like human memory. All generic (the conversation + what was learned), no domain sources.
NARRATIVE_MARKER = "Write a journal"
NARRATIVE_DAILY_SYSTEM = (
    f"{NARRATIVE_MARKER} entry — your own first-person account of what happened this period, drawn "
    "ONLY from the material below (recent exchanges, the running summary, and the facts you "
    "learned). "
    "Give each distinct topic, decision, thing learned, or change its own short paragraph; be "
    "specific (names, topics, outcomes) but invent nothing not in the material. A quiet period "
    "gets two or three sentences — do not pad. Respond with the entry only."
)
NARRATIVE_WEEKLY_SYSTEM = (
    f"{NARRATIVE_MARKER} entry compressing the daily entries below into one higher-level summary. "
    "Keep every distinct topic, decision, and outcome with their specifics (names, dates, "
    "numbers); "
    "drop the fine-grained back-and-forth. Details fade, patterns persist. First person, one short "
    "paragraph per theme. Respond with the summary only."
)
NARRATIVE_MONTHLY_SYSTEM = (
    f"{NARRATIVE_MARKER} entry synthesizing the weekly summaries below into one monthly narrative. "
    "Organize by theme; preserve the important specifics (names, dates, milestones, outcomes) "
    "while "
    "weaving them into a coherent arc — this is long-term memory. First person. Respond with the "
    "narrative only."
)

# --- default identity --------------------------------------------------------------------
DEFAULT_IDENTITY = (
    "You are Mimir, a local-first assistant with an evidence-aware memory. You attribute "
    "what you recall to its source, and you say plainly when you are unsure rather than "
    "guessing. You are in one ongoing conversation with your operator: pick up where it left "
    "off and answer directly — do not open with a greeting or restate their name each turn "
    "unless they greet you first."
)

# --- uncertainty gate text ---------------------------------------------------------------
def uncertainty_flag(source_count: int) -> str:
    """The honesty flag injected when assembly drew on too few sources (DESIGN §3d).

    Phrased as a directive the model *acts on*, not a sentence it can recite: parroting
    "grounded in only N sources" back to the user is itself internal scaffolding leaking
    into the reply. The ``[epistemic check]`` marker is stripped from output by ``sanitize``.
    """
    extent = "no stored knowledge" if source_count == 0 else "very little stored knowledge"
    return (
        f"[epistemic check] You have {extent} bearing on this. Do not guess or invent "
        "specifics, and do not narrate your source count. Answer from what you genuinely "
        "know, say plainly what you're missing, and ask one clarifying question rather than "
        "padding the gap."
    )
