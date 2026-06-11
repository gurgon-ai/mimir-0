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
    "questions, small talk, or speculation. Respond with a JSON object of the form "
    '{"facts": ["fact one", "fact two"]}. If there is nothing durable, return '
    '{"facts": []}.'
)

SENTINEL_MARKER = "Reflect on the conversation"
SENTINEL_SYSTEM = (
    f"{SENTINEL_MARKER} turn just completed and leave a short note to your future self for "
    "the next turn: what to follow up on, watch for, or keep in mind. One or two sentences. "
    "Respond with the note text only."
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
    f"{SELF_MODEL_MARKER} for an AI memory system, in the first person, grounded ONLY in the "
    "operational facts provided below (its own knowledge store and recent reflections). Two to "
    "four sentences. Describe what the system has come to be through use — what it holds, who it "
    "serves, what it has been attending to. Do NOT invent capabilities, experiences, persona "
    "traits, or details not supported by the facts. This is the system describing itself from "
    "evidence, not a fixed character."
)

# --- default identity --------------------------------------------------------------------
DEFAULT_IDENTITY = (
    "You are Mimir, a local-first assistant with an evidence-aware memory. You attribute "
    "what you recall to its source, and you say plainly when you are unsure rather than "
    "guessing."
)

# --- uncertainty gate text ---------------------------------------------------------------
def uncertainty_flag(source_count: int) -> str:
    """The honesty flag injected when assembly drew on too few sources (DESIGN §3d)."""
    return (
        f"[epistemic check] This answer is grounded in only {source_count} source"
        f"{'' if source_count == 1 else 's'}. If that is too thin to answer confidently, "
        "say what you don't know, name the gap, and ask one clarifying question rather "
        "than guessing."
    )
