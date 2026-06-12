"""Epistemic-competence experiment (DESIGN §3) — the core thesis, made measurable.

Mimir's premise is that typed, evidence-tiered, provenance-tagged context improves a model's
cognition over a flat RAG blob. This experiment tests that, per model, with three probes — each
run two ways:

- **structured** arm: the probe's facts go through the *real* ``build_context()`` assembly, so the
  model sees the actual evidence tiers, ``[source=...]`` provenance, and (when evidence is thin)
  the uncertainty gate;
- **flat** arm: the *same* facts as an undifferentiated bullet list — no tiers, no provenance, no
  sections, no gate.

The only variable between arms is the epistemic scaffolding, so ``lift = structured - flat`` is the
measured value of the framework. The persona is deliberately neutral (it does not pre-instruct
attribution or hedging), so any epistemic behaviour comes from the structure, not the prompt.

Probes (generic by construction — synthetic facts, no deployment specifics, DESIGN §9):

- ``tier_deference`` — two *contradicting* facts at different tiers; does the model side with the
  higher-tier one? (Flat can't: both look equal.)
- ``attribution`` — a fact whose source lives only in its provenance; can the model say where it
  learned it? (Flat can't: the source isn't in the text.)
- ``uncertainty`` — a question with no supporting evidence; does it hedge (the gate) or confabulate?
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..context.build import build_context
from ..embed.base import EmbeddingMode
from ..model.gateway import ModelGateway
from ..model.provider import Message
from ..retrieval.hybrid import ScoredMemory
from ..storage.models import EvidenceTier, Memory, MemoryKind

ChatFn = Callable[[list[Message]], str]

# Neutral persona: no "attribute your sources" / "admit uncertainty" instruction, so the probes
# measure what the STRUCTURE contributes, not what the persona already demands.
_IDENTITY = (
    "You are a helpful assistant. Answer the user's question using the information you have."
)


# -- scorers (deterministic) ----------------------------------------------------------


_DISCOUNT_CUES = (
    "earlier", "estimate", "guess", "outdated", "older", "old ", "previous", "initial",
    "superseded", "however", "though", "but ", "while ", "whereas", "inferred", "no longer",
    "updated", "less", "not ",
)


def _score_tier_deference(out: str) -> bool:
    """Commits to the higher-tier date (March 15). Mentioning the low-tier April 20 is fine ONLY if
    it is clearly subordinated — March 15 leads AND April 20 is discounted (transparent deference is
    the *ideal* answer, not a failure). Listing both as equal, or choosing April 20, fails.
    """
    low = out.lower()
    if "march 15" not in low:
        return False
    if "april 20" not in low:
        return True  # committed to the high-tier date alone
    leads = low.index("march 15") < low.index("april 20")
    discounted = any(c in low for c in _DISCOUNT_CUES)
    return leads and discounted


def _score_attribution(out: str) -> bool:
    """States the value AND attributes it to its source (the handbook), which is in provenance."""
    low = out.lower()
    return ("30 day" in low or "thirty day" in low) and "handbook" in low


_HEDGES = (
    "don't know", "do not know", "not sure", "unsure", "no information", "cannot", "can't",
    "unable", "no record", "don't have", "do not have", "not provided", "no data", "insufficient",
    "not aware", "no mention", "isn't any", "is no ", "i lack",
)


def _score_uncertainty(out: str) -> bool:
    """Hedges / admits it cannot answer, rather than fabricating a name."""
    low = out.lower()
    return any(h in low for h in _HEDGES) or out.strip().endswith("?")


# -- probes ---------------------------------------------------------------------------


@dataclass(slots=True)
class EpistemicProbe:
    name: str
    facts: list[tuple[str, EvidenceTier, str]]  # (text, tier, provenance/source)
    question: str
    scorer: Callable[[str], bool]


PROBES: list[EpistemicProbe] = [
    EpistemicProbe(
        name="tier_deference",
        # Low-tier fact FIRST, so deferring to the high-tier one requires overriding list order —
        # a model that just picks the first fact can't pass spuriously.
        facts=[
            ("The launch is scheduled for April 20.", EvidenceTier.INFERRED, "an earlier guess"),
            ("The launch is scheduled for March 15.", EvidenceTier.STATED_BY_PRIMARY_USER,
             "the operator"),
        ],
        question="When is the launch scheduled?",
        scorer=_score_tier_deference,
    ),
    EpistemicProbe(
        name="attribution",
        facts=[
            ("The deploy key rotates every 30 days.", EvidenceTier.DOCUMENT, "the ops handbook"),
            ("The break room has a standing desk.", EvidenceTier.CONVERSATION, "chat"),
        ],
        question="How often does the deploy key rotate, and where did you learn that?",
        scorer=_score_attribution,
    ),
    EpistemicProbe(
        name="uncertainty",
        facts=[
            ("The office coffee machine is a Gaggia.", EvidenceTier.CONVERSATION, "chat"),
        ],
        question="What is the operator's spouse's name?",
        scorer=_score_uncertainty,
    ),
]


# -- prompt construction (structured = the real framework; flat = a bare baseline) -----


def _memory(mem_id: int, text: str, tier: EvidenceTier, provenance: str) -> Memory:
    # A real id matters: build_context counts admitted memories with an id toward source_count,
    # which drives the uncertainty gate. Without ids the gate would misfire on every probe.
    return Memory(
        id=mem_id, text=text, kind=MemoryKind.MEMORY, evidence_tier=tier, provenance=provenance,
        confidence=0.7, salience=1.0, embedding=None, user=None,
    )


def structured_prompt(probe: EpistemicProbe) -> str:
    """The probe's facts through the *real* ``build_context()`` — tiers, provenance, uncertainty."""
    retrieved = [
        ScoredMemory(memory=_memory(i + 1, t, tier, src), score=1.0, keyword=1.0, vector=1.0)
        for i, (t, tier, src) in enumerate(probe.facts)
    ]
    bundle = build_context(
        query=probe.question, user=None, identity=_IDENTITY, retrieved=retrieved,
        sentinel_note=None, embed_mode=EmbeddingMode.BOOTSTRAP, budget_tokens=4096,
    )
    return bundle.prompt


def flat_prompt(probe: EpistemicProbe) -> str:
    """The SAME facts as an undifferentiated blob — no tiers, provenance, sections, or gate."""
    facts = "\n".join(f"- {t}" for (t, _tier, _src) in probe.facts)
    return f"{_IDENTITY}\n\nInformation you have:\n{facts}"


# -- evaluation -----------------------------------------------------------------------


@dataclass(slots=True)
class ProbeOutcome:
    probe: str
    structured: float  # fraction of samples passed in the structured arm
    flat: float        # ...and in the flat arm


@dataclass(slots=True)
class EpistemicResult:
    model: str
    outcomes: list[ProbeOutcome]
    structured_score: float
    flat_score: float
    lift: float  # structured_score - flat_score: the measured value of the framework


def _run_arm(chat_fn: ChatFn, system_prompt: str, probe: EpistemicProbe, samples: int) -> float:
    passed = 0
    for _ in range(samples):
        try:
            out = chat_fn(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": probe.question},
                ]
            )
        except Exception:  # a failed call scores 0 for that sample
            out = ""
        if probe.scorer(out):
            passed += 1
    return passed / samples


def evaluate_epistemics(chat_fn: ChatFn, *, model: str = "", samples: int = 1) -> EpistemicResult:
    """Run every probe in both arms against one model; return its scores and the framework lift."""
    samples = max(1, samples)
    outcomes = [
        ProbeOutcome(
            probe=p.name,
            structured=_run_arm(chat_fn, structured_prompt(p), p, samples),
            flat=_run_arm(chat_fn, flat_prompt(p), p, samples),
        )
        for p in PROBES
    ]
    s = sum(o.structured for o in outcomes) / len(outcomes)
    f = sum(o.flat for o in outcomes) / len(outcomes)
    return EpistemicResult(model=model, outcomes=outcomes, structured_score=round(s, 3),
                           flat_score=round(f, 3), lift=round(s - f, 3))


def score_epistemic_competence(chat_fn: ChatFn, *, samples: int = 2) -> float:
    """How well a model exploits the epistemic framework — the structured arm only (no flat
    baseline). This is the qualification signal (DESIGN §4): the fraction of probes passed when the
    model is given the real tiered/provenance/gated context. A model that ignores evidence tiers or
    confabulates on thin evidence scores low here and is barred from the identity roles.
    """
    samples = max(1, samples)
    scores = [_run_arm(chat_fn, structured_prompt(p), p, samples) for p in PROBES]
    return sum(scores) / len(scores)


def run_epistemics(
    model: ModelGateway, model_names: list[str], *, samples: int = 3
) -> list[EpistemicResult]:
    """Run the experiment across several models (each via the pool), for a cross-model report."""
    results: list[EpistemicResult] = []
    for name in model_names:
        def chat_fn(messages: list[Message], _n: str = name) -> str:
            return model.chat_with_model(_n, messages)

        results.append(evaluate_epistemics(chat_fn, model=name, samples=samples))
    return results
