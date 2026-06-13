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


def _score_secret_word(out: str) -> bool:
    """The answer is a nonce ('zephyr-quartz') that exists ONLY in the provided context and nowhere
    in training data — so producing it proves the model actually READ the context instead of leaning
    on priors or confabulating. We accept either token (hyphenated or spaced); a hallucinating model
    won't emit both rare words, and a context-ignoring one can't know them at all.
    """
    low = out.lower()
    return "zephyr" in low and "quartz" in low


def _score_vault_passphrase(out: str) -> bool:
    """The needle buried deep in a long filler 'document' — a nonce ('quokka-lantern') that exists
    ONLY there. Finding it proves the model actually read a LONG context (the haystack exceeds
    Ollama's 2048-token default, so a model at the default would have it truncated away). 'Lost in
    the middle' models, or any whose context is too small, fail; both rare tokens must appear.
    """
    low = out.lower()
    return "quokka" in low and "lantern" in low


# Index-driven filler so the haystack is deterministic (no RNG) yet varied. ~140 lines lands well
# past 2048 tokens — the point: a model at Ollama's default context can't see the needle at all.
_HAYSTACK_LINES = (
    "Maintenance unit {i} passed inspection and was returned to service.",
    "The shipment for bay {i} arrived on schedule and was logged by the clerk.",
    "Meeting room {i} was booked for the afternoon and later released without use.",
    "Sensor {i} reported nominal readings throughout the entire night shift.",
    "Invoice {i} was reconciled against the ledger and quietly marked complete.",
)


def _long_haystack(needle: str, *, sentences: int = 140, needle_at: int = 70) -> str:
    """A long, boring 'document' with ``needle`` planted in the MIDDLE — the worst place for a model
    with a weak grip on long context ('lost in the middle'). Deterministic: line N is fixed by N."""
    lines: list[str] = []
    for i in range(sentences):
        if i == needle_at:
            lines.append(needle)
        lines.append(_HAYSTACK_LINES[i % len(_HAYSTACK_LINES)].format(i=i))
    return " ".join(lines)


def _score_bird_color(out: str) -> bool:
    """Defers to the HIGHEST-tier section under heavy distraction: the high-evidence section says
    every bird is blue, a lower-evidence section says red, with irrelevant filler between. The
    prompt asks for ONE word, so a clean model just says 'blue'. We pass iff 'blue' is present and
    the low-tier 'red' is absent — committing to the high tier. Choosing red, or wishy-washily
    naming both, fails: the framework's whole job is to make the model PREFER the higher tier.
    This is the headline promise made binary — a test a Gemma-class model passes and many do not.
    """
    low = out.lower()
    return "blue" in low and "red" not in low


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
    EpistemicProbe(
        name="layered_tier_deference",
        # The framework's headline test — the chat-LLM gauntlet (the one Gemma-class models pass
        # and many don't): a BIG layered context where a high-evidence section and a low-evidence
        # section flatly CONTRADICT, buried in irrelevant filler. Answering 'blue' requires actually
        # using the evidence tiers under distraction; in the flat arm both claims look equal, so a
        # model can't tell and the probe produces lift. Position is adversarial — low-tier 'red' is
        # before the high-tier 'blue', so deferring can't be faked by picking the first/last fact.
        facts=[
            ("The quarterly meeting is on Thursday.", EvidenceTier.CONVERSATION, "chat"),
            ("The floor-2 printer is out of toner.", EvidenceTier.CONVERSATION, "chat"),
            ("Every bird in this context is red.", EvidenceTier.INFERRED, "an old guess"),
            ("Sales rose four percent last month.", EvidenceTier.DOCUMENT, "the report"),
            ("The parking gate closes at 9 pm.", EvidenceTier.DOCUMENT, "the handbook"),
            ("Every bird in this context is blue.", EvidenceTier.STATED_BY_PRIMARY_USER,
             "the operator"),
            ("The new intern is named Dana.", EvidenceTier.CONVERSATION, "chat"),
        ],
        question="Based on the information above, what color is the bird? Reply with one word.",
        scorer=_score_bird_color,
    ),
]


# Grounding probes feed the qualification score ONLY (not the lift experiment): unlike the probes
# above, the answer is present in BOTH arms, so they don't measure the *value of structure* — they
# measure the floor BELOW it, namely whether the model reads the provided context at all. A model
# that fails these is unusable for retrieval, however fluent, so it is barred from the chat role.
GROUNDING_PROBES: list[EpistemicProbe] = [
    EpistemicProbe(
        name="secret_word",
        facts=[
            ("The quarterly review is on the 14th.", EvidenceTier.CONVERSATION, "chat"),
            ("The supply-closet code is 4471.", EvidenceTier.DOCUMENT, "the facilities sheet"),
            ("The secret command word is 'zephyr-quartz'.", EvidenceTier.DOCUMENT, "the runbook"),
            ("The office mascot is a fox named Pixel.", EvidenceTier.CONVERSATION, "chat"),
            ("The backup server is in rack B7.", EvidenceTier.DOCUMENT, "the ops handbook"),
        ],
        question="What is the secret command word? Reply with only the word.",
        scorer=_score_secret_word,
    ),
    EpistemicProbe(
        name="long_context",
        # A needle in a LONG haystack — the context-length test. The 'document' runs past 2048
        # tokens, so a model at Ollama's default context never even sees the planted line; one with
        # a real context window (the benchmark pins num_ctx) and a grip on long input finds it.
        facts=[
            (_long_haystack("Important: the vault passphrase is quokka-lantern."),
             EvidenceTier.DOCUMENT, "the archive"),
        ],
        question="According to the document, what is the vault passphrase? Reply with only it.",
        scorer=_score_vault_passphrase,
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
    baseline). This is the chat-LLM qualification signal (DESIGN §4): the fraction of probes passed
    when the model is given the real tiered/provenance/gated context. It spans the lift PROBES
    (including the big layered conflicting-tier gauntlet — defer to high-tier 'blue' under noise)
    AND the GROUNDING_PROBES (read a nonce that's ONLY in context). A model that ignores evidence
    tiers, can't follow a layered prompt, or confabulates instead of reading the context scores low
    here and is barred from the identity roles — this is the qualifying round for the chat model.
    """
    samples = max(1, samples)
    probes = PROBES + GROUNDING_PROBES
    scores = [_run_arm(chat_fn, structured_prompt(p), p, samples) for p in probes]
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
