"""Identity anchors: foundational, operator-established identity (the init interview).

A fresh Mimir has no history, so its synthesized self-model is thin. The identity anchors solve
that cold start: a short, fixed set of **universal** identity questions whose answers ground the
self-model from the first boot — the system's name, who it serves, where it is, and what it is
for. The answers are operator-provided (interactively via the interview, or declaratively in
config); the *questions* are domain-neutral, so the core stays generic — no deployment keys here.

Anchors are injected verbatim at the top of the always-on self-model section every turn — so the
foundational facts are reliably present, not at the mercy of the synthesizer paraphrasing them —
and they also feed the self-model synthesis brief so the evolving narrative stays consistent.
"""

from __future__ import annotations

from ..storage.gateway import StorageGateway
from ..storage.repo import get_identity_anchors, set_identity_anchor

# The universal identity dimensions. Order is the interview order. Extend deliberately —
# every anchor must be domain-neutral (no deployment-specific keys in core).
ANCHORS: list[tuple[str, str]] = [
    ("name", "What is your name?"),
    ("operator", "Who do you serve — who is your primary user or operator?"),
    ("location", "Where are you — what is your setting or deployment context?"),
    ("purpose", "What is your purpose — what are you here to do?"),
]
ANCHOR_KEYS = [key for key, _ in ANCHORS]

# How each established anchor is stated, first-person, in the self-model.
_ANCHOR_TEMPLATES = {
    "name": "My name is {value}.",
    "operator": "I serve {value}.",
    "location": "I am situated in {value}.",
    "purpose": "My purpose is {value}.",
}


def establish_identity(storage: StorageGateway, answers: dict[str, str]) -> dict[str, str]:
    """Record the provided anchors (upsert). Unknown keys and blank answers are ignored.

    Returns the full set of anchors after the update.
    """
    for key in ANCHOR_KEYS:
        value = answers.get(key)
        if value and value.strip():
            set_identity_anchor(storage, key, value.strip())
    return current_anchors(storage)


def current_anchors(storage: StorageGateway) -> dict[str, str]:
    """The established anchors, restricted to the known universal keys."""
    stored = get_identity_anchors(storage)
    return {k: stored[k] for k in ANCHOR_KEYS if k in stored}


def pending_questions(storage: StorageGateway) -> list[tuple[str, str]]:
    """The (key, question) pairs not yet answered — what the interview still needs."""
    have = current_anchors(storage)
    return [(key, question) for key, question in ANCHORS if key not in have]


def render_anchors(anchors: dict[str, str]) -> str | None:
    """Render established anchors as first-person grounding, or ``None`` if there are none."""
    lines = [
        _ANCHOR_TEMPLATES.get(key, "{value}").format(value=anchors[key])
        for key in ANCHOR_KEYS
        if key in anchors
    ]
    return " ".join(lines) if lines else None
