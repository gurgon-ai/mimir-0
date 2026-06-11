"""The identity initialization interview — a thin interactive runner over the core API.

Mimir 0 is a library, so the *mechanism* (the anchor questions, ``establish_identity``) lives in
the core and the *interaction* lives here, at the edge. This module asks the operator the pending
identity questions on the terminal and records the answers. A host app (CLI, UI, voice) can
implement its own runner the same way against ``Mimir.pending_identity_questions()`` /
``Mimir.establish_identity()``.

Run it against a config:

    python -m mimir.interview --config mimir.toml
"""

from __future__ import annotations

import argparse
import sys

from .brain import Mimir
from .cognition.identity import ANCHORS
from .config import load_config


def _print_anchors(anchors: dict[str, str]) -> None:
    for key, value in anchors.items():
        print(f"  {key}: {value}")


def run_interview(brain: Mimir, *, revise: bool = False) -> dict[str, str]:
    """Ask the identity questions interactively and establish the answers. Re-runnable.

    Default: asks only the *pending* (unanswered) anchors. With ``revise=True`` it re-asks
    **all** anchors, showing each current value — pressing Enter keeps it. Returns the full
    anchor set afterward.
    """
    current = brain.identity_anchors()

    if revise:
        questions = ANCHORS
        print("Revising identity. Press Enter to keep the current value.\n")
    else:
        questions = brain.pending_identity_questions()
        if not questions:
            print("Identity already established:")
            _print_anchors(current)
            print("\n(Run with --revise to change existing answers.)")
            return current
        print("Let's establish a foundational identity. Press Enter to skip any question.\n")

    answers: dict[str, str] = {}
    for key, question in questions:
        existing = current.get(key)
        suffix = f" [current: {existing}]" if (revise and existing) else ""
        try:
            reply = input(f"  {question}{suffix} ").strip()
        except EOFError:
            break
        if reply:
            answers[key] = reply

    if answers:
        brain.establish_identity(answers)
    final = brain.identity_anchors()
    print("\nIdentity now:")
    _print_anchors(final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mimir's identity initialization interview.")
    parser.add_argument("--config", required=True, help="path to mimir.toml")
    parser.add_argument(
        "--revise",
        action="store_true",
        help="re-ask all anchors (Enter keeps the current value) instead of only the unanswered",
    )
    args = parser.parse_args(argv)

    brain = Mimir(load_config(args.config))
    try:
        run_interview(brain, revise=args.revise)
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
