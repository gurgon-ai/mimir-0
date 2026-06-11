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
from .config import load_config


def run_interview(brain: Mimir) -> dict[str, str]:
    """Ask the pending identity questions interactively and establish the answers.

    Returns the full anchor set afterward. Blank answers are skipped (asked again next run).
    """
    pending = brain.pending_identity_questions()
    if not pending:
        print("Identity already established:")
        for key, value in brain.identity_anchors().items():
            print(f"  {key}: {value}")
        return brain.identity_anchors()

    print("Let's establish a foundational identity. Press Enter to skip any question.\n")
    answers: dict[str, str] = {}
    for key, question in pending:
        try:
            reply = input(f"  {question} ").strip()
        except EOFError:
            break
        if reply:
            answers[key] = reply

    anchors = brain.establish_identity(answers)
    print("\nIdentity established:")
    for key, value in anchors.items():
        print(f"  {key}: {value}")
    return anchors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mimir's identity initialization interview.")
    parser.add_argument("--config", required=True, help="path to mimir.toml")
    args = parser.parse_args(argv)

    brain = Mimir(load_config(args.config))
    try:
        run_interview(brain)
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
