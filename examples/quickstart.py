"""Quickstart: watch the §6 loop breathe — boot → bake → recall → sentinel.

Runs with ZERO setup: a deterministic mock provider + the bootstrap embedder, so no Ollama,
no GPU, no network, no account. For a real conversation, point it at a mimir.toml (see the
bottom of this file and docs/SETUP.md).

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mimir import Config, Mimir
from mimir.config import ProviderSpec, RoleSpec
from mimir.embed.base import EmbeddingMode


def mock_brain(storage_path: str) -> Mimir:
    """A fully offline brain: deterministic mock provider, bootstrap embeddings."""
    role = RoleSpec(model="mock")
    config = Config(
        storage_path=storage_path,
        roles={"chat": role, "bake": role, "reasoning": role},
        provider=ProviderSpec(type="mock"),
        embed_mode=EmbeddingMode.BOOTSTRAP,
        primary_user="alex",
    )
    return Mimir(config)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        brain = mock_brain(str(Path(tmp) / "quickstart.db"))
        try:
            print("=" * 70)
            print("TURN 1 — the user states a fact. Mimir bakes it.")
            print("=" * 70)
            r1 = brain.turn("My favorite color is teal.", user="alex")
            print("  user> My favorite color is teal.")
            print(f"  mimir> {r1.reply}")
            print(f"  [baked: {[m.text for m in r1.baked]}]")

            # The sentinel runs async; the next turn joins it automatically, but we wait
            # here so the demo prints in order.
            brain.wait_for_sentinel()

            print()
            print("=" * 70)
            print("TURN 2 — a later question. Mimir recalls, attributed to its source.")
            print("=" * 70)
            r2 = brain.turn("What is my favorite color?", user="alex")
            print("  user> What is my favorite color?")
            print(f"  mimir> {r2.reply}")

            print()
            print("What went into the prompt (build_context introspection):")
            print(json.dumps(r2.context.introspect(), indent=2))

            print()
            print("The assembled prompt itself:")
            print("-" * 70)
            print(r2.context.prompt)
            print("-" * 70)
        finally:
            brain.close()

    print()
    print("For a real conversation with a local model, write a mimir.toml (see")
    print("mimir.toml.example + docs/SETUP.md) and use:")
    print('    brain = Mimir.from_config("mimir.toml")')


if __name__ == "__main__":
    main()
