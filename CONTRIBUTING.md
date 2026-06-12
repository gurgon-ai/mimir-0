# Contributing to Mimir 0

Thanks for your interest. Mimir 0 is a **cognition core**, not an agent framework — the bar for
what goes in core is high and the discipline is specific. This document is the contract.

> Read `DESIGN.md` first — it is the spec. `CLAUDE.md` holds the working conventions. This file is
> about *how to contribute without breaking the doctrine*.

## The runtime contract (the law)

Core runs on **exactly this and nothing else**: Python 3.11+, SQLite (stdlib), one chat endpoint,
one embeddings endpoint. **Core has zero third-party runtime dependencies.** If your change needs
more than that to boot the loop, it is wrong. Heavy or optional things (PDF extraction, etc.) go
behind an extra (`[documents]`, …), never in core.

## The two seams are law

Every write to the store goes through the **storage gateway**; every model/embedding call goes
through the **model gateway**. There is no other path. Add capabilities *behind* the seams; never
bypass them.

## Fail loud, self-check, stay observable (DESIGN §10)

The disease this project guards against is *silence*. So:

- **No bare `except`** in core without a re-raise or an explicit, logged downgrade. A swallowed
  error is a banned pattern (`ruff` enforces `E722`).
- **Schema changes** append a migration to `storage/schema.py` (never edit a past one) and update
  the startup `EXPECTED_SHAPE` check. Misconfiguration must fail loud with an instruction — never a
  silent fallback to another store.
- Background/async work must never break the `turn → bake → recall` loop; its failure is a logged
  downgrade, off the hot path.

## Keep core layers generic

Core cognition layers derive from **universal signals only**. Anything deployment- or
domain-specific enters through the two sanctioned channels: the seed identity in config, and
registered context sources. Don't put domain keys, household specifics, or personal data in core.

## Process

- **Each load-bearing claim in `DESIGN.md` gets a test** asserting it (an executable spec).
- **A change to core behavior updates `DESIGN.md` in the same PR** — prose drift is a defect.
- Match the surrounding code's style, comment density, and idiom.

## Running the checks

Everything CI runs, locally:

```bash
pip install -e ".[dev]"
ruff check .          # lint (no bare except, import order, line length)
mypy src/mimir        # types, strict
pytest -q             # the full executable spec
python -m mimir.selftest   # the §6 acceptance loop + canary, on the mock provider
```

All four must pass. Tests use a deterministic mock provider, so they need no model server, GPU, or
network.

## Scope discipline

Mimir 0's value is in cognition, and in *not* building breadth before the core is rock-solid. If
you're proposing a large new subsystem, open an issue first and point at the `DESIGN.md` section it
implements. A 30-line working example earns more trust than an architecture diagram.

## License

By contributing you agree your contributions are licensed under the project's Apache-2.0 license.
