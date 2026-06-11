# Mimir 0

**A local-first cognition core for evidence-aware memory and context assembly.**

Mimir 0 is a small Python library that gives a language model a memory that behaves like a
mind's, not a database's. Knowledge is **typed, provenance-tracked, and evidence-tiered**, and
it's assembled into the prompt with an explicit epistemic structure. You tell it something;
later it recalls that fact, cites where it came from, and tells you when it's reasoning from
thin evidence instead of confabulating.

> **Status: design phase / pre-alpha.** The architecture is specified; runnable code is being
> built spine-first (boot → bake a memory → recall it with provenance → reflective review).
> Not yet usable. The full design lives in `DESIGN.md`.

## What makes it different

Most memory libraries store text in a vector blob and retrieve by similarity. Mimir 0 adds:

- **Typed knowledge layers** — facts, learned conclusions, and entity relationships live in
  separate stores with separate retrieval, not one flat index.
- **Evidence tiers + provenance** — every fact knows who said it and how reliable that source
  is. Injected context is attributed, never flattened into "you told me."
- **Confidence vs. salience, decoupled** — truth and relevance are separate axes. A fact
  doesn't become *false* just because it hasn't been used lately; it just becomes less *salient*.
- **An uncertainty gate** — when the system is reasoning from thin evidence, it says so and
  asks a clarifying question, instead of guessing confidently.
- **An async "second mind"** — a reflective pass reviews each turn and leaves a note for the next.

## Runtime contract

Runs on **Python + SQLite + one chat endpoint + one embeddings endpoint.** No GPU requirement,
no cloud, no peripherals. Bring a local model (Ollama, etc.) or an API — it's provider-agnostic,
and it can pool several local inference nodes across a LAN for distributed local inference.

## License

Apache-2.0. *(The full `LICENSE` file is added before the first public release.)*

---

*Mimir 0 is the general, reusable cognition core extracted and rebuilt from a larger private
home-AI system. None of the original's hardware integrations or personal data come with it —
just the memory and reasoning architecture.*
