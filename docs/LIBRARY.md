# The Library layer — gist in SQLite, detail in Markdown (planned, staged)

**Status: design, not built.** Phase 1 builds on the current spine; Phase 2 (model-driven fetch)
layers cleanly on top of Phase 1. This brief is the spec to build from when it's greenlit.

> **Not the `[wiki]` block.** The existing `[wiki]` is a *read-only Kiwix/ZIM offline encyclopedia*
> (`cognition/wiki.py`) — an external reference source. The **Library** is the system's *own*
> long-form knowledge — "books I've read", curated/compiled notes — a separate knowledge layer that
> sits **adjacent to memory and never replaces it** (its own `build_context` section, like documents
> or the graph).

## The idea: progressive disclosure

A deep knowledge base can't fit a small operational window (Mimir runs at a fixed ~24k `num_ctx`).
So split knowledge by **depth**, not store:

- **Gist tier** — per-page **summaries + high-impact facts**, embedded, **always cheap to surface**.
  The model always "knows what it knows": which books exist and roughly what each says.
- **Detail tier** — the full page text, **fetched on demand** only when the gist isn't enough.

This is how a careful reader works: you carry the gist of everything you've read; you reopen the book
for specifics. It keeps the window honest (you never dump the corpus) while making the *whole* library
reachable.

## Influence + how this differs (the honest note)

Influenced by the **"LLM wiki" pattern popularized by Karpathy and others** — compile sources into
readable markdown and query by loading files. The general principle (distillation over raw-chunk RAG;
an LLM-maintained knowledge base) is widely converged, not proprietary. This design departs from the
pure-files version in three ways that matter for Mimir:

1. **Provenance.** A link table ties every page back to its sources, and the gist is tier-marked as
   *derived* (a summary), never asserted as source-of-truth fact (DESIGN §3b/§3c).
2. **Small window.** We don't dump the corpus into a huge context; we keep a cheap gist resident and
   fetch one page on demand (DESIGN: continuity = RAG + compression, not raw window).
3. **Non-destructive.** Pages are *re-derived* from sources, not mutated-in-place on every ingest, so
   errors don't compound, and hand edits are never silently clobbered.

## Architecture

```
                 ┌── gist tier (SQLite, always-on, embedded) ──┐
  query ──hybrid─┤  library_pages: summary + key_facts + vec    │── inject "Library" section
                 └──────────────────────────────────────────────┘        (adjacent to memory)
                          │ link table (provenance)
                          ▼
                 ┌── detail tier (Markdown files on disk) ──┐
   on demand ───▶│  full page text — fetched, not resident   │── injected only when pulled
                 └────────────────────────────────────────────┘
```

- **SQLite is canonical for the gist + the index** (and a *derived, rebuildable* index over the MD;
  if it's lost, regenerate it from the folder — a projection, not a second source of truth, so the
  §10 "never silently a second store" doctrine holds).
- **Markdown is canonical for the detail pages** — human-editable, git-versionable, portable,
  loadable on demand.
- **A link table is the provenance graph** — page → the documents/memories it was synthesized from;
  traceable both ways.

### Schema stub (a new migration → adds to the ladder + `EXPECTED_SHAPE`)

```sql
CREATE TABLE library_pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    path         TEXT    NOT NULL UNIQUE,   -- the MD file under [library] folder
    summary      TEXT    NOT NULL DEFAULT '',   -- gist: a few sentences
    key_facts    TEXT    NOT NULL DEFAULT '',   -- gist: high-impact bullet facts
    embedding    BLOB,                       -- of summary+key_facts, for hybrid retrieval
    content_hash TEXT    NOT NULL,           -- of the MD file, for change detection
    edited       INTEGER NOT NULL DEFAULT 0, -- 1 = hand-edited; re-derive must not clobber
    created_at   REAL    NOT NULL,
    indexed_at   REAL    NOT NULL
);
CREATE TABLE library_sources (              -- provenance: which sources fed this page
    page_id INTEGER NOT NULL REFERENCES library_pages(id) ON DELETE CASCADE,
    source  TEXT    NOT NULL                 -- a document source path / memory id
);
CREATE INDEX idx_library_sources_page ON library_sources(page_id);
```

(Gist lives in its own table, not `memories` — it is a *separate layer*, kept out of the knowledge
recall, surfaced as its own section. Raw memory and raw document chunks stay in `memories` unchanged.)

### Retrieval flow

1. Hybrid (keyword + embedding, reuse `retrieval.hybrid.retrieve`) over `library_pages` → the matched
   gist rows.
2. Inject the matched gist into a dedicated **"Library"** section in `build_context()` — adjacent to
   the memory/knowledge block, clearly framed as derived ("from my library, summarized").
3. The full detail is **not** resident — it is pulled only by the Load button (Phase 1) or the model
   (Phase 2), via the shared fetch path.

---

## Phase 1 — system-driven gist + a UI Load button (buildable on the current spine)

1. **Schema** — `library_pages` + `library_sources` (migration; `EXPECTED_SHAPE`).
2. **Build/import pages** — pages come from hand-authored MD in the `[library] folder`, and/or are
   **compiled in idle time** from ingested documents (a sleep phase): derive `summary` + `key_facts`
   with one `reasoning` call, embed, content-hash, link to sources. Fail-soft (no fleet → skip,
   re-derive later), non-destructive (don't overwrite `edited=1` pages).
3. **Retrieve + inject** — hybrid over `library_pages`; the matched gist becomes the Library section.
4. **UI** — after each reply, show the relevant page(s) as chips: *📖 Title — [Load]*. **Load**
   fetches the MD (`GET /api/library/page?id=`), **displays it**, and **pins it into the next turn's
   context** via a small "active sources" tray (✕ to drop). Budget-aware: loaded detail is the only
   thing that spends the window on full text, and only because the user asked. (The chips double as a
   **sources/provenance** affordance: "here's what the answer drew on — click to verify or go deeper.")
5. **Sync** — content-hash scan (reuse the documents-ledger pattern): changed file → re-derive gist +
   re-embed; new file → index; deleted file → drop rows. Manual "Scan library" button.
6. **Endpoints** — `GET /api/library` (list), `GET /api/library/page?id=` (fetch one),
   `POST /api/library/scan`.

---

## Phase 2 — model-driven fetch (the planned tool usage)

Goal: the model itself opens a page mid-turn when the gist isn't enough — *"let the model press the
same Load button."* It reuses the **exact Phase-1 fetch path**, so Phase 2 is small once Phase 1 ships.

### The tool contract (two equivalent surfaces — pick per model capability)

**A. Native tool call** (models that support tools):

```jsonc
{
  "name": "read_library_page",
  "description": "Open a Library page for full detail when the resident gist isn't enough to answer. "
                 "Use a page_id (or title) from the Library section already in your context. "
                 "Returns the page's markdown; cite it as a Library source.",
  "parameters": {
    "type": "object",
    "properties": {
      "page_id": { "type": "integer", "description": "id from the Library section in context" },
      "section": { "type": "string",  "description": "optional heading to fetch just one section" }
    },
    "required": ["page_id"]
  }
}
```

**B. In-band fallback** (models without tool-calling — keeps the "works on a dummy" floor): the model
emits a marker in its draft, e.g. `<FETCH page="17" section="Reciprocal altruism">`, intercepted on
the **same path as the `<RECALL>`/epistemic-tag sanitizer** (`sanitize.py`), then stripped from the
user-facing reply.

### The loop (gated, hot-path, opt-in)

```
turn → draft (model sees the gist) → detect fetch request(s)
     → load page/section (cap N per turn; budget-checked; fail-soft: a missing page = a noted gap,
       never a crash) → re-prompt with the page injected as an active source → final answer
```

- **Opt-in by config** (`[library] model_fetch = false` by default) — it's a deliberate second pass,
  the same spend the user's Load button makes, just model-initiated. Off by default keeps turns fast.
- **Caps:** `max_fetches_per_turn`, budget check before each load (respect the 24k window).
- **Shared plumbing:** the Load button (Phase 1) and the tool (Phase 2) hit the same fetch + the same
  "inject as active source" code path. Building Phase 1 *is* most of Phase 2.

---

## Cross-cutting decisions / gotchas

- **Gist is tiered `derived`** — honest about being a summary; the model can escalate to the verbatim
  page when precision matters (the Load button / the tool).
- **No-clobber hand edits** — re-derivation respects `edited=1`; regenerating a page is deliberate.
- **Section-granular** indexing so a fetch can pull one section, not always a whole page (24k budget).
- **Fail-soft file IO** — a renamed/missing MD logs and is skipped, never breaks the turn (§10).
- **Sync is content-hash driven** — the SQLite index is always rebuildable from the folder.

## Config (planned)

```toml
[library]
folder = "library"          # MD detail pages (separate from [documents] raw drops)
enabled = true              # the gist layer + retrieval
gist_top_k = 3              # how many page gists to surface per turn
model_fetch = false         # Phase 2: let the model open pages itself (opt-in; off = faster turns)
max_fetches_per_turn = 2    # Phase 2 cap
```

## Build order

1. Phase 1 §1–3 (schema, compile/import, retrieve + inject) — the gist layer working end-to-end.
2. Phase 1 §4–6 (UI Load button, sync, endpoints) — the human-in-the-loop detail fetch.
3. Phase 2 (the tool) — once Phase 1's fetch path is solid; start with the in-band fallback (works on
   any model), add native tool-calling for models that support it.
