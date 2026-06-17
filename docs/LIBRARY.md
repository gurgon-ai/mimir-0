# The Library layer — three tiers of truth (documents → claims → composites)

**Status: foundation built (the data layer); pipeline + UI staged below.** The system's own long-form
knowledge — "books I've read" — as a layer **adjacent to memory, never replacing it** (its own
`build_context` section, like documents or the graph).

> **Not the `[wiki]` block.** `[wiki]` is a *read-only Kiwix/ZIM offline encyclopedia*
> (`cognition/wiki.py`), an external reference. The **Library** is the system's *own* compiled
> knowledge, with full provenance back to the source.

## Three tiers of truth (the DB is the spine)

The key idea (and the realization that shaped this): the DB isn't an index, it's the **provenance
spine** connecting three tiers of decreasing fidelity / increasing readability:

1. **Source documents = ground truth.** The ingested files, **left where the user dropped them**
   (the `[documents]` folder), recorded by **exact filename + size + hash + title**. Never rewritten;
   always re-readable down to a cited line/page.
2. **Database = short, cited claims.** Atomic facts distilled from the sources, each carrying its
   **source document + exact locator (page/line/section)** and an embedding. This is the always-on,
   queryable, *citable* layer — "the database is actually a database." Honest and exact.
3. **Markdown composites = fuzzy understanding.** The LLM's synthesized, cross-referenced pages — a
   "cogenesis of LLM logic" — in a **separate `[library]` folder/tree**, fetched on demand. Derived,
   never source of truth.

```
 source doc (title, p.14, line)  ──cited by──▶  claim (DB: short fact + locator + vector)
   ground truth, re-readable                       always-on, retrievable, citable
                                                        │ composed (library_page_claims)
                                                        ▼
                                              MD composite page (fuzzy understanding)
                                                  separate folder, fetched on demand
```

**Traversable both ways:** a claim cites its source line *and* knows which composite(s) it fed; a
composite lists its claims → down to source lines. So the system can **cite anything, verify
anything, and go back to the source if necessary** — which is exactly what document-creation-with-
citations needs, and what keeps the whole thing epistemically honest.

## Influence + how this differs

Influenced by the **"LLM wiki" pattern popularized by Karpathy and others** (distillation over
raw-chunk RAG; an LLM-maintained knowledge base) — a widely converged idea, not proprietary. This
differs in the ways that matter for Mimir: **provenance** (every claim cites a source line; composites
trace to claims), **small window** (claims are cheap and resident; composites/sources load on demand —
we never dump the corpus), and **non-destructive** (composites are re-derived from claims, not
mutated-in-place, so errors don't compound and hand edits aren't clobbered).

## Schema (built — migration 21)

```sql
library_documents  (id, path, filename, size_bytes, content_hash, title, ingested_at)  -- ground truth
library_claims     (id, document_id, text, locator, embedding, confidence, created_at) -- the DB spine
library_pages      (id, path, title, summary, content_hash, created_at, updated_at)    -- MD composite
library_page_claims(page_id, claim_id)                                                 -- provenance link
```

`storage/models.py` (`LibraryDocument` / `LibraryClaim` / `LibraryPage`) + `storage/repo.py` CRUD
(`upsert_library_document`, `replace_document_claims`, `claims_for_document`, `upsert_library_page`,
`set_page_claims`, `claims_for_page`, `delete_library_document` (cascades), …) are the data layer.

## Retrieval = a depth ladder over the three tiers

- **Always-on:** hybrid over `library_claims` → short facts injected into a "Library" section, each
  **shown with its citation** (title, p.14). Exact + honest.
- **On demand (composite):** the MD page (the fuzzy understanding) — the Load button / Phase-2 tool.
- **On demand (verbatim):** the exact source line/page via a claim's `locator` — for quoting/checking.

## Pipeline — both extraction and composition run in **idle**

Source intake = the `[documents]` folder (the existing drop folder). The Library pipeline is a sleep
phase:
1. **Extract claims** from each source doc: for each extracted unit (which carries a locator —
   `p.4`, a heading), an LLM pass distils atomic facts; each claim is stored with that locator +
   `document_id` + an embedding. Content-hashed, so unchanged docs are skipped; re-extraction
   replaces a doc's claims.
2. **Compose composites** from the claims: the LLM synthesizes a Markdown page (fuzzy understanding)
   and links it to the claims it used (`library_page_claims`). Written to the `[library]` folder.
   Re-derived (non-destructive), hand edits respected.

## Staging

- **Phase 1a — data foundation. ✅ built.** Schema (migration 21), models, repo CRUD, the
  provenance links + cascade, round-trip tests.
- **Phase 1b — claims spine. ✅ built.** Idle claim extraction (`ingest_pending_library`: each source
  doc → `library_documents` + cited `library_claims`, one per fact with its locator + embedding) and
  hybrid claim retrieval (`_library_gist` → `retrieve_claims`/`render_claims`) into the "Library"
  `build_context` section — each fact shown with its citation `[title, locator]`. A `library` sleep
  phase. Source = the `[documents]` folder; `[library] claims_top_k`.
- **Phase 1c — composites + UI. ✅ built.** The idle pass compiles a Markdown composite from each
  document's claims (`_compile_composite`/`compose_page`) into the `[library]` folder, linked to its
  claims (`set_page_claims`), non-destructive (a hand-edited page is left alone). A **Library tab**
  lists composite pages + source documents; clicking a page shows its full Markdown + its source
  citations (claim → title + locator); a **pin-to-chat** toggle adds a page to an "active sources"
  tray, and pinned pages are loaded into the next turn (`turn(loaded_pages=…)` →
  `_merge_loaded_library`). Endpoints: `GET /api/library{,/page,/source}`, `POST /api/library/scan`.
  *Follow-up:* after-reply chips (surface which sources the answer drew on for one-click load).
- **Phase 2 — model-driven fetch.** A `read_library_page(page_id, section?)` native tool + an in-band
  `<FETCH …>` fallback (works on any model, intercepted like `<RECALL>`), reusing the Phase-1c fetch
  path — "let the model open the book / pull the source itself." Opt-in, capped, fail-soft.

## Config (planned)

```toml
[library]
folder = "library"          # MD composite pages (separate folder/tree; the [documents] folder is source)
enabled = true
claims_top_k = 5            # how many cited claims to surface per turn
model_fetch = false         # Phase 2: let the model fetch composites/sources itself (opt-in)
```

## Cross-cutting decisions / gotchas

- **Claims are tiered as derived but always cited** — the model answers from a fact *and* its source,
  and can escalate to the verbatim line or the composite when needed.
- **Non-destructive composites; no-clobber hand edits.**
- **Content-hash sync** — the DB (claims + composite index) is a rebuildable projection over the
  source files + the MD folder; no second source of truth.
- **Fail-soft file IO** — a missing/renamed source is a noted gap, never a crash (§10).
- **Section-granular** where possible, for the 24k window.
