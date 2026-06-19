# Changelog

All notable changes to Mimir 0. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
Pre-1.0: the API and schema may change between releases.

## [Unreleased]

First fixes from real single-machine + LAN use after the feature-complete cut.

### Added
- **The "Fleet" tab is now "Models".** It holds all the model information (roles, qualification,
  per-node placement), so the label matches what's there. The "Save schedule" button on the Sleep tab
  is now "Save settings" — it persists the whole settings form (timezone, sleep window, inner life,
  *and* context size), so it reads right sitting under the Context-size section.
- **A "⏭ Skip model" button on the gauntlet progress bar.** If a model is grinding (a thinking model,
  a slow node), skip it — `should_skip(model)` is checked between every dimension and before every
  call, so it aborts that model's battery promptly (raising `_SkipModel`, which — unlike a failover —
  does *not* retry on another node; the user wants this model skipped) and moves on. Wired
  benchmark_model → benchmark_fleet → brain → server (`POST /api/fleet/benchmark/skip`), for both the
  benchmark and the tournament gauntlet.
- **A Context-size slider (Small / Medium / Large / X-Large) — one knob for the KV-cache window and
  how much you recall.** VRAM and KV-cache size vary by box, and there's no point pushing in more
  unique facts than the window can hold — so the two move together. Each preset sets the operational
  `num_ctx` injected into *every* model call (consistent, so a warm model isn't reloaded between
  callers), the `context_budget_tokens` we assemble, and the `benchmark_num_ctx` we qualify at:
  Small 4096/2048 · Medium 8192/4096 · Large 32768/12288 · X-Large 65536/24576. A runtime setting
  (kv override, default Medium), applied live on the Sleep tab — changing it reloads warm models at
  the new window. `brain._apply_context_size()` / `ModelGateway.set_operational_num_ctx()`.
- **The Finals "Your champions" picker is now the onboarding surface: per-role colour, role
  descriptions, and a checkable council pool.** Each role's dropdown is coloured by what *that* role
  needs (🟢 strong · 🟡 ok · 🔴 weak), defaulting to the top pick, with the role's description shown
  to the right (the board is wide and this is where you first meet the roles). **Council** moved to
  the bottom (after vision) and lists *every* eligible model with a **checkbox each** — toggle a
  model in/out of the adversarial pool (`council_excluded`, persisted; benched from deliberation
  without disabling it everywhere) via `POST /api/fleet/council/member`.
- **A points-based recommendation + a harder, de-saturating battery — so the picker stops crowning
  the wrong model.** Two compounding flaws made it recommend a small old model over a clearly better
  big one: (1) the battery was too easy — `talk`/`tools`/`code` were 1.00 for nearly everyone, so
  ~16 models jammed between 0.92–1.00 and the ranking was noise; (2) ties then broke by speed, handing
  it to the *smaller/faster* model. Fixes: the **reasoning** dimension's cases are now chosen
  empirically (probed across a 3B→32B fleet spread; kept only the ones that discriminate — bat-and-
  ball, the "all but" trap, Sally's-sisters, a painted-cube count, etc.; dropped the saturated ones),
  so quality actually spreads (e.g. a think-off deepseek-r1 that can't reason scores ~2/10, not top).
  And recommendation is now a transparent **points total** — quality for that role (dominant) + speed
  (strong, *every* role: a 4-minute background worker is useless) + a faint size prior to break
  near-ties toward capacity. Each pick carries its `score` + a `quality/speed/size` breakdown, shown
  in the Finals picker.
- **"＋ Qualify new" — benchmark only models you've added, not the whole fleet.** Installing one model
  no longer means an hour-long full re-run to re-learn what you already know. A **merge-scan**
  discovers newly-installed models while preserving every existing score (`merge_catalogue` /
  `scan_fleet(merge=True)`, vs. the full run's clear-then-rebuild), then only the unranked models
  (`quality is None`) are benchmarked (`brain.qualify_new_models`, `POST /api/fleet/benchmark/new`).
  A new Fleet-tab button runs it on the same live board; finds nothing → says so.
- **Interactive "Your champions" + a tidier Fleet tab.** The tournament Finals is now a per-role
  **picker**: each role is a dropdown of its eligible models, defaulting to the recommendation;
  changing one pins the role immediately. **Council** shows as the whole eligible pool (multi-model
  adversarial reasoning, not a single pick). The manual role-assignment list is **compressed** — the
  per-role description moved to a hover tooltip, the pick list is narrower, and the state tag is now
  an emoji (📌 pinned · ⚙️ auto). The **embed role** finally lists embedding models (they're excluded
  from the chat placement matrix, so the dropdown was empty even with an embedder on every node — it
  now has its own per-node list). **Vision** moved to its own column after Speed, set off by a
  divider, with a note that it's a capability check that does **not** affect the quality score.
- **`--reembed` — rebuild the whole vector store with the current embed model.** `Mimir.reembed()`
  re-embeds every memory, library claim, and procedure trigger (CLI: stop the server, run
  `python -m mimir.server --config mimir.toml --reembed`, restart). Use it after changing
  `[roles.embed]`: vectors from different models are incomparable *even at the same dimension*, so a
  model swap silently degrades recall until the store is rebuilt. Non-destructive — rows whose embed
  call fails are left untouched, and it aborts cleanly if the embedder is degraded.
- **`.docx` ingestion (the `[documents]` extra now pulls `python-docx`).** Word documents are
  extracted with heading-style sections as locators (e.g. `report.docx:Methods`), so they flow through
  the 📎 upload, the drop folder, the Library, and `brain.ingest(...)` exactly like `.txt`/`.md`/`.pdf`.
  Without the extra a `.docx` fails loud with the install instruction (`.doc` legacy format isn't
  supported — `.docx` only).
- **The Library layer — the system's own long-form knowledge as three tiers of truth (docs/LIBRARY.md).**
  The DB is the *provenance spine*: source documents (ground truth, left in place; recorded by exact
  filename + size + hash + title) → short **cited claims** (DB: atomic facts, each carrying its source
  document + locator + an embedding) → Markdown composites (the fuzzy LLM understanding; Phase 1c).
  Linked both ways so the system can cite, verify, and re-read the source line — for citations and
  epistemic honesty. *Built so far:* the data foundation (migration 21: `library_documents` /
  `library_claims` / `library_pages` / `library_page_claims`) and **Phase 1b — the claims spine**:
  an idle pass (`ingest_pending_library`, a `library` sleep phase) distils each source document into
  cited claims, and a hybrid retrieval surfaces the most relevant claims into a "Library" context
  section adjacent to memory, **each fact shown with its `[title, locator]` citation**. **Phase 1c**
  adds the **composites + Load UI**: the idle pass synthesizes a Markdown composite (the fuzzy
  understanding) from each document's claims into a separate `[library]` folder, linked to its claims
  (non-destructive — a hand-edited page is left alone); a **Library tab** lists pages + sources, shows
  a page's full Markdown with its source citations, and lets you **pin a page to chat** (loaded into
  the next turn). After a reply that drew on the library, **Load chips** show the source page(s) for
  one-click pinning. `GET /api/library{,/page,/source}`, `POST /api/library/scan`. **Phase 2 —
  model-driven fetch** (opt-in `[library] model_fetch`): the model may reply `<FETCH id=N>` to open a
  page itself; the turn loads it and re-answers with the detail (capped, off by default, on the
  non-streaming `turn()`/agent path; the UI uses the chips). The Library is feature-complete.
  Influenced by the Karpathy "LLM wiki" pattern, made provenance-preserving + small-window +
  non-destructive.
- **A documents drop folder + 📎 upload → idle-built local "wiki" (DESIGN §8).** A `[documents]
  folder` the UI's paperclip saves into and you can drop files into directly; idle time (a new
  `documents` sleep phase, or the Docs tab's "Scan folder now") ingests new/changed files into
  recallable `document`-tier knowledge — content-hashed so unchanged files are skipped — and writes a
  short summary of each, a small browsable wiki on the Docs tab the model also draws on. Upload is
  immediate (recallable now; summary follows on the next idle pass); a changed file is re-ingested and
  re-summarized. `.txt`/`.md` in core, `.pdf` via the `[documents]` extra (`pip install
  'mimir-0[documents]'`). New routes: `POST /api/documents/upload` (base64), `POST
  /api/documents/scan`, `GET /api/documents`. `Mimir.upload_document` / `ingest_pending_documents` /
  `documents`. SETUP §5b documents the drop folder, PDF setup, and the integration path.
- **The inner life now feeds the forum (inner-life → council escalation).** When the idle loop lands
  on a genuine, *fresh* conflict it occasionally convenes the full **council** on it — a forum thread
  + verdict — instead of a solo musing, so the council stirs during the day on what the system's own
  attention surfaces, not only in the nightly sleep pass. Gated to a daytime trickle: a conflict
  stimulus, the self-directed council enabled, a healthy fleet, an hourly cooldown, and it shares the
  sleep deliberation's seen-set so neither re-argues the other. "Think now" stays a quick solo musing.
  `Mimir._should_escalate_to_council` / `_escalate_to_council`. (Answers "the forum went quiet": the
  doc-chunk junk is filtered and the old conflicts were already argued, so the forum now fills from
  fresh tensions — and the inner life actively drives it during idle.)
- **`[roles.embed] = "auto"` — discover the embedding model instead of pinning a brittle tag.** You
  provide whatever chat LLM(s) you like and at least one embedding model; Mimir finds them. Chat roles
  already auto-resolved (benchmark-ranked); now `embed` does too — but specially, because embedding
  models define the *vector space*: it **discovers and remembers** one (persisted in `kv`), prefers
  the remembered choice across restarts, and if that model goes unreachable it stays pinned and lets
  the embedder degrade to keyword recall rather than silently switching to an incompatible model. The
  gateway's `auto` stop-gap pick is now deterministic (sorted), and `resolve_auto_model` uses the
  proper `is_embedding_model` check so a chat role never grabs an embedding model (`all-minilm`, `bge`,
  …). `Mimir._resolve_embed_model`. SETUP/example default `embed` to `auto`.
- **Inner life, Slice 2 — idle thoughts now earn their way into conversation (DESIGN §5a).** A
  reflection is a *framed* thought, not a knowledge fact, so inner-life memories are split out of the
  knowledge recall; at turn time the one most relevant to the current input — if it clears a relevance
  bar — is surfaced as a single tentative background note ("while idle I'd been thinking…", weighed as
  the system's own idle thought, not as fact). Gated and framed, never force-injected; off-topic
  musings stay silent. `Mimir._surface_inner_life`. (Slice 1 stored thoughts but only surfaced them
  passively via ordinary recall, where a faint musing rarely won a slot — and would have read as a
  fact if it did.)
- **A peer-AI source tier — ontology, not just trust level (DESIGN §3b).** The trust policy
  conflated *who* a speaker is (trust) with *what kind* of thing they are. A peer AI is its own
  category: it emits generated text that may be confabulated, or may *echo* something this system
  itself said (so two agents agreeing fakes corroboration). New:
  - **`speaker_kind` on the turn API** (`"human"` default / `"ai_peer"`) — the caller declares the
    kind; an unknown value is rejected (the policy never resolves ambiguity by elevating).
  - **`[identity] peer_agents`** — operator-side enforcement: identities known to be AIs always bake
    as peer, even if they send `speaker_kind="human"`.
  - **`STATED_BY_PEER` evidence tier (0.95)** — below human conversation, attributed and marked
    AI-sourced (`provenance="stated by peer AI <name>"`), and a *decaying* tier so peer chatter fades
    and archives like other low-value content. **Kind wins over identity** — an agent can't reach a
    human tier by also being named primary/trusted.
  Build your own front-end → `speaker_kind="human"` (users are treated as users). Wire another agent
  in → `speaker_kind="ai_peer"`. `bake._tier_and_provenance(..., is_peer=)`, `normalize_speaker_kind`.

### Fixed
- **The tournament board now shows the FAILED test results too — every machine the run tested, not
  just the ones with a passing model.** The board grouped the live `results` by machine (already the
  right layout), but `results` only holds models that *scored*, so a model that timed out — or a
  machine where every model did — vanished entirely. A failed test is still a test result, and the
  gap made it look like the speed round never ran. Fix is data-only (`withAllMachines()`): the same
  table is fed the scored results **plus** the pairings the run actually tested on each machine — a
  recorded score *or* time, a timed-out probe counting as a (slow) time — gated to the run's filters
  (size band, enabled machines). Failed pairings render as blank-quality rows in the existing style;
  nothing shows on a machine until it's been tested there, and the full catalogue (untested,
  oversized, disabled) is never dumped in. (Backend: `placement` added to the tournament status; the
  per-round veto keeps one keep-checkbox per scored model.) `_docx_table_lines` de-duped merged
  cells by `id(cell._tc)`, but that proxy is a temporary — once CPython GC'd it, the same address
  could be handed to a *later* cell's proxy, so a fresh cell collided with a stale id and was silently
  dropped. (It only surfaced in the full test run, where memory/GC timing differs — a table cell would
  vanish from the extracted text.) Now `seen` holds the cell **elements themselves**: dedupe by
  identity, and the live reference keeps the id from recurring.
- **Larger Large/X-Large context presets.** Large 16384→**32768** window (12288 budget), X-Large
  32768→**65536** (24576 budget) — roughly 2× Large, still within many models' max — so the slider's
  top end actually reaches a long-context window. Small/Medium unchanged.
- **Cleaner qualification board: per-model "test X/N" progress, one time per machine.** Each model
  runs the whole battery (talk, tools, code, discipline, reasoning, latency, epistemics, vision), so
  the gauntlet now shows a per-model sub-bar — *test X/8 · scoring reasoning… (3 left)* — under the
  round progress (`benchmark_model` reports each dimension via `on_step`; threaded to the bench +
  tournament status as `current_step`). And the Speed column shows just **that machine's** time in
  seconds (a node-grouped table was wrongly listing *other* machines' times under one machine's
  heading); the model-centric leaderboard shows its single fastest node. The full per-machine
  breakdown (every model on each node, all stats) stays in 📊 Per-node placement. The Speed-test
  button also resolves correctly after a reload during the auto speed-test phase.
- **The full speed-test is now an automatic, visible phase of the benchmark/tournament — not a
  skippable manual step.** The recommendation's speed term uses each model's *fastest known* node,
  but the benchmark only timed the one node it scored each model on — so until every `(model, node)`
  was timed, "fastest known" could be wrong (a node can be far faster than the one a model was scored
  on). That timing lived behind a manual "3 · Speed-test" button that was easy to skip and left no
  visible trace. Now scoring is followed by a **speed-test phase that times every remaining pairing**
  (shown live as "⏱ Speed-testing every (model, node) pairing — i/total"), so the recommendation is
  only computed once speeds are complete, and the per-node times then appear in the leaderboard's
  **Speed** column. (The manual button remains for re-runs, e.g. after re-enabling a node.)
- **"Apply best" / the tournament finals now survive a restart.** Applying roles changed only the
  in-memory config, so after a reboot the role silently reverted to its default while the
  recommendation still showed it — looked like the apply did nothing. Apply now persists each role as
  a pin (the same store `set_role` uses and `_restore_role_pins` reads on boot), so it sticks and
  appears pinned in the picker. (The interactive per-role dropdown already persisted; only the Apply
  buttons didn't.)
- **The role picker colours each model PER ROLE, by that role's requirements — not one deceptive
  generic status.** A model strong for chat but weak at vision/code was painted red in *every* role's
  dropdown, implying it was a bad chat pick when it's a great one. Now green/yellow/red reflects only
  the dims the role actually needs (`brain.role_requirements()` → `role_caps`): green = every required
  dim ≥0.8, yellow = all ≥0.5 (eligible), red = a required dim falls short. Vision colours as a
  capability (none/sees/full), matching its column.
- **Vision is scored best-across-nodes — a model isn't marked blind because it was tested on a node
  whose Ollama mangles its vision.** Vision turned out to be *per-node*, not model-wide: a
  byte-identical model file (same digest, same vision projector) reads the probe image perfectly
  under Ollama 0.20/0.22 but returns garbage under 0.30.7 — a runtime regression, not GPU/quant/pull.
  So the same model scored 1.0 on one node and 0.0 on another, and which node it happened to be
  benchmarked on decided red-vs-green. Now, for a model that actually carries a vision projector
  (checked via `/api/show`, which is consistent across versions — the `/api/tags` capabilities the
  catalogue stores are not), the benchmark probes vision on its other nodes and records the **best**
  (stops at full). Text-only models are never probed (the projector check gates it), so it adds no
  cost to the common case. The per-node probe is hard-bounded (`_VISION_PROBE_TIMEOUT`, warm-then-
  read, skip a node that can't load in budget or is busy) so it can't stall — an early build let it
  cold-load a multimodal model across four nodes at a 120s-per-call timeout, parking a *fast*
  (3.3s/turn) model for ~10 minutes.
- **A bad node no longer hangs a model's benchmark for 20 minutes — it fails fast and fails over.**
  A node can pass the quick speed probe yet hang the real battery (intermittent, or it loads a
  1-token warm but stalls on actual generation), so all ~12 scoring calls hit the per-call timeout in
  turn — ~700–1200s stuck on one model, which then records a false ~0 even though the model runs fine
  elsewhere. Now the battery aborts a node after a few transport failures (or a failed warm) and
  **fails over to the next node the model is installed on** (capability is never failed on speed —
  DESIGN §6), scoring it on a node that works instead of zeroing it on one that doesn't. A completion
  signal (`on_done`) fires for every model — scored, failed-over, or no-viable-node — so the
  progress view never leaves a ghost model climbing past 20 minutes.
- **Ctrl-C stops the server even mid-benchmark.** A running benchmark/tournament drives a
  `ThreadPoolExecutor` whose workers are non-daemon (`concurrent.futures`, 3.9+) and blocked in
  uninterruptible model calls; the interpreter's `atexit` handler joins them, so Ctrl-C hung the
  terminal until a slow remote call returned (and daemon HTTP threads spewed `storage gateway is
  closed` 500s in the meantime). `serve()` now does a clean shutdown — stop serving, `server_close()`
  the socket, `brain.close()` (storage commits per op, so abandoning in-flight scoring is safe, at
  worst a half-scored catalogue a re-run completes) — then force-terminates so background work can't
  hold the terminal hostage.
- **Benchmark speed is now true decode throughput (TPS), not contaminated wall-clock — so a fast MoE
  stops losing to a slower dense model.** `return_time` was timed as wall-clock (model load +
  prompt-eval + decode) and normalized by an *estimated* token count, so a fast MoE (gemma4:26b, 4B
  active, ~210 tok/s) caught mid VRAM-swap recorded a fake ~38s/turn and lost speed-weighted roles to
  a genuinely slower dense model (deepseek-r1:14b), which a terse think-off reply made look fast. The
  provider now exposes Ollama's `eval_count`/`eval_duration` (`chat_timed`), and both the battery and
  the pre-gate probe compute **seconds-per-256-token-turn from pure decode** — load-immune, identical
  whether the sample is 64 or 600 tokens, so the leaderboard, the placement matrix, and role
  selection finally agree on one number. The pre-gate probe also grew 64→128 tokens to clear the
  decode warmup ramp (a 64-token probe under-read a fast MoE ~20%).
- **The leaderboard is one row per model again (per-node speed lives in the cell, not in duplicate
  rows).** A per-node expansion had blown each model into one row per node it was *timed* on —
  duplicating models, leaving gaps where a node wasn't timed (reading like a per-node "fail"), and
  generally a mess. Capability is model-wide (scored once), so the board shows it once, ranked by
  quality (speed breaks ties), with a **Speed/turn** cell listing every node it runs on (🖥️ local ·
  🌐 LAN, fastest first). The per-`(node, model)` breakdown stays in 📊 Per-node placement.
- **The "scoring…" header no longer sticks on a finished model for minutes.** Scoring runs one worker
  per node concurrently, but progress tracked a single `current` = the last model to *enter* scoring,
  so when a fast node finished a model (already in the list) while a slow node ground on, the header
  showed the finished model's name with a timer climbing past 10 minutes. Status now tracks an
  in-flight map and reports the **longest-running in-flight model** (the real bottleneck — never one
  already done), its true elapsed, and a `+N more` count, for both the tournament and the benchmark.
- **Council membership is the diverse pool it was meant to be, not a near-empty list.** It required
  `reasoning ≥ 0.50` (the full identity-role floor), which dropped most yellow models — so the
  "second lineup" collapsed to a handful. Council is diversity-first: it now admits every yellow/green
  model (`quality ≥ 0.50`) that can reason at all (a light `reasoning ≥ 0.25` floor), via a
  per-(role, capability) floor-override mechanism (`_ROLE_FLOORS`). (The single-model "council" line
  in the Finals is its strongest member; the diverse roster is the Council tab.)
- **Vision reads yellow when a model sees but can't OCR — it was wrongly red.** Vision is capability
  *detection*, but the board ran it through the quality scale (`<0.5 → red`), so a model that counted
  the probe's shapes (it sees) but missed the pseudoword OCR scored 0.4 and rendered red, as if blind.
  It now bands as a capability: **❌ none · 🟡 sees (partial) · ✅ full**, and the vision role gate
  agrees (any model that passes a vision case, `≥ 0.4`, is vision-capable; only 0.0 is barred).
- **A downed embedding backend no longer crashes the brain — it degrades to keyword recall, loudly.**
  Found a live run where the only embedding-capable node was offline, so every embed (query recall,
  bake, procedures, inner life) raised `model "nomic-embed-text:v1.5" not found` and the loop spammed
  tracebacks. Now the endpoint embedder is wrapped in a `ResilientEmbedder`: on a backend outage it
  logs one throttled warning (captured into the error ring → shown in the Mind tab) and returns
  `None`, which every caller already treats as the keyword-only path — so turns keep working in
  degraded mode and resume semantic recall on their own when the model returns (DESIGN §10: fail
  loud, keep working; not a silent backend swap — same store, announced). Inner life skips a cycle
  cleanly when embeddings are down rather than storing an un-deduplicatable musing.
- **Embed-model `"auto"` could strand a store on an unreachable model.** `"auto"` remembers the first
  reachable embed model and stays pinned to preserve the vector space — but a live run showed the
  remembered `embeddinggemma:300m` going away while only `nomic-embed-text:v1.5` stayed up, leaving
  every embed degraded to keyword-only with no automatic recovery (it won't silently switch vector
  spaces). Documented the safer alternative (pin the exact tag) and shipped `--reembed` as the clean
  recovery path; `mimir.toml.example` and SETUP now spell both out.
- **Dropped the coherence dimension — it measured nothing.** The peer-judged coherence pass scored
  every model mid-range (🟡) regardless of quality: the probe was trivial (every model answered it),
  the same fixed panel graded everyone's ~identical answer, and a vague "rate 0.0–1.0" prompt makes
  LLM judges cluster ~0.65 — so it never discriminated *and* it duplicated what `epistemics` already
  measures deterministically. Removed the judge panel + canary + the coherence column (and its uniform
  drag on `quality`). The DB column is kept (NULL) to avoid a migration. (A genuine multi-turn
  coherence probe — the one thing nothing else measures — is noted as a future *finalists-only* test,
  too costly to run fleet-wide.)
- **Verifiable dims scored greedily (temperature 0) — no more luck-flips.** The single-correct-answer
  dimensions (talk/tools/code/reasoning/vision) now run at temperature 0, so a near-tied model isn't
  knocked to 🟡 by one unlucky high-temperature draw (the observed "gemma4:e2b all-green but the
  stronger e4b yellow on one dim" bug). `discipline` and `epistemics` keep the sampled temperature on
  purpose — their signal is consistency across runs (a stochastic tag-leak / repeated gauntlet).
- **Vision in documents + image upload — the scanned-doc gap closed.** Drop an image in the documents
  folder (or upload one with 📎: `.png/.jpg/.jpeg/.webp/.gif/.bmp`) and the **`vision`-role model
  describes + transcribes it** (`VISION_DESCRIBE_SYSTEM`: description + verbatim OCR) into recallable
  document-tier text — so it answers about photos and scanned pages. Gated on a vision model existing
  (the empirical benchmark + `[roles.vision]`); with none, an image ingest **fails loud** with the
  reason (never silent). `[vision] describe_images` toggles it off entirely. The Library tab shows the
  active vision model (or "no vision model — images won't ingest"). New `ingest_text()` shares the
  chunk/embed/store path; `list_documents` now includes images. (Next/limit: still no OCR for
  image-only *pages inside* a PDF/DOCX — standalone images + photos work now.)
- **Role-assignment dropdown colours models by benchmark status.** Once benchmarked, each model in
  the Manual Role assignment picker is tinted: **green** if every scored dimension is ≥0.8, **yellow**
  if any is 0.50–0.79, **red** if any failed (<0.5), and left **white** if not yet benchmarked — so
  you can see at a glance which models are fully qualified for a role. (Vision is informational, so
  it's excluded from the status.)
- **Vision role + self-describing role assignment.** A new optional **`vision`** role (the model that
  reads images, for the upcoming image/document-vision path): `[roles.vision] = "auto"` binds the best
  model that *passes the vision probe* (gated on the `vision` score, ranked by quality) — a non-vision
  fleet simply gets no pick. The role-assignment tab now shows a **one-line description under each
  role** (chat / bake / reasoning / embed / background / council / vision), so what each does is
  obvious without renaming — notably clarifying that `reasoning` is *background cognition* (summaries,
  reflections, distillation), distinct from the loose `background` (inner-life) role. The vision probe
  is now OCR-weighted (0.6 word / 0.4 count) so a text model's lucky count-guess stays below the role
  floor and can't pose as vision-capable.
- **Vision benchmark dimension — capability determined empirically (DESIGN §4 "Round 4").** A model's
  vision is *tested*, not read from advertised metadata: the benchmark sends a fixed probe image
  (`assets/vision_probe.png` — the word GLYPHON + three red circles) and scores reading the word (OCR)
  + counting the shapes. A text-only model can't read GLYPHON or see the circles, so it scores ~0 —
  that failure IS the determination. Informational like coherence (kept out of `quality`, never
  role-gating). Stored as a new catalogue column (`vision`, schema → v22) and shown in the fleet
  leaderboard / placement / tournament tables. (No provider change needed — Ollama already forwards a
  per-message `images` field.) Next: vision in documents + image upload, gated on a vision-capable
  model existing.
- **Draft-RAG — opt-in two-pass recall (a chat toggle).** The model generates a short *draft* answer
  first (capped via `max_tokens`), memory is re-retrieved against that draft — it surfaces what the
  reply is *about*, which the user's literal wording can miss — and the new hits fold into the prompt
  the real answer is generated from. Two LLM calls per turn, so **replies are slower**: off by
  default, a checkbox in the chat toggle row with a one-time confirm warning. Distinct from the
  existing burst-worker output-RAG (which reuses the reply you already made for the *next* turn);
  draft-RAG spends an extra call to help *this* turn. `[draft_rag] enabled/top_k/draft_tokens`;
  `turn(draft_rag=…)`; `gateway.chat(params=…)` now allows a per-call param override.
- **Docs + Library tabs consolidated into one Library tab.** They managed the same source documents
  through two pipelines (chunks+wiki-summary vs cited-claims+composites) with duplicate lists and two
  scan buttons. Now a single tab: one unified document list (each row joins chunks + claims + summary
  + ingest/index time + the include toggle + delete; click the name to expand), one **"Scan & index
  now"** button that runs both passes, and the composite pages, ingest-by-path, and Kiwix/ZIM tucked
  into collapsible `<details>` sections. The data layers are unchanged — only the UI merged. The
  `/api/library` overview now returns the unified per-document rows + the resolved drop-folder path.
- **Context-layer + per-document toggles — choose what's in context (and get recall speed back).**
  Chat now has a row above the bar to toggle whole **layers** per turn — *memory*, *documents*, *wiki*
  (and *deep read*) — for "what am I doing right now." The Library tab lists each document **compactly**
  (an *include-in-context* checkbox + the name; click the name to expand size, index time, and delete),
  so an unselected book is excluded from recall — and crucially **at the SQL load layer** (its chunks
  and claims aren't even loaded), which is the speed lever once a library gets large. `turn(...,
  include_memory=/include_library=/include_wiki=)`, `POST /api/library/enable`; toggles persist.
- **Per-document index time** is recorded and shown (answers "how long per doc"): the documents pass
  stores `ingest_seconds`, the library pass `index_seconds` (the claim-extraction cost), surfaced in
  the Library tab's expanded doc detail.
- **Delete a document and purge everything derived from it (two directions).** The Library tab now
  lists each source document with a **🗑 delete** button (confirm: "Are you sure?") that calls one
  `forget_document` primitive — removing the doc's memory chunks, its library document + cited claims,
  the composite page (DB row + Markdown file), the wiki ledger entry, and the source file. The inverse
  works on its own too: **just delete the file** from the drop folder and the next idle scan
  reconciles and forgets it across every layer. Both keyed by the shared source path
  (`memories.source` == `library_documents.path`); idempotent. (`POST /api/library/forget`; the docs
  scan now also returns `forgotten:[…]`.)
- **Chat renders short `**bold**` spans.** Gemma/Qwen lean on Markdown bold heavily; the chat showed
  the literal asterisks. The UI now turns a SHORT `**…**` span (≤10 words, no line break) into real
  bold and drops the asterisks, in both streamed and replayed (history) assistant messages. A long or
  unbalanced run is left literal (no bolding a whole paragraph or mangling stray `*`), the user's own
  text stays verbatim, and everything is HTML-escaped first (XSS-safe). `fmtInline` in `server.py`.
- **DOCX locators are now the full heading path** (e.g. `Biohazards > Infections > Hantavirus`) — a
  heading stack plus short bold "pseudo-heading" lines (Word's common sub-labels) deepen the path, so
  a claim cites a precise spot rather than just the nearest section. (Detail nested inside table cells
  stays section-level — that structure isn't visible at the paragraph layer.)
- **Citation guard — flag a reply that cites a source the system doesn't hold (DESIGN §10).** A
  deterministic, zero-model-cost post-check: any bracketed citation whose named source matches nothing
  in the system's documents/library (an invented `[National Fire Code 2020]`) gets a fail-loud note
  appended — never silently deleted. Verifies the *source exists* (not that the content is truly in
  it); conservative matching so a real citation is never wrongly accused. On by default
  (`[library] citation_guard`), works on both the streaming and non-streaming paths
  (`cognition/citations.py`). It checks **all** injected sources — documents *and* library — which is
  what my own too-narrow "library-claims only" check missed when it wrongly called a real Hantavirus
  citation fabricated (the content was in the document chunks, just not distilled into a claim).
- **Harden the library framing against fabricated citations.** Observed on a live run: asked about a
  topic adjacent to but absent from the documents (Hantavirus, when the library had bloodborne-pathogen
  claims but nothing on rodents), a small model answered from training-data general knowledge and
  stamped it with a real-looking document citation — a hallucination wearing a source, the worst
  failure for an evidence-tiered system. The Library section now explicitly instructs: cite ONLY what
  appears in the provided entries; if the answer needs something not there, say so and answer from
  general knowledge WITHOUT a citation — never attach a document source to a fact that isn't in the
  entries. (A bigger `chat` model follows this far better; a deterministic citation guard — verifying
  each citation against the sources actually injected this turn — is the architectural follow-up.)
- **"Deep read" toggle — pull full library pages, not just the cited claims.** A switch by the chat
  box (off by default) injects the *whole composite page(s)* of the document the surfaced claims came
  from, automatically — the deterministic, one-pass way to give a capable model the full text for a
  focused question, without hand-pinning each page via Load. Works on the streaming path; full pages
  count toward grounding. (`turn(..., deep_read=True)` / `deep_read` in the turn API body.)
- **Model-driven library fetch now works while streaming (the UI path).** Phase-2 fetch
  (`[library] model_fetch`, opt-in) let a capable model open a full composite page itself with
  `<FETCH id=N>` — but only on the non-streaming `turn()`, so it was inert in the streaming UI. The
  streaming path now peeks the opening tokens, intercepts the marker (never shown to the user), loads
  the page, and streams the final answer with it in context (`_stream_chat_with_fetch`). Completes
  the "toggle now, fetch next" pair.
- **The uncertainty gate now counts library claims as grounding — document Q&A stopped deflecting.**
  `source_count` (which decides whether to inject the "you have very little stored knowledge" honesty
  flag) summed recalled memories + graph edges + wiki passages but **omitted cited library claims** —
  a whole grounding layer. So a question answered from one's own reading (e.g. 462 ingested claims)
  saw `source_count ≤ 1`, tripped the gate, and the model was explicitly told it was on thin ice —
  producing "consult your supervisor / I don't have the manual" deflections on material it actually
  held. Library claims (and user-loaded pages) now count. Also widened the candidate pools that
  document ingestion had starved: knowledge recall `DEFAULT_TOP_K` 6 → 10, `[library] claims_top_k`
  5 → 8 (the token budget still caps what's admitted).
- **DOCX extraction now reads tables — not just paragraphs.** `_extract_docx` walked only
  `document.paragraphs`, so a **table-structured** Word doc (a safety matrix, a form, a spec sheet)
  came through nearly empty — one real file dropped from ~273k characters of table text to ~1k of
  headings, ingesting "successfully" on almost nothing. The extractor now walks the document body in
  order, pulling paragraphs **and** table cells (joined `cell | cell` per row), de-duping merged cells
  and recursing into nested tables, so table content lands under its preceding heading. Both scan
  endpoints gained a **`force`** option (Docs tab "force re-ingest" / Library "force re-distil"
  checkboxes) to re-read unchanged files after an extractor change like this one.
- **"Scan folder now" no longer hides ingestion failures — it reports them.** A drop folder full of
  `.docx`/`.pdf` files with the `[documents]` extra missing (or any per-file extract error) reported a
  bare "Ingested 0," reading as "there are no documents." The scan now returns `failed` (each with its
  error message) and `unsupported` (wrong-type files it skipped), and the Docs tab surfaces both —
  failures in red (with the reason, e.g. the `pip install 'mimir-0[documents]'` instruction), skipped
  types in amber. A single bad file no longer aborts the scan (per-file isolation, DESIGN §10).
- **The Docs tab now shows the drop folder's resolved absolute path + whether it exists.** It used to
  print the bare configured value (e.g. `documents`), so a relative folder gave no hint of *where* it
  actually resolved — and if the server ran from a different working directory it silently watched the
  wrong place, reading as "not set." `/api/documents` now returns `folder_abs` + `folder_exists`, and
  the UI flags a missing folder in amber with a note that relative paths resolve against the server's
  working directory.
- **A "model not found" 404 now fails fast.** Ollama returns 404 both transiently (while
  loading/unloading) and permanently ("model not found, try pulling it first"); the latter was being
  retried 3× per node, burning time and spamming the log. A 404 whose body says "not found" is now
  classified non-transient. Plus a SETUP note: in endpoint mode the embed model must stay reachable
  on the fleet, the `[roles.embed]` tag must match what's installed, and keep one model per store.
- **Inner life stopped fixating on reference docs and its own output.** The idle loop picked the
  *most salient* memory to muse on, which was usually a high-salience DOCUMENT chunk (README, from the
  self-knowledge bake) or an INFERRED council verdict / prior musing — so it produced repetitive
  "technical components…"/"that timeout suggests…" loops. It now muses only on *stated beliefs*
  (shares `deliberation.NON_BELIEF_TIERS`, excluding DOCUMENT/INFERRED + archived) — the same fix as
  the council deliberation. (Verified against a live run that had been looping on exactly these.)
- **The Mind tab now shows inner-life thoughts.** There was no window onto them — a musing is kept
  out of the knowledge recall, so it was invisible. New "Inner life — thoughts while idle" panel
  (`Mimir.recent_thoughts`, surfaced via `/api/mind`).
- **Honest sleep-cycle log for forced runs.** A manual "Run sleep"/"Deliberate now" bypasses the
  window budget, which the phase-start log printed as a confusing `-1 min left` (it read like a
  negative budget). It now says `forced, no window limit`. The budgeting itself was correct — it
  re-reads the clock per phase and skips a phase that won't fit. `sleep_cycle._budget_label`.
- **The council stopped trying to chat with embedding models.** The embedding-model filter was a bare
  `"embed" in name` check everywhere (routing, `auto` role resolution, council roster), so embedding
  models without "embed" in the name — `all-minilm`, `bge`, `gte`, `mxbai` — slipped through and the
  council kept assigning one as a persona, 400-ing ("does not support chat") every deliberation.
  Centralized `model.provider.is_embedding_model()` (a broader name heuristic) and used it in the
  pool, gateway, and council.
- **Self-directed deliberation stopped "reconciling" reference material and its own musings.** The
  near-duplicate conflict scan included DOCUMENT chunks (overlapping README/DESIGN windows from the
  self-knowledge bake looked like near-duplicates) and INFERRED memories (inner-life musings, prior
  council verdicts) — so the council argued giant doc fragments against each other and looped on its
  own output. It now considers only *stated* beliefs (excludes DOCUMENT/INFERRED and archived rows).
- **Forum readability.** Long, multi-line, markdown-laden questions no longer render as one giant bold
  blob: the thread list shows a clipped single-line summary, and the thread view wraps the full
  question in a height-capped, scrollable block.

### Changed
- **Memory distillation: the store now fades and prunes instead of only accumulating.** A bridge-test
  audit found over-retention (aux-store rows piling up, the salience axis flat at 1.0, nothing ever
  archived). Three fixes, all in the consolidation pass and tier-aware so they never touch a
  primary-user fact:
  - **Salience decays faster for the decaying tiers** (conversation/inferred — peer chatter and
    self-generated rumination): a 10-day half-life vs 30 for authority/document facts, so low-value
    provisional content goes dormant in weeks instead of months (`PROVISIONAL_SALIENCE_HALF_LIFE_DAYS`).
  - **Archival is now tier-based, not confidence-gated.** A memory is archived when it has fallen below
    the salience floor *and* is a decaying tier — so stale conversational content goes dormant
    regardless of its stored confidence, while authority/document facts are still never archived for
    disuse (no death spiral). Archiving preserves confidence (DESIGN §3c).
  - **Sentinel notes are now pruned** like working-memory and self-model rows (keep the most recent
    `SENTINEL_NOTE_KEEP=10`); they're fetched by recency, so older ones were pure dead weight.
- **Inner-life musings start faint and don't repeat.** Self-generated reflections now bake at salience
  0.25 (was 0.6) so they fade and archive within weeks unless recall revives them, and a near-duplicate
  guard (`_is_duplicate_musing`) skips a musing that's near-identical to a recent one — so the idle
  loop distils rather than piling up verbatim repeats.
- **`Mimir.retier_speaker(name, tier)` — re-tier a speaker's baked memories.** Maintenance for when a
  speaker was ingested at the wrong trust level (e.g. a peer AI baked as `stated_by_primary_user`
  before `[identity] primary_user` was set): drops every memory with provenance `stated by <name>` to
  a lower tier (default `conversation` — attributed, not believed as fact). `repo.retier_by_provenance`.
- **A server-side trust policy for who gets believed (`[identity] trusted_users`).** The caller
  declares the speaker (`user`); the *config* decides how much that speaker is believed — never the
  caller — so an exposed API can't self-assert trust. `primary_user` → top tier (1.30);
  `trusted_users` (new) → trusted tier (1.20); any **other named speaker** (an unknown API caller, a
  peer AI, a guest) is now **attributed but baked at CONVERSATION tier, not as fact** (previously any
  named non-primary speaker was auto-trusted). Zero-config single-user is unchanged: with no policy
  set, the lone named speaker is the primary, so a build-your-own-UI "just works." This makes
  agent-to-agent safe (a peer's hallucinations don't enter as top-tier memory) without any bespoke
  wiring — Mimir stays a clean responding participant.
- **Consolidation prunes stale single-latest rows.** Working-memory and self-model rows accumulate
  one-per-synthesis, but only the latest of each is ever used (recency, limit 1) — pure dead weight on
  a long-lived deployment. The sleep cycle's consolidate pass now prunes them (keeps 2 working-memory,
  3 self-model), reported as `pruned` in the sleep report.
- **Working memory now keeps recent turns raw and folds only the oldest (true rolling compression).**
  Previously a fold summarized *all* buffered exchanges and **deleted every one**, so right after a
  compression the prompt had only the summary and no recent verbatim turns. Now, once
  `fold_threshold` (default 10) exchanges accumulate, it folds the **oldest** into the rolling summary
  (the prior summary folded in too — older material compressed harder each pass) and **keeps the most
  recent `keep_recent` (default 4) raw**. The trigger is **count-based** (fires when enough have
  built up) instead of a fixed turn cadence, and it runs off the hot path in the burst worker right
  after the reply streams — so the extra LLM call lands while you're composing, not in the reply
  latency. The summary is now a short couple of paragraphs (was three sentences). It stays
  cross-session and already feeds the daily narratives. Matches the home-AI's compress-keep-recent
  scheme. New config: `[working_memory] fold_threshold`, `keep_recent` (the old `refresh_every` is
  deprecated/back-compat only).
- **Timezone works without a package — UTC offsets + a cleaner host-local default.** Setting an IANA
  zone (e.g. `America/Vancouver`) needs a tz database, which bare Windows lacks — so Mimir now also
  accepts **UTC offsets** (`UTC`, `UTC-08:00`, `GMT+5:30`), resolved with pure stdlib arithmetic, no
  `tzdata`. The picker leads with offsets and a **"System local time (recommended)"** default, and the
  status line shows the actual offset in use (`system local time, UTC-07:00`) instead of an alarming
  "timezone not resolved" warning — it only notes the `tzdata` extra (or suggests an offset) when an
  IANA name genuinely can't resolve. For the common home case (the machine is in your timezone),
  leaving it blank is correct and silent. New `temporal.resolve_timezone()`.
- **The inner council fans across the whole fleet (DESIGN §5).** Council personas (user-convened or
  the sleep-cycle's self-deliberation) are now **pinned one-per-node** across every reachable machine
  instead of piling onto the best node for a model — so a deliberation lights up the entire fleet in
  parallel and finishes far faster. Each node runs a model it has, chosen greedily to be **distinct
  across nodes** where inventory allows (more minds, not just more copies); concurrency scales to the
  node count (cap 16, was 5). A pinned node that's gone or flaky falls back to ordinary routing, so no
  persona is lost. New: `ProviderPool.council_placements()`/`chat_on()`,
  `ModelGateway.council_placements()`/`chat_on_node()`; `Position` now records the node that argued
  it. Single-provider/local installs are unaffected (no placement info → prior model-routing).
- **A Sleep tab with live schedule + timezone settings (no config edit needed).** The sleep window,
  enable toggle, and timezone are now editable in the web UI (new **Sleep** tab) and take effect
  live — they persist as runtime overrides in the `kv` store, layered over the `[sleep]`/`[locale]`
  config defaults (config stays the headless default; the UI is the live preference). The scheduler
  reads the effective settings each tick, so a change applies without a restart. The interview ends
  with a quick "when are you usually asleep/away?" step that seeds the window + timezone. `Run sleep
  now` moved here too. New: `Mimir.settings()`/`update_settings()`/`available_timezones()`,
  `GET/POST /api/settings`, `GET /api/timezones`. **Timezone** drives all wall-clock reads (storage
  is already epoch-UTC; only display shifts); IANA zones need the OS tz database or the optional
  `tzdata` extra (`pip install mimir-0[timezone]`) — without it the core degrades to host-local **and
  says so** in the status line, never silently.
- **The chat UI wears the assistant's chosen name.** The onboarding "what would you like to call me?"
  answer (the `name` identity anchor) now drives the chat input placeholder ("Say something to …") and
  the speaker label on the assistant's bubbles, instead of a hard-coded "Mimir". Applied on load and
  re-applied the moment the name is set/revised (interview strip, Profile panel, or Identity tab).
- **Seeding interview expanded to 19 questions, with a Core-12 off-ramp.** Padded the get-to-know-you
  with the Core + Expanded items from `docs/mimir_foundational_interview.md` §8 — mission/purpose,
  values, voice, answer style, uncertainty handling, hard limits, memory policy, and local grounding
  (scope, conditions, vocabulary) + a catch-all. Seven now mirror identity anchors (name, operator,
  location, purpose, values, voice, boundaries), so the interview seeds more of the always-on
  self-model. After the **first 12 (the essentials)** the strip offers an **off-ramp** — *Finish here*
  or *Continue · 7 more* (good to keep going while the fleet benchmark runs); the Profile panel marks
  where the optional deeper questions begin. All skippable/editable.
- **Memory graph as a drifting galaxy.** The visual graph now lays out as a slow-rotating galaxy:
  importance (degree + salience + usage + how foundational a memory is) pulls the biggest, brightest
  blobs to the **centre** and loose ones to the rim. **The seeding-interview memories** (provenance
  `onboarding`) and operator-stated facts get the strongest pull — biggest, dead centre — so who you
  are and who's around you form the core. Added **scroll-to-zoom (toward the cursor), drag-to-pan, and
  double-click-reset**, a continuous gentle drift + rotation (slow enough to click comfortably), and a
  **white-hot→blue glow** palette (white cores at the centre, deep-blue at the rim, with halos) for a
  lightning-like look.
- **The web chat now streams.** The composer was calling the non-streaming `/api/turn` (the UI froze
  until the whole reply landed — painful on slow edge nodes); it now uses the existing SSE
  `/api/turn/stream` so tokens appear as they're generated, with a pulsing **"thinking…"** indicator
  until the first token so you can tell it's working.
- **Manual per-role model assignment + Fleet/Models tabs merged.** The web UI's **Fleet** tab now has
  a **Role assignment** control: a dropdown per role to leave it on **auto** (best-qualified, re-picked
  on rescan) or **pin** a specific model (honoured exactly, never substituted) — backed by
  `Mimir.set_role(role, model)` and `POST /api/fleet/role`. The separate **Models** tab is gone; its
  model pool (enable/disable, roles served/barred) folded into **Fleet**, which shares most of the same
  data. `/api/fleet/pool` now also returns `available` (live model names) so the dropdown has options
  before any benchmark.

### Fixed
- **Fleet robustness — a dead/slow node no longer wedges turns (and the API).** A node that timed out
  was retried 120 s × 3 *and* failed over to the next slow node, so one turn could take minutes and —
  since `_turn` holds the brain lock — hang the whole API. Now a **timeout is not retried** (the node
  took the full deadline; retrying just burns another) and the node is **cooled down** (skipped by
  routing for a window) on a single strike. New `ProviderError.timeout`, `pool._cooldown()`.
- **Disabling a model now actually stops routing to it.** Disabling a model in the pool used to affect
  only `auto` selection + recommendations, not live routing — so a role pinned (or config-set) to it
  kept using it. The veto is now pushed to the gateway (`set_disabled_models`): a role pointing at a
  disabled model **re-resolves to the best enabled model**, and disabled models drop out of fallback
  chains.
- **Manual role pins survive a restart.** A model picked for a role in the UI was in-memory only and
  reverted to the config default on restart. Pins (model + node) now persist in `kv` and are
  re-applied on boot (overriding config + auto).
- **Memory graph no longer "explodes" on first load.** The force sim had no distance floor on
  repulsion (near-coincident nodes — e.g. high-importance ones all pulled toward the tiny central
  ring — got astronomical kicks), no velocity clamp, and no position bound, so dots flew off-screen
  for ~a minute before settling. Now repulsion is floored, per-frame velocity is clamped, positions
  are hard-capped to the viewport, and the layout **pre-settles off-screen** so the first visible
  frame is already calm. (Verified bounded even when every node starts coincident.)
- **The web UI columns scroll independently; the chat header + input bar stay put.** The page used a
  brittle `height: calc(100vh - 50px)` and missing flex `min-height:0`, so the whole window scrolled
  as one. Now the app fills the viewport and never scrolls as a whole: the left column's **chat log
  (or graph/forum takeover view) scrolls on its own** between a pinned session bar and a pinned
  composer, and the **right column scrolls separately**.
- **Manual role override can now target a specific edge node — not just a model.** The role dropdown
  collapsed to one entry per model name (routed to its fastest node, usually the local beast), so you
  couldn't pin a role onto an edge box, and a model living on several nodes showed only once. The
  picker now lists **every `(node, model)` placement**, grouped per model (`gemma3:12b · .189 · 4.2s`,
  `… · .190 · 6.1s`, plus a `· any node` option), built from the per-node placement matrix. Choosing
  a node **pins the role to that machine** (`set_role(role, model, node)` → `chat_on_node`), preferred
  with fallback to routing if it's down — so you can keep inference on the edge and off the beast
  (pair with the node disable toggle for a hard exclude). Streaming honours the pin too. New:
  `ModelGateway.role_nodes()`, node arg on `set_role`/`set_role_model`, `pool.chat_stream(node=…)`.
- **Interview wording: the values question read backwards.** "What principles or values should guide
  how you act?" implied the *operator* acts; it's asking what should guide *Mimir* → "…how I act?"
- **The model greeted the operator by name every turn.** Each turn is sent as just `[system, user]`
  (no prior assistant messages), so a model reads it as a fresh start and opens with "Greetings, …".
  The default identity now instructs it to treat the session as one ongoing conversation — pick up
  where it left off, no greeting or name-restating unless greeted first. (A fuller fix — threading
  recent turns in as real chat messages — is a later improvement.)
- **A failed latency probe was recorded as 0.0s — making a timing-out model rank as the *fastest*.**
  `_measure_turn_latency` caught a probe failure (timeout/transport error) and returned `0.0`, the
  best possible sort key — so a model that aced the short capability probes but timed out on the
  longer latency generation showed `0.0s/turn`, won "fastest node," and sailed under any `max_latency_s`
  cap (observed live: `phuzzy/darknemo:latest` passing every dimension at 0.0s). It now returns
  **None** ("unmeasured"); every consumer already treats None as not-fast / not-viable
  (`return_time or 1e9`), the per-node speed is left unwritten so the matrix re-times it, and the
  board shows `·` instead of a bogus 0.0. The sibling `_measure_node_speed` got the same hardening so
  an *instant* transport failure (elapsed ≈ 0) can't be recorded as fastest either — a failed probe
  must never sort fast.
- **An inverted size band silently qualified nothing.** If the size fields were transposed
  (`min_model_size_b` > `max_model_size_b` — e.g. min 28, max 7), `min ≤ size ≤ max` is unsatisfiable,
  so the benchmark found **0 eligible models** and parked an empty, unexplained round that looked
  broken. An inverted band is always a transposition, so it's now **swapped, with a loud log**; and
  any empty round (whatever the cause) now says *why* in the UI ("nothing matched your size band
  min X / max Y — widen it") instead of rendering blank.
- **Per-node speed was clobbered model-wide, wrecking the placement matrix.** `update_catalogue_scores`
  wrote `return_time` with `WHERE model=?`, so the *one* node a model was tested on had its latency
  stamped onto **every** node's row — making a 12B look like 1.5s/turn on a Pi *and* the beast. Speed
  is per-node: it now lives solely in `update_catalogue_speed` (per `(node, model)`), and
  `update_catalogue_scores` writes `return_time` only when explicitly given (legacy/single-node
  callers), omitting it otherwise. The benchmark records each node's real speed and leaves unmeasured
  pairings blank — so the matrix is now honest. The Fleet tab's manual 1·2·3 buttons also light up as
  the tournament runs (scan → score → apply), instead of staying grey.

### Fixed (benchmark hangs)
- **The latency cap didn't actually bound the benchmark — a 7s cap could still hang for minutes.**
  Three compounding causes: (1) the pre-gate **warmup was untimed** (120s ceiling) *and* generated
  freely, so a thinking model (e.g. qwen3) could reason for the full 120s during the load, before the
  cap was ever checked; (2) the scoring **battery calls** ran on the pool's production 120s socket
  timeout, which the cap never touched; (3) the pool **retried** that 120s timeout up to 3× on the
  same slow node (~6 min on one model). Now: warmups load with a **single token** (`num_predict=1`)
  so they can't reason; scoring calls carry a **tight per-call timeout** (~2× the latency budget,
  45–90s) and run with **no pool retries**, so a slow/wedged model fails fast and the round continues
  (it still fails *over* to another node that has the model — it just doesn't retry the slow one).
  New plumbing: a reserved `__timeout_s__` param the Ollama provider honours per call, and a
  `max_retries` override on the pool/gateway.
- **The latency cap must never cut *capability* — a model slow on one weak node may be excellent on
  another.** Capability is per-model (test it once, anywhere); latency is per-(model, node) and only
  governs routing. The benchmark no longer skips a model because it exceeds the user's cap; it only
  skips a node too slow to *test* on within a generous budget (`max(30s, cap)` per turn), records the
  per-node speed for routing, and the cap is applied at finals/routing — not as a quality filter.
  (This reverses an earlier "cap skips early" choice. Per-node *requeue* — try a faster node before
  giving up — and concurrent distribution across nodes are the next build.)
- **The latency cap's pre-gate measured the wrong thing, so it never skipped slow models.** The gate
  timed `"reply ok"` — one token, instant for anything — so a model that's snappy on a token but
  takes ~13s on a real turn sailed through a 7s cap, then crawled through the ~15-call battery
  (~160s total, none of it skipped). The pre-gate now times a **representative ~64-token generation**
  and normalizes it to **seconds per ~256-token turn** (the cap's actual units), so a 13s/turn model
  is skipped *before* the battery under a 7s cap. Per-node speed is now stored in the same normalized
  units, so routing's fastest-node pick reflects real per-turn latency.
- **A page refresh lost the whole tournament/benchmark view.** The resume logic only ran on the
  Fleet-tab click, so a fresh load never reconnected — the run kept going server-side but the UI
  forgot it. It now reattaches on page load (and tab-open), and shows a per-model elapsed timer so a
  slow model reads as grinding, not hung.

### Added
- **The live inner life — thinking in the long quiet (DESIGN §5a).** A new idle loop
  (`cognition/inner_life.py`) reclaims the time *between* conversations (the burst worker only
  reclaims the few seconds after a reply). On a slow, user-tunable cadence (default one thought every
  ~5 min) a daemon picks ONE universal stimulus — a recent error, an un-deliberated conflict, the most
  salient memory, the working-memory thread — and composes a brief first-person reflection with a
  cheap background model. The thought is stored as a low-confidence, decaying memory
  (`provenance="inner life"`, `INFERRED`, confidence 0.3) that **earns its way** back via ordinary
  recall — never force-injected. Built to two hard rules: **chat priority** (routes *off* the chat
  model, yields the instant a turn starts via `should_think`, holds a post-turn idle floor, long
  cadence) and **edge cost** (one model call per cycle, paused when the fleet is down, **off by
  default**). On/off + cadence are live in the Sleep tab and `[inner_life]` config; a "Think now"
  button (`POST /api/inner_life/run`) forces one cycle. This is the first step of idle-takeover
  continuous mode; the deep-idle two-voice dialogue is still to come. `Mimir.run_inner_life_tick`.
- **Bidirectional (output-triggered) RAG (DESIGN §5a).** Retrieval no longer only fires on the
  user's input — after the model replies, a burst task retrieves memory relevant to **its own reply**
  and surfaces it into the **next turn's** context, so a thread the model itself opened gets grounded
  (not just what the user asked). Off the hot path (one embed + local retrieval in the idle window);
  excludes the facts just baked from that reply (no echo); top-K configurable. `[output_rag] enabled`
  (default on), `top_k`. New: `Mimir._output_rag()`.
- **Integration API + a security layer — a brain with endpoints, no built-in hands (DESIGN §8).**
  Mimir 0 deliberately ships no IO of its own; instead the HTTP surface is now a documented,
  optionally-authenticated **integration API** so you build your own (voice, avatar, Home Assistant,
  social, agent frameworks — or a middle layer where **two Mimirs talk to each other**). Set
  `[server] api_token` (or env `MIMIR_API_TOKEN`, which wins) and every `/api/*` route requires
  `Authorization: Bearer <token>` (constant-time checked); unset = open localhost as before. The page
  shell stays open so the bundled UI prompts for the token once and stores it. `[server] cors_origins`
  enables browser frontends on other origins (with an `OPTIONS` preflight). The `user` field on
  `POST /api/turn` is the **speaker identity** — the seam for agent-to-agent. The token's env var
  name is configurable (`[server] api_token_env`, default `MIMIR_API_TOKEN`) so two instances on one
  machine don't collide. **The local browser UI is token-exempt by default** (a same-machine request
  skips the token, so a fresh run is never blocked by a token wall — the token still guards
  remote/integration callers); `[server] secure_ui = true` requires it for the local UI too, and
  `GET /api/health` is always exempt. Full contract in [`docs/API.md`](docs/API.md). New config:
  `[server] api_token`, `api_token_env`, `cors_origins`, `secure_ui`.
- **Self-knowledge — the system bakes its own README into memory so it knows what it is.** A
  `self_knowledge` phase in the sleep cycle ingests a configured doc (default `README.md`) through the
  document pipeline into recallable, `DOCUMENT`-tier memory tagged with its source — so "what are you
  / how do you work?" is answered from its own docs (and the self-model, which reads the store, draws
  on it). Content-hashed, so it re-embeds only when the doc changes; "Run sleep now" bakes it on
  demand; fail-soft if the file's missing. Config: `[self_knowledge] doc` (empty disables). New:
  `Mimir.bake_self_knowledge()`.
- **Self-observability — the system sees its own recent errors (DESIGN §10).** The doctrine was
  fail-*loud* (every downgrade logged); now it's also fail-*aware*. A bounded ring
  (`diagnostics.py`) captures `WARNING`+ off the `mimir` logger, and two surfaces use it: (1) a
  **system-health section in the turn's context** lists recent errors (within a window, capped) so the
  model knows when it's degraded and can say "I've had an error" instead of carrying on oblivious;
  (2) a **`health` phase in the sleep cycle** digests the period's errors (counts + samples) to `kv`
  so the nightly cycle reviews what went wrong and it survives a restart. The context block also
  carries **backend pool health + per-node speeds** (nodes up/down, saturation, s/turn) when the
  fleet is degraded — so the model knows "node …189 is down, …190 at 6.1s" — and the Mind tab shows
  it live (`Mimir.pool_health()`, `get_stats()` now lists down nodes).
  Config: `[diagnostics] surface_errors` (default on), `error_context_window_s` (30 min),
  `error_context_max`. New: `Mimir.recent_errors()`/`digest_errors()`/`health_digest()`.
- **The council forum — deliberations you can read, comment on, and keep house in (DESIGN §5a).** A
  `🏛 Forum` toggle swaps the chat panel for a forum (like the graph view): every deliberation —
  whether you convened it, asked from the forum, or the sleep cycle self-initiated it — now persists
  as a **thread** with one **post per persona** (tagged with the node + model that argued it, so the
  fleet fan-out is visible), the synthesized **verdict**, and your **comments**. Full-admin
  housekeeping: comment, close/reopen, delete a post, delete a thread; an "Ask the council" box seeds
  a fresh (fleet-distributed) deliberation. Comments are annotations — they don't feed back into the
  reasoning. New tables `forum_threads`/`forum_posts` (schema v20), forum repo CRUD, `Mimir.forum_*`,
  `GET /api/forum` + `/api/forum/thread`, `POST /api/forum` (ask/comment/close/reopen/delete).
- **Self-directed adversarial reasoning in the sleep cycle (DESIGN §5a).** The inner council is no
  longer only something you invoke by hand — during sleep the system **surfaces its own conflicts and
  argues them**. A new `deliberate` phase (after consolidate) deterministically surfaces tensions
  consolidation doesn't resolve — *graph tensions* (a subject with two+ objects under the same
  **non-functional** relation, e.g. "wants X" vs "wants Y") and *divergent near-duplicates* (memory
  pairs in a cosine tension band whose text differs) — a **hybrid curator** picks the few most worth
  arguing (an LLM ranks; deterministic weight order is the fallback), and each goes to the council;
  verdicts are stored as recallable understanding (`provenance="sleep deliberation"`). Conflicts
  argued in the last 30 days are skipped so it doesn't loop nightly. Toggle in the Sleep tab
  (`[deliberation] enabled`, default on) + a **"Deliberate now"** manual trigger
  (`POST /api/deliberate/run`). New: `cognition/deliberation.py`, `Mimir.deliberate_open_questions()`.
- **A wall-clock sleep cycle — heavy maintenance gets its own quiet window (DESIGN §5a).** The
  post-turn *burst* worker assumes the model idles while you read the reply, but with **streaming**
  chat the model is busy to the last token and on a **slow machine** one turn eats the window — so
  consolidation (dedup/decay/archive/contradiction hygiene) + narrative roll-ups now run in a
  user-set nightly window instead of fighting for scraps. A `[sleep]` block sets `enabled`
  (default **on**), `window_start`/`window_end` (cross-midnight aware), and `check_interval_s`. A
  daemon checks the clock; inside the window — and not already done today, and not mid-turn — it runs
  the phases **in order, skipping any that won't fit the time left**, checkpointed per-day so a
  same-night restart resumes and it never runs twice a day, with **catch-up before noon** if the
  window was missed (powered-off/restarted host). Manual "Run sleep now" any time (Mind tab → forces
  the full cycle and stamps the day). New: `cognition/sleep_cycle.py`, a generic `kv` table
  (schema v19) for the checkpoint, `Mimir.run_sleep_cycle()`/`sleep_cycle_status()`, and
  `GET /api/sleep/status`. The window is the new primary path; the old turn-cadence `[sleep] every`
  stays (default off) for back-compat.
- **Offline encyclopedia as a live reference layer (Kiwix/ZIM, DESIGN §9) — zero new dependency.**
  Point a `[wiki]` config block at a running `kiwix-serve` over any ZIM (Wikipedia nopic, a medical
  wiki, a top-50k slice, …) and each turn's query is searched live; the top articles' lead text is
  injected as an attributed `Reference — … (Wikipedia)` section that counts as grounding. Mimir talks
  to Kiwix over **stdlib HTTP** (no `libzim`, no compiled wheel, no GPL in the tree — the user runs a
  static binary, like Ollama), so the whole encyclopedia "populates" the knowledge layer at no ingest
  cost. Fully optional and **fail-open** (a missing/slow server yields no section, never an error or
  stall); trivial turns skip the lookup. `cognition/wiki.py` (`WikiSource`, stdlib `urllib` +
  `html.parser`), `WikiConfig`, `build_context(wiki_context=…)`. A `GET /api/wiki/status` reachability
  check (`Mimir.wiki_status()`) surfaces as a live status line in the web UI's **Docs** tab.
- **Empty recall is stated, not silent (DESIGN §3d).** When memory recall comes up empty,
  `build_context()` now renders the knowledge section anyway — "No stored memory is relevant to this…
  say you have no memory of it, do not guess" — so the model answers "I don't have any memory of this"
  instead of confabulating. It still counts as zero grounding, so the uncertainty gate also fires.
- **Visual memory graph — switch the chat to a relational map (DESIGN §3a).** A **🕸 Graph** toggle
  above the chat swaps the conversation for a force-directed map of **memory "blobs"** (sized by
  salience, coloured by evidence tier) and the **entities** from the triple graph, linked by relation
  edges and a light "mentions" edge from a memory to entities it names. Click a blob to **review and
  edit** it — text + salience — or delete it; click an entity to see its connections. Pure vanilla SVG
  + a small force simulation (no deps). New: `cognition.build_graph_map`, `repo.update_memory`,
  `Mimir.{graph_map,edit_memory,forget_memory}`, `GET /api/graph/map`, `POST /api/memory`
  (update/delete). Borrows the home AI's graph-orb concept, stripped to the relational essentials.
- **Session history + restore.** A durable conversation log (schema v16 `conversation`, one row per
  exchange, pruned to a rolling window) — the lasting full turn history, distinct from the capped
  EXCHANGE recency buffer (cleared on compression) and `interactions` (timestamps only). It does
  three things: (1) the web UI **restores the conversation on load** (`GET /api/history` →
  `restoreHistory()`), so a refresh/restart no longer wipes the chat; (2) recent turns are **replayed
  to the model as real `user`/`assistant` messages** (`_history_messages`), giving genuine continuity
  instead of summary-only context — the root-cause fix for the "fresh start" greeting and a coherence
  win on small models; (3) it survives a process restart (a new `Mimir` on the same DB has the
  history). `Mimir.history()`, `repo.{record_conversation_turn,recent_conversation}`. A reasonable,
  single-stream approximation of the home AI's session system.
- **Conversations as selectable sessions (home-AI style).** Turns are grouped into **sessions** (a
  new one on an explicit "+ New" or a long idle gap; schema v17 `conversation.session_id`). A
  **dropdown above the chat** lists past conversations, each with a one-line summary (its first
  message) + turn count + date; **Restore** loads a conversation back into the chat and continues it,
  **+ New** starts a fresh one. Replayed model context is scoped to the active session, so a new
  conversation starts clean and resuming an old one replays its tail. `Mimir.{sessions,
  start_new_session, resume_session}`, `GET /api/sessions`, `POST /api/session`,
  `GET /api/history?session=…`. (Replaces the flat History tab.)
- **The burst worker — post-response cognition as a scheduled idle-window pool (brain slice 3;
  DESIGN §5a).** `cognition/burst.py` is the generic scheduler extracted from the home AI, stripped to
  a universal skeleton: **pent-up-demand priority** (`effective = base − starved_s × rate`, so starved
  work floats up), **two task classes** (user-driven runs continuously; autonomous is slot-capped),
  **interruptibility** (an injected `is_busy` predicate — foreground always wins), and **surfaces**
  (a task can emit a note injected into the next reply as a `[…follow-up thinking…]` section). The
  brain now routes all post-response work — sentinel, self-model, working memory, sleep/narratives —
  through one worker instead of four independent threads; `turn()`/`turn_stream()` *signal* it after
  the reply and the next turn *settles* it (so the note/identity is ready), preserving the
  acceptance-loop contract. Pure, per-instance state + injected clock → deterministic
  (`signal()`+`drain_once()` tested without threads). `build_context(background_notes=…)`,
  `Mimir.wait_for_sentinel()` now waits on the worker; removed the ad-hoc `_spawn_sentinel`/`_maybe_*`
  plumbing. +9 tests.
- **Temporal narratives — a hierarchical, lossy-by-design journal (brain slice 2; DESIGN §3a/§3e).**
  The system now has a sense of *what happened* over time: `cognition/narratives.py` writes a
  first-person **daily** entry, compresses dailies older than 3 days into a **weekly** summary, and
  weeklies beyond the 5 newest into a **monthly** narrative — each tier lossier than the last (details
  fade, patterns persist). Generated off the hot path in the consolidation/sleep pass from generic
  sources only (the running summary + recent exchanges + facts learned that period — no integrations),
  idempotent per period, retained per tier (10 / 5 / 13). The recent entries are injected as a
  `[Recent history:]` prompt section (coarsest first), so a turn weeks later still has the shape of
  what came before. New: schema v15 `narratives`, `repo.{save_narrative,list_narratives,get_narrative,
  prune_narratives}`, `Mimir.generate_narratives()`, `build_context(recent_history=…)`; runs in
  `Mimir.sleep()` and the periodic sleep task.
- **Temporal grounding — the brain's clock/calendar sense (DESIGN §3e).** First slice of the
  thinking-components extraction from the home AI, stripped to a universal skeleton (no integrations).
  `cognition/temporal.py` injects a compact "It is Thursday, January 15 2026, 2:30 PM. Season: winter
  (spring in 64 days)." line into every prompt, and every recalled fact now carries a relative-age tag
  (`…; 3 days ago`) so the model reasons about recency instead of guessing. Explicit time/date/season
  questions are answered by a **deterministic intercept** (`Mimir.maybe_time_answer`, used
  automatically by `turn()`) with zero model cost. Timezone + hemisphere are `[locale]` config
  (default: host zone, northern seasons) — universal, no place baked into core. Pure functions
  (the moment is passed in), so it's deterministic and fully tested. `build_context()` gains
  `time_context` and `now_ts`.
- **Temporal-awareness baselines — the system feels time passing (DESIGN §3e).** A durable
  interaction log (one timestamp per turn, schema v14 `interactions`, pruned to a rolling window)
  powers a deterministic "you've been away longer than usual" note: pure statistics over the user's
  own gap distribution (median/p90/longest), surfaced as a `[Temporal awareness:]` prompt section
  only when the current gap is genuinely notable — silent within normal rhythm, zero model cost.
  `cognition/temporal.gap_insight`, `repo.{record_interaction,interaction_history}`,
  `build_context(temporal_awareness=…)`. The same baseline machinery extends to entity/topic
  staleness later.
- **The seeding interview (Phase 1) — the operator's first, highest-provenance facts.** A short
  get-to-know-you — what to call the assistant, who the operator is and what they do, their week,
  location, household, pets, interests — paired with the qualifying tournament and **re-runnable any
  time**. Each answer is **captured model-free and persisted immediately** (crash-safe; works before
  any chat model qualifies) as a `stated_by_primary_user` (top tier, 1.30×), `provenance="onboarding"`
  memory — the orientation everything else builds on. The facts **live in one place** (one editable
  row per question, keyed by `meta.onboarding_key`; re-answering updates in place) surfaced in a new
  **Profile** tab; name/operator/location also mirror into the identity anchors that inject into the
  self-model. UI: when the tournament board is up it now takes ~85% of the chat pane and a
  one-question-at-a-time **interview strip** sits below; a first-run prompt invites it. New:
  `cognition/onboarding.py`, `Mimir.{onboarding_profile,pending_onboarding,record_onboarding_answer}`,
  `repo.delete_memory`, `GET /api/onboarding` + `POST /api/onboarding/answer`. The LLM parse pass
  (one answer → several typed facts + graph triples, review-before-commit) is Phase 2. See
  `docs/mimir_foundational_interview.md`.
- **Speed- and health-aware routing — live node latency, learned from real traffic.** The provider
  pool now routes a call to the node with the lowest **expected wait** (`latency × current load`),
  not just the least-loaded one. Node speed is measured **passively on every real call** (no wasted
  synthetic calls) and folded into a per-`(node, model)` EWMA in the benchmark's own
  "seconds-per-~256-token-turn" unit; a rare **idle heartbeat** (`[backend] idle_probe_interval_s`,
  default 30 min, decoupled from the faster health refresh) tops up nodes that have gone quiet.
  Estimates **seed** from the catalogue's qualification `return_time` (routing is informed from turn
  one) and are **written back** so the placement matrix/leaderboard show live, current speed. A failed
  probe is recorded as *unmeasured*, never as fast. New: `model/latency.py` (the pure EWMA +
  verbosity-independent normalizer, shared with the benchmark), `ProviderPool.{seed_latency,
  latency_snapshot,idle_nodes,probe_latency,known_models}`, `Mimir.node_health()`, and
  `[backend] latency_alpha`.
- **Ranked fallback per role — a heterogeneous fleet still serves every role.** A role now resolves to
  an **ordered chain of acceptable models** (its qualified ranking, best first), not a single model.
  The gateway walks the chain: each model routes to its fastest healthy node; if every node for a
  model is down, routing falls to the next acceptable model — so a fleet where node A has only Gemma
  and node B only Qwen still serves `chat` (Gemma@A → Qwen@B). The chain is pruned to currently
  reachable models, and a **pinned** model is never substituted. `ModelGateway.set_role_fallbacks` /
  `fallbacks_view`; the brain derives each chain from `roster_for` on every (re)resolve.
- **The bridge into the brain harness — `background`/`council` roles + a roster query.** The second
  lineup is now first-class in the role gate: `background` (off-the-record reasoning) and `council`
  (adversarial pool) are wired into `ROLE_NEEDS` as **loose** roles — a reasoning-competence floor
  only, deliberately **not** discipline/epistemics-gated, so the big/undisciplined models the voice
  can't use are available to cognition that never speaks *as* the assistant. The harness staffs itself
  by **querying the roster** instead of a human reading a view: `Mimir.roster_for(role, n)` ("give me
  N models for role R"), plus `background_model()` and `council_members(n)`, all honouring the same
  model/node vetoes as every other pick. `council` routes to the diversity picker (families before
  depth); single-best roles return the top-N. The single-best path (`recommend_roles`) and the pooled
  path now share **one** ranking over **one** gate (`_bar_reason`), so the seated roster can never
  disagree with the eligibility the leaderboard renders (DESIGN §5a, §10).
- **The council-pool grading pass — qualify the big models caps-off.** The second lineup's selection
  is caps-off, but the big models (≥ the chat size cap — the 30–36B coders, the 122B MoE) were never
  *graded*: the benchmark's size cap skipped them, so they never entered the catalogue with scores and
  couldn't be drawn into the council. **🏋️ Qualify big models** (on the council view) grades exactly
  those — models above the chat cap, with the caps **off** (no upper size limit, no latency gate) and
  the full gauntlet (so their quality is comparable to the main pool). Crucially it grades them **in
  place — no rescan** (the `complete_speed_matrix` discipline), so the main pool's hard-won scores
  survive; run a tournament first, then this fills in the big council models without touching the rest.
  `/api/fleet/benchmark/council` → `benchmark_council_pool()`, reusing the benchmark board/progress.
- **The second lineup — a diversity-first adversarial council roster.** Adversarial / council /
  background reasoning gets its value from **family diversity** (different model families fail in
  different ways — a council of five qwen variants is worth far less than five different families),
  not from the top-N ranking. `council_roster()` ranks models *within* each family, then round-robins
  *across* families — each family's best first — so a 5-seat council pulls from 5 distinct families
  before it ever doubles up. It's **capacity-bound, not latency-gated**: any model clearing the
  quality floor on an enabled node qualifies, big-and-slow included (those are prime council members
  a chat cap would wrongly exclude). New **🏟️ Council roster** view (tournament-done panel + benchmark
  header) with a 3/5/7-seat toggle, the families represented, and the bench (qualified but not seated).
  `/api/fleet/council` → `council_roster()`. (The big ≥30B/122B models still need grading with the
  size cap off to enter the pool — benchmark with max size = 0.)
- **A per-node placement matrix — every model on every node it runs on, with each node's winner.**
  The results board shows each model once, under the single node it was capability-tested on — so a
  strong *multi-node* worker (e.g. a mid-size model installed across the LAN) was collapsed to one
  row and effectively invisible as a per-node worker, even though the speed-test had timed it on
  every node. The new **📊 Per-node placement** view (on the tournament-done panel and the
  benchmark-complete header) groups by node and lists every model installed there with **that node's**
  speed, the capability scores, and its role eligibility — and crowns each node's **🏆 winner** (best
  quality, this-node speed breaking ties) and **⚡ fastest**. Reads the live catalogue
  (`/api/fleet/placement` → `placement_matrix()`), so it reflects exactly what the speed-test
  measured. This is the display side of the background-worker roster.
- **A "what these scores mean" banner above the leaderboard.** A one-line, expandable note at the top
  of the Fleet and Models tabs: Mimir ranks models by *operational fitness for its own roles on your
  hardware* — best for **this system as built**, under this battery, on your fleet — **not** "best
  model overall." Collapsed by default (one line); expands to the full framing (model-agnostic; speed
  is per-node and shifts with load so routing re-selects live; coherence is experimental; a narrow win
  isn't a landslide). Defuses the "users overfit to the spectacle / read scores as universal truth"
  risk without dulling the tournament. Zero-dep `<details>`/`<summary>`.
- **The model pool explains its verdict — no silent role bars.** `recommend_roles` quietly dropped any
  model that missed a capability floor; you saw the winners but never *why* a model wasn't one. The
  pool now shows, per model, the roles it **qualifies for** (✓) alongside the ones it's **barred from**
  with the reason — e.g. "⊘ chat: discipline 0.00 < 0.50" — never a silent drop (DESIGN §10). The bar
  reason comes from a single shared gate (`_bar_reason`) that `recommend_roles` *also* uses to pick
  winners, so the board's explanation can never contradict the actual decision. (`fleet.py`,
  model-pool tab.)
- **The long-context probe now scales with the window you qualify at, and the default is the real
  operational window.** It used to plant the needle in a fixed ~2k-token haystack — enough to clear
  Ollama's 2048 default but no real test of the window a deployment actually runs. The haystack now
  **sizes to `benchmark_num_ctx`** (~60% of it), built from a wide pool of invented, coherent,
  public-clean filler sentences (no gibberish — a model shouldn't treat it specially — and nothing
  proprietary), with the needle in the **middle** (the "lost in the middle" worst case). The default
  `benchmark_num_ctx` is now **24576 (24k)** — qualify at the size you deploy at, because a different
  window in production rebuilds the warm KV cache on the first real turn and makes the benchmark's
  latencies lies. 24k is a proven window for a RAG + compression system: a fraction of models'
  128k–256k theoretical max, which you neither need nor want (KV-cache cost + attention degradation).
  Continuity comes from curated RAG memory and compression, not a giant raw window. Lower it only if
  your edge nodes can't physically hold 24k — and then it's a *placement* fact (which node serves
  which model at the window), not a capability cut.
- **The final time trial — per-node placement matrix.** A new Fleet-tab action ("⏱ Speed-test
  remaining nodes") and `brain.complete_speed_matrix()` that, after qualification, speed-tests each
  **acceptable** model (quality ≥ a floor) on every enabled node it's *installed on but wasn't timed
  on* — so the catalogue learns *which edge can run what, how fast*. Slow results are **recorded, not
  dropped** (a slow `(model, node)` is still a real backend resource for capacity-bound council/
  background work); the probe timeout is generous so a slow-but-real number is captured. Reads the
  existing catalogue (no rescan — preserves the quality scores), runs concurrently across nodes, and
  the per-node times fill into the fleet view. This is the input to the background-worker roster.
- **Concurrent, distributed qualification.** The benchmark now runs **one worker per enabled node**
  (the worker is the node's VRAM lock — one model at a time), with each model's candidate nodes
  **rotated** so different models start on different boxes (spread → real parallelism). Each model
  **falls back across its nodes** — pick the fastest that's quick enough to test on, skip a node
  that's down, and if none is quick, test on the fastest that *ran* — so **capability is never
  failed for being slow** (a model great on the beast but slow on a Pi still qualifies). Per-node
  speed is recorded as it goes (the start of the placement matrix). `benchmark_model(node=…)` pins a
  model's whole battery to one warm node (direct provider, no pool thrash). Mock/single-local stays
  **sequential and order-deterministic**, so concurrency engages only on a real multi-node fleet.
- **Per-node veto (schema v13).** Each discovered edge node can be toggled off in the Fleet tab —
  excluded from the pool's routing (with a fail-safe if *every* node is vetoed, so chat never
  hard-blocks), from qualification, and from recommendations, even if it's reachable. "Don't use that
  box, even though it's there." Mirrors the per-model enable/disable; `node_prefs` table + a
  `/api/fleet/node` endpoint.
- **The qualifying tournament — a staged, human-veto model knock-out** (Fleet tab → "🏆 Run
  qualifying tournament"). Built on the benchmark's new staging primitives (subset / triage /
  ephemeral): **Round 0 · Qualifying** scores the *whole* fleet fast and cheap (capabilities only,
  nothing saved) → you **untick** who shouldn't advance → **🥊 FIGHT → Round 1 · Gauntlet** re-tests
  only the survivors through the full framework qualification (reasoning + the epistemic layered/
  grounding/long-context probes, scores saved) → veto again → **Round 2 · Finals** champions each
  role *among your finalists only* (the veto wins over the global best), then Apply. Round 3 (vision)
  is reserved.
  The board takes over the chat pane; rounds run in the background with live progress and resume on
  tab-switch. New endpoints under `/api/fleet/tournament/{start,advance,apply,status}`.
- **A `reasoning` dimension in the benchmark (schema v12).** The old battery only tested *format*
  compliance (say `PONG`, return a weather JSON, write `def add`) — every competent model passed, so
  `quality` saturated near 1.0 and a fluent model that *couldn't actually solve anything* could sweep
  every role. The new dimension scores deterministic, regex-checkable **problems** (multi-step
  arithmetic, letter-counting, sequence completion, a code-trace, an instruction transform) — wrong
  answers fail however fluent the prose. The `chat`/`reasoning`/`code` roles now **gate** on it.
- **A chat-LLM epistemic qualifying round.** `score_epistemic_competence` now includes a big
  **layered, conflicting-tier gauntlet** (high-evidence section says "blue", a lower section says
  "red", buried in irrelevant filler → defer to the high tier under noise — the structured arm can,
  a flat blob can't), a **grounding floor** (recall a nonce that exists *only* in the provided
  context), and a **long-context needle** (a nonce planted in the middle of a ~2k+-token document —
  past Ollama's 2048 default, so it doubles as proof the `num_ctx` pin works). A model that can't
  follow a layered prompt, won't read context, or can't handle long input is barred from `chat`.
- **`backend.min_model_size_b` (a size floor; 0 = off).** The sibling of `max_model_size_b`: on
  capable hardware, an imperfect test lets a tiny model that scores "high enough" and wins on latency
  keep beating a bigger, genuinely-better one a second behind at the same score. The floor excludes
  models below it from scoring (and therefore from recommendations). Exposed as a UI scope field.
- **`backend.benchmark_num_ctx` (default 8192).** Ollama defaults `num_ctx` to a tiny 2048 unless
  told otherwise; the benchmark now pins an explicit, consistent context for every model so the
  layered prompts aren't silently truncated (which would cut off the high-tier fact). Raise to test
  longer context. The pre-gate warmup uses the same value, so a model loads once and stays warm.

### Changed
- **Benchmark latency now reflects a real turn, not a 3-token reply.** `return_time` was the mean
  wall-time of the battery's tiny calls, which can't tell a slow remote 12B from a snappy local 3B —
  so a big model looked "instant" and won even speed-weighted roles. It's now measured from one
  real-length generation, normalized to seconds per ~256-token turn.
- **Routing objective made explicit: the best-scoring model *for this system* that you're willing to
  wait for.** Latency is now a hard **cap** (`max_latency_s` excludes too-slow models before
  scoring), not a soft penalty. The `chat` role dropped its quality-minus-speed "balanced" formula
  for pure **quality-under-the-cap** (every role now ranks this way): within the cap a dominant model
  wins outright and speed only breaks ties — so a 26B that's a second behind a 4B at a higher score
  wins, because under the cap you've already decided the wait is worth it.
- **The benchmark scorecard is grouped by node/IP**, so it's obvious which machine each model runs
  on (LAN-only leftovers cluster under their IP instead of looking local), and gained a `Reason`
  column.
- **The Fleet tab leads with the tournament** as the recommended path; the manual
  Find / Benchmark / Apply buttons move below as the one-step-at-a-time equivalents.
- **Docs synced** — INFERENCE_ENGINE, DESIGN §4, README, and SETUP now describe the `reasoning`
  dimension, the epistemic gauntlet, real-turn latency, the size floor, `benchmark_num_ctx`, the
  quality-under-cap routing objective, and the qualifying tournament.

### Fixed
- **Benchmark timed cold model-loads, not warm performance.** With models swapping in and out of
  VRAM, every measurement included a one-time load cost — which inflated `return_time` *and* unfairly
  tripped the latency gate (a 26B that's fast warm but slow to load could be wrongly skipped). Now
  each model is **loaded with an untimed warmup call first**, then timed *warm* — for the latency
  probe/gate and the capability battery. A model that can't load within a 120s window is reported as
  unusably slow and skipped. Measurements now reflect steady-state, the way the model actually runs.
- **Thinking mode was never controlled (and couldn't be turned off).** All role params went into
  Ollama's `options`, but `think` is a *top-level* field — so thinking models thought by default
  (slow) and a `think` set in config was silently ignored. Now `think` defaults **off** (it slows
  generation and rarely improves output, per testing across models) and is sent top-level; opt in
  per role with `think = true` only where it helps (e.g. some models on tool selection). `think=false`
  is accepted by non-thinking models too, so it's safe everywhere.
- **Benchmark only ever tested the 8 smallest models — so a 4B "won" every role while the user's
  much better 26B model was never benchmarked at all.** The default capped the run at the 8 smallest
  approved models ≤30B, smallest-first, so mid/large models (gemma3:12b, gemma4:26b, …) were silently
  excluded and the recommendations were dominated by tiny models. Now the benchmark covers **all**
  approved models up to a **user-set size cap** (`[backend] max_model_size_b`, default 30B — only the
  user knows their hardware), and reports coverage ("benchmarked N of M; K skipped as too large") in
  the UI and logs. **Per-model latency timeout:** before the expensive battery, each model gets a
  trivial-prompt probe; one that exceeds the budget (`max_latency_s` if set, else a 30s default) is
  **skipped** instead of stalling the whole run — so a slow big model can no longer hang the
  benchmark (or hold the lock so a second run "doesn't work"). The UI now also shows the **scan
  phase** ("scanning the fleet…") instead of a blank "0/0", and reports too-slow skips.
  **Live scoreboard:** each model's scores stream into the Fleet area *as it finishes* (best-first
  table: quality + all dimensions + speed) — so the otherwise-idle UI fills with useful results
  during the run instead of just a counter (`benchmark_fleet` gained an `on_result` callback).
  **Scope fields on the Fleet tab:** "Max model size (B)" and "Max latency (s)" inputs (pre-filled
  from config) override the cap/latency for a run — no `mimir.toml` editing needed to control what
  the benchmark tests.
- **Benchmark looked frozen / gave no progress.** The fleet benchmark ran synchronously while
  holding the brain lock, so the entire web UI (including header polling) blocked for the multi-minute
  run, and nothing logged per model — it was impossible to tell a running benchmark from a broken one.
  Now: the run happens in a background thread, `/api/state` and a new `/api/fleet/benchmark/status`
  are lock-free so the page stays responsive, the UI polls and shows **"Benchmarking i/N: model…"**,
  and `benchmark_fleet` logs every model start/finish (`[i/N] model …`) so the log/console show life.
  Errors surface in the status instead of dying silently.
- **Identity drift in the self-model.** A small model (`gemma3:4b`) synthesizing the self-model could
  hallucinate a name not in the operator-established anchors (observed: anchor name `Mimir` but the
  synthesis wrote "I am Arthur"), creating a contradiction the chat model then adopted and inverted
  ("you serve Greg"). The synthesizer is now forbidden from stating or inventing the name, operator,
  or location — those are the verbatim anchors' job — and the identity section is framed as
  authoritative. (DESIGN §3e.)
- **Internal epistemic tags leaking into replies.** Small models absorbed the `[tier=…; source=…]`
  provenance style from the prompt and emitted it on their own sentences (even inventing
  `[tier=question]` / `[tier=focus]`). These are now stripped deterministically by `mimir.sanitize`,
  with a streaming-safe stripper so a tag split across stream deltas is still removed and no double
  space is left behind — applied to both the live SSE display and the stored exchange. (DESIGN §3b,
  §10.)
- **Boot no longer blocks on fleet inventory.** Initial LAN node discovery/inventory now runs in a
  background thread, so the web server starts listening immediately (~2s) instead of waiting on a
  full multi-node scan; a "Starting Mimir…" line prints at once.
- **Uncertainty flag no longer recited.** The §3d honesty flag was phrased as a statement
  ("grounded in only N sources") that models parroted verbatim into the reply — the same
  scaffolding-leak class as the tags. It is now a directive the model acts on (answer from what
  you know, name the gap, ask one question) and is told not to narrate its source count.

### Added
- **Recommended-models registry (inference engine, Phase A).** A curated, versioned data file
  (`cognition/recommended_models.toml`, loaded by `cognition/registry.py`) of families Mimir has
  tested — gemma/qwen/llama/phi/mistral/command-r/deepseek/granite/internlm, with per-role fitness,
  measured score floors, and `judge_ok` flags. `auto` routing now prefers a present
  recommended-for-the-role model **before any benchmark** (then approved-family, then any reachable),
  so a fresh user with both `gemma3:4b` and `gemma4:e4b` installed gets `gemma4:e4b` for chat, not the
  known-weak one — closing the worst out-of-box failure mode. Measured scores still override the
  registry once benchmarking runs. Not a whitelist: any installed model can still be measured and
  used. Spec: `docs/INFERENCE_ENGINE.md` §4.
- **Epistemic-competence experiment (`cognition/epistemics.py`).** Makes the core §3 thesis —
  typed/tiered/provenance context improves cognition over flat RAG — *measurable* per model. Each of
  three probes (tier deference, attribution, uncertainty) runs through the real `build_context()`
  (structured arm) and as a flat blob of the same facts (flat arm); `lift = structured − flat` is the
  framework's value. `brain.evaluate_epistemics(models, samples)` runs it across the fleet. Live
  cross-model finding: **positive lift for every model tested** — attribution is a universal win
  (impossible without provenance), the uncertainty gate most helps the weakest models, and
  tier-deference is model-dependent (gemma3:12b/gemma4:e4b defer perfectly; qwen3.5:9b ignores tiers).
- **`epistemics` is now a qualification-battery dimension.** The benchmark scores each model's
  structured-arm epistemic competence (does it exploit the framework?), and the identity-bearing
  roles (`chat`, `reasoning`) gate on **both** `discipline` and `epistemics` — so a model that
  ignores evidence tiers is barred from speaking as the system, just like one that leaks tags. New
  catalogue column (`epistemics`, schema v11); `ROLE_NEEDS` now lists multiple required capabilities
  per role. This is what keeps the framework from being handed to a model that won't use it.
- **Automatic model selection (`model = "auto"`).** A role's `model` can be pinned, set to `"auto"`,
  or omitted (→ auto). Auto resolves from the fleet by a strict hierarchy — **pin > measured-best
  (benchmarked + role-gated) > approved-family heuristic > any reachable model** — re-resolving on
  every rescan so a freshly benchmarked model is picked up. Users can **disable** a model (a bias
  veto) via `brain.set_model_enabled(...)` and it's skipped everywhere; the gateway stop-gaps an
  unresolved `auto` role to any reachable model so a turn never fails while the fleet is still
  inventorying. Default stays **local-only** (the LAN fleet is opt-in). New `model_prefs` table
  (schema v10) and a `brain.model_pool()` view (qualified ✓, speed, size, nodes, enabled, roles
  served) behind the Model Pool UI.
- **Model Pool tab in the web UI.** Lists every routable model with a ✓ if it passed the
  qualification gate, its size/quality/discipline/speed/nodes, and which roles it serves. A
  checkbox per model toggles it in or out of the automatic pool (the bias veto) — disabling a model
  serving an auto role re-routes that role live. Shows the backend mode (local vs LAN fleet) and the
  auto roles. New endpoints: `GET /api/fleet/pool`, `POST /api/fleet/model`.
- **`discipline` capability in the fleet IQ test.** The benchmark battery now scores a fourth
  dimension: does the model honor prohibitions, above all **not reproducing the internal
  `[tier=...; source=...]` scaffolding it is shown**. The probe replicates the *production* condition
  that actually triggers the leak — a tag-saturated recall block under the real soft "don't copy the
  tags" instruction — and samples it several times, scoring the fraction of bracket-free replies
  (leakage is probabilistic; consistency is the signal, per DESIGN §4). A weak single-tag prompt was
  too easy and missed the failure. The identity-bearing roles (`chat`, `reasoning`) gate on
  discipline, so the recommender refuses to route them to a fluent-but-leaky model — caught in
  qualification, not production. New catalogue column (`discipline`, schema v9). Validated live:
  `gemma3:4b` scores 0.25 (barred) while `gemma4:e2b`/`e4b`/`qwen3.5:9b` score 1.00 and `gemma3:12b`
  0.75.

### Validation
- End-to-end live run against a real LAN Ollama node (`gemma3:12b` for chat/reasoning,
  `gemma3:4b` for bake, `nomic-embed-text:v1.5` for embed): clean self-model synthesis (no
  hallucinated name), correct non-inverted identity ("I am Mimir, and I serve Greg"), no leaked
  tags or flag text, and a working bake → recall with attribution.
- Broader subsystem validation against the live 4-node fleet (43 models): document ingest →
  recall with provenance; the inner council deliberating across 5 distinct models with a coherent
  verdict; sleep/consolidation (salience decay) running clean; and a fleet benchmark with the
  coherence-judge canary passing and per-model quality/return-time scored.

## [0.1.0] — pre-alpha, feature-complete

The first feature-complete pre-alpha: the whole `DESIGN.md` architecture is implemented and
verified end-to-end against a live multi-node LAN. Still unhardened and untuned.

### The spine
- The §6 acceptance loop: boot empty → converse → bake a memory → a later turn recalls it with
  correct evidence tier and provenance via `build_context()` → the sentinel fires async and leaves
  a note. Runs as an automated self-test with a canary.
- Two chokepoint gateways: storage (single-writer thread, priority queue, batching, coalescing,
  retry-on-locked, flush) and model (provider pool with priority tiers, retry/backoff,
  transient-fail signaling, saturation breaker, failover).
- SQLite schema with versioned migrations (v1–v8), a startup schema check, and the fail-loud
  doctrine throughout (no silent swallow).

### Knowledge & epistemics
- Three-mode embeddings: stdlib bootstrap (locality hashing), endpoint, degraded — active mode
  reported loudly.
- Typed knowledge with evidence tiers + provenance, hybrid retrieval, and a deterministic
  uncertainty gate.
- Document ingestion (`ingest()`): text + markdown in core, PDF via the `[documents]` extra.
- Entity graph: subject–relation–object triples with 1–2 hop traversal.
- Working memory: rolling recency + periodic compression.
- Self-model: an evolving, generic identity synthesized from the store's own history, plus a
  re-runnable 8-anchor identity interview.
- Procedural memory: learned trigger → procedure habits.

### Async cognition
- Sentinel: a reflective pass that leaves a note for the next turn.
- Sleep / consolidation: dedup, salience/confidence decay (with the death-spiral guard), archival,
  and contradiction resolution.
- Inner council: adversarial deliberation across auto-discovered models, synthesized into a verdict.

### Distributed model fleet
- LAN auto-discovery of Ollama nodes (zero setup on the nodes), model-aware routing (a request
  goes only to a node that has the model), active health checks, and least-loaded selection.
- A persisted catalogue and benchmarking — a capability "IQ test" (talk / tools / code) plus a
  coherence vote by a panel of other models, guarded by a canary pair.
- Per-role recommendations from the benchmarked catalogue.

### Surface
- A zero-dependency stdlib reference web UI: streaming chat, the identity interview, mind / memory
  / graph / habits browsers, the inner council, document ingest, and the fleet (scan / benchmark /
  recommend).
- The library API, plus `python -m mimir.{selftest,interview,server}`.
