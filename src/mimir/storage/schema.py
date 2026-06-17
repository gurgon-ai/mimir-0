"""The SQLite schema, as a versioned migration ladder.

``schema_version`` is the first table created and the first thing checked at boot.
DESIGN §10: schema versioning + a tiny migration runner + a startup check are a v0
doctrine, not a later nicety. A misconfigured store fails loud — it NEVER silently
falls back to another store.

To evolve the schema: append a new ``(version, [statements])`` tuple to ``MIGRATIONS``
and bump nothing else — ``CURRENT_SCHEMA_VERSION`` is derived from the ladder length.
Never edit a past migration; only append. Each migration runs once, in order, inside a
transaction (see ``migrate.py``).
"""

from __future__ import annotations

# Each entry: (target_version, [SQL statements to reach it from the previous version]).
# Migration N takes the DB from version N-1 to version N.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # The version marker itself. Single-row table; the runner maintains it.
            "CREATE TABLE schema_version (version INTEGER NOT NULL)",
            # The one knowledge table for the v0 spine. Typed layers beyond `memory`
            # (understanding, procedural, …) arrive as new `kind` values and/or new
            # tables in later migrations — never by mutating this one in place.
            """
            CREATE TABLE memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT    NOT NULL,
                kind          TEXT    NOT NULL DEFAULT 'memory',
                evidence_tier TEXT    NOT NULL DEFAULT 'conversation',
                confidence    REAL    NOT NULL DEFAULT 0.7,
                salience      REAL    NOT NULL DEFAULT 1.0,
                embedding     BLOB,
                provenance    TEXT    NOT NULL DEFAULT 'conversation',
                user          TEXT,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                meta          TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX idx_memories_user ON memories(user)",
            "CREATE INDEX idx_memories_kind ON memories(kind)",
        ],
    ),
    (
        2,
        [
            # v0.1 document ingestion: a chunk is just a memory with evidence_tier='document'
            # and a `source` pointing at the file it came from. The column lets re-ingest
            # replace a document's chunks cleanly (delete-by-source). NULL for non-document rows.
            "ALTER TABLE memories ADD COLUMN source TEXT",
            "CREATE INDEX idx_memories_source ON memories(source)",
        ],
    ),
    (
        3,
        [
            # Identity anchors: foundational, operator-established identity facts (name, operator,
            # location, purpose) that ground the always-on self-model from the very first boot,
            # before any history exists. Key-value with upsert semantics — re-answering updates.
            "CREATE TABLE identity ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL,"
            " established_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        4,
        [
            # Entity graph: subject-relation-object triples — the 'what is connected' layer
            # (DESIGN §3a). Indexed on subject and object for 1-2 hop traversal; a unique
            # expression index dedupes case-insensitively so the same triple isn't stored twice.
            "CREATE TABLE triples ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " subject TEXT NOT NULL,"
            " relation TEXT NOT NULL,"
            " object TEXT NOT NULL,"
            " user TEXT,"
            " provenance TEXT NOT NULL DEFAULT 'conversation',"
            " confidence REAL NOT NULL DEFAULT 0.8,"
            " created_at REAL NOT NULL"
            ")",
            "CREATE INDEX idx_triples_subject ON triples(subject)",
            "CREATE INDEX idx_triples_object ON triples(object)",
            "CREATE UNIQUE INDEX idx_triples_unique ON triples("
            " lower(subject), lower(relation), lower(object), coalesce(user, ''))",
        ],
    ),
    (
        5,
        [
            # Sleep/consolidation: archived memories are excluded from active recall but kept in
            # the store (archiving ≠ disbelieving, DESIGN §3c) — a resurfaced one is still trusted.
            "ALTER TABLE memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0",
            "CREATE INDEX idx_memories_archived ON memories(archived)",
        ],
    ),
    (
        6,
        [
            # Procedural memory: learned reasoning habits as trigger→procedure pairs (DESIGN §3a).
            # The trigger is embedded for cosine matching; `uses` tracks how proven a habit is.
            "CREATE TABLE procedures ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " trigger TEXT NOT NULL,"
            " procedure TEXT NOT NULL,"
            " trigger_embedding BLOB,"
            " user TEXT,"
            " confidence REAL NOT NULL DEFAULT 0.7,"
            " uses INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        7,
        [
            # The fleet catalogue (DESIGN §5): a persisted snapshot of every (node, model) found,
            # with its weight/family/quant. return_time + quality are filled by Phase 2 benchmarking
            # (NULL until then). Rebuilt on each scan.
            "CREATE TABLE model_catalogue ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " node TEXT NOT NULL,"
            " model TEXT NOT NULL,"
            " family TEXT NOT NULL DEFAULT '',"
            " params_b REAL NOT NULL DEFAULT 0,"
            " quantization TEXT NOT NULL DEFAULT '',"
            " context_length INTEGER NOT NULL DEFAULT 0,"
            " capabilities TEXT NOT NULL DEFAULT '[]',"
            " return_time REAL,"
            " quality REAL,"
            " scanned_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        8,
        [
            # Phase 2 benchmark scores per model (DESIGN §4): the capability 'IQ test' dimensions.
            # NULL until benchmarked. quality (above) is the aggregate; these are the breakdown.
            "ALTER TABLE model_catalogue ADD COLUMN talk REAL",
            "ALTER TABLE model_catalogue ADD COLUMN tools REAL",
            "ALTER TABLE model_catalogue ADD COLUMN code REAL",
            "ALTER TABLE model_catalogue ADD COLUMN coherence REAL",
        ],
    ),
    (
        9,
        [
            # The 'discipline' dimension (DESIGN §4): does the model honor prohibitions — above all,
            # NOT reproducing the internal [tier=...; source=...] scaffolding it is shown. This is
            # the signal that separates an identity-safe chat/reasoning model from one that leaks
            # the prompt's tags (the failure that drove the output sanitizer; DESIGN §10).
            "ALTER TABLE model_catalogue ADD COLUMN discipline REAL",
        ],
    ),
    (
        10,
        [
            # Per-model user preference for automatic routing (DESIGN §4): a model is enabled by
            # default; a user who distrusts one disables it here and `auto` resolution skips it.
            # The catalogue is rebuilt every scan, so this preference lives separately and survives.
            "CREATE TABLE model_prefs ("
            " model TEXT PRIMARY KEY,"
            " enabled INTEGER NOT NULL DEFAULT 1,"
            " updated_at REAL NOT NULL DEFAULT 0"
            ")",
        ],
    ),
    (
        11,
        [
            # The 'epistemics' dimension (DESIGN §3/§4): does the model exploit Mimir's tiered/
            # provenance/gated context — defer to higher-tier facts, attribute to source, hedge on
            # thin evidence? The identity-bearing roles gate on it, so the framework is never handed
            # to a model that ignores it (e.g. one that disregards evidence tiers).
            "ALTER TABLE model_catalogue ADD COLUMN epistemics REAL",
        ],
    ),
    (
        12,
        [
            # The 'reasoning' dimension (DESIGN §4): can the model actually SOLVE a problem, not
            # just follow a format? The rest of the battery (PONG, a weather JSON, def add) measures
            # format compliance — every competent model passes, so quality saturates and can't
            # separate a capable model from one that merely complies. This dimension scores
            # deterministic, regex-checkable problems (arithmetic, counting, pattern, code-trace,
            # transforms) so a model that 'can't do the job' scores low even when well-behaved.
            "ALTER TABLE model_catalogue ADD COLUMN reasoning REAL",
        ],
    ),
    (
        13,
        [
            # Per-node user preference (DESIGN §5): an edge node is part of the fleet by default; a
            # user who doesn't want a reachable box used disables it here, and discovery, the
            # qualification, and routing all skip it. Survives the per-scan catalogue rebuild.
            "CREATE TABLE node_prefs ("
            " node TEXT PRIMARY KEY,"
            " enabled INTEGER NOT NULL DEFAULT 1,"
            " updated_at REAL NOT NULL DEFAULT 0"
            ")",
        ],
    ),
    (
        14,
        [
            # Interaction log: one tiny row per turn (timestamp + user), durable and append-only —
            # unlike the EXCHANGE recency log (capped/cleared on compression), this is the lasting
            # record of WHEN the user engaged. It powers temporal-awareness baselines (DESIGN §3e):
            # "you haven't been around in 14h (typical ~6h)". Pruned to a rolling window so it never
            # grows unbounded. Just a timestamp — no content, so it's cheap and privacy-light.
            "CREATE TABLE interactions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL NOT NULL,"
            " user TEXT"
            ")",
            "CREATE INDEX idx_interactions_ts ON interactions(ts)",
        ],
    ),
    (
        15,
        [
            # Temporal narratives (DESIGN §3a/§3e): hierarchical daily→weekly→monthly journal rows,
            # lossy by design (details fade, patterns persist). One row per (scope, period); the
            # unique index makes regeneration an idempotent replace, and old entries are pruned to a
            # per-scope retention cap. Generated off the hot path in the consolidation pass.
            "CREATE TABLE narratives ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " scope TEXT NOT NULL,"        # 'daily' | 'weekly' | 'monthly'
            " period TEXT NOT NULL,"       # e.g. '2026-06-14' or '2026-06-01_to_2026-06-07'
            " narrative TEXT NOT NULL,"
            " source_count INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL"
            ")",
            "CREATE UNIQUE INDEX idx_narratives_scope_period ON narratives(scope, period)",
        ],
    ),
    (
        16,
        [
            # Conversation log: the durable, full turn history (one row per exchange — user text +
            # reply) so the conversation survives a restart and the UI can RESTORE it on load, so
            # recent turns can be replayed to the model as real messages for continuity. Unlike
            # the EXCHANGE recency buffer (capped/cleared on compression) and from `interactions`
            # (timestamps only). Pruned to a rolling window so it never grows unbounded.
            "CREATE TABLE conversation ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " user TEXT,"
            " user_text TEXT NOT NULL,"
            " reply TEXT NOT NULL,"
            " created_at REAL NOT NULL"
            ")",
            "CREATE INDEX idx_conversation_created ON conversation(created_at)",
        ],
    ),
    (
        17,
        [
            # Sessions: tag each conversation turn with the session (distinct conversation) it's in,
            # so the UI can list past conversations in a dropdown and restore/continue one. A new
            # session starts on a long idle gap or an explicit "new conversation". Legacy rows
            # (NULL) read as one early session. Just a grouping key — no separate table needed.
            "ALTER TABLE conversation ADD COLUMN session_id TEXT",
            "CREATE INDEX idx_conversation_session ON conversation(session_id)",
        ],
    ),
    (
        18,
        [
            # Backfill: turns logged before v17 have a NULL session_id, which the UI can't select or
            # restore (an empty dropdown value). Group them into one restorable "legacy" session
            # so every turn belongs to a named session. (No-op for a fresh DB.)
            "UPDATE conversation SET session_id = 'legacy' WHERE session_id IS NULL",
        ],
    ),
    (
        19,
        [
            # A tiny generic key→value store for small bits of durable cognition state that don't
            # warrant their own table. First user: the sleep cycle's per-day checkpoint (the phases
            # run on each date), so a wall-clock maintenance window can resume after a same-night
            # restart and never run twice in a day (DESIGN §5a). Value is opaque JSON text.
            "CREATE TABLE kv ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  updated_at REAL NOT NULL"
            ")",
        ],
    ),
    (
        20,
        [
            # The council forum (DESIGN §5a): deliberations persisted as browsable threads so the
            # user can read the debate, comment, and keep house (close/delete). A thread is one
            # question; posts are the persona positions (tagged with the node+model that argued
            # them), the synthesized verdict, and user comments.
            "CREATE TABLE forum_threads ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  question TEXT NOT NULL,"
            "  status TEXT NOT NULL DEFAULT 'open',"   # open | closed
            "  source TEXT NOT NULL DEFAULT 'council',"  # council | sleep deliberation | user
            "  verdict TEXT NOT NULL DEFAULT '',"
            "  created_at REAL NOT NULL"
            ")",
            "CREATE TABLE forum_posts ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  thread_id INTEGER NOT NULL,"
            "  author TEXT NOT NULL,"                  # persona name | 'synthesis' | user name
            "  kind TEXT NOT NULL,"                    # position | verdict | comment
            "  model TEXT NOT NULL DEFAULT '',"
            "  node TEXT NOT NULL DEFAULT '',"
            "  content TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")",
            "CREATE INDEX idx_forum_posts_thread ON forum_posts(thread_id)",
        ],
    ),
    (
        21,
        [
            # The Library layer (docs/LIBRARY.md): the system's own long-form knowledge as THREE
            # tiers of truth. The DB is the provenance spine connecting them.
            #
            # 1) library_documents — the ground-truth source files (left where the user dropped
            #    them), by exact filename + size + hash + title, so it can go back and re-read.
            "CREATE TABLE library_documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  path TEXT NOT NULL UNIQUE,"
            "  filename TEXT NOT NULL,"
            "  size_bytes INTEGER NOT NULL DEFAULT 0,"
            "  content_hash TEXT NOT NULL,"
            "  title TEXT NOT NULL DEFAULT '',"
            "  ingested_at REAL NOT NULL"
            ")",
            # 2) library_claims — the DB SPINE: short atomic facts, each citing its source document
            #    + exact locator (page/line/section), embedded for retrieval. The citable layer.
            "CREATE TABLE library_claims ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  document_id INTEGER NOT NULL,"
            "  text TEXT NOT NULL,"
            "  locator TEXT NOT NULL DEFAULT '',"
            "  embedding BLOB,"
            "  confidence REAL NOT NULL DEFAULT 0.8,"
            "  created_at REAL NOT NULL"
            ")",
            "CREATE INDEX idx_library_claims_document ON library_claims(document_id)",
            # 3) library_pages — the MD COMPOSITE index: the LLM's fuzzy synthesized understanding,
            #    stored as a Markdown file at `path` (a separate folder/tree), fetched on demand.
            "CREATE TABLE library_pages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  path TEXT NOT NULL UNIQUE,"
            "  title TEXT NOT NULL,"
            "  summary TEXT NOT NULL DEFAULT '',"
            "  content_hash TEXT NOT NULL DEFAULT '',"
            "  created_at REAL NOT NULL,"
            "  updated_at REAL NOT NULL"
            ")",
            # The link: which claims composed which page (provenance both ways — page → claims →
            # source line; claim → the pages it fed). Lets the system cite, verify, and re-read.
            "CREATE TABLE library_page_claims ("
            "  page_id INTEGER NOT NULL,"
            "  claim_id INTEGER NOT NULL"
            ")",
            "CREATE INDEX idx_library_page_claims_page ON library_page_claims(page_id)",
            "CREATE INDEX idx_library_page_claims_claim ON library_page_claims(claim_id)",
        ],
    ),
]

# Derived, never hand-edited: the version this code expects an opened DB to be at.
CURRENT_SCHEMA_VERSION: int = MIGRATIONS[-1][0] if MIGRATIONS else 0

# What the startup check asserts actually exists, so a half-migrated or foreign DB is
# caught loudly rather than limping along. Maps table -> required columns.
EXPECTED_SHAPE: dict[str, set[str]] = {
    "schema_version": {"version"},
    "memories": {
        "id",
        "text",
        "kind",
        "evidence_tier",
        "confidence",
        "salience",
        "embedding",
        "provenance",
        "user",
        "created_at",
        "last_accessed",
        "access_count",
        "meta",
        "source",
        "archived",
    },
    "identity": {"key", "value", "established_at"},
    "triples": {
        "id",
        "subject",
        "relation",
        "object",
        "user",
        "provenance",
        "confidence",
        "created_at",
    },
    "procedures": {
        "id",
        "trigger",
        "procedure",
        "trigger_embedding",
        "user",
        "confidence",
        "uses",
        "created_at",
    },
    "model_catalogue": {
        "id",
        "node",
        "model",
        "family",
        "params_b",
        "quantization",
        "context_length",
        "capabilities",
        "return_time",
        "quality",
        "scanned_at",
        "talk",
        "tools",
        "code",
        "coherence",
        "discipline",
        "epistemics",
        "reasoning",
    },
    "model_prefs": {"model", "enabled", "updated_at"},
    "node_prefs": {"node", "enabled", "updated_at"},
    "interactions": {"id", "ts", "user"},
    "narratives": {"id", "scope", "period", "narrative", "source_count", "created_at"},
    "conversation": {"id", "user", "user_text", "reply", "created_at", "session_id"},
    "kv": {"key", "value", "updated_at"},
    "forum_threads": {"id", "question", "status", "source", "verdict", "created_at"},
    "forum_posts": {"id", "thread_id", "author", "kind", "model", "node", "content", "created_at"},
    "library_documents": {
        "id", "path", "filename", "size_bytes", "content_hash", "title", "ingested_at",
    },
    "library_claims": {
        "id", "document_id", "text", "locator", "embedding", "confidence", "created_at",
    },
    "library_pages": {"id", "path", "title", "summary", "content_hash", "created_at", "updated_at"},
    "library_page_claims": {"page_id", "claim_id"},
}
