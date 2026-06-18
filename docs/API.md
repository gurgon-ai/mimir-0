# Mimir 0 — integration API

Mimir 0 is a **brain, not an app**: it deliberately ships **no built-in hands** (no voice, no
avatar, no Home Assistant, no social). Instead it exposes a small, stable HTTP surface so you can
build whatever IO you want against it — voice in/out, an avatar, a home-assistant bridge, a social
bot, an agent framework, or a middle layer that lets **two Mimirs talk to each other**.

Two ways in:

- **Library** — `from mimir import Mimir; m.turn("hello", user="greg")`. The cleanest "system in/out"
  if you're in Python. Everything below is just the HTTP adapter over this.
- **HTTP** — `python -m mimir.server --config mimir.toml` serves the reference web UI *and* the API
  on the same port. The API is what you integrate against.

> The server is a **reference adapter, not a hardened public service.** Bind it to localhost (the
> default) or a trusted LAN, turn on the API token, and put a real reverse proxy (TLS, rate limits)
> in front if you expose it to the internet.

## Security

Auth is **off by default** (open on localhost — convenient for dev). Turn it on by setting a token,
two ways (env wins, so secrets needn't live in a file):

```toml
# mimir.toml
[server]
api_token = "a-long-random-string"
cors_origins = ["https://my-avatar.local"]   # browser origins allowed to call the API; ["*"] = any
```
```bash
export MIMIR_API_TOKEN="a-long-random-string"   # overrides the config value
```

**Running more than one instance on the same machine?** The env var name is configurable, so a token
set for one doesn't bleed into another. Point each instance at its own variable:

```toml
[server]
api_token_env = "MIMIR0_TOKEN"   # this instance reads $MIMIR0_TOKEN (default: MIMIR_API_TOKEN)
```

(Sharing one token across instances is fine too — a token is just a shared secret, and it even
saves a step when one instance calls another. Separate vars only matter if you want to rotate or
revoke them independently.)

When a token is set, **remote** callers must send it:

```
Authorization: Bearer a-long-random-string
```

…or the API returns **`401 {"error":"unauthorized"}`**.

**The local browser UI is exempt by default.** Requests from the same machine (`127.0.0.1`) skip the
token, so opening the UI on the box that runs Mimir is never blocked by a token wall — even with a
token set. The token guards **remote/integration** callers; the operator at the local UI just works.
Set `[server] secure_ui = true` to require the token for the local UI too (a shared box, or behind a
reverse proxy where every request looks local — enable this, or have the proxy do auth). When the UI
*is* gated (remote, or `secure_ui`), it prompts for the token once and stores it locally.

`GET /api/health` is **always** exempt (liveness probes need no credentials). CORS preflight
(`OPTIONS`) is unauthenticated, as browsers require; the allowed-origin header is echoed only for
origins in `cors_origins`.

## The turn endpoint

`POST /api/turn` — one turn in, one reply out (the whole cognition loop runs: recall, context
assembly, generation, then async bake/sentinel).

Request:
```json
{ "text": "what did I tell you about the garden?", "user": "greg", "speaker_kind": "human" }
```
- `text` (required) — the message.
- `user` (optional) — **the speaker's identity.** This is the seam for multi-speaker and
  **agent-to-agent**: set it to whoever is talking (`"greg"`, `"mimir-home"`, …). Memory and
  recency are tracked per speaker.
- `speaker_kind` (optional, default `"human"`) — **what kind of speaker this is**: `"human"` or
  `"ai_peer"`. A human's statements are believed per the trust policy below; a peer AI's are baked at
  a **lower tier** (`stated_by_peer`, 0.95 — below human conversation), attributed and marked
  AI-sourced, because they're generated text, not observation. So if you build your own interface,
  leave it `"human"`; if another agent is talking to this one, send `"ai_peer"`. (Alias: `"kind"`.)
  An unknown value is rejected with 400 — the policy never resolves ambiguity by elevating a caller.

Response:
```json
{
  "reply": "You said the garlic goes in around October…",
  "introspect": {
    "embed_mode": "endpoint",
    "budget_tokens": 8192,
    "requested_tokens": 5120, "admitted_tokens": 4990,
    "source_count": 7,
    "uncertainty_triggered": false,
    "warnings": [],
    "sections": [ { "name": "knowledge", "tier": "HIGH", "...": "..." } ]
  }
}
```
`introspect` is the context accounting (what was in the prompt, how big, how many grounding sources)
— useful for an avatar that wants to show confidence/sources, and for debugging.

### Streaming

`POST /api/turn/stream` — same request body, a **Server-Sent-Events** stream: `token` events as the
reply generates, then a final `done` event carrying `introspect`. Use this for low-latency voice/chat
frontends. (Send the `Authorization` header here too.)

## Examples

```bash
curl -s http://127.0.0.1:8765/api/turn \
  -H "Authorization: Bearer $MIMIR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"hello","user":"greg"}'
```

```python
import requests
r = requests.post("http://127.0.0.1:8765/api/turn",
                  headers={"Authorization": f"Bearer {TOKEN}"},
                  json={"text": "hello", "user": "greg"})
print(r.json()["reply"])
```

## Two Mimirs talking (the middle layer)

Mimir 0 provides no orchestration — that's yours — but it's built for it: each instance is a clean,
authenticated turn endpoint that accepts a **speaker identity**. A minimal relay:

```python
def relay(msg, a_url, a_tok, b_url, b_tok, a_name="mimir-a", b_name="mimir-b", turns=20):
    speaker, text = a_name, msg
    for _ in range(turns):
        target, tok = (b_url, b_tok) if speaker == a_name else (a_url, a_tok)
        reply = post_turn(target, tok, text, user=speaker)   # POST /api/turn
        print(f"{speaker}: {text}\n→ {reply}\n")
        speaker = b_name if speaker == a_name else a_name
        text = reply
```

Each side sees the other as just another speaker (`user="mimir-a"` / `"mimir-b"`), remembers the
exchange, and builds its own context — so the two accumulate a shared history and genuinely converse.
Give your home Mimir the same endpoint and they can run side by side and "hang out."

### Identity vs. trust (important)

The `user` field is the speaker's **identity** and `speaker_kind` is its **kind** — both caller-set.
How much that speaker is **believed** is **server-side config**, not the caller's to declare (so an
exposed endpoint can't inject top-tier "facts"). The policy (`[identity]` in `mimir.toml`):

- `primary_user = "greg"` → that (human) speaker's statements bake at the top evidence tier (1.30).
- `trusted_users = ["julien"]` → trusted tier (1.20).
- **any other named human** (an unknown caller, a guest) → attributed but baked at **CONVERSATION**
  tier (1.00) — recorded as "X said it," never as established fact.
- a **peer AI** — either `speaker_kind="ai_peer"` on the turn, or the identity listed in
  `peer_agents = ["mimir-home"]` — bakes at **`stated_by_peer`** (0.95): below human conversation,
  attributed, and marked AI-sourced. The kind wins over identity, so an agent can't reach a human
  tier by also being named primary/trusted. `peer_agents` is the operator-side enforcement (a known
  peer can't avoid it by sending `speaker_kind="human"`).
- With **no policy set at all**, Mimir is single-user: the lone named *human* speaker is treated as
  primary (so a simple custom UI works with zero config).

So: build your own front-end → send `speaker_kind="human"` and your users are treated as users. Wire
another agent to this one → send `speaker_kind="ai_peer"` (and/or list it in `peer_agents`) and its
claims are remembered-as-said-by-an-AI but kept below human input — safe against one AI's
hallucinations (or an echo between two agents) being mistaken for fact. The caller picks the name and
kind; the server picks the trust.

## Documents

Feed documents in three ways (all become `document`-tier, recallable knowledge; an idle pass also
writes a short summary per doc — the local "wiki"):

- `POST /api/ingest` `{"path": "..."}` — ingest a file the **server** can read by path.
- `POST /api/documents/upload` `{"name": "notes.pdf", "data": "<base64>"}` — upload bytes (what the
  📎 button uses); saved to the `[documents] folder` and ingested. `name` is sanitized to a basename;
  the type must be supported (`.txt`/`.md` in core; `.pdf`/`.docx` need the `[documents]` extra —
  `pip install 'mimir-0[documents]'`).
- `POST /api/documents/scan` — ingest any new/changed files dropped into the folder + fill summaries.
  Returns `{folder, ingested:[name], summarized, failed:[{name,error}], unsupported:[name],
  forgotten:[name]}` — per-file failures (e.g. a missing extra) and wrong-type drops are reported,
  never swallowed; `forgotten` lists docs whose source file vanished and were auto-cleaned.
- `POST /api/library/enable` `{"source": "<path or filename>", "enabled": false}` — toggle a
  document's "include in context." A disabled document's chunks + claims are excluded from recall (at
  the SQL load layer — recall speed back for a big library); the data is kept, re-enable to restore.
- The turn endpoints accept per-turn **layer toggles** (each defaults `true`): `include_memory`,
  `include_library`, `include_wiki` — skip a whole context layer for the turn.
- `POST /api/library/forget` `{"source": "<path or filename>", "delete_file": true}` — purge a
  document and everything derived from it: its memory chunks, library document + cited claims,
  composite page (row + Markdown file), and wiki ledger entry; `delete_file` also removes the source
  file (so an idle scan won't re-ingest it). Idempotent. Returns
  `{source, memory_chunks, library_doc, pages, file_deleted}`. The inverse also works on its own:
  delete the file from the folder and the next scan auto-forgets it.
- `GET /api/documents` — `{folder, folder_abs, folder_exists, documents:[{name, chunks, summary,
  ingested_at, source}]}`. `folder_abs` is the resolved absolute path (relative folders follow the
  server's working directory).

## Library (docs/LIBRARY.md)

The system's own long-form knowledge as three tiers — source documents (ground truth) → short **cited
claims** (always-on in chat, with their source title + locator) → Markdown **composites** (the
synthesized understanding, fetched on demand). Built in idle from the `[documents]` folder.

- `GET /api/library` — `{documents:[{id,filename,title,size_bytes,claims,…}], pages:[{id,title,summary,…}]}`.
- `GET /api/library/page?id=N` — a composite's full Markdown + its **citations** (each claim → title +
  locator). The Load button / fetch path.
- `GET /api/library/source?id=N` — a source document's **verbatim** text (for quoting/checking).
- `POST /api/library/scan` — (re)distil sources into cited claims + composites now.
- **Load into a turn:** `POST /api/turn` accepts `"library_pages": [id, …]` — the full Markdown of
  those composites is added to that turn's context (what the UI's pin-to-chat / Load chips send). The
  turn response includes `"library_sources": [{page_id,title}]` — the pages the answer drew on.
- **Model-driven fetch** (opt-in `[library] model_fetch`): the model may reply `<FETCH id=N>` to open
  a page itself; the turn loads it and re-answers (capped, off by default; non-streaming path).

## Other routes

The turn endpoints above are what most integrations need. The web UI is driven by a wider set —
identity, onboarding, mind, memories, graph, sessions, settings, sleep, deliberate, council, forum,
fleet/*, wiki/status, ingest, documents/* — all under `/api/` and so all behind the same token
(except `/api/health`). The authoritative list is the `do_GET`/`do_POST` dispatch in `server.py`.
