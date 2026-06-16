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

When a token is set, **every `/api/*` route** requires it:

```
Authorization: Bearer a-long-random-string
```

Without it (or with the wrong one) the API returns **`401 {"error":"unauthorized"}`**. The page
shell (`/`) stays open so the bundled web UI can prompt you for the token once (it stores it locally
and sends it thereafter). CORS preflight (`OPTIONS`) is unauthenticated, as browsers require; the
allowed-origin header is echoed only for origins in `cors_origins`.

## The turn endpoint

`POST /api/turn` — one turn in, one reply out (the whole cognition loop runs: recall, context
assembly, generation, then async bake/sentinel).

Request:
```json
{ "text": "what did I tell you about the garden?", "user": "greg" }
```
- `text` (required) — the message.
- `user` (optional) — **the speaker's identity.** This is the seam for multi-speaker and
  **agent-to-agent**: set it to whoever is talking (`"greg"`, `"mimir-parent"`, …). Memory and
  recency are tracked per speaker.

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

The `user` field is the speaker's **identity** — caller-set. How much that speaker is **believed** is
**server-side config**, not the caller's to declare (so an exposed endpoint can't inject top-tier
"facts"). The policy (`[identity]` in `mimir.toml`):

- `primary_user = "greg"` → that speaker's statements bake at the top evidence tier (1.30).
- `trusted_users = ["julien", "home-mimir"]` → trusted tier (1.20).
- **any other named speaker** (an unknown caller, a peer AI you haven't listed) → attributed but
  baked at **CONVERSATION** tier — recorded as "X said it," never as established fact.
- With **no policy set at all**, Mimir is single-user: the lone named speaker is treated as primary
  (so a simple custom UI works with zero config).

So for a peer AI: leave it off `trusted_users` and its claims are remembered-as-said but not believed
(safe against hallucinations); add it to `trusted_users` only if you want its statements to carry
weight. The caller picks the name; the server picks the trust.

## Other routes

The full route list is in `server.py`'s module docstring (identity, onboarding, memories, graph,
sessions, sleep/deliberate, fleet, forum, settings, diagnostics). They're all under `/api/` and so
all behind the same token. The turn endpoints above are the ones most integrations need.
