# Security Policy

## Status

Mimir 0 is **pre-alpha and not yet hardened.** It is a local-first library plus a *reference* web
server — the server is meant for localhost or a trusted LAN, **not** as a hardened public service.
If you expose it, put it behind a reverse proxy (TLS, auth, rate limits), turn on the API token, and
bind identity at your integration layer (see [`docs/API.md`](docs/API.md) — `user` is a *claim*, not
an authenticated identity).

## Reporting a vulnerability

Please report security issues **privately** through GitHub's private vulnerability reporting:

- Go to the repository's **Security** tab → **Report a vulnerability** (GitHub Security Advisories).

This opens a private channel with the maintainer; please do **not** open a public issue for a
suspected vulnerability. Include what you found, how to reproduce it, and the impact you expect.

There is no formal SLA (this is a solo, pre-1.0 project), but reports are read and triaged. Once a
fix lands, the advisory is published with credit if you'd like it.

## Scope

In scope:

- The core library (`src/mimir/`) — memory, context assembly, the two gateways, the trust policy.
- The reference server (`src/mimir/server.py`) and its `/api/*` surface — auth/token handling,
  CORS, path confinement for document ingestion, input validation.

Out of scope (by design — these are yours to provide and secure):

- The model/embeddings backend (Ollama or any endpoint) and the models themselves.
- Any front-end, relay, or integration you build on top of the API.
- Network exposure decisions — Mimir defaults to localhost; exposing it is your call to secure.

## Defense-in-depth already in place

These are documented invariants, not aspirations (see [`DESIGN.md`](DESIGN.md) §10):

- **Zero core runtime dependencies** — the attack surface is the standard library + your backend.
- **Parameterized SQL** throughout; all writes go through one storage gateway.
- **Trust policy** — a caller can't launder claims into trusted memory, and a peer AI can't reach a
  human trust tier by renaming (`speaker_kind` wins over identity).
- **Path confinement** — document ingestion/forget is confined to the configured folders.
- **Fail-loud doctrine** — no silent fallbacks; errors surface rather than degrade quietly.
