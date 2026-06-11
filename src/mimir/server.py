"""Reference web UI — a stdlib HTTP server over the core library (DESIGN §8 adapter).

Mimir 0's core is a library; this is the canonical *human* surface, kept deliberately outside the
core and built on Python's stdlib ``http.server`` so it adds **zero dependencies** — the runtime
contract (Python + SQLite + endpoints, nothing else) still holds. It serves a single-page UI and a
small JSON API:

    GET  /                 the web UI (chat + identity interview + document ingest + status)
    GET  /api/state        embedding mode, memory count, anchors established
    GET  /api/identity     current anchors + the questions still pending
    POST /api/identity     {"answers": {...}}  establish/revise identity anchors
    POST /api/turn         {"text": "...", "user": "..."}  → {"reply", "introspect"}
    POST /api/ingest       {"path": "..."}  ingest a local document

Run it:  ``python -m mimir.server --config mimir.toml``  (then open http://127.0.0.1:8765).

Brain access is serialized by a lock — a turn isn't built for concurrent callers, and this is a
single-operator local tool. It is a reference adapter, not a hardened public web service: bind it
to localhost and put a real reverse proxy in front if you expose it.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .brain import Mimir
from .cognition.self_model import gather_signals
from .errors import IngestError, MimirError
from .storage.models import Memory, MemoryKind
from .storage.repo import browse_memories, count_memories, latest_self_model

log = logging.getLogger("mimir.server")


class MimirHTTPServer(ThreadingHTTPServer):
    """A threading HTTP server that holds the shared brain and serializes access to it."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], brain: Mimir) -> None:
        super().__init__(address, _Handler)
        self.brain = brain
        self.brain_lock = threading.Lock()


def create_server(brain: Mimir, host: str = "127.0.0.1", port: int = 8765) -> MimirHTTPServer:
    """Build (but do not start) a server bound to ``host:port`` (port 0 picks a free one)."""
    return MimirHTTPServer((host, port), brain)


class _Handler(BaseHTTPRequestHandler):
    server: MimirHTTPServer
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _identity_payload(self) -> dict[str, Any]:
        brain = self.server.brain
        return {
            "anchors": brain.identity_anchors(),
            "pending": brain.pending_identity_questions(),
        }

    # -- routing ----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        route, params = parsed.path, parse_qs(parsed.query)
        try:
            if route == "/":
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/state":
                self._send_json(self._state())
            elif route == "/api/identity":
                self._send_json(self._identity_payload())
            elif route == "/api/mind":
                self._send_json(self._mind())
            elif route == "/api/memories":
                self._send_json(self._memories(params))
            elif route == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:  # bad query params
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # never leak a stack to the client; log loud (DESIGN §10)
            log.exception("GET %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            if self.path == "/api/turn":
                self._send_json(self._turn(body))
            elif self.path == "/api/identity":
                self._send_json(self._establish(body))
            elif self.path == "/api/ingest":
                self._send_json(self._ingest(body))
            else:
                self._send_json({"error": "not found"}, status=404)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
        except ValueError as exc:  # request validation (missing/!malformed fields)
            self._send_json({"error": str(exc)}, status=400)
        except IngestError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except MimirError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)

    # -- operations (all under the brain lock) ----------------------------------------

    def _state(self) -> dict[str, Any]:
        brain = self.server.brain
        with self.server.brain_lock:
            return {
                "embed_mode": brain._embedder.mode.value,
                "embed_banner": brain._embedder.mode.banner(),
                "memories": count_memories(brain._storage, kind=MemoryKind.MEMORY),
                "anchors_set": len(brain.identity_anchors()),
            }

    def _turn(self, body: dict[str, Any]) -> dict[str, Any]:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ValueError("'text' is required")
        user = body.get("user") or None
        with self.server.brain_lock:
            result = self.server.brain.turn(text, user=user)
            self.server.brain.wait_for_sentinel()  # let the note/self-model settle
        return {"reply": result.reply, "introspect": result.context.introspect()}

    def _establish(self, body: dict[str, Any]) -> dict[str, Any]:
        answers = body.get("answers") or {}
        if not isinstance(answers, dict):
            raise ValueError("'answers' must be an object")
        with self.server.brain_lock:
            self.server.brain.establish_identity({str(k): str(v) for k, v in answers.items()})
            return self._identity_payload()

    def _ingest(self, body: dict[str, Any]) -> dict[str, Any]:
        path = str(body.get("path", "")).strip()
        if not path:
            raise ValueError("'path' is required")
        with self.server.brain_lock:
            result = self.server.brain.ingest(path)
        return {
            "source": result.source,
            "units": result.units,
            "chunks_written": result.chunks_written,
            "chunks_replaced": result.chunks_replaced,
        }

    def _mind(self) -> dict[str, Any]:
        brain = self.server.brain
        with self.server.brain_lock:
            signals = gather_signals(brain._storage)
            self_model = latest_self_model(brain._storage)
            anchors = brain.identity_anchors()
        return {
            "self_model": self_model.text if self_model else None,
            "anchors": anchors,
            "stats": {
                "total": signals.total_memories,
                "documents": signals.documents,
                "reflections": signals.reflections,
                "users": signals.distinct_users,
                "by_tier": signals.tier_counts,
            },
            "recent_reflections": signals.recent_reflections,
        }

    def _memories(self, params: dict[str, list[str]]) -> dict[str, Any]:
        kind_str = params.get("kind", ["memory"])[0]
        try:
            kind = MemoryKind(kind_str)
        except ValueError as exc:
            raise ValueError(f"unknown kind {kind_str!r}") from exc
        limit = max(1, min(500, int(params.get("limit", ["100"])[0])))
        query = (params.get("q", [""])[0] or "").strip() or None
        with self.server.brain_lock:
            memories = browse_memories(
                self.server.brain._storage, kind=kind, query=query, limit=limit
            )
        return {"memories": [_memory_to_dict(m) for m in memories]}


def _memory_to_dict(mem: Memory) -> dict[str, Any]:
    """Serialize a memory for the browser (provenance and epistemics on display)."""
    return {
        "id": mem.id,
        "text": mem.text,
        "evidence_tier": mem.evidence_tier.key,
        "confidence": round(mem.confidence, 3),
        "salience": round(mem.salience, 3),
        "provenance": mem.provenance,
        "source": mem.source,
        "user": mem.user,
        "access_count": mem.access_count,
        "created_at": mem.created_at,
    }


def serve(config_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Boot a brain from config and serve the web UI until interrupted."""
    brain = Mimir.from_config(config_path)
    server = create_server(brain, host, port)
    bound_port = server.server_address[1]  # the real port (when port=0, the OS-assigned one)
    log.info("Mimir web UI on http://%s:%s  (Ctrl-C to stop)", host, bound_port)
    print(f"Mimir web UI: http://{host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        brain.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Serve Mimir's reference web UI.")
    parser.add_argument("--config", required=True, help="path to mimir.toml")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    args = parser.parse_args(argv)
    serve(args.config, host=args.host, port=args.port)
    return 0


# The single-page UI. No framework, no build step — vanilla HTML/CSS/JS that talks to the JSON API.
_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mimir 0</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, sans-serif; background:#0e1116; color:#d7dde5; }
  header { padding:12px 20px; border-bottom:1px solid #232a35; display:flex; align-items:center; gap:14px; }
  header h1 { font-size:18px; margin:0; letter-spacing:.5px; }
  header .status { font-size:12px; color:#8a94a3; }
  main { display:grid; grid-template-columns: 1fr 360px; gap:0; height: calc(100vh - 50px); }
  #chat { display:flex; flex-direction:column; border-right:1px solid #232a35; }
  #log { flex:1; overflow-y:auto; padding:18px; }
  .msg { margin:0 0 14px; max-width:80%; }
  .msg .who { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#6f7a8a; margin-bottom:3px; }
  .msg .body { padding:10px 13px; border-radius:10px; white-space:pre-wrap; }
  .user { margin-left:auto; }
  .user .body { background:#1f6feb; color:#fff; }
  .mimir .body { background:#1a2029; }
  .meta { font-size:11px; color:#6f7a8a; margin-top:4px; }
  #composer { display:flex; gap:8px; padding:12px; border-top:1px solid #232a35; }
  #composer input[type=text] { flex:1; }
  input[type=text], textarea { background:#11161d; border:1px solid #2b333f; color:#d7dde5; border-radius:8px; padding:9px 11px; font:inherit; }
  button { background:#238636; color:#fff; border:0; border-radius:8px; padding:9px 14px; font:inherit; cursor:pointer; }
  button.secondary { background:#30363d; }
  button:disabled { opacity:.5; cursor:default; }
  aside { overflow-y:auto; padding:16px; }
  aside section { margin-bottom:26px; }
  aside h2 { font-size:13px; text-transform:uppercase; letter-spacing:.7px; color:#8a94a3; margin:0 0 10px; }
  .field { margin-bottom:9px; }
  .field label { display:block; font-size:12px; color:#9aa4b2; margin-bottom:3px; }
  .field input { width:100%; }
  .row { display:flex; gap:8px; }
  .hint { font-size:12px; color:#6f7a8a; margin-top:6px; }
  #ingestResult, #identMsg { font-size:12px; color:#7fd17f; margin-top:6px; min-height:14px; }
  .tabs { display:flex; gap:4px; margin-bottom:14px; border-bottom:1px solid #232a35; }
  .tabs button { background:none; border:0; border-bottom:2px solid transparent; color:#8a94a3; padding:7px 10px; border-radius:0; font-size:13px; }
  .tabs button.active { color:#d7dde5; border-bottom-color:#1f6feb; }
  .tabpane.hidden { display:none; }
  .selfmodel { background:#11161d; border:1px solid #232a35; border-radius:8px; padding:11px; font-size:13px; white-space:pre-wrap; color:#c3ccd8; }
  .stats { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; }
  .stat { background:#161c24; border:1px solid #232a35; border-radius:6px; padding:5px 9px; font-size:12px; }
  .stat b { color:#fff; }
  .mem { border:1px solid #232a35; border-radius:8px; padding:9px 11px; margin-bottom:8px; }
  .mem .text { font-size:13px; white-space:pre-wrap; }
  .mem .tags { margin-top:6px; display:flex; flex-wrap:wrap; gap:5px; }
  .tag { font-size:11px; padding:2px 6px; border-radius:4px; background:#1a2029; color:#9aa4b2; }
  .tag.tier { background:#16324a; color:#9fd0ff; }
  .searchrow { display:flex; gap:6px; margin-bottom:10px; }
  .searchrow input { flex:1; }
  .searchrow select { background:#11161d; border:1px solid #2b333f; color:#d7dde5; border-radius:8px; padding:0 8px; }
</style>
</head>
<body>
<header>
  <h1>Mimir 0</h1>
  <span class="status" id="status">connecting…</span>
</header>
<main>
  <div id="chat">
    <div id="log"></div>
    <form id="composer">
      <input type="text" id="text" placeholder="Say something to Mimir…" autocomplete="off"/>
      <button type="submit" id="send">Send</button>
    </form>
  </div>
  <aside>
    <div class="tabs">
      <button data-tab="identity" class="active">Identity</button>
      <button data-tab="mind">Mind</button>
      <button data-tab="memories">Memories</button>
      <button data-tab="docs">Docs</button>
    </div>

    <div class="tabpane" id="tab-identity">
      <div id="identFields"></div>
      <div class="row">
        <button class="secondary" id="reviseBtn" type="button">Revise all</button>
        <button id="saveIdent" type="button">Save</button>
      </div>
      <div id="identMsg"></div>
    </div>

    <div class="tabpane hidden" id="tab-mind">
      <h2>Self-model</h2>
      <div class="selfmodel" id="selfModel">—</div>
      <div class="stats" id="mindStats"></div>
      <h2>Recent reflections</h2>
      <div id="reflections"></div>
    </div>

    <div class="tabpane hidden" id="tab-memories">
      <div class="searchrow">
        <select id="memKind">
          <option value="memory">memories</option>
          <option value="sentinel_note">reflections</option>
          <option value="self_model">self-models</option>
        </select>
        <input type="text" id="memQuery" placeholder="search text…"/>
      </div>
      <div id="memList"></div>
    </div>

    <div class="tabpane hidden" id="tab-docs">
      <h2>Ingest a document</h2>
      <div class="field">
        <input type="text" id="docPath" placeholder="/path/to/notes.md (.txt .md .pdf)"/>
      </div>
      <button id="ingestBtn" type="button">Ingest</button>
      <div id="ingestResult"></div>
      <div class="hint">Path is read on the server (this is a local tool).</div>
    </div>
  </aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let reviseMode = false;

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

function addMsg(who, body, meta) {
  const div = document.createElement("div");
  div.className = "msg " + (who === "you" ? "user" : "mimir");
  div.innerHTML = '<div class="who"></div><div class="body"></div>';
  div.querySelector(".who").textContent = who;
  div.querySelector(".body").textContent = body;
  if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; div.appendChild(m); }
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
}

async function refreshState() {
  try {
    const s = await api("GET", "/api/state");
    $("status").textContent = `embeddings: ${s.embed_mode} · ${s.memories} memories · ${s.anchors_set}/8 anchors`;
  } catch (e) { $("status").textContent = "error: " + e.message; }
}

async function loadIdentity() {
  const data = await api("GET", "/api/identity");
  const fields = $("identFields");
  fields.innerHTML = "";
  const show = reviseMode ? null : data.pending;  // revise → show all; else only pending
  const items = reviseMode
    ? Object.keys({name:1,operator:1,location:1,purpose:1,values:1,scope:1,boundaries:1,voice:1})
        .map(k => [k, k])
    : data.pending.map(([k,q]) => [k,q]);
  if (items.length === 0) {
    fields.innerHTML = '<div class="hint">Identity established. Click “Revise all” to change it.</div>';
  }
  const qmap = {}; data.pending.forEach(([k,q]) => qmap[k]=q);
  for (const [k] of items) {
    const wrap = document.createElement("div"); wrap.className = "field";
    const cur = data.anchors[k] || "";
    wrap.innerHTML = `<label>${k}${cur ? ' — current: '+cur : ''}</label><input type="text" data-key="${k}" placeholder="${(qmap[k]||'').replace(/"/g,'')}"/>`;
    fields.appendChild(wrap);
  }
}

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("text").value.trim(); if (!text) return;
  addMsg("you", text); $("text").value = ""; $("send").disabled = true;
  try {
    const r = await api("POST", "/api/turn", { text, user: "operator" });
    const i = r.introspect || {};
    const meta = `sources: ${i.source_count} · ${i.embed_mode}` + (i.uncertainty_triggered ? " · ⚠ thin evidence" : "");
    addMsg("mimir", r.reply, meta);
  } catch (e) { addMsg("mimir", "[error] " + e.message); }
  $("send").disabled = false; $("text").focus(); refreshState();
});

$("saveIdent").addEventListener("click", async () => {
  const answers = {};
  document.querySelectorAll("#identFields input").forEach(inp => {
    if (inp.value.trim()) answers[inp.dataset.key] = inp.value.trim();
  });
  try {
    await api("POST", "/api/identity", { answers });
    $("identMsg").textContent = "Saved.";
    reviseMode = false; await loadIdentity(); refreshState();
    setTimeout(() => $("identMsg").textContent = "", 2000);
  } catch (e) { $("identMsg").textContent = "Error: " + e.message; }
});

$("reviseBtn").addEventListener("click", async () => { reviseMode = !reviseMode; $("reviseBtn").textContent = reviseMode ? "Show pending" : "Revise all"; await loadIdentity(); });

$("ingestBtn").addEventListener("click", async () => {
  const path = $("docPath").value.trim(); if (!path) return;
  $("ingestResult").textContent = "Ingesting…";
  try {
    const r = await api("POST", "/api/ingest", { path });
    $("ingestResult").textContent = `Ingested ${r.chunks_written} chunk(s) from ${r.units} unit(s)` + (r.chunks_replaced ? ` (replaced ${r.chunks_replaced})` : "");
    refreshState();
  } catch (e) { $("ingestResult").textContent = "Error: " + e.message; }
});

// --- tabs ---
const loaders = { mind: loadMind, memories: loadMemories };
document.querySelectorAll(".tabs button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p => p.classList.add("hidden"));
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
    if (loaders[btn.dataset.tab]) loaders[btn.dataset.tab]();
  });
});

async function loadMind() {
  try {
    const m = await api("GET", "/api/mind");
    $("selfModel").textContent = m.self_model || "(not yet synthesized — keep talking)";
    const s = m.stats || {};
    const tiers = Object.entries(s.by_tier || {}).map(([k,v]) => `${k}: ${v}`).join(", ") || "none";
    $("mindStats").innerHTML =
      `<div class="stat"><b>${s.total||0}</b> memories</div>` +
      `<div class="stat"><b>${s.documents||0}</b> docs</div>` +
      `<div class="stat"><b>${s.reflections||0}</b> reflections</div>` +
      `<div class="stat"><b>${s.users||0}</b> users</div>` +
      `<div class="stat">tiers — ${tiers}</div>`;
    const refl = $("reflections"); refl.innerHTML = "";
    (m.recent_reflections || []).forEach(t => {
      const d = document.createElement("div"); d.className = "mem";
      const tx = document.createElement("div"); tx.className = "text"; tx.textContent = t; d.appendChild(tx);
      refl.appendChild(d);
    });
    if (!(m.recent_reflections||[]).length) refl.innerHTML = '<div class="hint">No reflections yet.</div>';
  } catch (e) { $("selfModel").textContent = "error: " + e.message; }
}

async function loadMemories() {
  const kind = $("memKind").value;
  const q = $("memQuery").value.trim();
  try {
    const data = await api("GET", `/api/memories?kind=${encodeURIComponent(kind)}&q=${encodeURIComponent(q)}&limit=100`);
    const list = $("memList"); list.innerHTML = "";
    if (!data.memories.length) { list.innerHTML = '<div class="hint">No matching entries.</div>'; return; }
    data.memories.forEach(m => {
      const d = document.createElement("div"); d.className = "mem";
      const tx = document.createElement("div"); tx.className = "text"; tx.textContent = m.text; d.appendChild(tx);
      const tags = document.createElement("div"); tags.className = "tags";
      const add = (cls, t) => { const sp = document.createElement("span"); sp.className = "tag " + cls; sp.textContent = t; tags.appendChild(sp); };
      add("tier", m.evidence_tier);
      if (m.provenance) add("", m.provenance);
      if (m.user) add("", "user: " + m.user);
      add("", "conf " + m.confidence); add("", "sal " + m.salience);
      d.appendChild(tags); list.appendChild(d);
    });
  } catch (e) { $("memList").innerHTML = "error: " + e.message; }
}

$("memKind").addEventListener("change", loadMemories);
$("memQuery").addEventListener("input", () => { clearTimeout(window._mt); window._mt = setTimeout(loadMemories, 250); });

refreshState(); loadIdentity();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
