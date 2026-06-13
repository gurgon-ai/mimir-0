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
from logging.handlers import RotatingFileHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

from .brain import Mimir
from .cognition.self_model import gather_signals
from .cognition.working_memory import current_working_memory
from .errors import IngestError, MimirError
from .storage.models import Memory, MemoryKind, Procedure, Triple
from .storage.repo import (
    browse_memories,
    browse_triples,
    count_memories,
    count_procedures,
    count_triples,
    disabled_nodes,
    latest_self_model,
    list_procedures,
)

log = logging.getLogger("mimir.server")

# A client that closes its socket mid-response (reload, navigation, cancelled fetch). On Windows
# this surfaces as ConnectionAbortedError [WinError 10053]; elsewhere as broken-pipe/reset. Benign —
# never a server fault, so it must not log a stack trace.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

# The qualifying tournament's rounds. The first is the Qualifying round (numbered Round 0 — it
# qualifies the field before the real rounds); Round 3 (Vision) is reserved until the vision
# dimension ships. ``n`` is the internal 1-based index; ``label`` is what the user sees. Each round
# narrows the field; the user vetoes between rounds (DESIGN §4).
_TOURNEY_ROUNDS: tuple[dict[str, Any], ...] = (
    {"n": 1, "key": "qualifying", "label": "Round 0", "name": "Qualifying",
     "blurb": "Fast scoring to qualify the field — the cheap battery only, no gauntlet, nothing saved."},
    {"n": 2, "key": "gauntlet", "label": "Round 1", "name": "Framework gauntlet",
     "blurb": "The real test on the survivors: reasoning + the epistemic framework "
              "(layered tiers, grounding, long-context). Scores are saved."},
    {"n": 3, "key": "finals", "label": "Round 2", "name": "Finals",
     "blurb": "The per-role champions among your finalists — confirm, then Apply."},
)
_TOURNEY_TOTAL = 3  # active rounds (0–2); Round 3 (Vision) is reserved


class MimirHTTPServer(ThreadingHTTPServer):
    """A threading HTTP server that holds the shared brain and serializes access to it."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], brain: Mimir) -> None:
        super().__init__(address, _Handler)
        self.brain = brain
        self.brain_lock = threading.Lock()
        # Live benchmark progress (the run is multi-minute; the UI polls this so it never looks
        # frozen). Guarded by its own small lock so status reads never block on the benchmark.
        self.bench_lock = threading.Lock()
        self.bench_state: dict[str, Any] = {"running": False}
        # The qualifying tournament: a multi-round, human-veto narrowing built on the same
        # background-run + poll pattern as the benchmark. Its own lock so status reads never block.
        self.tourney_lock = threading.Lock()
        self.tourney_state: dict[str, Any] = {"active": False}
        # The "final time trial" — fills the per-node placement matrix in the background.
        self.matrix_lock = threading.Lock()
        self.matrix_state: dict[str, Any] = {"running": False}


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
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except _CLIENT_GONE:
            # The browser closed the connection before we finished writing (a reload, a navigation,
            # or a cancelled fetch). Benign — not a server fault. Don't spew a traceback.
            log.debug("client disconnected before response sent: %s", self.path)
            self.close_connection = True

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
            elif route == "/api/graph":
                self._send_json(self._graph(params))
            elif route == "/api/procedures":
                self._send_json(self._procedures())
            elif route == "/api/fleet":
                self._send_json(self._fleet())
            elif route == "/api/fleet/pool":
                self._send_json(self._model_pool())
            elif route == "/api/fleet/benchmark/status":
                self._send_json(self._benchmark_status())
            elif route == "/api/fleet/tournament/status":
                self._send_json(self._tournament_status())
            elif route == "/api/fleet/matrix/status":
                with self.server.matrix_lock:
                    self._send_json(dict(self.server.matrix_state))
            elif route == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:  # bad query params
            self._send_json({"error": str(exc)}, status=400)
        except _CLIENT_GONE:  # client vanished mid-request — benign, not a fault
            return
        except Exception as exc:  # never leak a stack to the client; log loud (DESIGN §10)
            log.exception("GET %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/turn/stream":
            self._turn_stream()  # manages its own (streaming) response
            return
        try:
            body = self._read_json()
            if route == "/api/turn":
                self._send_json(self._turn(body))
            elif route == "/api/identity":
                self._send_json(self._establish(body))
            elif route == "/api/ingest":
                self._send_json(self._ingest(body))
            elif route == "/api/sleep":
                self._send_json(self._sleep())
            elif route == "/api/council":
                self._send_json(self._council(body))
            elif route == "/api/procedures":
                self._send_json(self._learn_procedure(body))
            elif route == "/api/fleet/scan":
                self._send_json(self._scan_fleet())
            elif route == "/api/fleet/benchmark":
                self._send_json(self._benchmark_fleet(body))
            elif route == "/api/fleet/tournament/start":
                self._send_json(self._tournament_start(body))
            elif route == "/api/fleet/tournament/advance":
                self._send_json(self._tournament_advance(body))
            elif route == "/api/fleet/tournament/apply":
                self._send_json(self._tournament_apply())
            elif route == "/api/fleet/matrix":
                self._send_json(self._matrix_start())
            elif route == "/api/fleet/apply":
                self._send_json({"applied": self._apply_recommendations()})
            elif route == "/api/fleet/model":
                self._send_json(self._set_model_enabled(body))
            elif route == "/api/fleet/node":
                self._send_json(self._set_node_enabled(body))
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
        except _CLIENT_GONE:  # client vanished mid-request — benign, not a fault
            return
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)

    # -- operations (all under the brain lock) ----------------------------------------

    def _state(self) -> dict[str, Any]:
        # Lock-free: these are concurrent-safe reads (the storage gateway handles read concurrency
        # via WAL). Not taking the brain lock keeps the header live even while a benchmark holds it.
        brain = self.server.brain
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

    def _turn_stream(self) -> None:
        """Server-Sent-Events stream of a turn: token events, then a done event with introspect.

        Manages its own response (it can't use ``_send_json``). Validation errors are sent as a
        JSON 400 *before* the stream starts; once streaming, a failure is sent as an error event.
        """
        try:
            body = self._read_json()
            text = str(body.get("text", "")).strip()
            if not text:
                self._send_json({"error": "'text' is required"}, status=400)
                return
            user = body.get("user") or None
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event: str, data: Any) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        try:
            with self.server.brain_lock:
                stream = self.server.brain.turn_stream(text, user=user)
                while True:
                    try:
                        token = next(stream)
                    except StopIteration as stop:
                        self.server.brain.wait_for_sentinel()
                        emit("done", {"introspect": stop.value or {}})
                        break
                    emit("token", {"text": token})
        except _CLIENT_GONE:
            return  # the client went away mid-stream; the turn is interrupted
        except Exception as exc:
            log.exception("turn stream failed")
            try:
                emit("error", {"error": str(exc)})
            except OSError:
                pass

    def _fleet(self) -> dict[str, Any]:
        with self.server.brain_lock:
            report = self.server.brain.fleet_report()
            stats = self.server.brain._model.get_stats()
            be = self.server.brain.config.backend
            disabled = disabled_nodes(self.server.brain._storage)
        report["stats"] = stats
        report["max_model_size_b"] = be.max_model_size_b if be else 30.0
        report["min_model_size_b"] = be.min_model_size_b if be else 0.0
        report["max_latency_s"] = be.max_latency_s if be else 0.0
        report["disabled_nodes"] = sorted(disabled)
        return report

    def _scan_fleet(self) -> dict[str, Any]:
        with self.server.brain_lock:
            result = self.server.brain.scan_fleet()
        return {"nodes": result.nodes, "models": result.models}

    def _benchmark_fleet(self, body: dict[str, Any]) -> dict[str, Any]:
        """Kick off a benchmark in the background and return immediately. The run is multi-minute;
        the UI polls /api/fleet/benchmark/status. Holding the brain lock for the whole run (in the
        worker thread) preserves serialization against turns, but the status + state reads are
        lock-free, so the page never freezes (the bug this fixes).

        Optional body fields ``max_model_size_b`` / ``min_model_size_b`` / ``max_latency_s`` (the UI
        scope fields) override the configured cap/floor/latency for this run."""
        srv = self.server
        cap = float(body["max_model_size_b"]) if body.get("max_model_size_b") not in (None, "") else None
        floor = float(body["min_model_size_b"]) if body.get("min_model_size_b") not in (None, "") else None
        latency = float(body["max_latency_s"]) if body.get("max_latency_s") not in (None, "") else None
        with srv.bench_lock:
            if srv.bench_state.get("running"):
                return {"started": False, **srv.bench_state}  # already running
            srv.bench_state = {"running": True, "i": 0, "total": 0,
                               "current": "scanning the fleet…", "done": False, "results": []}

        def _progress(i: int, total: int, model: str, eta: float | None) -> None:
            with srv.bench_lock:
                srv.bench_state.update(i=i, total=total, current=model, eta=eta)

        def _on_result(b: Any, node: str) -> None:
            with srv.bench_lock:
                srv.bench_state.setdefault("results", []).append({
                    "model": b.model, "quality": b.quality, "talk": b.talk, "tools": b.tools,
                    "code": b.code, "discipline": b.discipline, "epistemics": b.epistemics,
                    "reasoning": b.reasoning, "coherence": b.coherence,
                    "return_time": b.return_time, "node": node,
                })

        def _run() -> None:
            try:
                with srv.brain_lock:
                    result = srv.brain.benchmark_fleet(
                        max_params_b=cap, min_params_b=floor, latency_budget_s=latency,
                        progress=_progress, on_result=_on_result,
                    )
                with srv.bench_lock:
                    srv.bench_state.update(
                        running=False, done=True, current="",
                        benchmarked=result.benchmarked, judges_ok=result.judges_ok,
                        eligible=result.eligible, skipped_too_big=result.skipped_too_big,
                        skipped_too_small=result.skipped_too_small,
                        skipped_too_slow=result.skipped_too_slow,
                    )
            except Exception as exc:  # surfaced via status, never a silent death (DESIGN §10)
                log.exception("benchmark run failed")
                with srv.bench_lock:
                    srv.bench_state.update(running=False, done=True, error=str(exc))

        threading.Thread(target=_run, name="mimir-benchmark", daemon=True).start()
        return {"started": True}

    def _benchmark_status(self) -> dict[str, Any]:
        with self.server.bench_lock:
            return dict(self.server.bench_state)

    # -- qualifying tournament (multi-round, human-veto) ------------------------------

    def _tournament_status(self) -> dict[str, Any]:
        with self.server.tourney_lock:
            return dict(self.server.tourney_state)

    def _matrix_start(self) -> dict[str, Any]:
        """Kick off the final time trial in the background: speed-test acceptable models on the
        enabled nodes they're installed on but not yet timed on. UI polls /api/fleet/matrix/status."""
        srv = self.server
        with srv.matrix_lock:
            if srv.matrix_state.get("running"):
                return {"started": False, **srv.matrix_state}
            srv.matrix_state = {"running": True, "i": 0, "total": 0, "current": "scanning…",
                                "done": False}

        def _progress(i: int, total: int, what: str) -> None:
            with srv.matrix_lock:
                srv.matrix_state.update(i=i, total=total, current=what)

        def _run() -> None:
            try:
                with srv.brain_lock:
                    timed = srv.brain.complete_speed_matrix(progress=_progress)
                with srv.matrix_lock:
                    srv.matrix_state.update(running=False, done=True, current="", timed=timed)
            except Exception as exc:   # surfaced via status, never a silent death (DESIGN §10)
                log.exception("speed-matrix time trial failed")
                with srv.matrix_lock:
                    srv.matrix_state.update(running=False, done=True, error=str(exc))

        threading.Thread(target=_run, name="mimir-matrix", daemon=True).start()
        return {"started": True}

    def _tournament_start(self, body: dict[str, Any]) -> dict[str, Any]:
        """Begin the tournament at Round 1 (triage). Scope caps (size floor/ceiling, latency) are
        captured once here and reused for every round. Runs in the background; the UI polls status."""
        srv = self.server

        def _f(v: Any) -> float | None:
            return float(v) if v not in (None, "") else None

        with srv.tourney_lock:
            if srv.tourney_state.get("active") and srv.tourney_state.get("phase") == "running":
                return {"started": False, **srv.tourney_state}  # a round is already running
            srv.tourney_state = {"active": True, "scope": {
                "max_model_size_b": _f(body.get("max_model_size_b")),
                "min_model_size_b": _f(body.get("min_model_size_b")),
                "max_latency_s": _f(body.get("max_latency_s")),
            }}
        self._start_round(1, None)
        return {"started": True}

    def _tournament_advance(self, body: dict[str, Any]) -> dict[str, Any]:
        """The 'FIGHT' button: record the survivors the user kept and start the next round. Round 1→2
        re-benchmarks the survivors through the full gauntlet; Round 2→3 computes the finals."""
        srv = self.server
        keep = {m for m in (body.get("keep") or []) if isinstance(m, str)}
        with srv.tourney_lock:
            st = srv.tourney_state
            if not st.get("active") or st.get("phase") != "awaiting_veto":
                return {"advanced": False, "error": "no round is awaiting a decision"}
            cur = int(st.get("round", 0))
        if not keep:
            return {"advanced": False, "error": "keep at least one model to continue"}
        if cur == 1:
            self._start_round(2, keep)
            return {"advanced": True}
        if cur == 2:
            with srv.brain_lock:
                recs = srv.brain.tournament_finals(keep)
            with srv.tourney_lock:
                prev = [r for r in srv.tourney_state.get("results", []) if r["model"] in keep]
                meta = _TOURNEY_ROUNDS[2]
                srv.tourney_state.update(
                    round=3, round_name=meta["name"], round_key="finals",
                    round_label=meta["label"], blurb=meta["blurb"],
                    phase="done", current="", recommendations=recs, results=prev,
                    finalists=sorted(keep),
                )
            return {"advanced": True}
        return {"advanced": False, "error": "the tournament is already at the finals"}

    def _tournament_apply(self) -> dict[str, Any]:
        """Apply the finals: re-point roles to the champions among the kept finalists."""
        srv = self.server
        with srv.tourney_lock:
            finalists = {m for m in (srv.tourney_state.get("finalists") or []) if isinstance(m, str)}
        if not finalists:
            return {"applied": {}}
        with srv.brain_lock:
            return {"applied": srv.brain.apply_finals(finalists)}

    def _start_round(self, round_num: int, keep: set[str] | None) -> None:
        """Run one tournament round in a background thread (mirrors the benchmark worker exactly).
        Round 1 is triage (cheap + ephemeral); later rounds run the full gauntlet on the survivors
        and persist. On completion the round parks in ``awaiting_veto`` for the user's decision."""
        srv = self.server
        meta = _TOURNEY_ROUNDS[round_num - 1]
        triage = round_num == 1
        with srv.tourney_lock:
            scope = dict(srv.tourney_state.get("scope", {}))
            srv.tourney_state.update(
                active=True, round=round_num, round_name=meta["name"], round_key=meta["key"],
                round_label=meta["label"], blurb=meta["blurb"], total_rounds=_TOURNEY_TOTAL,
                phase="running", i=0, total=0, current="scanning the fleet…", eta=None,
                results=[], error=None, recommendations=None,
            )

        def _progress(i: int, total: int, model: str, eta: float | None) -> None:
            with srv.tourney_lock:
                srv.tourney_state.update(i=i, total=total, current=model, eta=eta)

        def _on_result(b: Any, node: str) -> None:
            with srv.tourney_lock:
                srv.tourney_state.setdefault("results", []).append({
                    "model": b.model, "quality": b.quality, "talk": b.talk, "tools": b.tools,
                    "code": b.code, "discipline": b.discipline, "epistemics": b.epistemics,
                    "reasoning": b.reasoning, "coherence": b.coherence,
                    "return_time": b.return_time, "node": node,
                })

        def _run() -> None:
            try:
                with srv.brain_lock:
                    srv.brain.benchmark_fleet(
                        # The tournament IS the qualification, so it considers EVERY reachable model
                        # (not just approved families — that heuristic is for the pre-benchmark guess);
                        # the user prunes by veto between rounds.
                        only_approved=False,
                        max_params_b=scope.get("max_model_size_b"),
                        min_params_b=scope.get("min_model_size_b"),
                        latency_budget_s=scope.get("max_latency_s"),
                        only_models=keep, framework=not triage, persist=not triage,
                        judge=not triage, progress=_progress, on_result=_on_result,
                    )
                with srv.tourney_lock:
                    srv.tourney_state.update(phase="awaiting_veto", current="")
            except Exception as exc:  # surfaced via status, never a silent death (DESIGN §10)
                log.exception("tournament round %d failed", round_num)
                with srv.tourney_lock:
                    srv.tourney_state.update(phase="error", error=str(exc))

        threading.Thread(target=_run, name=f"mimir-tourney-r{round_num}", daemon=True).start()

    def _apply_recommendations(self) -> dict[str, str]:
        with self.server.brain_lock:
            return self.server.brain.apply_recommendations()

    def _model_pool(self) -> dict[str, Any]:
        with self.server.brain_lock:
            brain = self.server.brain
            pool = brain.model_pool()
            pool["lan_backend"] = brain.config.backend is not None
            pool["nodes_up"] = brain._model.get_stats().get("nodes_up", 0)
            be = brain.config.backend
            pool["max_model_size_b"] = be.max_model_size_b if be else 30.0
            pool["min_model_size_b"] = be.min_model_size_b if be else 0.0
            pool["max_latency_s"] = be.max_latency_s if be else 0.0
        return pool

    def _set_model_enabled(self, body: dict[str, Any]) -> dict[str, Any]:
        model = str(body.get("model", "")).strip()
        if not model:
            raise ValueError("model is required")
        enabled = bool(body.get("enabled", True))
        with self.server.brain_lock:
            moved = self.server.brain.set_model_enabled(model, enabled)
        return {"model": model, "enabled": enabled, "moved": moved}

    def _set_node_enabled(self, body: dict[str, Any]) -> dict[str, Any]:
        node = str(body.get("node", "")).strip()
        if not node:
            raise ValueError("node is required")
        enabled = bool(body.get("enabled", True))
        with self.server.brain_lock:
            moved = self.server.brain.set_node_enabled(node, enabled)
        return {"node": node, "enabled": enabled, "moved": moved}

    def _procedures(self) -> dict[str, Any]:
        with self.server.brain_lock:
            procs = list_procedures(self.server.brain._storage, limit=200)
        return {"procedures": [_procedure_to_dict(p) for p in procs]}

    def _learn_procedure(self, body: dict[str, Any]) -> dict[str, Any]:
        trigger = str(body.get("trigger", "")).strip()
        procedure = str(body.get("procedure", "")).strip()
        if not trigger or not procedure:
            raise ValueError("'trigger' and 'procedure' are both required")
        with self.server.brain_lock:
            proc = self.server.brain.learn_procedure(trigger, procedure)
        return _procedure_to_dict(proc)

    def _council(self, body: dict[str, Any]) -> dict[str, Any]:
        question = str(body.get("question", "")).strip()
        if not question:
            raise ValueError("'question' is required")
        with self.server.brain_lock:
            result = self.server.brain.deliberate(question)
        return {
            "question": result.question,
            "verdict": result.verdict,
            "positions": [
                {"persona": p.persona, "model": p.model, "text": p.text} for p in result.positions
            ],
        }

    def _sleep(self) -> dict[str, Any]:
        with self.server.brain_lock:
            report = self.server.brain.sleep()
        return {
            "deduped": report.deduped,
            "decayed": report.decayed,
            "archived": report.archived,
            "contradictions_resolved": report.contradictions_resolved,
            "total_changes": report.total_changes,
        }

    def _mind(self) -> dict[str, Any]:
        brain = self.server.brain
        with self.server.brain_lock:
            signals = gather_signals(brain._storage)
            self_model = latest_self_model(brain._storage)
            anchors = brain.identity_anchors()
            working_memory = current_working_memory(brain._storage)
        return {
            "self_model": self_model.text if self_model else None,
            "working_memory": working_memory,
            "anchors": anchors,
            "stats": {
                "total": signals.total_memories,
                "documents": signals.documents,
                "reflections": signals.reflections,
                "users": signals.distinct_users,
                "triples": count_triples(brain._storage),
                "procedures": count_procedures(brain._storage),
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

    def _graph(self, params: dict[str, list[str]]) -> dict[str, Any]:
        limit = max(1, min(500, int(params.get("limit", ["100"])[0])))
        query = (params.get("q", [""])[0] or "").strip() or None
        with self.server.brain_lock:
            triples = browse_triples(self.server.brain._storage, query=query, limit=limit)
        return {"triples": [_triple_to_dict(t) for t in triples]}


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
        "archived": mem.archived,
        "created_at": mem.created_at,
    }


def _procedure_to_dict(proc: Procedure) -> dict[str, Any]:
    return {
        "id": proc.id,
        "trigger": proc.trigger,
        "procedure": proc.procedure,
        "uses": proc.uses,
        "confidence": round(proc.confidence, 3),
        "user": proc.user,
    }


def _triple_to_dict(triple: Triple) -> dict[str, Any]:
    return {
        "subject": triple.subject,
        "relation": triple.relation,
        "object": triple.object,
        "confidence": round(triple.confidence, 3),
        "user": triple.user,
        "provenance": triple.provenance,
    }


def serve(config_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Boot a brain from config and serve the web UI until interrupted."""
    print(f"Starting Mimir from {config_path} …")
    print("(scanning the LAN for Ollama nodes can take a couple of seconds)", flush=True)
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


def _setup_logging(log_file: str | None) -> None:
    """Console logging always; a rotating file too (default ``mimir.log``) so long runs leave a
    reviewable trail — observability is the doctrine, and a vanished console is not observable.

    ``--log-file ""`` (empty) disables the file and logs to console only.
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    path = "mimir.log" if log_file is None else log_file
    if path:
        # 5 MB × 3 backups: bounded disk, weeks of normal use, never silently truncates a live run.
        fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    if path:
        log.info("logging to console and %s (rotating, 5MB x 3)", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Mimir's reference web UI.")
    parser.add_argument("--config", required=True, help="path to mimir.toml")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    parser.add_argument(
        "--log-file",
        default=None,
        help="rotating log file path (default: mimir.log; pass '' to log to console only)",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.log_file)
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
  button.working { background:#9e6a03; }   /* amber — in progress */
  button.done    { background:#1f7a37; }   /* green — completed */
  button.failed  { background:#b62324; }   /* red — errored */
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
  .tabs { display:flex; flex-wrap:wrap; gap:2px 4px; margin-bottom:14px; border-bottom:1px solid #232a35; }
  .tabs button { background:none; border:0; border-bottom:2px solid transparent; color:#8a94a3; padding:7px 9px; border-radius:0; font-size:13px; white-space:nowrap; }
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
  #benchBoard h2 { font-size:16px; margin:0 0 4px; display:flex; align-items:center; gap:10px; }
  #benchBoard table { width:100%; border-collapse:collapse; font-size:13px; margin-top:10px; }
  #benchBoard th { text-align:left; color:#8a94a3; font-weight:600; padding:7px 10px; border-bottom:1px solid #232a35; position:sticky; top:-18px; background:#0e1116; }
  #benchBoard td { padding:7px 10px; border-bottom:1px solid #1a2029; white-space:nowrap; }
  #benchBoard tr.top td { background:#13241a; }
  #benchBoard td.q { font-weight:700; color:#d7dde5; }
  #benchBoard .legend { font-size:12px; color:#6f7a8a; }
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
    <div id="benchBoard" style="display:none; flex:1; overflow:auto; padding:18px;"></div>
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
      <button data-tab="graph">Graph</button>
      <button data-tab="procedures">Habits</button>
      <button data-tab="council">Council</button>
      <button data-tab="fleet">Fleet</button>
      <button data-tab="models">Models</button>
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
      <button class="secondary" id="sleepBtn" type="button">Consolidate now</button>
      <div id="sleepResult" class="hint"></div>
      <h2>Working memory</h2>
      <div class="selfmodel" id="workingMemory">—</div>
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

    <div class="tabpane hidden" id="tab-graph">
      <div class="searchrow">
        <input type="text" id="graphQuery" placeholder="search entities / relations…"/>
      </div>
      <div id="graphList"></div>
    </div>

    <div class="tabpane hidden" id="tab-procedures">
      <div class="field"><label>When… (trigger)</label><input type="text" id="procTrigger" placeholder="the user asks for a summary"/></div>
      <div class="field"><label>…do this (procedure)</label><input type="text" id="procBody" placeholder="give 3 bullet points, then a one-line takeaway"/></div>
      <button id="procBtn" type="button">Teach habit</button>
      <div id="procMsg" class="hint"></div>
      <div id="procList"></div>
    </div>

    <div class="tabpane hidden" id="tab-council">
      <div class="field">
        <input type="text" id="councilQ" placeholder="Pose an open question for the council…"/>
      </div>
      <button id="councilBtn" type="button">Deliberate</button>
      <div class="hint">Convenes adversarial personas across your installed models — may take a while.</div>
      <h2>Verdict</h2>
      <div class="selfmodel" id="councilVerdict">—</div>
      <h2>Positions</h2>
      <div id="councilPositions"></div>
    </div>

    <div class="tabpane hidden" id="tab-fleet">
      <div class="row">
        <button id="fleetTourneyBtn" type="button" title="The recommended path. A staged knock-out: Round 0 qualifying (fast) → you pick survivors → Round 1 the full framework gauntlet → Round 2 finals. You choose who advances between rounds.">🏆 Run qualifying tournament</button>
      </div>
      <div class="hint" style="margin-top:6px;">The recommended way to qualify your fleet — narrows it in rounds: <b>Round 0 · Qualifying</b> (fast) → <b>🥊 you keep the survivors</b> → <b>Round 1 · Framework gauntlet</b> (the real test) → <b>Round 2 · Finals</b>. (Round 3 · Vision is reserved.) The scoreboard takes over the chat pane.</div>
      <div class="row" style="margin-top:10px;">
        <button class="secondary" id="fleetMatrixBtn" type="button" title="The final time trial: speed-test each qualified model on every node it's installed on but not yet timed, so we know which edge can run what (the background-worker map). Records even slow results.">⏱ Speed-test remaining nodes</button>
      </div>
      <div class="hint" style="margin-top:6px;">After qualifying, this fills the <b>placement matrix</b> — times every qualified model on the nodes it lives on but wasn't timed on, so you learn which edges can host which models for background/council work (slow is fine there). Per-node times then fill in below.</div>
      <div class="hint" style="margin-top:14px; opacity:0.8;">— or do it manually, one step at a time —</div>
      <div class="row" style="margin-top:6px;">
        <button class="secondary" id="fleetScanBtn" type="button" title="List what models are installed on each node. Fast — runs no models.">1 · Find models</button>
        <button class="secondary" id="fleetBenchBtn" type="button" title="Run each model through the test battery to score it. Slow — this is the expensive step.">2 · Benchmark (score)</button>
        <button class="secondary" id="fleetApplyBtn" type="button" title="Point each role at its top-scoring model from the benchmark.">3 · Apply best</button>
      </div>
      <div class="hint" style="margin-top:6px;"><b>Find</b> lists installed models (fast, no scoring) → <b>Benchmark</b> scores them all at once (slow) → <b>Apply</b> routes each role to the best.</div>
      <div class="row" style="margin-top:8px; align-items:center; gap:14px; flex-wrap:wrap;">
        <label class="hint" style="display:flex; align-items:center; gap:6px;">Benchmark — min model size (B)
          <input type="number" id="benchMinSize" min="0" step="1" autocomplete="off" style="width:70px;"/></label>
        <label class="hint" style="display:flex; align-items:center; gap:6px;">max model size (B)
          <input type="number" id="benchMaxSize" min="0" step="1" autocomplete="off" style="width:70px;"/></label>
        <label class="hint" style="display:flex; align-items:center; gap:6px;">max latency (s)
          <input type="number" id="benchMaxLatency" min="0" step="0.5" autocomplete="off" style="width:70px;"/></label>
        <span class="hint">skips models smaller than the min (0 = off), bigger than the max, or slower than the latency (0 = 30s default)</span>
      </div>
      <div id="fleetMsg" class="hint">Press <b>1 · Find models</b> to inventory the fleet, then <b>2 · Benchmark</b> to score them.</div>
      <div id="fleetList"></div>
    </div>

    <div class="tabpane hidden" id="tab-models">
      <div id="poolBackend" class="hint"></div>
      <div id="poolMsg" class="hint">Models Mimir can route to. A checked box keeps a model in the automatic pool; uncheck to exclude it. ✓ = passed the qualification gate.</div>
      <div id="poolList"></div>
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
  return div;
}

async function streamTurn(text) {
  const bubble = addMsg("mimir", "");
  const body = bubble.querySelector(".body");
  const resp = await fetch("/api/turn/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, user: "operator" }) });
  if (!resp.ok) { const e = await resp.json().catch(() => ({ error: "HTTP " + resp.status })); throw new Error(e.error); }
  const reader = resp.body.getReader(); const dec = new TextDecoder();
  let buf = "", introspect = null;
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\\n\\n")) >= 0) {
      const evt = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let ev = "message", data = "";
      evt.split("\\n").forEach(l => { if (l.startsWith("event:")) ev = l.slice(6).trim(); else if (l.startsWith("data:")) data += l.slice(5).trim(); });
      if (!data) continue;
      let obj; try { obj = JSON.parse(data); } catch (_) { continue; }
      if (ev === "token") { body.textContent += obj.text; $("log").scrollTop = $("log").scrollHeight; }
      else if (ev === "done") { introspect = obj.introspect; }
      else if (ev === "error") { body.textContent += (body.textContent ? "\\n" : "") + "[error] " + obj.error; }
    }
  }
  if (introspect) {
    const m = document.createElement("div"); m.className = "meta";
    m.textContent = `sources: ${introspect.source_count} · ${introspect.embed_mode}` + (introspect.uncertainty_triggered ? " · ⚠ thin evidence" : "");
    bubble.appendChild(m);
  }
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
const loaders = { mind: loadMind, memories: loadMemories, graph: loadGraph, procedures: loadProcedures, fleet: loadFleet, models: loadModels };
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
    $("workingMemory").textContent = m.working_memory || "(empty)";
    const s = m.stats || {};
    const tiers = Object.entries(s.by_tier || {}).map(([k,v]) => `${k}: ${v}`).join(", ") || "none";
    $("mindStats").innerHTML =
      `<div class="stat"><b>${s.total||0}</b> memories</div>` +
      `<div class="stat"><b>${s.documents||0}</b> docs</div>` +
      `<div class="stat"><b>${s.reflections||0}</b> reflections</div>` +
      `<div class="stat"><b>${s.users||0}</b> users</div>` +
      `<div class="stat"><b>${s.triples||0}</b> connections</div>` +
      `<div class="stat"><b>${s.procedures||0}</b> habits</div>` +
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
      if (m.archived) add("", "archived");
      add("", "conf " + m.confidence); add("", "sal " + m.salience);
      d.appendChild(tags); list.appendChild(d);
    });
  } catch (e) { $("memList").innerHTML = "error: " + e.message; }
}

async function loadGraph() {
  const q = $("graphQuery").value.trim();
  try {
    const data = await api("GET", `/api/graph?q=${encodeURIComponent(q)}&limit=200`);
    const list = $("graphList"); list.innerHTML = "";
    if (!data.triples.length) { list.innerHTML = '<div class="hint">No connections yet — they form as you talk.</div>'; return; }
    data.triples.forEach(t => {
      const d = document.createElement("div"); d.className = "mem";
      const tx = document.createElement("div"); tx.className = "text"; tx.textContent = `${t.subject}  —  ${t.relation}  →  ${t.object}`; d.appendChild(tx);
      const tags = document.createElement("div"); tags.className = "tags";
      const add = (cls, x) => { const sp = document.createElement("span"); sp.className = "tag " + cls; sp.textContent = x; tags.appendChild(sp); };
      if (t.user) add("", "user: " + t.user);
      add("", "conf " + t.confidence);
      d.appendChild(tags); list.appendChild(d);
    });
  } catch (e) { $("graphList").innerHTML = "error: " + e.message; }
}

$("sleepBtn").addEventListener("click", async () => {
  $("sleepResult").textContent = "Consolidating…";
  try {
    const r = await api("POST", "/api/sleep");
    $("sleepResult").textContent = `Deduped ${r.deduped} · decayed ${r.decayed} · archived ${r.archived} · contradictions ${r.contradictions_resolved}.`;
    loadMind(); refreshState();
  } catch (e) { $("sleepResult").textContent = "Error: " + e.message; }
});

$("memKind").addEventListener("change", loadMemories);
$("memQuery").addEventListener("input", () => { clearTimeout(window._mt); window._mt = setTimeout(loadMemories, 250); });
$("graphQuery").addEventListener("input", () => { clearTimeout(window._gt); window._gt = setTimeout(loadGraph, 250); });

async function loadProcedures() {
  try {
    const data = await api("GET", "/api/procedures");
    const list = $("procList"); list.innerHTML = "";
    if (!data.procedures.length) { list.innerHTML = '<div class="hint">No habits taught yet.</div>'; return; }
    data.procedures.forEach(p => {
      const d = document.createElement("div"); d.className = "mem";
      const tx = document.createElement("div"); tx.className = "text"; tx.textContent = `When ${p.trigger}: ${p.procedure}`; d.appendChild(tx);
      const tags = document.createElement("div"); tags.className = "tags";
      const sp = document.createElement("span"); sp.className = "tag"; sp.textContent = "used " + p.uses + "×"; tags.appendChild(sp);
      d.appendChild(tags); list.appendChild(d);
    });
  } catch (e) { $("procList").innerHTML = "error: " + e.message; }
}

async function loadFleet() {
  try {
    const data = await api("GET", "/api/fleet");
    const up = (data.stats && data.stats.nodes_up) || 0;
    // Pre-fill the benchmark scope fields from the current config (only fill if untouched).
    if ($("benchMinSize") && document.activeElement !== $("benchMinSize") && !$("benchMinSize").value) $("benchMinSize").value = data.min_model_size_b ?? 0;
    if ($("benchMaxSize") && document.activeElement !== $("benchMaxSize") && !$("benchMaxSize").value) $("benchMaxSize").value = data.max_model_size_b ?? 30;
    if ($("benchMaxLatency") && document.activeElement !== $("benchMaxLatency") && !$("benchMaxLatency").value) $("benchMaxLatency").value = data.max_latency_s ?? 0;
    $("fleetMsg").textContent = `${data.nodes} node(s), ${up} up, ${data.models} models found. "2 · Benchmark" to score them.`;
    const list = $("fleetList"); list.innerHTML = "";
    const recs = data.recommendations || {};
    const recRoles = Object.entries(recs).filter(([_r, v]) => v);
    if (recRoles.length) {
      const box = document.createElement("div"); box.className = "selfmodel";
      box.innerHTML = "<b>Recommendations</b>";
      recRoles.forEach(([role, r]) => {
        const line = document.createElement("div"); line.style.marginTop = "5px";
        const q = (r.quality != null) ? `q${r.quality}` : "";
        const t = (r.return_time != null) ? ` · ${r.return_time}s` : "";
        line.textContent = `${role} → ${r.model} (${q}${t}, prefers ${r.prefer}) on ${r.node}`;
        box.appendChild(line);
      });
      list.appendChild(box);
    }
    const offNodes = new Set(data.disabled_nodes || []);
    Object.entries(data.by_node || {}).forEach(([node, models]) => {
      const off = offNodes.has(node);
      const d = document.createElement("div"); d.className = "mem"; if (off) d.style.opacity = "0.5";
      const head = document.createElement("label"); head.style.display = "flex"; head.style.alignItems = "center"; head.style.gap = "8px"; head.style.cursor = "pointer";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !off;
      cb.title = "Use this node in the fleet (qualification + routing). Untick to exclude it even if reachable.";
      cb.addEventListener("change", () => setNodeEnabled(node, cb.checked));
      const h = document.createElement("span"); h.className = "text"; h.textContent = `${shortNode(node)}  (${models.length} models)${off ? " — disabled" : ""}`;
      head.appendChild(cb); head.appendChild(h); d.appendChild(head);
      const tags = document.createElement("div"); tags.className = "tags";
      models.slice(0, 12).forEach(mm => { const sp = document.createElement("span"); sp.className = "tag"; const q = (mm.quality != null) ? ` · q${mm.quality}` : ""; const t = (mm.return_time != null) ? ` · ${mm.return_time}s` : ""; sp.textContent = `${mm.model} ${mm.params_b}B${q}${t}`; tags.appendChild(sp); });
      d.appendChild(tags); list.appendChild(d);
    });
    if (!Object.keys(data.by_node || {}).length) list.innerHTML = '<div class="hint">No models yet — click "1 · Find models".</div>';
    resumeFleetWork();
  } catch (e) { $("fleetList").innerHTML = "error: " + e.message; }
}

// Re-attach the UI to any benchmark/tournament still running server-side. Called on page load (so a
// refresh never loses the board) AND when the Fleet tab opens. Background runs survive a refresh —
// this just reconnects the view, so you always know whether something is still going.
async function resumeFleetWork() {
  try {
    const ms = await api("GET", "/api/fleet/matrix/status");
    if (ms.running) pollMatrix();
    const bs = await api("GET", "/api/fleet/benchmark/status");
    if (bs.running) { _benchBoardClosed = false; pollBenchmark(); return; }
    const ts = await api("GET", "/api/fleet/tournament/status");
    if (!ts.active) return;
    _benchBoardClosed = false;
    if (ts.phase === "running") pollTournament();   // re-attach the live poll
    else renderTourney(ts);                          // awaiting_veto / done / error → re-show it
  } catch (e) { /* fleet endpoints unavailable (no backend) — nothing to resume */ }
}

function fmtDuration(sec) {
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60), mm = m % 60;
  return mm ? `${h}h ${mm}m` : `${h}h`;
}

// Live benchmark scoreboard — takes over the (idle) chat pane for full width, best model first,
// with emoji status so you can scan pass/fail at a glance. User can close it to get the chat back.
let _benchBoardClosed = false;
function benchShow(on) {
  $("benchBoard").style.display = on ? "block" : "none";
  $("log").style.display = on ? "none" : "";
  $("composer").style.display = on ? "none" : "";
}
function closeBench() { _benchBoardClosed = true; benchShow(false); }
function _emoji(v) { return v == null ? "·" : v >= 0.8 ? "✅" : v >= 0.5 ? "🟡" : "❌"; }
function _medal(i) { return i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : (i + 1) + "."; }
function _stars(q) { const n = Math.max(0, Math.min(5, Math.round((q || 0) * 5))); return "★".repeat(n) + "☆".repeat(5 - n); }

function shortNode(n) {
  if (!n) return "❓ unknown node";
  const m = n.replace("http://", "").replace(":11434", "");
  return (m === "127.0.0.1" || m === "localhost") ? "🖥️ localhost (this machine)" : "🌐 " + m;
}

function renderBenchResults(results, header) {
  if (_benchBoardClosed) return;
  benchShow(true);
  const all = (results || []).slice();
  const best = all.slice().sort((a, b) => (b.quality || 0) - (a.quality || 0))[0];
  let h = `<h2>${header || "🏁 Benchmarking…"} <button class="secondary" style="margin-left:auto; padding:4px 10px;" onclick="closeBench()">✕ Close</button></h2>`;
  h += '<div class="legend">✅ ≥ 0.80 · 🟡 0.50–0.79 · ❌ &lt; 0.50 &nbsp;|&nbsp; ★ = quality &nbsp;|&nbsp; grouped by node</div>';
  if (!all.length) { $("benchBoard").innerHTML = h + '<div class="hint" style="margin-top:12px;">Warming up the first model…</div>'; return; }
  // group by node so it's obvious which machine each model lives on
  const groups = {};
  all.forEach(r => { (groups[r.node || ""] = groups[r.node || ""] || []).push(r); });
  const order = Object.keys(groups).sort((a, b) => {
    const la = a.includes("127.0.0.1"), lb = b.includes("127.0.0.1");
    if (la !== lb) return la ? -1 : 1;   // localhost group first
    return a.localeCompare(b);
  });
  h += "<table><tr><th></th><th>Model</th><th>Quality</th><th>Talk</th><th>Tools</th><th>Code</th><th>Reason</th><th>Discipline</th><th>Epistemics</th><th>Coherence</th><th>Speed/turn</th></tr>";
  order.forEach(node => {
    const rows = groups[node].sort((a, b) => (b.quality || 0) - (a.quality || 0));
    h += `<tr><td colspan="11" class="nodehdr">${shortNode(node)} · ${rows.length} model(s)</td></tr>`;
    rows.forEach((r, i) => {
      const rank = (best && r.model === best.model) ? "🏆" : _medal(i);
      h += `<tr class="${i === 0 ? "top" : ""}"><td>${rank}</td><td>${r.model}</td>`
        + `<td class="q">${_stars(r.quality)} <span style="color:#8a94a3; font-weight:400;">${(r.quality ?? 0).toFixed(2)}</span></td>`
        + `<td>${_emoji(r.talk)}</td><td>${_emoji(r.tools)}</td><td>${_emoji(r.code)}</td><td>${_emoji(r.reasoning)}</td><td>${_emoji(r.discipline)}</td><td>${_emoji(r.epistemics)}</td><td>${_emoji(r.coherence)}</td>`
        + `<td>${r.return_time != null ? r.return_time.toFixed(1) + "s" : "·"}</td></tr>`;
    });
  });
  $("benchBoard").innerHTML = h + "</table>";
}

// -- the qualifying tournament: same board, plus round chrome + keep-checkboxes + the FIGHT button.
function renderFinals(recs) {
  let h = '<div class="selfmodel" style="margin:8px 0;"><b>🏆 Finals — your champions</b>';
  const roles = Object.entries(recs || {}).filter(([_r, v]) => v);
  if (!roles.length) h += '<div class="hint" style="margin-top:6px;">No finalist cleared a role\\'s gate — keep more models or re-run.</div>';
  roles.forEach(([role, r]) => {
    const q = (r.quality != null) ? `q${r.quality}` : "";
    const t = (r.return_time != null) ? ` · ${r.return_time}s` : "";
    h += `<div style="margin-top:5px;">${role} → <b>${r.model}</b> (${q}${t}) on ${shortNode(r.node)}</div>`;
  });
  return h + "</div>";
}

function tourneyTable(results, showChecks, round) {
  if (!results.length) return '<div class="hint" style="margin-top:12px;">Warming up the first model…</div>';
  const best = results.slice().sort((a, b) => (b.quality || 0) - (a.quality || 0))[0];
  const groups = {};
  results.forEach(r => { (groups[r.node || ""] = groups[r.node || ""] || []).push(r); });
  const order = Object.keys(groups).sort((a, b) => { const la = a.includes("127.0.0.1"), lb = b.includes("127.0.0.1"); if (la !== lb) return la ? -1 : 1; return a.localeCompare(b); });
  const full = round >= 2;   // triage (round 1) didn't measure epistemics/coherence — hide those columns
  const span = 9 + (full ? 2 : 0) + (showChecks ? 1 : 0);
  let cols = `<th></th>${showChecks ? "<th>Keep</th>" : ""}<th>Model</th><th>Quality</th><th>Talk</th><th>Tools</th><th>Code</th><th>Reason</th><th>Discipline</th>`;
  if (full) cols += "<th>Epistemics</th><th>Coherence</th>";
  cols += "<th>Speed/turn</th>";
  let h = `<table><tr>${cols}</tr>`;
  order.forEach(node => {
    const rows = groups[node].sort((a, b) => (b.quality || 0) - (a.quality || 0));
    h += `<tr><td colspan="${span}" class="nodehdr">${shortNode(node)} · ${rows.length} model(s)</td></tr>`;
    rows.forEach((r, i) => {
      const rank = (best && r.model === best.model) ? "🏆" : _medal(i);
      h += `<tr class="${i === 0 ? "top" : ""}"><td>${rank}</td>`;
      if (showChecks) h += `<td style="text-align:center;"><input type="checkbox" class="tkeep" data-model="${r.model}" checked></td>`;
      h += `<td>${r.model}</td>`
        + `<td class="q">${_stars(r.quality)} <span style="color:#8a94a3; font-weight:400;">${(r.quality ?? 0).toFixed(2)}</span></td>`
        + `<td>${_emoji(r.talk)}</td><td>${_emoji(r.tools)}</td><td>${_emoji(r.code)}</td><td>${_emoji(r.reasoning)}</td><td>${_emoji(r.discipline)}</td>`;
      if (full) h += `<td>${_emoji(r.epistemics)}</td><td>${_emoji(r.coherence)}</td>`;
      h += `<td>${r.return_time != null ? r.return_time.toFixed(1) + "s" : "·"}</td></tr>`;
    });
  });
  return h + "</table>";
}

function renderTourney(s) {
  if (_benchBoardClosed) return;
  benchShow(true);
  const round = s.round || 1, phase = s.phase, label = s.round_label || `Round ${round}`;
  let h = `<h2>🏆 Tournament — ${label} · ${s.round_name || ""} <button class="secondary" style="margin-left:auto; padding:4px 10px;" onclick="closeBench()">✕ Close</button></h2>`;
  h += `<div class="legend">${s.blurb || ""} &nbsp;|&nbsp; ✅ ≥ 0.80 · 🟡 0.50–0.79 · ❌ &lt; 0.50</div>`;
  if (phase === "running") {
    if (s.current !== _tourneyCur) { _tourneyCur = s.current; _tourneyCurT = Date.now(); }
    const onModel = s.current ? ` · ${Math.round((Date.now() - _tourneyCurT) / 1000)}s on this model` : "";
    const slow = s.current && (Date.now() - _tourneyCurT) > 90000 ? " ⏳ (slow — a latency cap would skip models like this)" : "";
    const eta = (s.eta != null) ? ` · ~${fmtDuration(s.eta)} left` : "";
    h += `<div class="hint" style="margin:6px 0;">${s.total ? `Scoring ${s.i}/${s.total}: ${s.current}…${onModel}${eta}${slow}` : (s.current || "Preparing…")}</div>`;
  }
  if (s.round_key === "finals") h += renderFinals(s.recommendations);
  // An empty round that's NOT still running means nothing qualified — explain why (don't look broken).
  if (phase !== "running" && !(s.results || []).length) {
    const sc = s.scope || {};
    h += `<div class="hint" style="margin:10px 0; color:#ff8a8a;">No models qualified this round. Your scope may exclude everything — size band min <b>${sc.min_model_size_b ?? 0}</b>B / max <b>${sc.max_model_size_b ?? "∞"}</b>B (an inverted band excludes all). Widen the size fields and re-run.</div>`;
  }
  h += tourneyTable((s.results || []).slice(), phase === "awaiting_veto", round);
  if (phase === "awaiting_veto") {
    const next = round === 1 ? "🥊 FIGHT → Round 1 (gauntlet)" : "🏁 Compute finals (Round 2)";
    h += `<div class="row" style="margin-top:12px; gap:10px; align-items:center;"><button id="tourneyAdvanceBtn" type="button" onclick="advanceTourney()">${next}</button><span class="hint">Untick any model you don't want to advance, then ${round === 1 ? "fight" : "finalize"}.</span></div>`;
  } else if (phase === "done") {
    h += '<div class="row" style="margin-top:12px; gap:10px;"><button id="tourneyApplyBtn" type="button" onclick="applyTourney()">✅ Apply finals to roles</button><button class="secondary" type="button" onclick="closeBench()">Done</button></div>';
  } else if (phase === "error") {
    h += `<div class="hint" style="color:#ff8a8a; margin-top:10px;">Error: ${s.error}</div>`;
  }
  $("benchBoard").innerHTML = h;
}

let _tourneyPolling = false;
let _tourneyCur = "", _tourneyCurT = 0;   // current model + when it started, for an elapsed timer
async function pollTournament() {
  if (_tourneyPolling) return;
  _tourneyPolling = true; $("fleetTourneyBtn").disabled = true; btnState("fleetTourneyBtn", "working");
  try {
    while (true) {
      const s = await api("GET", "/api/fleet/tournament/status");
      if (!s.active) break;
      renderTourney(s);
      // Reflect the tournament's progress on the manual 1·2·3 buttons (it does scan→score→apply
      // under the hood), so they're not stuck grey while the tournament runs.
      btnState("fleetScanBtn", "done");   // the tournament scans the fleet first
      btnState("fleetBenchBtn", s.phase === "running" ? "working" : "done");
      if (s.phase === "running") {
        const eta = (s.eta != null) ? ` · ~${fmtDuration(s.eta)} left` : "";
        const lbl = s.round_label || `Round ${s.round}`;
        $("fleetMsg").textContent = s.total ? `${lbl}: ${s.i}/${s.total} ${s.current}…${eta}` : `${lbl}: ${s.current || "preparing"}…`;
        await new Promise(r => setTimeout(r, 1500));
        continue;
      }
      if (s.phase === "awaiting_veto") { $("fleetMsg").textContent = `${s.round_label || ("Round " + s.round)} done — pick who advances, then FIGHT.`; btnState("fleetTourneyBtn", "done"); }
      else if (s.phase === "done") { $("fleetMsg").textContent = "🏆 Tournament complete — review the finals."; btnState("fleetTourneyBtn", "done"); }
      else if (s.phase === "error") { $("fleetMsg").textContent = "Tournament error: " + s.error; btnState("fleetTourneyBtn", "failed"); }
      break;   // interactive now — stop polling until the user acts
    }
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
  _tourneyPolling = false; $("fleetTourneyBtn").disabled = false;
}

async function advanceTourney() {
  const keep = Array.from(document.querySelectorAll(".tkeep")).filter(c => c.checked).map(c => c.dataset.model);
  if (!keep.length) { $("fleetMsg").textContent = "Keep at least one model to continue."; return; }
  const btn = $("tourneyAdvanceBtn"); if (btn) { btn.disabled = true; btn.textContent = "Working…"; }
  try {
    const r = await api("POST", "/api/fleet/tournament/advance", { keep });
    if (!r.advanced) { $("fleetMsg").textContent = "Tournament: " + (r.error || "could not advance"); if (btn) { btn.disabled = false; } return; }
    pollTournament();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; if (btn) btn.disabled = false; }
}

async function applyTourney() {
  const btn = $("tourneyApplyBtn"); if (btn) { btn.disabled = true; btn.textContent = "Applying…"; }
  try {
    const r = await api("POST", "/api/fleet/tournament/apply", {});
    const n = Object.keys(r.applied || {}).length;
    $("fleetMsg").textContent = n ? `Applied finals to ${n} role(s): ` + Object.entries(r.applied).map(([k, v]) => `${k}=${v}`).join(", ") : "Nothing to apply.";
    if (btn) btn.textContent = n ? "✅ Applied" : "Nothing to apply";
    btnState("fleetApplyBtn", n ? "done" : "");   // light the manual "3 · Apply" too
    refreshState();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; if (btn) { btn.disabled = false; btn.textContent = "✅ Apply finals to roles"; } }
}

// Workflow buttons light up by state: amber (working) → green (done) → red (failed).
function btnState(id, state) {
  const b = $(id); if (!b) return;
  b.classList.remove("working", "done", "failed");
  if (state) b.classList.add(state);
}

$("fleetScanBtn").addEventListener("click", async () => {
  $("fleetMsg").textContent = "Finding models on the fleet…"; $("fleetScanBtn").disabled = true; btnState("fleetScanBtn", "working");
  try {
    const r = await api("POST", "/api/fleet/scan");
    $("fleetMsg").textContent = `Found ${r.models} models across ${r.nodes} node(s).`;
    btnState("fleetScanBtn", "done"); loadFleet(); refreshState();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; btnState("fleetScanBtn", "failed"); }
  $("fleetScanBtn").disabled = false;
});

// Poll the (background) benchmark and show live progress. Reusable: called both when YOU click
// Benchmark and when loadFleet() finds one already running (started elsewhere or before a tab
// switch), so the Fleet tab always reflects reality. Self-guards against running twice.
let _benchPolling = false;
async function pollBenchmark() {
  if (_benchPolling) return;
  _benchPolling = true; $("fleetBenchBtn").disabled = true; btnState("fleetBenchBtn", "working");
  try {
    while (true) {
      const s = await api("GET", "/api/fleet/benchmark/status");
      if (s.error) { $("fleetMsg").textContent = "Benchmark error: " + s.error; btnState("fleetBenchBtn", "failed"); break; }
      if (s.done || !s.running) {
        btnState("fleetBenchBtn", "done");
        if (s.benchmarked !== undefined) {   // a finished run (not just the idle initial state)
          const skips = [];
          if (s.skipped_too_big) skips.push(`${s.skipped_too_big} too large`);
          if (s.skipped_too_small) skips.push(`${s.skipped_too_small} too small`);
          if (s.skipped_too_slow) skips.push(`${s.skipped_too_slow} too slow`);
          const skipped = skips.length ? ` (${skips.join(", ")} skipped)` : "";
          $("fleetMsg").textContent = `Benchmarked ${s.benchmarked || 0} of ${s.eligible || 0} eligible model(s)${skipped}` + (s.judges_ok ? "" : " — coherence judges untrusted") + ".";
          const best = (s.results || []).slice().sort((a, b) => (b.quality || 0) - (a.quality || 0))[0];
          renderBenchResults(s.results, `✅ Benchmark complete — ${s.benchmarked || 0} scored${best ? `, top 🏆 ${best.model}` : ""}`);
        }
        loadFleet();
        break;
      }
      // Until the first model starts, total is 0 — show the scan phase rather than "0/0".
      const eta = (s.eta != null) ? `  ·  ~${fmtDuration(s.eta)} left` : "";
      $("fleetMsg").textContent = s.total ? `Benchmarking ${s.i}/${s.total}: ${s.current}…${eta}` : `${s.current || "Preparing…"}`;
      const header = s.total ? `🏁 Benchmarking ${s.i}/${s.total}${eta}` : `🔎 ${s.current || "Preparing…"}`;
      renderBenchResults(s.results, header);   // live scoreboard takes over the chat pane
      await new Promise(r => setTimeout(r, 1500));
    }
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
  _benchPolling = false; $("fleetBenchBtn").disabled = false;
}

$("fleetBenchBtn").addEventListener("click", async () => {
  $("fleetMsg").textContent = "Starting benchmark…"; btnState("fleetBenchBtn", "working");
  _benchBoardClosed = false;   // a fresh run re-opens the scoreboard even if you closed the last one
  try {
    const scope = { min_model_size_b: $("benchMinSize").value, max_model_size_b: $("benchMaxSize").value, max_latency_s: $("benchMaxLatency").value };
    await api("POST", "/api/fleet/benchmark", scope);   // returns immediately; runs in the background
    pollBenchmark();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; btnState("fleetBenchBtn", "failed"); }
});

$("fleetTourneyBtn").addEventListener("click", async () => {
  _benchBoardClosed = false;   // a fresh tournament re-opens the board even if you closed the last one
  $("fleetMsg").textContent = "Starting the tournament…"; btnState("fleetTourneyBtn", "working");
  try {
    const scope = { min_model_size_b: $("benchMinSize").value, max_model_size_b: $("benchMaxSize").value, max_latency_s: $("benchMaxLatency").value };
    await api("POST", "/api/fleet/tournament/start", scope);   // returns immediately; runs in the background
    pollTournament();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; btnState("fleetTourneyBtn", "failed"); }
});

let _matrixPolling = false;
async function pollMatrix() {
  if (_matrixPolling) return;
  _matrixPolling = true; $("fleetMatrixBtn").disabled = true; btnState("fleetMatrixBtn", "working");
  try {
    while (true) {
      const s = await api("GET", "/api/fleet/matrix/status");
      if (s.error) { $("fleetMsg").textContent = "Time trial error: " + s.error; btnState("fleetMatrixBtn", "failed"); break; }
      if (s.done || !s.running) {
        btnState("fleetMatrixBtn", "done");
        if (s.timed !== undefined) $("fleetMsg").textContent = `⏱ Time trial done — timed ${s.timed} (model, node) pairing(s). Per-node times below are now complete.`;
        loadFleet();
        break;
      }
      $("fleetMsg").textContent = s.total ? `⏱ Time trial ${s.i}/${s.total}: ${s.current}…` : `⏱ ${s.current || "preparing"}…`;
      await new Promise(r => setTimeout(r, 1200));
    }
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
  _matrixPolling = false; $("fleetMatrixBtn").disabled = false;
}

$("fleetMatrixBtn").addEventListener("click", async () => {
  $("fleetMsg").textContent = "Starting the time trial…"; btnState("fleetMatrixBtn", "working");
  try {
    await api("POST", "/api/fleet/matrix", {});   // returns immediately; runs in the background
    pollMatrix();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; btnState("fleetMatrixBtn", "failed"); }
});

$("fleetApplyBtn").addEventListener("click", async () => {
  btnState("fleetApplyBtn", "working");
  try {
    const r = await api("POST", "/api/fleet/apply");
    const n = Object.keys(r.applied || {}).length;
    $("fleetMsg").textContent = n ? `Applied recommendations to ${n} role(s): ` + Object.entries(r.applied).map(([k,v]) => `${k}=${v}`).join(", ") : "Nothing to apply (benchmark first).";
    btnState("fleetApplyBtn", n ? "done" : "");
    refreshState();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; btnState("fleetApplyBtn", "failed"); }
});

async function loadModels() {
  try {
    const data = await api("GET", "/api/fleet/pool");
    const backend = data.lan_backend ? `LAN fleet (${data.nodes_up} node(s) up)` : "Local only";
    const auto = (data.auto_roles || []).join(", ") || "none (all roles pinned)";
    $("poolBackend").innerHTML = `<b>Backend:</b> ${backend} &middot; <b>Auto roles:</b> ${auto}. Locality is set by [backend] lan_backend in mimir.toml (restart to change).`;
    const list = $("poolList"); list.innerHTML = "";
    const models = data.models || [];
    if (!models.length) { list.innerHTML = '<div class="hint">No models yet — open the Fleet tab and click "1 · Find models".</div>'; return; }
    models.forEach(m => {
      const row = document.createElement("div"); row.className = "mem"; if (!m.enabled) row.style.opacity = "0.5";
      const label = document.createElement("label"); label.style.display = "flex"; label.style.alignItems = "center"; label.style.gap = "8px"; label.style.cursor = "pointer";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!m.enabled;
      cb.addEventListener("change", () => setModelEnabled(m.model, cb.checked));
      const name = document.createElement("span"); name.className = "text";
      name.textContent = `${m.passed ? "✓ " : (m.benchmarked ? "✗ " : "· ")}${m.model}`;
      name.title = m.passed ? "passed the qualification gate" : (m.benchmarked ? "benchmarked, below the gate" : "not yet benchmarked");
      label.appendChild(cb); label.appendChild(name); row.appendChild(label);
      const tags = document.createElement("div"); tags.className = "tags";
      const bits = [];
      if (m.params_b) bits.push(`${m.params_b}B`);
      if (m.quality != null) bits.push(`q${m.quality}`);
      if (m.reasoning != null) bits.push(`reason ${m.reasoning}`);
      if (m.discipline != null) bits.push(`disc ${m.discipline}`);
      if (m.return_time != null) bits.push(`${m.return_time}s/turn`);
      bits.push(`${(m.nodes || []).length} node(s)`);
      if (m.approved) bits.push("approved");
      (m.roles || []).forEach(r => bits.push("▶ " + r));
      bits.forEach(b => { const sp = document.createElement("span"); sp.className = "tag"; sp.textContent = b; tags.appendChild(sp); });
      row.appendChild(tags); list.appendChild(row);
    });
  } catch (e) { $("poolList").innerHTML = "error: " + e.message; }
}

async function setModelEnabled(model, enabled) {
  $("poolMsg").textContent = `${enabled ? "Enabling" : "Disabling"} ${model}…`;
  try {
    const r = await api("POST", "/api/fleet/model", { model, enabled });
    const moved = Object.entries(r.moved || {});
    $("poolMsg").textContent = moved.length
      ? `${model} ${enabled ? "enabled" : "disabled"} — re-routed ` + moved.map(([k,v]) => `${k}→${v}`).join(", ")
      : `${model} ${enabled ? "enabled" : "disabled"}.`;
    loadModels(); refreshState();
  } catch (e) { $("poolMsg").textContent = "Error: " + e.message; loadModels(); }
}

async function setNodeEnabled(node, enabled) {
  $("fleetMsg").textContent = `${enabled ? "Enabling" : "Disabling"} ${node}…`;
  try {
    const r = await api("POST", "/api/fleet/node", { node, enabled });
    const moved = Object.entries(r.moved || {});
    $("fleetMsg").textContent = moved.length
      ? `${node} ${enabled ? "enabled" : "disabled"} — re-routed ` + moved.map(([k,v]) => `${k}→${v}`).join(", ")
      : `${node} ${enabled ? "enabled" : "disabled"} for the fleet.`;
    loadFleet(); refreshState();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; loadFleet(); }
}

$("procBtn").addEventListener("click", async () => {
  const trigger = $("procTrigger").value.trim(); const procedure = $("procBody").value.trim();
  if (!trigger || !procedure) { $("procMsg").textContent = "Both fields are required."; return; }
  try {
    await api("POST", "/api/procedures", { trigger, procedure });
    $("procTrigger").value = ""; $("procBody").value = "";
    $("procMsg").textContent = "Learned."; loadProcedures(); refreshState();
    setTimeout(() => $("procMsg").textContent = "", 2000);
  } catch (e) { $("procMsg").textContent = "Error: " + e.message; }
});

$("councilBtn").addEventListener("click", async () => {
  const question = $("councilQ").value.trim(); if (!question) return;
  $("councilBtn").disabled = true; $("councilVerdict").textContent = "Deliberating…"; $("councilPositions").innerHTML = "";
  try {
    const r = await api("POST", "/api/council", { question });
    $("councilVerdict").textContent = r.verdict;
    r.positions.forEach(p => {
      const d = document.createElement("div"); d.className = "mem";
      const tx = document.createElement("div"); tx.className = "text"; tx.textContent = p.text; d.appendChild(tx);
      const tags = document.createElement("div"); tags.className = "tags";
      const add = (cls, x) => { const sp = document.createElement("span"); sp.className = "tag " + cls; sp.textContent = x; tags.appendChild(sp); };
      add("tier", p.persona); add("", p.model);
      d.appendChild(tags); $("councilPositions").appendChild(d);
    });
    refreshState();
  } catch (e) { $("councilVerdict").textContent = "Error: " + e.message; }
  $("councilBtn").disabled = false;
});

refreshState(); loadIdentity(); resumeFleetWork();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
