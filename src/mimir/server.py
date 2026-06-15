"""Reference web UI — a stdlib HTTP server over the core library (DESIGN §8 adapter).

Mimir 0's core is a library; this is the canonical *human* surface, kept deliberately outside the
core and built on Python's stdlib ``http.server`` so it adds **zero dependencies** — the runtime
contract (Python + SQLite + endpoints, nothing else) still holds. It serves a single-page UI and a
small JSON API:

    GET  /                 the web UI (chat + identity interview + document ingest + status)
    GET  /api/state        embedding mode, memory count, anchors established
    GET  /api/identity     current anchors + the questions still pending
    POST /api/identity     {"answers": {...}}  establish/revise identity anchors
    GET  /api/onboarding   the seeding interview: every question + answer, what's pending
    POST /api/onboarding/answer  {"key": "...", "answer": "..."}  store/update one answer (blank clears)
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
from .errors import ConfigError, IngestError, MimirError
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
            elif route == "/api/onboarding":
                self._send_json(self._onboarding_payload())
            elif route == "/api/history":
                self._send_json(self._history(params))
            elif route == "/api/sessions":
                self._send_json(self._sessions())
            elif route == "/api/mind":
                self._send_json(self._mind())
            elif route == "/api/memories":
                self._send_json(self._memories(params))
            elif route == "/api/graph":
                self._send_json(self._graph(params))
            elif route == "/api/graph/map":
                with self.server.brain_lock:
                    self._send_json(self.server.brain.graph_map())
            elif route == "/api/wiki/status":
                self._send_json(self.server.brain.wiki_status())  # lock-free (quick external check)
            elif route == "/api/sleep/status":
                self._send_json(self.server.brain.sleep_cycle_status())  # lock-free (reads state)
            elif route == "/api/settings":
                self._send_json(self.server.brain.settings())  # lock-free (kv read)
            elif route == "/api/forum":
                self._send_json({"threads": self.server.brain.forum_threads()})
            elif route == "/api/forum/thread":
                self._send_json(self._forum_thread(params))
            elif route == "/api/timezones":
                self._send_json({"zones": self.server.brain.available_timezones()})
            elif route == "/api/procedures":
                self._send_json(self._procedures())
            elif route == "/api/fleet":
                self._send_json(self._fleet())
            elif route == "/api/fleet/pool":
                self._send_json(self._model_pool())
            elif route == "/api/fleet/placement":
                with self.server.brain_lock:
                    self._send_json(self.server.brain.placement_matrix())
            elif route == "/api/fleet/council":
                try:
                    size = int((params.get("size") or ["5"])[0])
                except ValueError:
                    size = 5
                with self.server.brain_lock:
                    self._send_json(self.server.brain.council_roster(size=max(1, min(12, size))))
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
            elif route == "/api/onboarding/answer":
                self._send_json(self._onboarding_answer(body))
            elif route == "/api/session":
                self._send_json(self._session_action(body))
            elif route == "/api/memory":
                self._send_json(self._memory_action(body))
            elif route == "/api/ingest":
                self._send_json(self._ingest(body))
            elif route == "/api/sleep":
                self._send_json(self._sleep())
            elif route == "/api/deliberate/run":
                self._send_json(self._deliberate_now())
            elif route == "/api/forum":
                self._send_json(self._forum_action(body))
            elif route == "/api/settings":
                self._send_json(self._update_settings(body))
            elif route == "/api/council":
                self._send_json(self._council(body))
            elif route == "/api/procedures":
                self._send_json(self._learn_procedure(body))
            elif route == "/api/fleet/scan":
                self._send_json(self._scan_fleet())
            elif route == "/api/fleet/benchmark":
                self._send_json(self._benchmark_fleet(body))
            elif route == "/api/fleet/benchmark/council":
                self._send_json(self._benchmark_council())
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
            elif route == "/api/fleet/role":
                self._send_json(self._set_role(body))
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

    def _history(self, params: dict[str, list[str]]) -> dict[str, Any]:
        """The durable conversation log (optionally one session), for restoring the chat."""
        try:
            limit = max(1, min(200, int((params.get("limit") or ["50"])[0])))
        except ValueError:
            limit = 50
        session = (params.get("session") or [None])[0] or None
        return {"turns": self.server.brain.history(
            user="operator", limit=limit, session_id=session)}

    def _memory_action(self, body: dict[str, Any]) -> dict[str, Any]:
        """Edit or delete one memory from the graph viewer."""
        try:
            mem_id = int(body.get("id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("'id' (integer) is required") from exc
        action = str(body.get("action", "update")).strip()
        with self.server.brain_lock:
            if action == "delete":
                self.server.brain.forget_memory(mem_id)
                return {"deleted": mem_id}
            if action == "update":
                text = body.get("text")
                sal = body.get("salience")
                mem = self.server.brain.edit_memory(
                    mem_id,
                    text=str(text) if text is not None else None,
                    salience=float(sal) if sal is not None else None,
                )
                return {"memory": _memory_to_dict(mem) if mem else None}
            raise ValueError("'action' must be 'update' or 'delete'")

    def _sessions(self) -> dict[str, Any]:
        """Past conversations (summary + timestamps) for the chat dropdown."""
        return {"sessions": self.server.brain.sessions(user="operator")}

    def _session_action(self, body: dict[str, Any]) -> dict[str, Any]:
        action = str(body.get("action", "")).strip()
        if action == "new":
            return {"session_id": self.server.brain.start_new_session()}
        if action == "resume":
            sid = str(body.get("session_id", "")).strip()
            if not sid:
                raise ValueError("'session_id' is required to resume")
            self.server.brain.resume_session(sid)
            return {"session_id": sid}
        raise ValueError("'action' must be 'new' or 'resume'")

    def _onboarding_payload(self) -> dict[str, Any]:
        """The seeding interview: every question + current answer, and what's still pending."""
        brain = self.server.brain
        profile = brain.onboarding_profile()
        pending = brain.pending_onboarding()
        return {
            "profile": profile,
            "pending": pending,
            "complete": not pending,
            # First-run prompt: nudge the interview only when nothing's been captured yet.
            "started": any(q["answer"] for q in profile),
        }

    def _onboarding_answer(self, body: dict[str, Any]) -> dict[str, Any]:
        key = str(body.get("key", "")).strip()
        if not key:
            raise ValueError("'key' is required")
        answer = str(body.get("answer", ""))
        # Lock-free by design (like /api/state): the interview is meant to run DURING the qualifying
        # tournament, which holds brain_lock for whole rounds (minutes). Taking it here would hang
        # "Next" until the round finished. Capture is a storage write + embed — the storage gateway is
        # the thread-safe single writer and the embedder is pool-/stdlib-safe, so no global lock needed.
        self.server.brain.record_onboarding_answer(key, answer)
        return self._onboarding_payload()

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

    def _run_benchmark_bg(
        self, run: Any, *, scanning: str = "scanning the fleet…"
    ) -> dict[str, Any]:
        """Shared background-run scaffold for any benchmark-style pass: sets up bench_state, the
        progress/on_result callbacks, and the worker thread; ``run(progress, on_result)`` does the
        actual work and returns a FleetBenchmarkResult. Holding the brain lock for the whole run (in
        the worker) preserves serialization against turns, but status reads stay lock-free so the
        page never freezes. The UI polls /api/fleet/benchmark/status for all of them."""
        srv = self.server
        with srv.bench_lock:
            if srv.bench_state.get("running"):
                return {"started": False, **srv.bench_state}  # already running
            srv.bench_state = {"running": True, "i": 0, "total": 0,
                               "current": scanning, "done": False, "results": []}

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
                    result = run(_progress, _on_result)
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

    def _benchmark_fleet(self, body: dict[str, Any]) -> dict[str, Any]:
        """Kick off the main fleet benchmark in the background. Optional body fields
        ``max_model_size_b`` / ``min_model_size_b`` / ``max_latency_s`` (the UI scope fields)
        override the configured cap/floor/latency for this run."""
        cap = float(body["max_model_size_b"]) if body.get("max_model_size_b") not in (None, "") else None
        floor = float(body["min_model_size_b"]) if body.get("min_model_size_b") not in (None, "") else None
        latency = float(body["max_latency_s"]) if body.get("max_latency_s") not in (None, "") else None
        return self._run_benchmark_bg(lambda p, r: self.server.brain.benchmark_fleet(
            max_params_b=cap, min_params_b=floor, latency_budget_s=latency,
            progress=p, on_result=r,
        ))

    def _benchmark_council(self) -> dict[str, Any]:
        """Grade the council pool — the big models above the chat cap, caps off — in place (no
        rescan, so the main pool's scores survive). Then they enter the council roster."""
        return self._run_benchmark_bg(
            lambda p, r: self.server.brain.benchmark_council_pool(progress=p, on_result=r),
            scanning="grading the council pool (big models, caps off)…",
        )

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
        # The time trial spins up its OWN concurrent per-node probing, which would fight a running
        # benchmark/tournament for the same per-node VRAM locks (thrash, bad numbers). Refuse loudly
        # rather than collide — DESIGN §10. The UI also greys the button out, but this is the real
        # guard (a stale/forced click can't start it mid-qualification).
        with srv.bench_lock:
            bench_busy = bool(srv.bench_state.get("running"))
        with srv.tourney_lock:
            # A tournament is 'busy' for the matrix only while a round is actually running, or paused
            # for a veto (FIGHT will resume the next round → it would collide). At phase 'done' the
            # tournament is terminal and nothing runs on the nodes — that's exactly when the speed-test
            # is the intended next step, so allow it.
            tourney_busy = (bool(srv.tourney_state.get("active"))
                            and srv.tourney_state.get("phase") in ("running", "awaiting_veto"))
        if bench_busy or tourney_busy:
            what = "a benchmark" if bench_busy else "the tournament"
            return {"started": False, "busy": True,
                    "reason": f"Can't run the time trial while {what} is in progress — it shares the "
                              "same GPUs. Let it finish first."}
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
            # Live model names, so the role-assignment dropdown has options even before a benchmark.
            pool["available"] = sorted(brain._model.available_models())
            # Per-(node, model) placements + current node pins, so manual override can target a
            # specific edge node — a model on several nodes is selectable per node, not collapsed.
            pool["placement"] = brain.placement_matrix()
            pool["role_nodes"] = brain.role_nodes()
        return pool

    def _set_role(self, body: dict[str, Any]) -> dict[str, Any]:
        role = str(body.get("role", "")).strip()
        model = str(body.get("model", "")).strip()
        node = str(body.get("node", "")).strip() or None  # optional: pin to a specific fleet node
        if not role or not model:
            raise ValueError("'role' and 'model' are required")
        with self.server.brain_lock:
            return {"roles": self.server.brain.set_role(role, model, node)}

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
        # Manual "run sleep now": force the full cycle (ignores window + daily guard) so a click
        # also stamps today's checkpoint, and the night scheduler then skips a redundant run.
        with self.server.brain_lock:
            cycle = self.server.brain.run_sleep_cycle(force=True)
            report = self.server.brain._last_sleep_report
        out: dict[str, Any] = {
            "ran": cycle.ran,
            "skipped": cycle.skipped,
            "failed": cycle.failed,
            "completed": cycle.completed,
        }
        if report is not None:
            out.update({
                "deduped": report.deduped,
                "decayed": report.decayed,
                "archived": report.archived,
                "contradictions_resolved": report.contradictions_resolved,
                "total_changes": report.total_changes,
            })
        return out

    def _deliberate_now(self) -> dict[str, Any]:
        # Manual trigger for self-directed deliberation: surface conflicts → council → verdicts.
        with self.server.brain_lock:
            return self.server.brain.deliberate_open_questions(force=True)

    def _forum_thread(self, params: dict[str, list[str]]) -> dict[str, Any]:
        thread_id = int((params.get("id") or ["0"])[0])
        thread = self.server.brain.forum_thread(thread_id)
        if thread is None:
            raise MimirError(f"no thread {thread_id}")
        return thread

    def _forum_action(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forum housekeeping + 'ask the council'. One action per call (keeps the UI simple)."""
        action = body.get("action")
        brain = self.server.brain
        if action == "ask":
            question = str(body.get("question", "")).strip()
            if not question:
                raise MimirError("question is required")
            with self.server.brain_lock:  # a real deliberation — fans across the fleet
                result = brain.deliberate(question, user="operator")
            return {"thread_id": result.thread_id}
        if action == "comment":
            brain.forum_comment(int(body["thread_id"]), str(body.get("text", "")).strip(),
                                user="operator")
        elif action in ("close", "reopen"):
            brain.forum_set_status(int(body["thread_id"]),
                                   "closed" if action == "close" else "open")
        elif action == "delete_thread":
            brain.forum_delete_thread(int(body["thread_id"]))
        elif action == "delete_post":
            brain.forum_delete_post(int(body["post_id"]))
        else:
            raise MimirError(f"unknown forum action: {action!r}")
        return {"ok": True}

    def _update_settings(self, body: dict[str, Any]) -> dict[str, Any]:
        changes = body.get("settings", body)  # accept {"settings": {...}} or a bare {...}
        if not isinstance(changes, dict):
            raise MimirError("settings must be an object")
        try:
            return self.server.brain.update_settings(changes)
        except (ConfigError, ValueError) as exc:  # bad tz / time → a clean 400-style message
            raise MimirError(str(exc)) from exc

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
  /* App fills the viewport and never scrolls as a whole: header pinned, the two columns scroll on
     their own. (min-height:0 lets a flex/grid child actually shrink so its child can scroll.) */
  body { margin:0; font:15px/1.5 system-ui, sans-serif; background:#0e1116; color:#d7dde5;
    height:100vh; display:flex; flex-direction:column; overflow:hidden; }
  header { flex:none; padding:12px 20px; border-bottom:1px solid #232a35; display:flex; align-items:center; gap:14px; }
  header h1 { font-size:18px; margin:0; letter-spacing:.5px; }
  header .status { font-size:12px; color:#8a94a3; }
  main { display:grid; grid-template-columns: 1fr 360px; gap:0; flex:1; min-height:0; }
  #chat { display:flex; flex-direction:column; border-right:1px solid #232a35; min-height:0; }
  /* The left column's scrolling region (chat log or a takeover view) — header + footer stay put. */
  #log, #benchBoard, #forumView, #graphView { min-height:0; }
  #log { flex:1; overflow-y:auto; padding:18px; }
  .msg { margin:0 0 14px; max-width:80%; }
  .msg .who { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#6f7a8a; margin-bottom:3px; }
  .msg .body { padding:10px 13px; border-radius:10px; white-space:pre-wrap; }
  .user { margin-left:auto; }
  .user .body { background:#1f6feb; color:#fff; }
  .mimir .body { background:#1a2029; }
  .msg .body.thinking { color:#8a94a3; font-style:italic; animation: pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:.4 } 50% { opacity:1 } }
  .meta { font-size:11px; color:#6f7a8a; margin-top:4px; }
  #graphSvg { display:block; background:radial-gradient(circle at 50% 45%, #0a1422, #05080d); cursor:grab; }
  #graphSvg.panning { cursor:grabbing; }
  #graphSvg line { stroke:#4a9fe0; stroke-opacity:.18; }
  #graphSvg .node { cursor:pointer; }
  #graphSvg .node text { fill:#bcd7f5; font-size:10px; pointer-events:none; opacity:.85; }
  #graphSvg .node .halo { stroke:none; }
  #graphSvg .node .core { stroke:#0a1422; stroke-width:1; }
  #graphSvg .node.sel .core { stroke:#eaf6ff; stroke-width:2.5; }
  #graphInspect { display:none; position:absolute; top:12px; right:12px; width:300px; max-height:88%;
    overflow:auto; background:#11161d; border:1px solid #2b333f; border-radius:10px; padding:14px; }
  #graphInspect textarea { width:100%; min-height:90px; resize:vertical; }
  #sessionBar { flex:none; display:flex; gap:8px; align-items:center; padding:8px 12px; border-bottom:1px solid #232a35; }
  #sessionSelect { flex:1; min-width:0; background:#11161d; border:1px solid #2b333f; color:#d7dde5; border-radius:8px; padding:6px 9px; font:inherit; }
  #composer { flex:none; display:flex; gap:8px; padding:12px; border-top:1px solid #232a35; }
  #composer input[type=text] { flex:1; }
  input[type=text], textarea { background:#11161d; border:1px solid #2b333f; color:#d7dde5; border-radius:8px; padding:9px 11px; font:inherit; }
  button { background:#238636; color:#fff; border:0; border-radius:8px; padding:9px 14px; font:inherit; cursor:pointer; }
  button.secondary { background:#30363d; }
  button.working { background:#9e6a03; }   /* amber — in progress */
  button.done    { background:#1f7a37; }   /* green — completed */
  button.failed  { background:#b62324; }   /* red — errored */
  button:disabled { opacity:.5; cursor:default; }
  aside { overflow-y:auto; min-height:0; padding:16px; }   /* right column scrolls independently */
  aside section { margin-bottom:26px; }
  aside h2 { font-size:13px; text-transform:uppercase; letter-spacing:.7px; color:#8a94a3; margin:0 0 10px; }
  .field { margin-bottom:9px; }
  .field label { display:block; font-size:12px; color:#9aa4b2; margin-bottom:3px; }
  .field input { width:100%; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
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
  /* the seeding-interview strip: sits in the bottom slice of the chat pane under the tournament board */
  #interviewStrip { flex:none; max-height:34%; overflow:auto; border-top:2px solid #1f6feb; padding:12px 14px; background:#0d1320; }
  #interviewStrip .ivhead { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#6f8ad0; margin-bottom:6px; display:flex; align-items:center; gap:10px; }
  #interviewStrip .ivq { font-size:15px; color:#e7edf5; margin-bottom:9px; }
  #ivForm { display:flex; gap:8px; }
  #ivForm input { flex:1; }
  #ivProgress { font-size:12px; color:#6f7a8a; margin-top:7px; }
  .profile-fact { border:1px solid #232a35; border-radius:8px; padding:9px 11px; margin-bottom:9px; }
  .profile-fact label { display:block; font-size:12px; color:#9aa4b2; margin-bottom:4px; }
  .profile-fact .ans { width:100%; }
  .profile-fact .anchorbadge { font-size:10px; color:#6f8ad0; margin-left:6px; }
</style>
</head>
<body>
<header>
  <h1>Mimir 0</h1>
  <span class="status" id="status">connecting…</span>
</header>
<main>
  <div id="chat">
    <div id="sessionBar">
      <span class="hint" style="white-space:nowrap;">Conversation:</span>
      <select id="sessionSelect" title="Past conversations — pick one and Restore to view/continue it."></select>
      <button class="secondary" id="sessionRestore" type="button" title="Load the selected conversation and continue it.">Restore</button>
      <button class="secondary" id="sessionNew" type="button" title="Start a fresh conversation.">+ New</button>
      <button class="secondary" id="graphToggle" type="button" title="Switch between the chat and a visual map of your memories." style="margin-left:auto;">🕸 Graph</button>
      <button class="secondary" id="forumToggle" type="button" title="The council forum — read deliberations, comment, and keep house.">🏛 Forum</button>
    </div>
    <div id="benchBoard" style="display:none; flex:1; overflow:auto; padding:18px;"></div>
    <div id="forumView" style="display:none; flex:1; overflow:auto; padding:16px;"></div>
    <div id="graphView" style="display:none; flex:1; position:relative; overflow:hidden;">
      <svg id="graphSvg" width="100%" height="100%"></svg>
      <div id="graphLegend" class="hint" style="position:absolute; top:8px; left:10px; pointer-events:none;"></div>
      <div id="graphInspect"></div>
    </div>
    <div id="log"></div>
    <div id="interviewStrip" style="display:none;">
      <div class="ivhead"><span>🌱 Getting to know you</span>
        <span id="ivProgressTop" style="color:#6f7a8a;"></span>
        <button class="secondary" type="button" id="ivSkip" style="margin-left:auto; padding:3px 9px;">Skip</button>
        <button class="secondary" type="button" id="ivDone" style="padding:3px 9px;">Later</button>
      </div>
      <div class="ivq" id="ivQ">…</div>
      <form id="ivForm">
        <input type="text" id="ivInput" placeholder="Type your answer…" autocomplete="off"/>
        <button type="submit" id="ivSend">Next</button>
      </form>
      <div id="ivProgress"></div>
    </div>
    <form id="composer">
      <input type="text" id="text" placeholder="Say something to Mimir…" autocomplete="off"/>
      <button type="submit" id="send">Send</button>
    </form>
  </div>
  <aside>
    <div class="tabs">
      <button data-tab="identity" class="active">Identity</button>
      <button data-tab="profile">Profile</button>
      <button data-tab="mind">Mind</button>
      <button data-tab="sleep">Sleep</button>
      <button data-tab="memories">Memories</button>
      <button data-tab="graph">Graph</button>
      <button data-tab="procedures">Habits</button>
      <button data-tab="council">Council</button>
      <button data-tab="fleet">Fleet</button>
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

    <div class="tabpane hidden" id="tab-profile">
      <h2>You &amp; this place</h2>
      <div class="hint" style="margin-bottom:10px;">The seeding interview — your highest-trust facts
        (<span class="tag tier">stated_by_primary_user</span>), the orientation everything else builds
        on. Edit any answer below, or run the guided interview. Everything's editable anytime.</div>
      <div class="row" style="margin-bottom:12px;">
        <button id="runInterviewBtn" type="button">🌱 Run interview</button>
        <button class="secondary" id="profileReloadBtn" type="button">Reload</button>
      </div>
      <div id="profileFacts"></div>
      <div id="profileMsg" style="font-size:12px; color:#7fd17f; margin-top:6px; min-height:14px;"></div>
    </div>

    <div class="tabpane hidden" id="tab-mind">
      <h2>Self-model</h2>
      <div class="selfmodel" id="selfModel">—</div>
      <div class="stats" id="mindStats"></div>
      <h2>Working memory</h2>
      <div class="selfmodel" id="workingMemory">—</div>
      <h2>Recent reflections</h2>
      <div id="reflections"></div>
    </div>

    <div class="tabpane hidden" id="tab-sleep">
      <h2>Sleep cycle</h2>
      <div class="hint" style="margin-bottom:10px;">When you're away, Mimir does its heavy upkeep —
        consolidating memory, resolving contradictions, writing its journal (and, soon, reasoning
        adversarially over its own open questions). Set the quiet window below. All times are in your
        timezone; everything is stored in UTC and shifted to it.</div>
      <div class="field">
        <label>Timezone</label>
        <select id="setTz"><option value="">System local time (recommended)</option></select>
      </div>
      <div class="field" style="display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap;">
        <div><label>Quiet hours start</label><input type="time" id="setStart"/></div>
        <div><label>Quiet hours end</label><input type="time" id="setEnd"/></div>
        <label style="font-weight:normal;"><input type="checkbox" id="setEnabled"/> Enabled</label>
      </div>
      <label style="font-weight:normal; display:block; margin-bottom:8px;">
        <input type="checkbox" id="setDeliberate"/> Reason adversarially over my own conflicts during sleep
      </label>
      <button id="saveSleep" type="button">Save schedule</button>
      <span id="settingsMsg" class="hint" style="margin-left:10px;"></span>
      <h2 style="margin-top:18px;">Status</h2>
      <div id="sleepStatus" class="hint">—</div>
      <button class="secondary" id="sleepBtn" type="button">Run sleep now</button>
      <button class="secondary" id="delibBtn" type="button">Deliberate now</button>
      <div id="sleepResult" class="hint"></div>
      <div id="delibResult" style="margin-top:8px;"></div>
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
      <h2 style="margin-top:0;">Role assignment</h2>
      <div class="hint" style="margin-bottom:8px;">Pick the model for each role, or leave it on <b>auto</b> (the system picks the best-qualified). Setting one manually <b>pins</b> it — a rescan won't change it.</div>
      <div id="roleAssign"></div>
      <div id="roleMsg" class="hint" style="min-height:14px;"></div>

      <h2 style="margin-top:22px;">Qualify your fleet</h2>
      <details class="hint" style="margin-bottom:10px; border:1px solid #333; border-radius:6px; padding:6px 10px;">
        <summary style="cursor:pointer;">ℹ️ What these scores mean — <b>best for <em>this</em> system, not the world</b></summary>
        <div style="margin-top:6px; line-height:1.5;">
          Mimir ranks models by <b>operational fitness for its own roles on your hardware</b> — not “best model overall.” A winner is the best model <b>for the system as built</b>, under this test battery, on your fleet — never a universal benchmark. Any installed model can compete (model-agnostic). Speed is <b>per-node and shifts with load</b>, so routing re-selects live. Coherence is <b>experimental</b> (a peer-review annotation, not a gate). And a narrow win isn’t a landslide.
        </div>
      </details>
      <div class="row">
        <button id="fleetTourneyBtn" type="button" title="The recommended path. A staged knock-out: Round 0 qualifying (fast) → you pick survivors → Round 1 the full framework gauntlet → Round 2 finals. You choose who advances between rounds.">🏆 Run qualifying tournament</button>
      </div>
      <div class="hint" style="margin-top:6px;">The recommended way to qualify your fleet — narrows it in rounds: <b>Round 0 · Qualifying</b> (fast) → <b>🥊 you keep the survivors</b> → <b>Round 1 · Framework gauntlet</b> (the real test) → <b>Round 2 · Finals</b>. (Round 3 · Vision is reserved.) The scoreboard takes over the chat pane.</div>
      <div class="hint" style="margin-top:14px; opacity:0.8;">— or do it manually, one step at a time —</div>
      <div class="row" style="margin-top:6px;">
        <button class="secondary" id="fleetScanBtn" type="button" title="List what models are installed on each node. Fast — runs no models.">1 · Find models</button>
        <button class="secondary" id="fleetBenchBtn" type="button" title="Run each model through the test battery to score it. Slow — this is the expensive step.">2 · Benchmark</button>
        <button class="secondary" id="fleetMatrixBtn" type="button" title="The time trial: speed-test each qualified model on every node it's installed on but not yet timed, so we know which edge can run what (the background-worker map). Records even slow results. Disabled while a benchmark/tournament is running.">3 · Speed-test</button>
        <button class="secondary" id="fleetApplyBtn" type="button" title="Point each role at its top-scoring model from the benchmark.">4 · Apply best</button>
      </div>
      <div class="hint" style="margin-top:6px;"><b>Find</b> lists installed models (fast, no scoring) → <b>Benchmark</b> scores them all (slow) → <b>Speed-test</b> fills the <b>placement matrix</b> (times each qualified model on the other nodes it lives on, so you learn which edges can host which models for background/council work — slow is fine there) → <b>Apply</b> routes each role to the best.</div>
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

      <h2 style="margin-top:22px;">Models</h2>
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

      <h2 style="margin-top:22px;">Offline encyclopedia</h2>
      <div class="hint" style="margin-bottom:8px;">A live reference layer over a local Kiwix/ZIM (set the <code>[wiki]</code> block in mimir.toml). Optional.</div>
      <div id="wikiStatus" class="hint">checking…</div>
      <button class="secondary" id="wikiRecheck" type="button" style="margin-top:8px;">Recheck</button>
    </div>
  </aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let reviseMode = false;
// The assistant's chosen name (onboarding "what would you like to call me?" → the `name` anchor).
// Drives the chat input placeholder and the speaker label on its bubbles; defaults until loaded.
let ASSISTANT_NAME = "Mimir";
function applyAssistantName(name) {
  ASSISTANT_NAME = (name && name.trim()) || "Mimir";
  $("text").placeholder = "Say something to " + ASSISTANT_NAME + "…";
  document.querySelectorAll(".msg.mimir .who").forEach(el => { el.textContent = ASSISTANT_NAME; });
}

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
  div.querySelector(".who").textContent = who === "you" ? "you" : ASSISTANT_NAME;
  div.querySelector(".body").textContent = body;
  if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; div.appendChild(m); }
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
  return div;
}

async function streamTurn(text) {
  const bubble = addMsg("mimir", "");
  const body = bubble.querySelector(".body");
  // Thinking indicator: a pulsing placeholder until the first token arrives, so you can tell it's
  // working even when an edge node is slow to start generating.
  body.classList.add("thinking"); body.textContent = "thinking…";
  let started = false;
  const begin = () => { if (!started) { started = true; body.classList.remove("thinking"); body.textContent = ""; } };
  const resp = await fetch("/api/turn/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, user: "operator" }) });
  if (!resp.ok) { body.classList.remove("thinking"); const e = await resp.json().catch(() => ({ error: "HTTP " + resp.status })); throw new Error(e.error); }
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
      if (ev === "token") { begin(); body.textContent += obj.text; $("log").scrollTop = $("log").scrollHeight; }
      else if (ev === "done") { introspect = obj.introspect; }
      else if (ev === "error") { begin(); body.textContent += (body.textContent ? "\\n" : "") + "[error] " + obj.error; }
    }
  }
  if (!started) { body.classList.remove("thinking"); body.textContent = "(no response)"; }
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
  applyAssistantName(data.anchors && data.anchors.name);
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
  // Stream the reply token-by-token (with a thinking indicator) so a slow edge node shows progress
  // instead of a frozen UI.
  try { await streamTurn(text); }
  catch (e) { addMsg("mimir", "[error] " + e.message); }
  $("send").disabled = false; $("text").focus(); refreshState();
  // Refresh the conversation dropdown (a brand-new session appears; counts update) and keep the
  // current (most recent) one selected.
  loadSessions().then(s => { if (s.length) $("sessionSelect").value = s[0].session_id || ""; });
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

// --- session dropdown (past conversations: select + restore + new) ---
async function loadSessions(selectId) {
  try {
    const data = await api("GET", "/api/sessions");
    const sel = $("sessionSelect"); sel.innerHTML = "";
    const sessions = data.sessions || [];
    if (!sessions.length) {
      const o = document.createElement("option"); o.value = ""; o.textContent = "(no past conversations)";
      sel.appendChild(o); return sessions;
    }
    sessions.forEach(s => {
      const o = document.createElement("option"); o.value = s.session_id || "";
      const when = s.last ? new Date(s.last * 1000).toLocaleDateString() : "";
      const sum = (s.summary || "(empty)").replace(/\\s+/g, " ").slice(0, 50);
      o.textContent = `${sum} · ${s.count} msg · ${when}`;
      sel.appendChild(o);
    });
    if (selectId != null) sel.value = selectId;
    return sessions;
  } catch (e) { return []; }
}

async function restoreSession(sessionId) {
  $("log").innerHTML = "";
  const q = sessionId ? `&session=${encodeURIComponent(sessionId)}` : "";
  const data = await api("GET", "/api/history?limit=200" + q);
  (data.turns || []).forEach(t => { addMsg("you", t.user_text); addMsg("mimir", t.reply); });
}

$("sessionRestore").addEventListener("click", async () => {
  const sid = $("sessionSelect").value; if (!sid) return;
  try { await api("POST", "/api/session", { action: "resume", session_id: sid }); await restoreSession(sid); }
  catch (e) { addMsg("mimir", "[error] " + e.message); }
});
$("sessionNew").addEventListener("click", async () => {
  try { await api("POST", "/api/session", { action: "new" }); $("log").innerHTML = ""; await loadSessions(""); }
  catch (e) { addMsg("mimir", "[error] " + e.message); }
});

// --- memory graph: a drifting galaxy of memory blobs + entities (zoom/pan, click to review/edit) ---
const GNS = "http://www.w3.org/2000/svg";
const graph = { on: false, nodes: [], links: [], byId: {}, raf: 0, sel: null,
                w: 600, h: 500, k: 1, px: 0, py: 0, root: null, frozen: false, phase: 0 };

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function graphSize() {
  const r = $("graphSvg").getBoundingClientRect();
  graph.w = r.width || 600; graph.h = r.height || 500;
}
function applyTransform() {
  if (graph.root) graph.root.setAttribute("transform",
    `translate(${graph.px.toFixed(1)},${graph.py.toFixed(1)}) scale(${graph.k.toFixed(3)})`);
}
// White-hot at the centre (important) → deep blue at the rim, for the lightning look.
function glowColor(t) {
  const a = [59, 130, 246], b = [233, 246, 255];
  const m = i => Math.round(a[i] + (b[i] - a[i]) * t);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}

async function loadGraphMap() {
  let data; try { data = await api("GET", "/api/graph/map"); } catch (e) { return; }
  graphSize();
  const cx = graph.w / 2, cy = graph.h / 2, R = Math.min(graph.w, graph.h) * 0.42;
  graph.nodes = (data.nodes || []).map((n, i) => {
    const a = i * 2.3999, rr = 30 + (i % 9) / 9 * R;  // golden-angle scatter to start
    return Object.assign({}, n, { x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr, vx: 0, vy: 0 });
  });
  graph.byId = {}; graph.nodes.forEach(n => { graph.byId[n.id] = n; });
  graph.links = (data.links || []).filter(l => graph.byId[l.source] && graph.byId[l.target])
    .map(l => ({ source: graph.byId[l.source], target: graph.byId[l.target], label: l.label }));

  // Importance = degree + (for memories) salience + usage + how foundational it is. The seeding
  // interview (provenance "onboarding") and operator-stated facts are the bedrock, so they get the
  // biggest boost — biggest blobs, dead centre. Drives the ring, the size, and the brightness.
  const TIERW = { stated_by_primary_user: 3, stated_by_trusted: 2, document: 1.2,
                  inferred: 0.8, conversation: 0.5 };
  graph.nodes.forEach(n => { n.deg = 0; });
  graph.links.forEach(l => { l.source.deg++; l.target.deg++; });
  graph.nodes.forEach(n => {
    const sal = n.salience || 1, acc = n.access || 0;
    n.imp = n.deg + (n.type === "memory"
      ? sal * 0.6 + Math.log(1 + acc) * 0.8 + (TIERW[n.tier] || 0.5) * 1.5
        + (n.provenance === "onboarding" ? 5 : 0)   // foundational interview → bedrock
      : 0);
  });
  const imps = graph.nodes.map(n => n.imp);
  const lo = Math.min(...imps, 0), hi = Math.max(...imps, lo + 1e-6);
  const Rmax = Math.min(graph.w, graph.h) * 0.46;
  graph.nodes.forEach(n => {
    n.impN = (n.imp - lo) / (hi - lo);          // 0 (peripheral) … 1 (core)
    n.tr = 16 + (Rmax - 16) * (1 - n.impN);      // target ring radius — important = small = centre
    n.rad = (n.type === "memory" ? 5 : 3.5) + n.impN * 9;
  });

  buildGraphSvg();
  graph.px = 0; graph.py = 0; graph.k = 1; applyTransform();
  $("graphLegend").innerHTML =
    `${graph.nodes.length} nodes · ${graph.links.length} links — click a blob to edit` +
    `<br><span style="opacity:.7;">scroll = zoom · drag = pan · double-click = reset</span>`;
  graph.frozen = false; graph.phase = 0;
  cancelAnimationFrame(graph.raf); graphTick();
}

function buildGraphSvg() {
  const svg = $("graphSvg"); svg.innerHTML = "";
  // A shared white radial-gradient for the glow: opaque-ish near the core, fading to transparent at
  // the rim (objectBoundingBox units, so it scales to each halo). A thin, white, gradient halo.
  const defs = document.createElementNS(GNS, "defs");
  const grad = document.createElementNS(GNS, "radialGradient"); grad.setAttribute("id", "memGlow");
  [["0%", "#ffffff", "0.75"], ["60%", "#ffffff", "0.4"], ["100%", "#eaf6ff", "0"]].forEach(([o, c, op]) => {
    const st = document.createElementNS(GNS, "stop");
    st.setAttribute("offset", o); st.setAttribute("stop-color", c); st.setAttribute("stop-opacity", op);
    grad.appendChild(st);
  });
  defs.appendChild(grad); svg.appendChild(defs);
  const root = document.createElementNS(GNS, "g"); svg.appendChild(root); graph.root = root;
  graph.links.forEach(l => { l.el = document.createElementNS(GNS, "line"); root.appendChild(l.el); });
  graph.nodes.forEach(n => {
    const g = document.createElementNS(GNS, "g"); g.setAttribute("class", "node");
    const col = glowColor(n.impN);
    const halo = document.createElementNS(GNS, "circle");
    halo.setAttribute("class", "halo");        // thin white glow: ~4–10px beyond the core, gradient-faded
    halo.setAttribute("r", (n.rad + 4 + n.impN * 6).toFixed(1));
    halo.setAttribute("fill", "url(#memGlow)"); g.appendChild(halo);
    const c = document.createElementNS(GNS, "circle");
    c.setAttribute("class", "core"); c.setAttribute("r", n.rad.toFixed(1)); c.setAttribute("fill", col);
    g.appendChild(c);
    const tx = document.createElementNS(GNS, "text");
    tx.setAttribute("x", (n.rad + 3).toFixed(1)); tx.setAttribute("y", 4); tx.textContent = n.label || "";
    g.appendChild(tx);
    g.addEventListener("click", (e) => { e.stopPropagation(); selectNode(n); });
    root.appendChild(g); n.el = g;
  });
}

function graphTick() {
  const n = graph.nodes, L = graph.links, cx = graph.w / 2, cy = graph.h / 2;
  const SPIN = 0.0004;  // very slow galaxy rotation (~2.5°/s) — easy to click
  // Gentle "breathing": the target ring radius drifts in/out by a few percent on a slow sine.
  const breathe = 1 + 0.045 * Math.sin(graph.phase * 0.0045);
  graph.phase++;
  for (let i = 0; i < n.length; i++) {
    for (let j = i + 1; j < n.length; j++) {
      const a = n[i], b = n[j]; const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01;
      const d = Math.sqrt(d2), f = 900 / d2; const fx = dx / d * f, fy = dy / d * f;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    }
  }
  L.forEach(l => {
    const a = l.source, b = l.target; const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) + 0.01, f = (d - 70) * 0.006;
    const fx = dx / d * f, fy = dy / d * f; a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
  });
  const cs = Math.cos(SPIN), sn = Math.sin(SPIN);
  n.forEach(p => {
    const dx = p.x - cx, dy = p.y - cy, r = Math.sqrt(dx * dx + dy * dy) + 1e-3;
    const fr = (p.tr * breathe - r) * 0.02;       // pull toward this node's (breathing) target ring
    p.vx += dx / r * fr; p.vy += dy / r * fr;
    p.vx *= 0.9; p.vy *= 0.9;
    let nx = p.x + p.vx, ny = p.y + p.vy;
    const rx = nx - cx, ry = ny - cy;             // steady slow rotation about the centre
    p.x = cx + rx * cs - ry * sn; p.y = cy + rx * sn + ry * cs;
    p.el.setAttribute("transform", `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`);
  });
  L.forEach(l => {
    l.el.setAttribute("x1", l.source.x.toFixed(1)); l.el.setAttribute("y1", l.source.y.toFixed(1));
    l.el.setAttribute("x2", l.target.x.toFixed(1)); l.el.setAttribute("y2", l.target.y.toFixed(1));
  });
  if (graph.on && !graph.frozen) graph.raf = requestAnimationFrame(graphTick);  // freeze on select
}

function graphDeselect() {
  $("graphInspect").style.display = "none";
  if (graph.sel && graph.sel.el) { graph.sel.el.classList.remove("sel"); graph.sel = null; }
  if (graph.frozen) { graph.frozen = false; cancelAnimationFrame(graph.raf); graphTick(); }  // resume drift
}

function selectNode(n) {
  graph.frozen = true; cancelAnimationFrame(graph.raf);  // freeze the galaxy while you inspect/edit
  if (graph.sel && graph.sel.el) graph.sel.el.classList.remove("sel");
  graph.sel = n; n.el.classList.add("sel");
  const box = $("graphInspect"); box.style.display = "block"; box.innerHTML = "";
  if (n.type !== "memory") {
    const deg = graph.links.filter(l => l.source === n || l.target === n).length;
    box.innerHTML = `<h2 style="margin-top:0;">Entity</h2><div class="text">${escapeHtml(n.label)}</div>` +
      `<div class="hint" style="margin-top:6px;">${deg} connection(s)</div>`;
    return;
  }
  const h = document.createElement("h2"); h.style.marginTop = "0"; h.textContent = "Memory"; box.appendChild(h);
  const ta = document.createElement("textarea"); ta.value = n.text; box.appendChild(ta);
  const tags = document.createElement("div"); tags.className = "tags"; tags.style.margin = "8px 0";
  [`tier: ${n.tier || "?"}`, `source: ${n.provenance || "?"}`].forEach(t => {
    const s = document.createElement("span"); s.className = "tag"; s.textContent = t; tags.appendChild(s);
  });
  box.appendChild(tags);
  const sl = document.createElement("label"); sl.className = "hint";
  sl.style.cssText = "display:flex; align-items:center; gap:6px;"; sl.textContent = "salience";
  const sin = document.createElement("input"); sin.type = "number"; sin.step = "0.1"; sin.min = "0";
  sin.value = n.salience; sin.style.width = "70px"; sl.appendChild(sin); box.appendChild(sl);
  const row = document.createElement("div"); row.className = "row"; row.style.marginTop = "10px";
  const save = document.createElement("button"); save.textContent = "Save";
  save.addEventListener("click", async () => {
    try {
      await api("POST", "/api/memory",
        { action: "update", id: n.mid, text: ta.value, salience: parseFloat(sin.value) });
      box.style.display = "none"; loadGraphMap();
    } catch (e) { alert("Error: " + e.message); }
  });
  const del = document.createElement("button"); del.className = "secondary"; del.textContent = "Delete";
  del.addEventListener("click", async () => {
    if (!confirm("Delete this memory permanently?")) return;
    try { await api("POST", "/api/memory", { action: "delete", id: n.mid }); box.style.display = "none"; loadGraphMap(); }
    catch (e) { alert("Error: " + e.message); }
  });
  row.appendChild(save); row.appendChild(del); box.appendChild(row);
}

function toggleGraph() {
  graph.on = !graph.on;
  if (graph.on && typeof forum !== "undefined" && forum.on) {  // mutually exclusive takeover views
    forum.on = false; $("forumView").style.display = "none"; $("forumToggle").textContent = "🏛 Forum";
  }
  $("graphView").style.display = graph.on ? "block" : "none";
  $("log").style.display = graph.on ? "none" : "";
  $("composer").style.display = graph.on ? "none" : "";
  $("graphToggle").textContent = graph.on ? "💬 Chat" : "🕸 Graph";
  if (graph.on) loadGraphMap();
  else { cancelAnimationFrame(graph.raf); $("graphInspect").style.display = "none"; }
}
$("graphToggle").addEventListener("click", toggleGraph);

// --- the council forum: deliberations as browsable threads (comment + full-admin housekeeping) ---
const forum = { on: false, threadId: null };

function toggleForum() {
  forum.on = !forum.on;
  if (forum.on && graph.on) {  // the two takeover views are mutually exclusive
    graph.on = false; $("graphView").style.display = "none";
    $("graphToggle").textContent = "🕸 Graph"; cancelAnimationFrame(graph.raf);
  }
  $("forumView").style.display = forum.on ? "block" : "none";
  $("log").style.display = forum.on ? "none" : "";
  $("composer").style.display = forum.on ? "none" : "";
  $("forumToggle").textContent = forum.on ? "💬 Chat" : "🏛 Forum";
  if (forum.on) { forum.threadId = null; loadForumList(); }
}
$("forumToggle").addEventListener("click", toggleForum);

function forumWhen(ts) { return ts ? new Date(ts * 1000).toLocaleString() : ""; }

async function loadForumList() {
  const el = $("forumView");
  el.innerHTML = '<div class="hint">Loading…</div>';
  let threads;
  try { threads = (await api("GET", "/api/forum")).threads || []; }
  catch (e) { el.innerHTML = '<div class="hint">Error: ' + escapeHtml(e.message) + '</div>'; return; }
  const ask =
    '<div style="display:flex; gap:8px; margin-bottom:14px;">' +
    '<input type="text" id="forumAsk" placeholder="Ask the council a question…" style="flex:1;"/>' +
    '<button id="forumAskBtn" type="button">Ask the council</button></div>' +
    '<div id="forumAskMsg" class="hint" style="margin-bottom:10px;"></div>';
  if (!threads.length) {
    el.innerHTML = ask + '<div class="hint">No deliberations yet. Ask the council above, ' +
      'or let it argue its own conflicts during sleep.</div>';
  } else {
    el.innerHTML = ask + threads.map(t => {
      const badge = t.status === "closed"
        ? '<span style="color:#8896a6;">● closed</span>'
        : '<span style="color:#7fd17f;">● open</span>';
      return `<div class="mem forum-thread" data-id="${t.id}" style="cursor:pointer;">` +
        `<div class="text"><b>${escapeHtml(t.question)}</b></div>` +
        `<div class="meta">${badge} · ${escapeHtml(t.source)} · ${t.posts} posts · ${forumWhen(t.created_at)}</div></div>`;
    }).join("");
    el.querySelectorAll(".forum-thread").forEach(d =>
      d.addEventListener("click", () => openForumThread(parseInt(d.dataset.id))));
  }
  $("forumAskBtn").addEventListener("click", forumAsk);
  $("forumAsk").addEventListener("keydown", e => { if (e.key === "Enter") forumAsk(); });
}

async function forumAsk() {
  const q = $("forumAsk").value.trim(); if (!q) return;
  $("forumAskMsg").textContent = "The council is convening across the fleet… (this can take a while)";
  $("forumAskBtn").disabled = true;
  try {
    const r = await api("POST", "/api/forum", { action: "ask", question: q });
    if (r.thread_id) openForumThread(r.thread_id); else loadForumList();
  } catch (e) { $("forumAskMsg").textContent = "Error: " + e.message; }
  finally { $("forumAskBtn").disabled = false; }
}

async function openForumThread(id) {
  forum.threadId = id;
  const el = $("forumView");
  el.innerHTML = '<div class="hint">Loading…</div>';
  let t;
  try { t = await api("GET", "/api/forum/thread?id=" + id); }
  catch (e) { el.innerHTML = '<div class="hint">Error: ' + escapeHtml(e.message) + '</div>'; return; }
  const closed = t.status === "closed";
  const head =
    '<button class="secondary" id="forumBack" type="button">← All threads</button>' +
    `<h2 style="margin:10px 0 4px;">${escapeHtml(t.question)}</h2>` +
    `<div class="hint" style="margin-bottom:10px;">${escapeHtml(t.source)} · ${closed ? "closed" : "open"} · ${forumWhen(t.created_at)} ` +
    `<button class="secondary" id="forumStatus" type="button">${closed ? "Reopen" : "Close"}</button> ` +
    `<button class="secondary" id="forumDelThread" type="button">Delete thread</button></div>`;
  const posts = (t.posts || []).map(p => {
    const where = (p.node || p.model) ? ` · ${escapeHtml([p.node, p.model].filter(Boolean).join(" @ "))}` : "";
    const isVerdict = p.kind === "verdict";
    const style = isVerdict
      ? 'border-left:3px solid #7fa8d1; background:#1a2230;'
      : (p.kind === "comment" ? 'border-left:3px solid #6f7a8a;' : '');
    return `<div class="mem" style="${style}">` +
      `<div class="meta"><b>${escapeHtml(p.author)}</b> · ${escapeHtml(p.kind)}${where} · ${forumWhen(p.created_at)} ` +
      `<a href="#" class="forum-delpost" data-id="${p.id}" style="color:#c98; margin-left:6px;">delete</a></div>` +
      `<div class="text" style="white-space:pre-wrap;">${escapeHtml(p.content)}</div></div>`;
  }).join("");
  const comment =
    '<div style="display:flex; gap:8px; margin-top:12px;">' +
    '<input type="text" id="forumComment" placeholder="Add a comment…" style="flex:1;"/>' +
    '<button id="forumCommentBtn" type="button">Comment</button></div>';
  el.innerHTML = head + posts + comment;
  $("forumBack").addEventListener("click", loadForumList);
  $("forumStatus").addEventListener("click", async () => {
    await api("POST", "/api/forum", { action: closed ? "reopen" : "close", thread_id: id });
    openForumThread(id);
  });
  $("forumDelThread").addEventListener("click", async () => {
    await api("POST", "/api/forum", { action: "delete_thread", thread_id: id });
    loadForumList();
  });
  el.querySelectorAll(".forum-delpost").forEach(a => a.addEventListener("click", async (e) => {
    e.preventDefault();
    await api("POST", "/api/forum", { action: "delete_post", post_id: parseInt(a.dataset.id) });
    openForumThread(id);
  }));
  const send = async () => {
    const text = $("forumComment").value.trim(); if (!text) return;
    await api("POST", "/api/forum", { action: "comment", thread_id: id, text });
    openForumThread(id);
  };
  $("forumCommentBtn").addEventListener("click", send);
  $("forumComment").addEventListener("keydown", e => { if (e.key === "Enter") send(); });
}

// Zoom (scroll, toward the cursor) + pan (drag the background). A background click with no drag
// deselects. Node clicks are handled on the node (stopPropagation), so dragging never starts there.
(() => {
  const svg = $("graphSvg");
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const nk = Math.max(0.25, Math.min(5, graph.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
    graph.px = mx - (mx - graph.px) * (nk / graph.k);
    graph.py = my - (my - graph.py) * (nk / graph.k);
    graph.k = nk; applyTransform();
  }, { passive: false });
  let drag = null;
  svg.addEventListener("mousedown", (e) => {
    if (e.target.closest(".node")) return;  // let node clicks through
    drag = { x: e.clientX, y: e.clientY, px: graph.px, py: graph.py, moved: false };
    svg.classList.add("panning");
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    graph.px = drag.px + dx; graph.py = drag.py + dy; applyTransform();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    const moved = drag.moved; drag = null; svg.classList.remove("panning");
    if (!moved) graphDeselect();  // a plain click on the background
  });
  svg.addEventListener("dblclick", () => { graph.px = 0; graph.py = 0; graph.k = 1; applyTransform(); });
})();

async function loadWikiStatus() {
  const el = $("wikiStatus"); el.textContent = "checking…";
  try {
    const s = await api("GET", "/api/wiki/status");
    if (!s.enabled) {
      el.innerHTML = "Not configured — add a <code>[wiki]</code> block in mimir.toml " +
        "(point it at a running <code>kiwix-serve</code>). See docs/SETUP.md §5c.";
    } else if (s.reachable) {
      el.innerHTML = `<span style="color:#7fd17f;">✓ connected</span> — book ` +
        `<b>${s.book}</b> at ${s.url}`;
    } else {
      el.innerHTML = `<span style="color:#e0a0a0;">✗ not reachable</span> — ${s.url}` +
        ` (book ${s.book || "?"})${s.error ? " · " + s.error : ""}. Is kiwix-serve running?`;
    }
  } catch (e) { el.textContent = "error: " + e.message; }
}
$("wikiRecheck").addEventListener("click", loadWikiStatus);

// --- tabs ---
const loaders = { mind: loadMind, sleep: loadSleepTab, memories: loadMemories, graph: loadGraph, procedures: loadProcedures, fleet: loadFleet, docs: loadWikiStatus };
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

async function loadSleepStatus() {
  const el = $("sleepStatus");
  try {
    const s = await api("GET", "/api/sleep/status");
    const win = `${s.window_start}–${s.window_end}`;
    const sched = s.enabled
      ? `Quiet window <b>${win}</b>${s.in_window ? ' <span style="color:#7fd17f;">(open now)</span>' : ''}`
      : `Scheduler <b>off</b> — manual only`;
    let last = "never run";
    if (s.last_cycle_date) {
      const phases = Object.entries(s.phases || {}).map(([k,v]) => `${k}: ${v}`).join(", ");
      last = `last: ${s.last_cycle_date}${s.completed ? " ✓" : " (partial)"}${phases ? " — " + phases : ""}`;
    }
    const off = s.utc_offset ? "UTC" + s.utc_offset.replace(/(\\d{2})(\\d{2})$/, "$1:$2") : "";
    const sys = `system local time${off ? ", " + off : ""}`;
    let clock;
    if (!s.timezone) clock = `Now ${s.now_local} (${sys})`;
    else if (s.timezone_active) clock = `Now ${s.now_local} (${escapeHtml(s.timezone)})`;
    else clock = `Now ${s.now_local} (${sys}) <span class="hint">— “${escapeHtml(s.timezone)}” ` +
      `needs the <code>tzdata</code> extra; or pick a UTC offset for a zero-dep zone</span>`;
    el.innerHTML = `${sched}. ${last}. ${clock}.`;
  } catch (e) { el.textContent = "error: " + e.message; }
}

let _tzZones = null;
async function fillTz(sel, current) {
  if (!_tzZones) { _tzZones = (await api("GET", "/api/timezones")).zones || []; }
  sel.innerHTML = '<option value="">System local time (recommended)</option>';
  _tzZones.forEach(z => { const o = document.createElement("option"); o.value = z; o.textContent = z; sel.appendChild(o); });
  sel.value = current || "";
}

async function loadSleepTab() {
  try {
    const s = await api("GET", "/api/settings");
    await fillTz($("setTz"), s.timezone);
    $("setStart").value = s.sleep_window_start || "02:00";
    $("setEnd").value = s.sleep_window_end || "06:00";
    $("setEnabled").checked = !!s.sleep_enabled;
    $("setDeliberate").checked = !!s.deliberation_enabled;
  } catch (e) { $("settingsMsg").textContent = "error: " + e.message; }
  loadSleepStatus();
}

$("saveSleep").addEventListener("click", async () => {
  $("settingsMsg").textContent = "Saving…";
  try {
    await api("POST", "/api/settings", { settings: {
      timezone: $("setTz").value,
      sleep_window_start: $("setStart").value,
      sleep_window_end: $("setEnd").value,
      sleep_enabled: $("setEnabled").checked,
      deliberation_enabled: $("setDeliberate").checked,
    }});
    $("settingsMsg").textContent = "Saved.";
    setTimeout(() => $("settingsMsg").textContent = "", 1500);
    loadSleepStatus();
  } catch (e) { $("settingsMsg").textContent = "Error: " + e.message; }
});

$("sleepBtn").addEventListener("click", async () => {
  $("sleepResult").textContent = "Running sleep cycle…";
  try {
    const r = await api("POST", "/api/sleep");
    const counts = (r.deduped !== undefined)
      ? ` Deduped ${r.deduped} · decayed ${r.decayed} · archived ${r.archived} · contradictions ${r.contradictions_resolved}.`
      : "";
    $("sleepResult").textContent = `Ran ${(r.ran||[]).join(", ") || "nothing"}.${counts}`;
    loadSleepStatus(); loadMind(); refreshState();
  } catch (e) { $("sleepResult").textContent = "Error: " + e.message; }
});

$("delibBtn").addEventListener("click", async () => {
  $("delibResult").innerHTML = '<span class="hint">Surfacing conflicts and convening the council… (this can take a while)</span>';
  try {
    const r = await api("POST", "/api/deliberate/run");
    const ran = r.ran || [];
    if (!ran.length) {
      $("delibResult").innerHTML = `<span class="hint">No open conflicts to argue${r.surfaced ? ` (surfaced ${r.surfaced}, none fresh)` : ""}.</span>`;
      return;
    }
    $("delibResult").innerHTML = `<div class="hint">Argued ${ran.length} of ${r.surfaced} surfaced:</div>` +
      ran.map(d => `<div class="mem"><div class="text"><b>Q:</b> ${escapeHtml(d.question)}</div>` +
        `<div class="text" style="color:#9fb3c8;"><b>Verdict:</b> ${escapeHtml(d.verdict)}</div></div>`).join("");
    refreshState();
  } catch (e) { $("delibResult").innerHTML = '<span class="hint">Error: ' + escapeHtml(e.message) + '</span>'; }
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
    loadModels();        // the merged tab also shows role assignment + the model pool
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
    // Lock the time trial only mid-tournament (a round running, or a veto pending that FIGHT resumes).
    // At 'done' the tournament is terminal and the speed-test is the next step → leave it enabled.
    const midTourney = ts.phase === "running" || ts.phase === "awaiting_veto";
    setMatrixEnabled(!midTourney);
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
  // the composer is hidden while the board OR the interview owns the pane
  $("composer").style.display = (on || ivState.active) ? "none" : "";
  if (on) { maybeStartInterview(); }                       // pair the interview with the tournament
  else if (ivState.mode === "board") interviewShow(false); // board gone → retire a board-paired strip
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
  let h = `<h2>${header || "🏁 Benchmarking…"} <button class="secondary" style="margin-left:auto; padding:4px 10px;" onclick="showPlacement()">📊 Per-node placement</button> <button class="secondary" style="padding:4px 10px;" onclick="showCouncil()">🏟️ Council</button> <button class="secondary" style="padding:4px 10px;" onclick="closeBench()">✕ Close</button></h2>`;
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

// The per-node placement matrix — every model on EVERY node it runs on, with that node's speed and
// each node's winner. This is the speed-test's whole output, which the results board (one row per
// model on its test node) never shows. Reads /api/fleet/placement (the live catalogue).
async function showPlacement() {
  try { renderPlacement(await api("GET", "/api/fleet/placement")); }
  catch (e) { $("fleetMsg").textContent = "Error loading placement: " + e.message; }
}
function renderPlacement(data) {
  _benchBoardClosed = false; benchShow(true);
  let h = '<h2>📊 Per-node placement — what runs best on each node <button class="secondary" style="margin-left:auto; padding:4px 10px;" onclick="closeBench()">✕ Close</button></h2>';
  h += '<div class="legend">🏆 node winner (best quality, speed breaks ties) · ⚡ fastest here · ✅ ≥0.80 🟡 0.50–0.79 ❌ &lt;0.50 · speed is per-node · roles: green = eligible, ⊘ = barred</div>';
  h += '<div class="hint" style="margin:6px 0;">☑ Untick a machine to <b>exclude it from qualification + routing</b> — e.g. disable the GPU box to see what your edge nodes can do <i>on their own</i> (the whole point: useful home AI without killer compute). Takes effect on the next benchmark/tournament.</div>';
  const disabledNodes = new Set(data.disabled_nodes || []);
  const nodes = Object.keys(data.by_node || {}).sort((a, b) => {
    const la = a.includes("127.0.0.1"), lb = b.includes("127.0.0.1");
    if (la !== lb) return la ? -1 : 1; return a.localeCompare(b);
  });
  if (!nodes.length) { $("benchBoard").innerHTML = h + '<div class="hint" style="margin-top:12px;">No placement data yet — run a benchmark, then the speed-test.</div>'; return; }
  nodes.forEach(node => {
    const models = data.by_node[node];
    const off = disabledNodes.has(node);
    const champ = off ? null : models.find(m => m.champion);
    h += `<div style="${off ? "opacity:0.45;" : ""}">`;
    h += `<div class="nodehdr" style="margin-top:14px;"><label style="cursor:pointer;" title="Use this machine in the fleet. Untick to exclude it from qualification + routing."><input type="checkbox" ${off ? "" : "checked"} onchange="toggleNodeFromView('${node}', this.checked)"> ${shortNode(node)}</label> · ${models.length} model(s)${off ? " — <b>disabled</b> (excluded)" : (champ ? ` · winner 🏆 <b>${champ.model}</b> (q${(champ.quality ?? 0).toFixed(2)} · ${champ.return_time != null ? champ.return_time.toFixed(1) + "s" : "·"})` : "")}</div>`;
    h += "<table><tr><th></th><th>Model</th><th>Quality</th><th>Talk</th><th>Tools</th><th>Code</th><th>Reason</th><th>Disc</th><th>Epis</th><th>Coh</th><th>Speed</th><th>Roles</th></tr>";
    models.forEach(m => {
      const flag = (m.champion ? "🏆" : "") + (m.fastest ? "⚡" : "");
      const elig = (m.eligible_roles || []).join(", ");
      const barEntries = Object.entries(m.barred || {});
      const bars = barEntries.map(([r]) => "⊘" + r).join(" ");
      const barTitle = barEntries.map(([r, w]) => `${r}: ${w}`).join("; ");
      h += `<tr class="${m.champion ? "top" : ""}" style="${m.enabled ? "" : "opacity:0.5;"}"><td>${flag}</td><td>${m.model}</td>`
        + `<td class="q">${_stars(m.quality)} <span style="color:#8a94a3; font-weight:400;">${(m.quality ?? 0).toFixed(2)}</span></td>`
        + `<td>${_emoji(m.talk)}</td><td>${_emoji(m.tools)}</td><td>${_emoji(m.code)}</td><td>${_emoji(m.reasoning)}</td><td>${_emoji(m.discipline)}</td><td>${_emoji(m.epistemics)}</td><td>${_emoji(m.coherence)}</td>`
        + `<td>${m.return_time != null ? m.return_time.toFixed(1) + "s" : "·"}</td>`
        + `<td style="font-size:11px;"><span style="color:#7fd17f;">${elig}</span> <span style="color:#e0a0a0;" title="${barTitle}">${bars}</span></td></tr>`;
    });
    h += "</table></div>";
  });
  $("benchBoard").innerHTML = h;
}
// Toggle a machine on/off from the placement view (excludes it from qualification + routing), then
// re-render so the change is visible. The thesis: disable the GPU beast to test the edge-only fleet.
async function toggleNodeFromView(node, enabled) {
  try {
    await api("POST", "/api/fleet/node", { node, enabled });
    $("fleetMsg").textContent = `${shortNode(node)} ${enabled ? "enabled" : "disabled"} — applies on the next benchmark/tournament.`;
    showPlacement();
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
}

// The adversarial-council roster (the "second lineup") — a SPREAD of model families, not the top-N.
// Diversity is the point: different families fail differently, so a family-spread panel beats five
// variants of the best model. Reads /api/fleet/council.
let _councilSize = 5;
async function showCouncil(size) {
  if (size) _councilSize = size;
  try { renderCouncil(await api("GET", "/api/fleet/council?size=" + _councilSize)); }
  catch (e) { $("fleetMsg").textContent = "Error loading council: " + e.message; }
}
// Grade the big models above the chat cap (caps off, in place — no rescan), so they enter the
// council pool. Reuses the benchmark board/progress; reopen the council when it finishes.
async function qualifyCouncilPool() {
  _benchBoardClosed = false;
  $("fleetMsg").textContent = "Grading the council pool — big models, caps off…";
  try {
    const res = await api("POST", "/api/fleet/benchmark/council", {});
    if (res.started === false) { $("fleetMsg").textContent = "Busy: " + (res.error || "a run is already in progress"); return; }
    pollBenchmark();   // same background board; click 🏟️ Council again when it's done
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
}
function renderCouncil(data) {
  _benchBoardClosed = false; benchShow(true);
  const r = data.roster || [];
  let h = '<h2>🏟️ Adversarial council — the diverse second lineup <button class="secondary" style="margin-left:auto; padding:4px 10px;" onclick="closeBench()">✕ Close</button></h2>';
  h += `<div class="legend">A <b>spread of ${data.families.length} families</b> across ${data.size} seat(s) — diversity beats ranking for adversarial reasoning. Not latency-gated (big-and-slow welcome). Pool: ${data.pool} models in ${data.pool_families} families.</div>`;
  h += '<div class="row" style="margin:8px 0; gap:6px; align-items:center;"><span class="hint">Seats:</span>'
    + [3, 5, 7].map(n => `<button class="secondary" style="padding:3px 10px;${n === _councilSize ? "border-color:#2d8;" : ""}" onclick="showCouncil(${n})">${n}</button>`).join("")
    + '<button class="secondary" style="margin-left:14px; padding:3px 10px;" onclick="qualifyCouncilPool()" title="Grade the big models above the chat size cap (caps off, no rescan) so they enter the council pool. Run a tournament/benchmark first.">🏋️ Qualify big models</button></div>';
  if (!r.length) { $("benchBoard").innerHTML = h + '<div class="hint" style="margin-top:12px;">No qualified models seated yet — run a benchmark/tournament first. The big ≥cap council models (the 122B etc.) are skipped by the chat size cap, so click <b>🏋️ Qualify big models</b> to grade them caps-off into the pool.</div>'; return; }
  h += '<table><tr><th>Seat</th><th>Family</th><th>Model</th><th>Quality</th><th>Reason</th><th>Runs on</th></tr>';
  r.forEach((m, i) => {
    const t = m.return_time != null ? ` · ${m.return_time.toFixed(1)}s` : "";
    h += `<tr class="${i === 0 ? "top" : ""}"><td>${i + 1}</td><td><b>${m.family}</b></td><td>${m.model}</td>`
      + `<td class="q">${_stars(m.quality)} <span style="color:#8a94a3; font-weight:400;">${(m.quality ?? 0).toFixed(2)}</span></td>`
      + `<td>${_emoji(m.reasoning)}</td><td>${shortNode(m.node || "")}${t}</td></tr>`;
  });
  h += "</table>";
  const bench = data.bench || [];
  if (bench.length) {
    h += `<div class="hint" style="margin-top:10px;"><b>Bench</b> (qualified, next-up): ` + bench.slice(0, 10).map(b => `${b.model} <span style="opacity:.6;">(${b.family})</span>`).join(", ") + (bench.length > 10 ? ` +${bench.length - 10} more` : "") + "</div>";
  }
  $("benchBoard").innerHTML = h;
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
    h += '<div class="hint" style="margin-top:12px;">🏆 Tournament complete. <b>4 · Apply</b> routes your chat / bake / reasoning roles now. <b>3 · Speed-test</b> is optional — it times the remaining edges to fill the placement matrix (which edge can host which model, for future background/council work); slow edges take 30–70s each. You can Apply without it.</div>';
    h += '<div class="row" style="margin-top:8px; gap:10px;"><button class="secondary" type="button" onclick="speedTestFromTourney()">⏱ 3 · Speed-test remaining nodes</button><button class="secondary" type="button" onclick="showPlacement()">📊 Per-node placement</button><button class="secondary" type="button" onclick="showCouncil()">🏟️ Council roster</button><button id="tourneyApplyBtn" type="button" onclick="applyTourney()">✅ 4 · Apply finals to roles</button><button class="secondary" type="button" onclick="closeBench()">Done</button></div>';
    h += '<div id="tourneyMatrixMsg" class="hint" style="margin-top:6px;"></div>';
  } else if (phase === "error") {
    h += `<div class="hint" style="color:#ff8a8a; margin-top:10px;">Error: ${s.error}</div>`;
  }
  $("benchBoard").innerHTML = h;
}

let _tourneyPolling = false;
let _tourneyCur = "", _tourneyCurT = 0;   // current model + when it started, for an elapsed timer
async function pollTournament() {
  if (_tourneyPolling) return;
  _tourneyPolling = true; $("fleetTourneyBtn").disabled = true; btnState("fleetTourneyBtn", "working"); setMatrixEnabled(false);
  try {
    while (true) {
      const s = await api("GET", "/api/fleet/tournament/status");
      if (!s.active) { setMatrixEnabled(true); break; }
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
      else if (s.phase === "done") { $("fleetMsg").textContent = "🏆 Tournament complete — review the finals."; btnState("fleetTourneyBtn", "done"); setMatrixEnabled(true); }
      else if (s.phase === "error") { $("fleetMsg").textContent = "Tournament error: " + s.error; btnState("fleetTourneyBtn", "failed"); setMatrixEnabled(true); }
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

// The time trial shares GPUs with the benchmark/tournament, so the button is locked out (greyed)
// while either runs and reappears available after. The backend refuses it too — this is just the
// visible signal so you never wonder whether a mid-run click did something.
const _MATRIX_TITLE = ($("fleetMatrixBtn") || {}).title || "";
function setMatrixEnabled(on) {
  const b = $("fleetMatrixBtn"); if (!b) return;
  b.disabled = !on;
  b.title = on ? _MATRIX_TITLE
              : "Finish the benchmark / tournament first — the time trial shares the same GPUs and would collide with it.";
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
  _benchPolling = true; $("fleetBenchBtn").disabled = true; btnState("fleetBenchBtn", "working"); setMatrixEnabled(false);
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
  _benchPolling = false; $("fleetBenchBtn").disabled = false; setMatrixEnabled(true);
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
        const msg = (s.timed === 0)
          ? "✓ Speed-test: every eligible model is already timed — nothing to do. Go ahead and Apply."
          : (s.timed !== undefined ? `✓ Speed-test complete — timed ${s.timed} (model, node) pairing(s). Per-node times are now filled in.` : "✓ Speed-test done.");
        _setMatrixMsg(msg);
        loadFleet();
        break;
      }
      _setMatrixMsg(s.total ? `⏱ Speed-testing ${s.i}/${s.total}: ${s.current}… (slow edges can take 30–70s each)` : `⏱ ${s.current || "checking which pairings need timing"}…`);
      await new Promise(r => setTimeout(r, 1200));
    }
  } catch (e) { $("fleetMsg").textContent = "Error: " + e.message; }
  _matrixPolling = false; $("fleetMatrixBtn").disabled = false;
}

// Mirror time-trial status to BOTH the Fleet tab (fleetMsg) and the tournament board
// (tourneyMatrixMsg, when present), so the user gets feedback wherever they're looking.
function _setMatrixMsg(t) { $("fleetMsg").textContent = t; const b = $("tourneyMatrixMsg"); if (b) b.textContent = t; }

async function startMatrix() {
  _setMatrixMsg("Starting the time trial — checking which (model, node) pairings still need timing…");
  btnState("fleetMatrixBtn", "working");
  try {
    const r = await api("POST", "/api/fleet/matrix", {});   // returns immediately; runs in the background
    if (r.started === false) {   // backend refused (a benchmark/tournament is running) — don't poll
      _setMatrixMsg(r.reason || "The time trial is busy.");
      btnState("fleetMatrixBtn", null); setMatrixEnabled(!r.busy); return;
    }
    pollMatrix();
  } catch (e) { _setMatrixMsg("Error: " + e.message); btnState("fleetMatrixBtn", "failed"); }
}
$("fleetMatrixBtn").addEventListener("click", startMatrix);

// Programmatically activate a side-panel tab (reuses the tab button's own handler).
function switchTab(name) { const b = document.querySelector(`.tabs button[data-tab="${name}"]`); if (b) b.click(); }

// Run the speed-test from the tournament board (step 3 before step 4): the matrix progress lives on
// the Fleet tab (right panel), so switch there to show it; the tournament board (left/chat pane)
// stays put with its Apply button, so the user can apply finals once the matrix finishes.
function speedTestFromTourney() { switchTab("fleet"); startMatrix(); }

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

function ipTag(node) {
  // The node's last IP octet, e.g. "IP.189" — enough to tell edge boxes apart at a glance.
  if (!node) return "";
  const host = node.replace(/^https?:\\/\\//, "").replace(/:\\d+$/, "");
  if (host === "127.0.0.1" || host === "localhost") return "local";
  const parts = host.split("."); return "IP." + parts[parts.length - 1];
}

function renderRoleAssign(data) {
  // Per-role model picker. "auto" + EVERY (node, model) placement, grouped by model — so a model
  // that lives on several nodes is selectable per node (pin a role onto an edge box, off the beast).
  // Each per-node option shows the IP octet + that node's measured time. Choosing one pins the role
  // (POST /api/fleet/role with {model, node}); "<model> · any node" pins the model, routed live.
  const wrap = $("roleAssign"); if (!wrap) return;
  wrap.innerHTML = "";
  const active = data.active_roles || {};
  const autoRoles = new Set(data.auto_roles || []);
  const roleNodes = data.role_nodes || {};
  // model → [{node, t}], from the per-node placement matrix (the real per-(node,model) truth).
  const byModel = {};
  Object.entries((data.placement && data.placement.by_node) || {}).forEach(([node, models]) => {
    (models || []).forEach(m => {
      (byModel[m.model] = byModel[m.model] || []).push({ node, t: m.return_time });
    });
  });
  // Models discovered live but not yet in the catalogue still get an "any node" option.
  (data.available || []).forEach(m => { byModel[m] = byModel[m] || []; });
  Object.values(active).forEach(v => { if (v && v !== "auto") byModel[v] = byModel[v] || []; });
  const roles = Object.keys(active);
  if (!roles.length) { wrap.innerHTML = '<div class="hint">No roles configured.</div>'; return; }
  const enc = (m, n) => JSON.stringify({ m, n });
  roles.sort().forEach(role => {
    const curModel = active[role] || "auto";
    const curNode = roleNodes[role] || "";
    const isAuto = autoRoles.has(role);
    const row = document.createElement("div"); row.className = "field";
    row.style.display = "flex"; row.style.alignItems = "center"; row.style.gap = "10px";
    const lab = document.createElement("label"); lab.style.minWidth = "90px"; lab.style.margin = "0";
    lab.textContent = role;
    const sel = document.createElement("select");
    const autoOpt = document.createElement("option");
    autoOpt.value = "auto"; autoOpt.textContent = isAuto ? "auto (pick best)" : "auto (pick best)";
    sel.appendChild(autoOpt);
    Object.keys(byModel).sort().forEach(m => {
      const nodes = byModel[m];
      const grp = document.createElement("optgroup"); grp.label = m;
      const any = document.createElement("option");
      any.value = enc(m, ""); any.textContent = `${m} · any node`;
      if (!isAuto && m === curModel && !curNode) any.selected = true;
      grp.appendChild(any);
      nodes.slice().sort((a, b) => (a.t ?? 1e9) - (b.t ?? 1e9)).forEach(({ node, t }) => {
        const o = document.createElement("option");
        const ip = ipTag(node); const ts = (t != null) ? ` · ${t}s` : " · untimed";
        o.value = enc(m, node); o.textContent = `${m} · ${ip || node}${ts}`;
        if (!isAuto && m === curModel && node === curNode) o.selected = true;
        grp.appendChild(o);
      });
      sel.appendChild(grp);
    });
    if (isAuto) sel.value = "auto";
    sel.addEventListener("change", () => {
      if (!sel.value || sel.value === "auto") return;
      const { m, n } = JSON.parse(sel.value);
      setRole(role, m, n);
    });
    const tag = document.createElement("span"); tag.className = "tag";
    tag.textContent = isAuto ? "auto" : (curNode ? `pinned · ${ipTag(curNode) || curNode}` : "pinned");
    tag.title = isAuto ? "the system picks the best-qualified model for this role"
      : (curNode ? "fixed to one model on one node" : "fixed to one model, routed to its live-best node");
    row.appendChild(lab); row.appendChild(sel); row.appendChild(tag);
    wrap.appendChild(row);
  });
}

async function setRole(role, model, node) {
  const where = node ? ` on ${ipTag(node) || node}` : "";
  $("roleMsg").textContent = `Setting ${role} → ${model}${where}…`;
  try {
    await api("POST", "/api/fleet/role", { role, model, node: node || "" });
    $("roleMsg").textContent = `${role} pinned to ${model}${where}.`;
    loadModels(); refreshState();
    setTimeout(() => { $("roleMsg").textContent = ""; }, 2500);
  } catch (e) { $("roleMsg").textContent = "Error: " + e.message; }
}

async function loadModels() {
  try {
    const data = await api("GET", "/api/fleet/pool");
    renderRoleAssign(data);
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
      const serving = new Set(m.roles || []);
      (m.roles || []).forEach(r => bits.push("▶ " + r));                                    // serving now
      (m.eligible_roles || []).forEach(r => { if (!serving.has(r)) bits.push("✓ " + r); });  // qualifies for
      bits.forEach(b => { const sp = document.createElement("span"); sp.className = "tag"; sp.textContent = b; tags.appendChild(sp); });
      row.appendChild(tags);
      // Explain the verdict — never drop a barred role silently (DESIGN §10): show which role the
      // model is barred from and the floor it missed. Only for benchmarked models (an unbenchmarked
      // one already reads "·" above; listing "not benchmarked yet" for every role would be noise).
      if (m.benchmarked && m.barred && Object.keys(m.barred).length) {
        const bar = document.createElement("div"); bar.className = "tags";
        Object.entries(m.barred).forEach(([role, why]) => {
          const sp = document.createElement("span"); sp.className = "tag";
          sp.style.borderColor = "#7a3b3b"; sp.style.color = "#e0a0a0";
          sp.textContent = `⊘ ${role}: ${why}`; sp.title = `barred from ${role}: ${why}`;
          bar.appendChild(sp);
        });
        row.appendChild(bar);
      }
      list.appendChild(row);
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

// -- the seeding interview: a one-question-at-a-time strip under the tournament board -----------
// (and re-runnable any time from the Profile tab). Capture-only; answers persist immediately.
const ivState = { queue: [], i: 0, active: false, dismissed: false, mode: null, offramped: false };

function interviewShow(on) {
  ivState.active = on;
  $("interviewStrip").style.display = on ? "block" : "none";
  if (on) $("composer").style.display = "none";
  // when the interview ends, hand the composer back — unless the board still owns the pane
  else if ($("benchBoard").style.display === "none") $("composer").style.display = "";
}

async function maybeStartInterview() {
  if (ivState.active || ivState.dismissed) return;        // don't restart on every board poll
  let data; try { data = await api("GET", "/api/onboarding"); } catch (_) { return; }
  if (data.complete) return;
  startInterview(data.pending, "board");
}

function startInterview(queue, mode) {
  ivState.queue = queue; ivState.i = 0; ivState.dismissed = false; ivState.offramped = false;
  ivState.mode = mode || "manual";
  interviewShow(true);
  renderInterviewQ();
}

function renderOfframp() {
  // Reached the end of the Core 12 — offer to stop or keep going with the optional 7.
  $("ivForm").style.display = "none"; $("ivSkip").style.display = "none"; $("ivDone").style.display = "none";
  $("ivProgressTop").textContent = "";
  $("ivQ").textContent = "That's the essentials — you're all set. There are 7 more optional questions " +
    "for deeper grounding (the fleet benchmark takes a while anyway).";
  $("ivProgress").innerHTML =
    '<button class="secondary" type="button" id="ivMore" style="padding:3px 10px;">Continue · 7 more</button> ' +
    '<button class="secondary" type="button" id="ivStop" style="padding:3px 10px;">Finish here</button>';
  $("ivMore").addEventListener("click", () => { ivState.offramped = true; renderInterviewQ(); });
  $("ivStop").addEventListener("click", () => { ivState.dismissed = true; interviewShow(false); });
}

function renderInterviewQ() {
  const q = ivState.queue[ivState.i];
  if (q && q.core === false && !ivState.offramped) { renderOfframp(); return; }
  if (!q) {  // finished the queue → offer a quick schedule/timezone setup
    renderInterviewSchedule();
    return;
  }
  $("ivForm").style.display = ""; $("ivSkip").style.display = "";
  $("ivDone").style.display = ""; $("ivDone").textContent = "Later";
  $("ivQ").textContent = q.question;
  $("ivInput").value = q.answer || "";
  $("ivProgressTop").textContent = `${ivState.i + 1} / ${ivState.queue.length}`;
  $("ivProgress").textContent = "Press Enter to save · skippable · stored as your highest-trust facts.";
  $("ivInput").focus();
}

function renderInterviewSchedule() {
  // The final interview step: timezone + quiet hours, written to settings (not a memory).
  $("ivForm").style.display = "none"; $("ivSkip").style.display = "none";
  $("ivProgressTop").textContent = "";
  $("ivQ").textContent = "Last thing — when are you usually asleep or away? I'll do my background " +
    "upkeep then. (Change this anytime in the Sleep tab.)";
  $("ivProgress").innerHTML =
    '<div style="display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; margin-top:6px;">' +
    '<div><label style="display:block;font-size:11px;">Timezone</label><select id="ivTz"></select></div>' +
    '<div><label style="display:block;font-size:11px;">Quiet start</label><input type="time" id="ivStart" value="02:00"/></div>' +
    '<div><label style="display:block;font-size:11px;">Quiet end</label><input type="time" id="ivEnd" value="06:00"/></div>' +
    '<button class="secondary" type="button" id="ivSchedSave" style="padding:3px 10px;">Save</button>' +
    '<span id="ivSchedMsg" class="hint"></span></div>';
  $("ivDone").style.display = ""; $("ivDone").textContent = "Done";
  api("GET", "/api/settings").then(s => {
    fillTz($("ivTz"), s.timezone);
    $("ivStart").value = s.sleep_window_start || "02:00";
    $("ivEnd").value = s.sleep_window_end || "06:00";
  }).catch(() => fillTz($("ivTz"), ""));
  $("ivSchedSave").addEventListener("click", async () => {
    $("ivSchedMsg").textContent = "Saving…";
    try {
      await api("POST", "/api/settings", { settings: {
        timezone: $("ivTz").value, sleep_window_start: $("ivStart").value,
        sleep_window_end: $("ivEnd").value, sleep_enabled: true,
      }});
      $("ivSchedMsg").textContent = "Saved ✓ — all set.";
    } catch (e) { $("ivSchedMsg").textContent = "Error: " + e.message; }
  });
}

async function submitInterviewAnswer() {
  const q = ivState.queue[ivState.i]; if (!q) return;
  const answer = $("ivInput").value.trim();
  if (answer) {
    try { await api("POST", "/api/onboarding/answer", { key: q.key, answer }); }
    catch (e) { $("ivProgress").textContent = "Error: " + e.message; return; }
    refreshState();
    if (q.key === "assistant_name") applyAssistantName(answer);  // re-label the chat at once
    if (!$("tab-profile").classList.contains("hidden")) loadProfile();
  }
  ivState.i++; renderInterviewQ();
}

$("ivForm").addEventListener("submit", (e) => { e.preventDefault(); submitInterviewAnswer(); });
$("ivSkip").addEventListener("click", () => { ivState.i++; renderInterviewQ(); });
$("ivDone").addEventListener("click", () => { ivState.dismissed = true; interviewShow(false); });

// -- Profile tab: the editable 'one place' for the seeding facts --------------------------------
async function loadProfile() {
  let data; try { data = await api("GET", "/api/onboarding"); } catch (e) { $("profileMsg").textContent = "Error: " + e.message; return; }
  const wrap = $("profileFacts"); wrap.innerHTML = "";
  let dividerShown = false;
  data.profile.forEach(q => {
    if (q.core === false && !dividerShown) {  // mark where the optional, deeper questions begin
      dividerShown = true;
      const div = document.createElement("div"); div.className = "hint";
      div.style.cssText = "margin:14px 0 8px; opacity:.7; border-top:1px solid #232a35; padding-top:10px;";
      div.textContent = "— optional · deeper grounding —"; wrap.appendChild(div);
    }
    const d = document.createElement("div"); d.className = "profile-fact";
    const badge = q.anchor ? `<span class="anchorbadge">↳ self-model: ${q.anchor}</span>` : "";
    d.innerHTML = `<label>${q.question}${badge}</label>`;
    const inp = document.createElement("input"); inp.type = "text"; inp.className = "ans";
    inp.value = q.answer || ""; inp.placeholder = "(not answered)"; inp.dataset.key = q.key;
    inp.addEventListener("change", async () => {
      try {
        await api("POST", "/api/onboarding/answer", { key: q.key, answer: inp.value });
        if (q.key === "assistant_name") applyAssistantName(inp.value);
        $("profileMsg").textContent = "Saved."; refreshState();
        setTimeout(() => $("profileMsg").textContent = "", 1500);
      } catch (e) { $("profileMsg").textContent = "Error: " + e.message; }
    });
    d.appendChild(inp); wrap.appendChild(d);
  });
}

$("runInterviewBtn").addEventListener("click", async () => {
  let data; try { data = await api("GET", "/api/onboarding"); } catch (_) { return; }
  // re-runnable: walk every question, pre-filled with current answers, so it doubles as a refresh
  startInterview(data.profile.map(q => ({ key: q.key, question: q.question, answer: q.answer })), "manual");
});
$("profileReloadBtn").addEventListener("click", loadProfile);
loaders.profile = loadProfile;

// First-run nudge: if nothing's been captured yet, invite the interview (once, in the chat).
async function onboardingNudge() {
  try {
    const data = await api("GET", "/api/onboarding");
    if (!data.started) addMsg("mimir", "👋 Before we get going, I'd love to learn a little about you and this place — it makes me useful from the very first turn. Open the Profile tab and hit “Run interview” (or it'll appear under the tournament board while your fleet is being qualified). Everything's editable later, and you choose what I keep.");
  } catch (_) {}
}

// On load: populate the conversation dropdown and restore the most recent one into the chat (a
// refresh/restart no longer loses it). If there's no history yet, nudge the onboarding interview.
async function initConversation() {
  const sessions = await loadSessions();
  if (sessions.length && sessions[0].session_id) {
    $("sessionSelect").value = sessions[0].session_id;
    await restoreSession(sessions[0].session_id);
  } else {
    onboardingNudge();
  }
}

refreshState(); loadIdentity(); resumeFleetWork(); initConversation();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
