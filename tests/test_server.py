"""Executable spec for the reference web server (the human interaction surface)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from mimir.brain import Mimir
from mimir.config import Config
from mimir.server import create_server


def _json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get_html(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8")


@pytest.fixture
def base_url(mock_config: Config) -> Iterator[str]:
    brain = Mimir(mock_config)
    server = create_server(brain, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        brain.close()


def _sse(url: str, body: dict) -> tuple[str, dict | None]:
    """POST and parse the SSE stream into (joined token text, introspect dict)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode("utf-8")
    text, introspect = "", None
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event, payload = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload += line[5:].strip()
        if not payload:
            continue
        obj = json.loads(payload)
        if event == "token":
            text += obj["text"]
        elif event == "done":
            introspect = obj["introspect"]
    return text, introspect


def test_turn_stream_sse_recalls(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "alex"})
    text, introspect = _sse(
        base_url + "/api/turn/stream", {"text": "What is my favorite color?", "user": "alex"}
    )
    assert "teal" in text.lower()  # streamed tokens reconstruct the reply
    assert introspect is not None and introspect["source_count"] >= 1


def test_turn_stream_missing_text_is_4xx(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/turn/stream", {})
    assert status == 400
    assert "error" in data


def test_index_serves_ui(base_url: str) -> None:
    status, html = _get_html(base_url + "/")
    assert status == 200
    assert "Mimir 0" in html


def test_state_reports_embed_mode(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/state")
    assert status == 200
    assert data["embed_mode"] == "bootstrap"
    assert data["memories"] == 0


def test_identity_establish_over_http(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/identity", {"answers": {"name": "Mimir"}})
    assert status == 200
    assert data["anchors"]["name"] == "Mimir"
    assert "name" not in {k for k, _ in data["pending"]}


def test_turn_bakes_and_recalls_over_http(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "alex"})
    status, data = _json(
        "POST", base_url + "/api/turn", {"text": "What is my favorite color?", "user": "alex"}
    )
    assert status == 200
    assert "teal" in data["reply"].lower()
    assert data["introspect"]["source_count"] >= 1


def test_ingest_over_http(mock_config: Config, tmp_path: Path) -> None:
    import dataclasses
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Topic\nThe answer is 42.\n", encoding="utf-8")
    brain = Mimir(dataclasses.replace(mock_config, documents_folder=str(docs)))
    server = create_server(brain, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # a path inside the configured folder → ingested
        status, data = _json("POST", base + "/api/ingest", {"path": str(docs / "note.md")})
        assert status == 200 and data["chunks_written"] >= 1
        # a path OUTSIDE the folder → refused (path-traversal guard on the exposed endpoint)
        outside = tmp_path / "secret.md"
        outside.write_text("private", encoding="utf-8")
        status, _ = _json("POST", base + "/api/ingest", {"path": str(outside)})
        assert status == 400
    finally:
        server.shutdown()
        brain.close()


def test_mind_endpoint_reports_state(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "alex"})
    status, data = _json("GET", base_url + "/api/mind")
    assert status == 200
    assert data["stats"]["total"] >= 1
    assert isinstance(data["anchors"], dict)
    # turn 1 seeds a self-model, but off the hot path now (the endpoint no longer blocks on the
    # burst) — so poll until it settles instead of assuming it's done when /api/turn returns.
    self_model = data["self_model"]
    for _ in range(50):
        if self_model is not None:
            break
        time.sleep(0.1)
        self_model = _json("GET", base_url + "/api/mind")[1]["self_model"]
    assert self_model is not None


def test_notebooks_endpoint_returns_the_list_shape(base_url: str) -> None:
    # The read endpoint that makes notebooks visible to the UI/integrators (the gap a live API test
    # found). Empty to start; the facade-level content coverage lives in test_notebook.py.
    status, data = _json("GET", base_url + "/api/notebooks")
    assert status == 200 and data["notebooks"] == []


def test_memories_browser_lists_and_searches(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "alex"})

    status, data = _json("GET", base_url + "/api/memories?kind=memory")
    assert status == 200
    assert any("teal" in m["text"].lower() for m in data["memories"])
    assert all("evidence_tier" in m and "provenance" in m for m in data["memories"])

    _, hit = _json("GET", base_url + "/api/memories?kind=memory&q=teal")
    assert len(hit["memories"]) >= 1
    _, miss = _json("GET", base_url + "/api/memories?kind=memory&q=zzznope")
    assert miss["memories"] == []


def test_graph_endpoint_lists_connections(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "alex"})
    status, data = _json("GET", base_url + "/api/graph")
    assert status == 200
    assert any(t["object"].lower() == "teal" for t in data["triples"])
    # mind stats expose the connection count
    _, mind = _json("GET", base_url + "/api/mind")
    assert mind["stats"]["triples"] >= 1


def test_sleep_endpoint_runs_consolidation(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "g"})
    status, data = _json("POST", base_url + "/api/sleep", {})
    assert status == 200
    assert {"deduped", "decayed", "archived", "contradictions_resolved", "total_changes"} <= set(
        data
    )


def test_fleet_scan_and_report(base_url: str) -> None:
    status, scanned = _json("POST", base_url + "/api/fleet/scan", {})
    assert status == 200
    assert scanned["models"] >= 1  # the mock provider advertises a few models
    _, report = _json("GET", base_url + "/api/fleet")
    assert report["models"] >= 1
    assert "by_node" in report


def test_benchmark_runs_async_with_progress(base_url: str) -> None:
    import time

    _json("POST", base_url + "/api/fleet/scan", {})  # catalogue the mock models
    status, started = _json("POST", base_url + "/api/fleet/benchmark", {})
    assert status == 200 and started.get("started") is True  # returns immediately, not blocking

    st: dict = {}
    for _ in range(100):  # poll the lock-free status until done (mock benchmark is fast)
        _, st = _json("GET", base_url + "/api/fleet/benchmark/status")
        if st.get("done") or not st.get("running"):
            break
        time.sleep(0.1)
    # The async path completes cleanly and reports done (mock families aren't on the approved
    # list, so the default benchmark scores 0 of them — real fleets score normally).
    assert st.get("done") is True
    assert "error" not in st


def test_model_pool_lists_and_toggles(base_url: str) -> None:
    _json("POST", base_url + "/api/fleet/scan", {})  # catalogue the mock models
    status, pool = _json("GET", base_url + "/api/fleet/pool")
    assert status == 200
    assert "lan_backend" in pool and "auto_roles" in pool
    models = {m["model"]: m for m in pool["models"]}
    assert "mock-a" in models and models["mock-a"]["enabled"] is True

    # Disable a model; the pool reflects the user's veto.
    s2, r2 = _json("POST", base_url + "/api/fleet/model", {"model": "mock-a", "enabled": False})
    assert s2 == 200 and r2["enabled"] is False
    _, pool2 = _json("GET", base_url + "/api/fleet/pool")
    assert {m["model"]: m for m in pool2["models"]}["mock-a"]["enabled"] is False

    # A model name is required.
    s3, _ = _json("POST", base_url + "/api/fleet/model", {"enabled": True})
    assert s3 == 400


def test_procedures_teach_and_list(base_url: str) -> None:
    status, taught = _json(
        "POST",
        base_url + "/api/procedures",
        {"trigger": "user asks for a recap", "procedure": "summarize in 3 bullets"},
    )
    assert status == 200
    assert taught["trigger"] == "user asks for a recap"
    _, listing = _json("GET", base_url + "/api/procedures")
    assert any(p["procedure"] == "summarize in 3 bullets" for p in listing["procedures"])


def test_council_endpoint_returns_positions_and_verdict(base_url: str) -> None:
    status, data = _json(
        "POST", base_url + "/api/council", {"question": "Breadth or depth first?"}
    )
    assert status == 200
    assert data["verdict"]
    assert len(data["positions"]) >= 3
    assert all({"persona", "model", "text"} <= set(p) for p in data["positions"])


def _poll_tourney(base_url: str, want: str = "awaiting_veto") -> dict:
    import time
    st: dict = {}
    for _ in range(150):
        _, st = _json("GET", base_url + "/api/fleet/tournament/status")
        if st.get("phase") in (want, "error", "done") or not st.get("active"):
            break
        time.sleep(0.1)
    return st


def test_tournament_runs_rounds_with_human_veto(base_url: str) -> None:
    _json("POST", base_url + "/api/fleet/scan", {})  # catalogue the mock models

    # Round 1: triage — starts in the background, narrows the field, parks for the user's veto.
    status, started = _json("POST", base_url + "/api/fleet/tournament/start", {})
    assert status == 200 and started.get("started") is True
    st = _poll_tourney(base_url)
    assert st["round"] == 1 and st["phase"] == "awaiting_veto"
    triaged = {r["model"] for r in st["results"]}
    assert {"mock-a", "mock-b", "mock-c"} <= triaged  # the whole fleet triaged (approved-blind)

    # FIGHT → Round 2 (gauntlet) on the survivors the user kept; mock-c is vetoed out.
    s2, adv = _json("POST", base_url + "/api/fleet/tournament/advance",
                    {"keep": ["mock-a", "mock-b"]})
    assert s2 == 200 and adv["advanced"] is True
    st2 = _poll_tourney(base_url)
    assert st2["round"] == 2 and st2["phase"] == "awaiting_veto"
    assert {r["model"] for r in st2["results"]} == {"mock-a", "mock-b"}  # only the survivors

    # FIGHT → Round 3 (finals): compute champions among the finalists; the tournament is done.
    s3, adv3 = _json("POST", base_url + "/api/fleet/tournament/advance", {"keep": ["mock-a"]})
    assert s3 == 200 and adv3["advanced"] is True
    _, st3 = _json("GET", base_url + "/api/fleet/tournament/status")
    assert st3["round"] == 3 and st3["phase"] == "done"
    assert st3["finalists"] == ["mock-a"]

    # Apply is idempotent and returns the (possibly empty) role→model map without erroring.
    s4, applied = _json("POST", base_url + "/api/fleet/tournament/apply", {})
    assert s4 == 200 and "applied" in applied

    # Advancing with no kept models is a clean 200 with a reason, not a crash.
    _json("POST", base_url + "/api/fleet/tournament/start", {})
    _poll_tourney(base_url)
    s5, none_kept = _json("POST", base_url + "/api/fleet/tournament/advance", {"keep": []})
    assert s5 == 200 and none_kept["advanced"] is False


def test_memories_bad_kind_is_4xx(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/memories?kind=bogus")
    assert status == 400
    assert "error" in data


def test_bad_requests_fail_with_4xx(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/turn", {})  # missing text
    assert status == 400
    assert "error" in data

    status2, data2 = _json("POST", base_url + "/api/ingest", {"path": "/no/such/file.md"})
    assert status2 == 400
    assert "error" in data2


def test_onboarding_flow_persists_and_reports(base_url: str) -> None:
    # The seeding interview: starts empty/incomplete, an answer persists + lands in the profile, and
    # completion flips only once every question is answered.
    status, data = _json("GET", base_url + "/api/onboarding")
    assert status == 200
    assert data["started"] is False and data["complete"] is False
    assert any(q["key"] == "assistant_name" for q in data["profile"])
    n_questions = len(data["profile"])

    s2, after = _json("POST", base_url + "/api/onboarding/answer",
                      {"key": "assistant_name", "answer": "Mimir"})
    assert s2 == 200 and after["started"] is True
    answers = {q["key"]: q["answer"] for q in after["profile"]}
    assert answers["assistant_name"] == "Mimir"
    assert len(after["pending"]) == n_questions - 1

    # Answer the rest → complete.
    for q in after["pending"]:
        _json("POST", base_url + "/api/onboarding/answer", {"key": q["key"], "answer": "x"})
    _, done = _json("GET", base_url + "/api/onboarding")
    assert done["complete"] is True and done["pending"] == []


def test_onboarding_answer_requires_a_key(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/onboarding/answer", {"answer": "x"})
    assert status == 400 and "error" in data


def test_page_includes_interview_strip_and_profile_tab(base_url: str) -> None:
    status, html = _get_html(base_url + "/")
    assert status == 200
    assert 'id="interviewStrip"' in html
    assert 'data-tab="profile"' in html


def test_onboarding_capture_is_lockfree_during_long_ops(mock_config: Config) -> None:
    # The interview runs DURING the qualifying tournament, which holds brain_lock for whole rounds.
    # Capturing an answer must NOT take that lock, or "Next" hangs until the round ends. We simulate
    # a long round by holding the lock here and asserting the POST still returns promptly.
    brain = Mimir(mock_config)
    server = create_server(brain, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with server.brain_lock:  # stand in for an in-progress tournament round
            status, data = _json("POST", base + "/api/onboarding/answer",
                                 {"key": "pets", "answer": "a dog named Rex"})
        assert status == 200
        answers = {q["key"]: q["answer"] for q in data["profile"]}
        assert answers["pets"] == "a dog named Rex"
    finally:
        server.shutdown()
        brain.close()


def test_set_role_pins_a_model(base_url: str) -> None:
    # Manual override: pick a model for a role and it sticks (pinned, out of the auto set).
    status, data = _json("POST", base_url + "/api/fleet/role",
                        {"role": "chat", "model": "mock-b"})
    assert status == 200
    assert data["roles"]["chat"] == "mock-b"
    _, pool = _json("GET", base_url + "/api/fleet/pool")
    assert pool["active_roles"]["chat"] == "mock-b"
    assert "chat" not in pool["auto_roles"]
    assert "mock-b" in pool["available"]  # the dropdown's option source


def test_set_role_requires_role_and_model(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/fleet/role", {"role": "chat"})
    assert status == 400 and "error" in data


def test_fleet_and_models_tabs_are_merged(base_url: str) -> None:
    status, html = _get_html(base_url + "/")
    assert status == 200
    assert 'id="roleAssign"' in html          # role assignment lives in the merged Fleet tab
    assert 'data-tab="models"' not in html     # the separate Models tab is gone


def test_history_endpoint_and_restore(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "hello there", "user": "operator"})
    status, data = _json("GET", base_url + "/api/history")
    assert status == 200
    assert any("hello there" in t["user_text"] for t in data["turns"])
    _, html = _get_html(base_url + "/")
    assert 'id="sessionBar"' in html  # the conversation dropdown restores into the chat on load


def test_sessions_list_and_switch(base_url: str) -> None:
    # A turn creates the current session; listing shows it with a summary line.
    _json("POST", base_url + "/api/turn", {"text": "teal is my favorite", "user": "operator"})
    status, data = _json("GET", base_url + "/api/sessions")
    assert status == 200 and data["sessions"]
    first = data["sessions"][0]
    assert "teal is my favorite" in (first["summary"] or "") and first["count"] >= 1

    # Start a new conversation, then a turn lands in a DISTINCT session.
    s2, new = _json("POST", base_url + "/api/session", {"action": "new"})
    assert s2 == 200 and new["session_id"]
    _json("POST", base_url + "/api/turn", {"text": "different topic now", "user": "operator"})
    _, after = _json("GET", base_url + "/api/sessions")
    assert len(after["sessions"]) >= 2

    # Restoring the original session returns only its turns.
    sid = first["session_id"]
    _json("POST", base_url + "/api/session", {"action": "resume", "session_id": sid})
    _, hist = _json("GET", base_url + f"/api/history?session={sid}")
    assert all("different topic" not in t["user_text"] for t in hist["turns"])


def test_session_action_validates(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/session", {"action": "bogus"})
    assert status == 400 and "error" in data


def test_graph_map_and_memory_edit_delete(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "the gate is near the barn", "user": "operator"})
    status, gm = _json("GET", base_url + "/api/graph/map")
    assert status == 200 and "nodes" in gm
    mems = [n for n in gm["nodes"] if n["type"] == "memory"]
    assert mems
    mid = mems[0]["mid"]
    su, up = _json("POST", base_url + "/api/memory",
                   {"action": "update", "id": mid, "text": "edited via graph", "salience": 2.0})
    assert su == 200 and up["memory"]["text"] == "edited via graph"
    sd, dl = _json("POST", base_url + "/api/memory", {"action": "delete", "id": mid})
    assert sd == 200 and dl["deleted"] == mid


def test_page_has_graph_view(base_url: str) -> None:
    _, html = _get_html(base_url + "/")
    assert 'id="graphSvg"' in html and 'id="graphToggle"' in html


def test_wiki_status_endpoint(base_url: str) -> None:
    # The mock config has no [wiki] block → disabled (the status line shows "not configured").
    status, data = _json("GET", base_url + "/api/wiki/status")
    assert status == 200 and data == {"enabled": False}
    _, html = _get_html(base_url + "/")
    assert 'id="wikiStatus"' in html  # the Docs-tab status line


def test_sleep_status_endpoint(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/sleep/status")
    assert status == 200
    assert data["enabled"] is True  # default-on window scheduler
    assert data["window_start"] and data["window_end"]
    assert "in_window" in data
    _, html = _get_html(base_url + "/")
    assert 'id="sleepStatus"' in html  # the Mind-tab sleep panel


def test_settings_endpoints(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/settings")
    assert status == 200 and "sleep_window_start" in data
    status, updated = _json("POST", base_url + "/api/settings",
                            {"settings": {"sleep_window_start": "01:15"}})
    assert status == 200 and updated["sleep_window_start"] == "01:15"
    # a bad value comes back as a clean error, not a 500 stack
    status, err = _json("POST", base_url + "/api/settings",
                        {"settings": {"timezone": "Not/AZone"}})
    assert status >= 400 and "error" in err


def test_timezones_endpoint(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/timezones")
    assert status == 200 and isinstance(data["zones"], list) and len(data["zones"]) > 0


def test_sleep_tab_present(base_url: str) -> None:
    _, html = _get_html(base_url + "/")
    assert 'data-tab="sleep"' in html and 'id="setTz"' in html


def test_deliberate_endpoint(base_url: str) -> None:
    status, data = _json("POST", base_url + "/api/deliberate/run")
    assert status == 200 and "ran" in data  # no conflicts seeded → empty but well-formed
    _, html = _get_html(base_url + "/")
    assert 'id="delibBtn"' in html and 'id="setDeliberate"' in html


def test_forum_endpoints(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/forum")
    assert status == 200 and data["threads"] == []  # nothing deliberated yet
    # "ask the council" creates a thread (mock model), then it's listed + fetchable
    status, asked = _json("POST", base_url + "/api/forum", {"action": "ask", "question": "Why?"})
    assert status == 200 and asked["thread_id"]
    _, listed = _json("GET", base_url + "/api/forum")
    assert len(listed["threads"]) == 1
    status, thread = _json("GET", base_url + "/api/forum/thread?id=" + str(asked["thread_id"]))
    assert status == 200 and thread["question"] == "Why?" and thread["posts"]
    _, html = _get_html(base_url + "/")
    assert 'id="forumToggle"' in html and 'id="forumView"' in html


def test_mind_exposes_recent_errors(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/mind")
    assert status == 200 and "recent_errors" in data and "error_counts" in data
    _, html = _get_html(base_url + "/")
    assert 'id="systemHealth"' in html  # the Mind-tab system-health readout


@pytest.fixture
def secured_url(mock_config: Config) -> Iterator[str]:
    mock_config.api_token = "s3cret"
    mock_config.cors_origins = ["https://avatar.local"]
    mock_config.secure_ui = True  # enforce the token even locally (tests connect via 127.0.0.1)
    brain = Mimir(mock_config)
    server = create_server(brain, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        brain.close()


@pytest.fixture
def token_local_url(mock_config: Config) -> Iterator[str]:
    # A token is set, but secure_ui is left off (the default) — so the LOCAL UI is exempt.
    mock_config.api_token = "s3cret"
    brain = Mimir(mock_config)
    server = create_server(brain, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        brain.close()


def test_local_ui_exempt_from_token_by_default(token_local_url: str) -> None:
    # Token set + secure_ui off → a same-machine request needs no token (first run isn't blocked).
    assert _req("GET", token_local_url + "/api/state")[0] == 200
    assert _json("POST", token_local_url + "/api/turn",
                 {"text": "hi", "user": "alex"})[0] == 200


def _req(method: str, url: str, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers)


def test_api_token_gates_api_but_not_the_page(secured_url: str) -> None:
    assert _req("GET", secured_url + "/api/state")[0] == 401                       # no token
    assert _req("GET", secured_url + "/api/state",
                {"Authorization": "Bearer wrong"})[0] == 401                       # bad token
    assert _req("GET", secured_url + "/api/state",
                {"Authorization": "Bearer s3cret"})[0] == 200                      # good token
    assert _req("GET", secured_url + "/")[0] == 200          # the shell stays open (UI prompts)


def test_cors_preflight_and_headers(secured_url: str) -> None:
    # preflight is unauthenticated (browsers don't send Authorization on OPTIONS)
    status, headers = _req("OPTIONS", secured_url + "/api/turn",
                           {"Origin": "https://avatar.local"})
    assert status == 204
    assert headers.get("Access-Control-Allow-Origin") == "https://avatar.local"
    assert "Authorization" in headers.get("Access-Control-Allow-Headers", "")
    # an authorized GET from the allowed origin echoes the CORS header
    _, headers = _req("GET", secured_url + "/api/state",
                      {"Origin": "https://avatar.local", "Authorization": "Bearer s3cret"})
    assert headers.get("Access-Control-Allow-Origin") == "https://avatar.local"
    # a disallowed origin gets no CORS header
    _, headers = _req("OPTIONS", secured_url + "/api/turn", {"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in headers


def test_health_is_instant_unauthed_and_lockfree(base_url: str) -> None:
    status, data = _json("GET", base_url + "/api/health")
    assert status == 200 and data["ok"] is True
    assert "busy" in data and "embed_mode" in data


def test_health_works_even_when_token_is_set(secured_url: str) -> None:
    # liveness must not require the token (monitors/peers probe it without credentials)
    assert _req("GET", secured_url + "/api/health")[0] == 200
    assert _req("GET", secured_url + "/api/state")[0] == 401   # but everything else still gated
