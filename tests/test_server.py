"""Executable spec for the reference web server (the human interaction surface)."""

from __future__ import annotations

import json
import threading
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
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "greg"})
    text, introspect = _sse(
        base_url + "/api/turn/stream", {"text": "What is my favorite color?", "user": "greg"}
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
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "greg"})
    status, data = _json(
        "POST", base_url + "/api/turn", {"text": "What is my favorite color?", "user": "greg"}
    )
    assert status == 200
    assert "teal" in data["reply"].lower()
    assert data["introspect"]["source_count"] >= 1


def test_ingest_over_http(base_url: str, tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("# Topic\nThe answer is 42.\n", encoding="utf-8")
    status, data = _json("POST", base_url + "/api/ingest", {"path": str(doc)})
    assert status == 200
    assert data["chunks_written"] >= 1


def test_mind_endpoint_reports_state(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "greg"})
    status, data = _json("GET", base_url + "/api/mind")
    assert status == 200
    assert data["stats"]["total"] >= 1
    assert isinstance(data["anchors"], dict)
    # turn 1 seeds a self-model (the /api/turn handler waits for background to settle)
    assert data["self_model"] is not None


def test_memories_browser_lists_and_searches(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "greg"})

    status, data = _json("GET", base_url + "/api/memories?kind=memory")
    assert status == 200
    assert any("teal" in m["text"].lower() for m in data["memories"])
    assert all("evidence_tier" in m and "provenance" in m for m in data["memories"])

    _, hit = _json("GET", base_url + "/api/memories?kind=memory&q=teal")
    assert len(hit["memories"]) >= 1
    _, miss = _json("GET", base_url + "/api/memories?kind=memory&q=zzznope")
    assert miss["memories"] == []


def test_graph_endpoint_lists_connections(base_url: str) -> None:
    _json("POST", base_url + "/api/turn", {"text": "My favorite color is teal.", "user": "greg"})
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
