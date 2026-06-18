"""Library Phase 1b — the cited claims spine: extraction, retrieval, idle indexing, and the
cited Library section in a turn. (Data foundation is covered by test_library_storage.py.)"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mimir.brain import Mimir
from mimir.cognition.library import (
    ScoredClaim,
    extract_claims,
    render_claims,
    retrieve_claims,
)
from mimir.config import Config
from mimir.storage.models import LibraryClaim
from mimir.storage.repo import (
    browse_memories,
    claims_for_document,
    list_library_claims,
    list_library_documents,
    list_library_pages,
)


def test_extract_claims_parses_and_degrades() -> None:
    out = extract_claims(lambda m: '{"claims": ["Bees make honey", "A hive has one queen"]}', "...")
    assert out == ["Bees make honey", "A hive has one queen"]
    assert extract_claims(lambda m: "not json", "...") == []   # lenient: nothing parseable → []


def test_retrieve_and_render_claims_cite_sources() -> None:
    claims = [
        LibraryClaim(document_id=1, text="Garlic is planted in October", locator="p.2"),
        LibraryClaim(document_id=2, text="Torque is rotational force", locator="p.9"),
    ]
    hits = retrieve_claims("when do I plant garlic", None, claims, top_k=2)
    assert hits and hits[0].claim.text.startswith("Garlic")     # on-topic claim ranks first
    assert all(isinstance(h, ScoredClaim) for h in hits)
    rendered = render_claims(hits, {1: "Gardening", 2: "Cars"})
    assert "[Gardening, p.2]" in rendered                       # every fact carries its citation


def _libbrain(mock_config: Config, tmp_path) -> Mimir:
    # Source of truth = the documents folder; composites written to a separate library folder.
    cfg = dataclasses.replace(
        mock_config,
        documents_folder=str(tmp_path / "documents"),
        library_folder=str(tmp_path / "library"),
    )
    return Mimir(cfg)


def test_idle_extraction_records_document_and_cited_claims(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "garden.md").write_text(
            "# Garlic\n\nGarlic is planted in October. Harvest garlic in July.")
        report = brain.ingest_pending_library()
        assert "garden.md" in report["documents"] and report["claims"] >= 1

        doc = list_library_documents(brain._storage)[0]
        assert doc.filename == "garden.md" and doc.size_bytes > 0   # exact filename + size tracked
        claims = claims_for_document(brain._storage, doc.id)
        assert claims and all(c.locator for c in claims)            # each claim cites a locator
        assert any("Garlic" in c.text for c in claims)

        # Unchanged re-scan is a no-op; removing the file drops the doc + cascades its claims.
        assert brain.ingest_pending_library()["claims"] == 0
        (folder / "garden.md").unlink()
        assert brain.ingest_pending_library()["dropped"] == 1
        assert list_library_documents(brain._storage) == []
        assert list_library_claims(brain._storage) == []
    finally:
        brain.close()


def test_library_claims_surface_cited_in_a_turn(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("Beekeepers inspect hives. Each hive has a single queen.")
        brain.ingest_pending_library()
        result = brain.turn("tell me about hives")
        prompt = result.context.prompt
        assert "hive" in prompt.lower() and "[bees" in prompt    # cited library claim in the prompt
    finally:
        brain.close()


def test_idle_compiles_a_linked_composite_with_citations(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "garden.md").write_text(
            "# Garlic\n\nGarlic is planted in October. Harvest garlic in July.")
        report = brain.ingest_pending_library()
        assert report["composed"] >= 1

        overview = brain.library_overview()
        page = overview["pages"][0]
        assert Path(page["path"]).is_file()              # the composite MD is on disk
        full = brain.library_page(page["id"])
        assert full["markdown"]                          # full composite loaded on demand
        assert full["citations"] and all(c["title"] for c in full["citations"])  # traces to source

        # A verbatim source is fetchable for quoting/checking.
        doc = overview["documents"][0]
        assert "Garlic" in brain.library_source(doc["id"])["text"]
    finally:
        brain.close()


def test_hand_edited_composite_is_not_clobbered(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "note.md").write_text("# Note\n\nA fact about the farm.")
        brain.ingest_pending_library()
        page_path = Path(brain.library_overview()["pages"][0]["path"])
        page_path.write_text("# Note\n\nMY HAND-EDITED VERSION.")   # user edits the composite
        brain.ingest_pending_library(force=True)                    # re-derive attempt
        assert "HAND-EDITED" in page_path.read_text()               # respected, not clobbered
    finally:
        brain.close()


def test_no_source_folder_is_a_quiet_noop(brain: Mimir) -> None:
    assert brain.ingest_pending_library() == {
        "folder": None, "documents": [], "claims": 0, "composed": 0, "dropped": 0}
    assert brain._library_gist("anything", None) == (None, [], 0)


def test_loaded_page_is_pinned_into_the_next_turn(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fences.md").write_text("# Fences\n\nThe north fence is cedar.")
        brain.ingest_pending_library()
        page_id = brain.library_overview()["pages"][0]["id"]
        # A query that wouldn't surface the gist on its own; the pinned page is loaded regardless.
        result = brain.turn("what's the weather", loaded_pages=[page_id])
        prompt = result.context.prompt
        assert "Full pages you've loaded" in prompt and "Fences" in prompt
    finally:
        brain.close()


def test_deep_read_pulls_full_page_for_the_matching_doc(mock_config: Config, tmp_path) -> None:
    """Deep read injects the FULL composite of the doc the surfaced claims belong to — without the
    user having to pin it by hand. Off by default, so the normal turn stays the cheap cited gist."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "hives.md").write_text("# Hives\n\nEach hive has one queen. Bees make honey.")
        brain.ingest_pending_library()
        q = "tell me about hives and queens"
        # Default: cited gist only, no full page injected.
        assert "Full pages you've loaded" not in brain.turn(q).context.prompt
        # Deep read on: the matching page's full Markdown is pulled in automatically.
        assert "Full pages you've loaded" in brain.turn(q, deep_read=True).context.prompt
    finally:
        brain.close()


def test_turn_surfaces_library_sources_for_load_chips(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "hives.md").write_text("# Hives\n\nEach hive has one queen. Bees make honey.")
        brain.ingest_pending_library()
        result = brain.turn("tell me about hives and queens")
        assert result.library_sources                       # the answer drew on a library page
        assert all("page_id" in s and "title" in s for s in result.library_sources)
    finally:
        brain.close()


def test_model_fetch_opens_a_page_and_reanswers(mock_config: Config, tmp_path) -> None:
    cfg = dataclasses.replace(
        mock_config, documents_folder=str(tmp_path / "documents"),
        library_folder=str(tmp_path / "library"), library_model_fetch=True)
    brain = Mimir(cfg)
    try:
        folder = Path(cfg.documents_folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fence.md").write_text("# Fence\n\nThe north fence is cedar.")
        brain.ingest_pending_library()
        page_id = brain.library_overview()["pages"][0]["id"]

        calls: list = []
        real_chat = brain._model.chat

        def scripted(role, messages, *a, **k):
            if role == "chat":
                calls.append(messages)
                return f"<FETCH id={page_id}>" if len(calls) == 1 else "The fence is cedar."
            return real_chat(role, messages, *a, **k)

        brain._model.chat = scripted
        result = brain.turn("what is the fence made of")
        assert "FETCH" not in result.reply and "cedar" in result.reply.lower()
        # the second (re-answer) pass had the loaded page detail in its system prompt
        assert "Full pages you've loaded" in calls[1][0]["content"]
    finally:
        brain.close()


def test_model_fetch_intercepts_marker_when_streaming(mock_config: Config, tmp_path) -> None:
    """Streaming Phase-2 fetch: the model opens with the marker, which is intercepted (never shown),
    the page loads, and the FINAL answer streams with the page in context."""
    cfg = dataclasses.replace(
        mock_config, documents_folder=str(tmp_path / "documents"),
        library_folder=str(tmp_path / "library"), library_model_fetch=True)
    brain = Mimir(cfg)
    try:
        folder = Path(cfg.documents_folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fence.md").write_text("# Fence\n\nThe north fence is cedar.")
        brain.ingest_pending_library()
        page_id = brain.library_overview()["pages"][0]["id"]

        systems: list[str] = []

        def scripted_stream(role, messages, *a, **k):
            systems.append(messages[0]["content"])
            if len(systems) == 1:
                yield f"<FETCH id={page_id}>"          # first pass: just the marker
            else:
                yield from ["The ", "fence ", "is ", "cedar."]  # second pass: the real answer

        brain._model.chat_stream = scripted_stream
        out = "".join(brain.turn_stream("what is the fence made of"))
        assert "FETCH" not in out and "cedar" in out.lower()   # marker never reached the user
        assert "Full pages you've loaded" in systems[1]        # 2nd pass got the page detail
    finally:
        brain.close()


def test_forget_document_purges_every_layer(mock_config: Config, tmp_path) -> None:
    """The Library 'delete' path: forget a doc → its memory chunks, library doc + claims, composite
    page (row + MD file), wiki ledger entry, and (with delete_file) the source file are all gone."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / "manual.md"
        f.write_text("# Safety\n\nReport unsafe work to a supervisor. Wear a harness above 8 feet.")
        brain.ingest_pending_documents()   # document-tier memory chunks + wiki ledger
        brain.ingest_pending_library()     # library doc + cited claims + composite page
        s = brain._storage
        assert any(m.source for m in browse_memories(s))            # doc chunks present
        assert list_library_documents(s) and list_library_claims(s)
        page_path = Path(list_library_pages(s)[0].path)
        assert page_path.is_file()                                  # composite MD on disk

        res = brain.forget_document(str(f), delete_file=True)
        assert res["memory_chunks"] >= 1 and res["library_doc"] == 1 and res["file_deleted"]
        assert not list_library_documents(s) and not list_library_claims(s)
        assert not list_library_pages(s) and not page_path.exists()  # composite row + MD gone
        assert not any(m.source for m in browse_memories(s))         # doc chunks gone
        assert brain.documents() == [] and not f.exists()           # ledger + source file gone
    finally:
        brain.close()


def test_deleting_the_file_self_cleans_on_the_next_scan(mock_config: Config, tmp_path) -> None:
    """The inverse direction: just delete the source file, and an idle scan eventually forgets it
    across every layer (no explicit delete call needed)."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / "gone.md"
        f.write_text("# Topic\n\nA fact that will be deleted later.")
        brain.ingest_pending_documents()
        brain.ingest_pending_library()
        assert brain.documents() and list_library_documents(brain._storage)

        f.unlink()                                       # user removes the file directly
        report = brain.ingest_pending_documents()        # the idle scan reconciles
        assert "gone.md" in report["forgotten"]
        assert brain.documents() == [] and not list_library_documents(brain._storage)
        assert not list_library_claims(brain._storage)
    finally:
        brain.close()


def test_disabled_document_drops_out_of_recall(mock_config: Config, tmp_path) -> None:
    """A document toggled 'not in context' stops contributing cited claims to a turn, but its data
    is kept (re-enabling restores it). The '[bees' citation marker only comes from its claims."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("# Bees\n\nEach hive has a single queen. Bees make honey.")
        brain.ingest_pending_documents()
        brain.ingest_pending_library()
        q = "tell me about hives and queens"
        assert "[bees" in brain.turn(q).context.prompt              # cited while enabled

        brain.set_document_enabled(str(folder / "bees.md"), False)  # toggle OFF
        assert "[bees" not in brain.turn(q).context.prompt          # no library claims surface
        assert list_library_claims(brain._storage)                  # data kept, not deleted

        brain.set_document_enabled(str(folder / "bees.md"), True)   # back ON
        assert "[bees" in brain.turn(q).context.prompt
    finally:
        brain.close()


def test_disabled_document_chunks_excluded_at_load(mock_config: Config, tmp_path) -> None:
    """The per-doc toggle excludes the document's memory chunks at the SQL load layer (the speed
    lever for a big library) — not just the library claims."""
    from mimir.storage.models import MemoryKind
    from mimir.storage.repo import list_memories
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("# Bees\n\nEach hive has a single queen. Bees make honey.")
        brain.ingest_pending_documents()
        src = str((folder / "bees.md").resolve())
        s = brain._storage
        assert any(m.source == src for m in list_memories(s, kind=MemoryKind.MEMORY))
        kept = list_memories(s, kind=MemoryKind.MEMORY, exclude_sources={src})
        assert all(m.source != src for m in kept)                  # the doc's chunks aren't loaded
    finally:
        brain.close()


def test_layer_toggles_skip_whole_sections(mock_config: Config, tmp_path) -> None:
    """The per-turn chat toggles drop a whole layer: include_library=False removes the Library
    section + document chunks; include_memory=False removes personal memories."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("# Bees\n\nEach hive has a single queen.")
        brain.ingest_pending_documents()
        brain.ingest_pending_library()
        assert "[bees" in brain.turn("hives and queens").context.prompt
        # Library layer off → no cited claims surface this turn.
        assert "[bees" not in brain.turn("hives and queens", include_library=False).context.prompt
    finally:
        brain.close()


def test_ingest_records_per_document_index_time(mock_config: Config, tmp_path) -> None:
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bees.md").write_text("# Bees\n\nEach hive has a single queen. Bees make honey.")
        brain.ingest_pending_library()
        doc = brain.library_overview()["documents"][0]
        assert doc["index_seconds"] is not None and doc["index_seconds"] >= 0
        assert doc["enabled"] is True
    finally:
        brain.close()


def test_draft_rag_folds_in_memory_surfaced_by_the_draft(mock_config: Config) -> None:
    """Draft-RAG (two-pass): a short draft answer re-retrieves memory and folds the new hits into
    the recall set — surfacing a memory the user's literal wording started without."""
    import dataclasses
    from types import SimpleNamespace

    from mimir.storage.models import EvidenceTier, Memory
    from mimir.storage.repo import list_memories, save_memory

    brain = Mimir(dataclasses.replace(mock_config, draft_rag_enabled=True))
    try:
        save_memory(brain._storage, Memory(
            text="The latch is stainless steel.",
            evidence_tier=EvidenceTier.STATED_BY_PRIMARY_USER))
        cands = list_memories(brain._storage)
        # The draft names the latch; re-retrieval against it pulls that memory in.
        brain._model.chat = lambda role, messages, **k: "latch stainless steel"
        merged = brain._draft_rag(SimpleNamespace(prompt="sys"), cands, [], None, None, "the gate")
        assert any("latch" in s.memory.text.lower() for s in merged)
        # Fail-soft: a draft error leaves the input recall untouched.
        def boom(*a, **k):
            raise RuntimeError("model down")
        brain._model.chat = boom
        assert brain._draft_rag(SimpleNamespace(prompt="sys"), cands, [], None, None, "x") == []
    finally:
        brain.close()


def test_citation_guard_flags_an_invented_source_in_a_turn(mock_config: Config, tmp_path) -> None:
    """End-to-end: a reply that cites a document the system doesn't hold gets a fail-loud note; a
    reply citing a real held document does not."""
    brain = _libbrain(mock_config, tmp_path)
    try:
        folder = brain._library_source_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Servus OHS Manual.md").write_text("# Safety\n\nReport unsafe work to a boss.")
        brain.ingest_pending_library()
        real_chat = brain._model.chat

        def scripted(role, messages, *a, **k):
            if role == "chat":
                return scripted.reply
            return real_chat(role, messages, *a, **k)

        brain._model.chat = scripted
        scripted.reply = "Follow the standard [National Fire Code 2020] for this."
        assert "⚠ Unverified citation" in brain.turn("how do I handle a fire?").reply
        scripted.reply = "Report it to your supervisor [Servus OHS Manual, Safety]."
        assert "⚠ Unverified citation" not in brain.turn("who do I tell?").reply
    finally:
        brain.close()
